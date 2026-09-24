#!/usr/bin/env python3
"""Re-run analyzer scans that failed while Spotify had this Mac blocked.

2026-09-24: a Spotify rate-limit penalty (caused by a flood from an outreach
audit script on this machine) blocked every Spotify Web API call from the Mac
from 6:54am to ~6:03pm. Real users' URL scans need the rig, the rig needs the
Web API, so their scans failed. Re-queueing the old job rows would not help:
the matching, the report and the results email all happen inside the original
/api/analyze-url request, which gave up after ~150s.

So this waits until Spotify answers from this Mac again, then re-submits each
scan through the live analyzer with the user's OWN stored session token —
exactly the request they made — one at a time. The normal results email goes
out. One summary push at the end, not one per attempt.

Guards:
  - never fires while Spotify still returns 429 from this machine
  - skips a scan if that user already completed the same track in the meantime
  - sequential; one Spotify probe every few minutes (no flood)
  - gives up after MAX_WAIT_H and says so, once

Run:  python3 rerun_blocked_scans.py            (waits, then runs)
      python3 rerun_blocked_scans.py --dry      (shows what it would do)
"""
import base64
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://analyze.freshlybakedstudios.com/api/analyze-url"
FAILED_JOBS = ["0465d9ec", "3cfbd8e1", "3e6e5f70", "864ebbb6"]
FAILED_SINCE = "2026-09-24T18:00:00"          # UTC
MAX_WAIT_H = 4
DRY = "--dry" in sys.argv


def env():
    e = {}
    for line in open(os.path.join(HERE, ".env")):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            e.setdefault(k, v.strip().strip('"'))
    return e


ENV = env()
SUPA = ENV["SUPABASE_URL"].strip()
SKEY = ENV["SUPABASE_SERVICE_KEY"].strip()


def log(msg):
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def supa(path):
    req = urllib.request.Request(f"{SUPA}/rest/v1/{path}",
                                 headers={"apikey": SKEY, "Authorization": f"Bearer {SKEY}"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def push(title, message):
    tok = ENV.get("PUSHOVER_APP_TOKEN") or ENV.get("PUSHOVER_TOKEN")
    user = ENV.get("PUSHOVER_USER_KEY") or ENV.get("PUSHOVER_USER")
    if not (tok and user) or DRY:
        log(f"(push) {title}: {message}")
        return
    data = urllib.parse.urlencode({"token": tok, "user": user, "title": title, "message": message}).encode()
    try:
        urllib.request.urlopen("https://api.pushover.net/1/messages.json", data=data, timeout=15)
    except Exception as e:
        log(f"push failed: {e}")


def spotify_ok(track_id):
    """One gentle probe from this machine. Returns (ok, wait_seconds)."""
    cid, sec = ENV.get("SPOTIFY_CLIENT_ID"), ENV.get("SPOTIFY_CLIENT_SECRET")
    try:
        req = urllib.request.Request(
            "https://accounts.spotify.com/api/token", data=b"grant_type=client_credentials",
            headers={"Authorization": "Basic " + base64.b64encode(f"{cid}:{sec}".encode()).decode(),
                     "Content-Type": "application/x-www-form-urlencoded"})
        token = json.load(urllib.request.urlopen(req, timeout=20))["access_token"]
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.spotify.com/v1/tracks/{track_id}?market=US",
            headers={"Authorization": f"Bearer {token}"}), timeout=20)
        return True, 0
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return False, int(e.headers.get("Retry-After", 300))
        return False, 300
    except Exception:
        return False, 300


def load_scans():
    rows = supa("analysis_jobs?select=id,created_at,user_email,spotify_url,token"
                f"&created_at=gte.{FAILED_SINCE}&status=eq.error&order=created_at")
    rows = [r for r in rows if r["id"][:8] in FAILED_JOBS and r.get("token")]
    seen, scans = set(), []
    for r in rows:                              # one re-run per user+track
        key = (r["user_email"].lower(), r["spotify_url"].split("?")[0])
        if key in seen:
            continue
        seen.add(key)
        scans.append(r)
    return scans


def already_done(scan):
    track = scan["spotify_url"].split("?")[0].split("/")[-1]
    rows = supa("analysis_jobs?select=id,status,spotify_url"
                f"&user_email=eq.{urllib.parse.quote(scan['user_email'])}"
                f"&created_at=gt.{scan['created_at']}&status=eq.complete")
    return any(track in (r.get("spotify_url") or "") for r in rows)


def submit(scan):
    body = urllib.parse.urlencode({"spotify_url": scan["spotify_url"], "token": scan["token"]}).encode()
    req = urllib.request.Request(API, data=body, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=480) as resp:
            j = json.load(resp)
            return True, j.get("job_id")
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.read()[:160].decode(errors='replace')}"
    except Exception as e:
        return False, str(e)[:160]


def main():
    scans = load_scans()
    log(f"{len(scans)} scan(s) to re-run: " +
        ", ".join(f"{s['user_email']} {s['spotify_url'].split('/')[-1][:10]}" for s in scans))
    if not scans:
        return
    probe_track = scans[0]["spotify_url"].split("?")[0].split("/")[-1]

    deadline = time.time() + MAX_WAIT_H * 3600
    while True:
        ok, wait = spotify_ok(probe_track)
        if ok:
            log("Spotify answers from this machine again")
            break
        if time.time() > deadline:
            push("⏳ Blocked scans NOT re-run",
                 f"Spotify still blocking this Mac after {MAX_WAIT_H}h. {len(scans)} scan(s) still owed.")
            return
        nap = max(120, min(wait + 15, 600))
        log(f"still blocked (Retry-After {wait}s) — next probe in {nap}s")
        if DRY:
            log("--dry: would keep waiting")
            return
        time.sleep(nap)

    if DRY:
        for s in scans:
            log(f"--dry: would re-submit {s['user_email']} {s['spotify_url']}")
        return

    time.sleep(60)            # let GEMS / the rig settle once the block lifts
    done, failed, skipped = [], [], []
    for s in scans:
        who = f"{s['user_email']} …{s['spotify_url'].split('/')[-1][:8]}"
        if already_done(s):
            skipped.append(who); log(f"skip {who}: they already re-ran it"); continue
        log(f"re-submitting {who}")
        ok, info = submit(s)
        if ok:
            done.append(who); log(f"  complete, new job {str(info)[:8]}")
        else:
            failed.append(f"{who} ({info})"); log(f"  FAILED: {info}")
        time.sleep(20)

    msg = f"re-ran {len(done)} / {len(scans)}"
    if done: msg += "\n✓ " + "\n✓ ".join(done)
    if skipped: msg += "\n↷ already done: " + ", ".join(skipped)
    if failed: msg += "\n✗ " + "\n✗ ".join(failed)
    push("✅ Blocked scans re-run" if not failed else "⚠️ Blocked scans re-run (some failed)", msg)
    log(msg)


if __name__ == "__main__":
    main()
