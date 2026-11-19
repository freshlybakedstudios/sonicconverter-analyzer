#!/usr/bin/env python3
"""
Startup script for Railway deployment.
Downloads the GEMS cache from GCS if missing OR if GCS has a newer version than
the local copy. (Railway uses a persistent volume, so without the freshness
check daily cache rebuilds never reach prod — the local file always "exists".)

2026-11-18: prefers the SLIM object, gems_universe_slim.json (tier entries cut
to listeners/followers/snapshot_date/tier and brand_affinity emptied — the only
parts the web app reads; ~2.7 GB less RAM than the full file). Falls back to
the full gems_universe.json exactly as before when the slim object is not in
GCS. Whichever file is used is handed to track_matcher via GEMS_CACHE_PATH.
"""
import os
import subprocess
import sys
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

CACHE_DIR = Path(__file__).parent / 'cache' / 'universe'
GCS_BASE = 'https://storage.googleapis.com/fbs-static-assets/'

SLIM_NAME = 'gems_universe_slim.json'
FULL_NAME = 'gems_universe.json'


def download_cache(url, path):
    """Download a GEMS universe cache object from GCS. Goes to a .part file
    first so a dropped connection never leaves a truncated cache behind with
    a fresh mtime (which the freshness check would then trust)."""
    print(f"📦 Downloading {url} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    try:
        urllib.request.urlretrieve(url, part)
        os.replace(part, path)
        print(f"✅ Cache downloaded to {path}")
    except Exception as e:
        print(f"❌ Cache download failed: {e}")
        try:
            part.unlink()
        except OSError:
            pass
        raise


def gcs_head(url):
    """HEAD a GCS object. Returns (status, epoch): status 'ok' with the
    Last-Modified header as epoch seconds (None if the header is absent),
    'missing' on 404, or 'error' on anything else (callers then fall back to
    whatever is on the local volume)."""
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as resp:
            lm = resp.headers.get('Last-Modified')
            return 'ok', (parsedate_to_datetime(lm).timestamp() if lm else None)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return 'missing', None
        print(f"⚠️  GCS HEAD {url} failed (HTTP {e.code}); using local cache if present")
        return 'error', None
    except Exception as e:
        print(f"⚠️  GCS freshness check failed for {url} (using local cache): {e}")
        return 'error', None


def ensure_fresh(name, remote_mtime):
    """Make sure cache/universe/<name> exists and is at least as new as GCS."""
    path = CACHE_DIR / name
    url = GCS_BASE + name
    if path.exists():
        local_mtime = path.stat().st_mtime
        if remote_mtime and remote_mtime > local_mtime + 1:
            print(f"🔄 {name} stale (local mtime {local_mtime:.0f} < GCS {remote_mtime:.0f}) — refreshing")
            download_cache(url, path)
        else:
            print(f"✅ {name} up to date at {path}")
    else:
        download_cache(url, path)
    return path


def pick_cache():
    """The slim object when GCS has it (or a local copy exists while GCS is
    unreachable); otherwise the full object, handled exactly as before."""
    status, slim_mtime = gcs_head(GCS_BASE + SLIM_NAME)
    slim_path = CACHE_DIR / SLIM_NAME
    if status == 'ok':
        return ensure_fresh(SLIM_NAME, slim_mtime)
    if status == 'error' and slim_path.exists():
        print(f"⚠️  GCS unreachable — using local {SLIM_NAME}")
        return slim_path
    if status == 'missing':
        print(f"ℹ️  {SLIM_NAME} not in GCS — falling back to the full {FULL_NAME}")
    _, full_mtime = gcs_head(GCS_BASE + FULL_NAME)
    return ensure_fresh(FULL_NAME, full_mtime)


def main():
    cache_path = pick_cache()
    # Inherited by the uvicorn child below; TrackMatcher() reads it.
    os.environ['GEMS_CACHE_PATH'] = str(cache_path)
    print(f"🗂️  Universe cache for this boot: {cache_path.name} "
          f"({cache_path.stat().st_size / 1e9:.2f} GB on disk)")

    port = os.getenv('PORT', '8000')
    print(f"🚀 Starting server on port {port}...")

    subprocess.run([
        sys.executable, '-m', 'uvicorn',
        'app:app',
        '--host', '0.0.0.0',
        '--port', port
    ])


if __name__ == '__main__':
    main()
