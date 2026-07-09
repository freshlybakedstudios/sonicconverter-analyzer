"""
Deal-calculator lead nurture — recovers quote-builders who didn't book.

A lead who runs the rate calculator and enters their email (to unlock pricing)
but doesn't pay is a high-intent, already-paid-for prospect. This module sends
them a warm, personalized follow-up so we convert traffic we already bought
instead of buying more.

Ships DISABLED by default: with NURTURE_ENABLED != 'true', run_nurture() emails
the OWNER a preview digest (who *would* be contacted + a sample) and never
touches a real lead. Flip NURTURE_ENABLED=true once the copy/targeting look good.

State lives in each lead's deal_leads.metadata.nurture jsonb:
    { "t1_sent_at": iso, "t2_sent_at": iso, "unsubscribed": bool }

Suppression: any email with a step='paid' row (booked) is skipped. Cart
abandoners (checkout_started but never paid) are intentionally KEPT — they're
the hottest leads.
"""
import os
import hmac
import hashlib
from datetime import datetime, timezone, timedelta

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://freshlybakedstudios.com")
PUBLIC_API_BASE = os.getenv("PUBLIC_API_BASE", "https://analyze.freshlybakedstudios.com")
OWNER_EMAIL = os.getenv("OWNER_EMAIL", "freshlybakedstudios@gmail.com")
NURTURE_FROM = os.getenv("NURTURE_FROM", "deals@freshlybakedstudios.com")
NURTURE_ENABLED = os.getenv("NURTURE_ENABLED", "false").lower() == "true"
_UNSUB_SECRET = os.getenv("NURTURE_UNSUB_SECRET", "fbs-nurture-unsub")

# Wait this long after a lead abandons before touch 1, and after touch 1 before touch 2.
TOUCH1_DELAY = timedelta(days=1)
TOUCH2_DELAY = timedelta(days=4)
LOOKBACK = timedelta(days=30)  # don't chase leads older than this

SERVICE_LABELS = {
    "fullProduction": "Full Production",
    "mixing": "Mixing",
    "mastering": "Mastering",
    "dolbyAtmos": "Dolby Atmos",
}

# Never nurture internal / test addresses (owner's own domain, placeholder domains).
SUPPRESS_DOMAINS = {
    "freshlybakedstudios.com",
    "example.com",
    "artist.com",
    "test.com",
}
# Specific internal/test addresses (owner's own inboxes on public providers).
SUPPRESS_EMAILS = {OWNER_EMAIL.strip().lower(), "freshlybakedstudios@gmail.com"}


def _is_suppressed_email(email: str) -> bool:
    e = (email or "").strip().lower()
    if "@" not in e:
        return True
    if e in SUPPRESS_EMAILS:
        return True
    return e.split("@")[-1] in SUPPRESS_DOMAINS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc)


def _parse(ts):
    """Parse a Supabase ISO timestamp to an aware datetime (best-effort)."""
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def unsub_token(email: str) -> str:
    return hmac.new(
        _UNSUB_SECRET.encode(), email.strip().lower().encode(), hashlib.sha256
    ).hexdigest()[:24]


def verify_unsub(email: str, token: str) -> bool:
    return hmac.compare_digest(unsub_token(email), (token or ""))


def _lead_view(lead: dict) -> dict:
    """Pull the display fields for a lead, with friendly fallbacks."""
    name = (lead.get("name") or "").strip()
    artist = (lead.get("artist_name") or "").strip()
    meta = lead.get("metadata") or {}
    keys = meta.get("services") or []
    labels = [SERVICE_LABELS.get(k, k) for k in keys]
    if labels:
        svc = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " + " + labels[-1]
    else:
        svc = ""  # no services stored (e.g. legacy leads) — sentence adapts
    val = meta.get("deal_value")
    # Greeting: use a name if we have a distinct one, else something warm.
    greet = name if name else (artist or "there")
    return {
        "email": lead.get("email", ""),
        "greet": greet,
        "artist": artist,
        "service_str": svc,
        "value": val,
    }


# ---------------------------------------------------------------------------
# Email content (approved copy)
# ---------------------------------------------------------------------------
def build_touch(lead: dict, touch: int):
    """Return (subject, html) for touch 1 or 2."""
    v = _lead_view(lead)
    email = v["email"]
    unsub = f"{PUBLIC_API_BASE}/api/deal/nurture/unsubscribe?e={email}&t={unsub_token(email)}"
    rates = f"{FRONTEND_URL}/rates"
    for_artist = f" for {v['artist']}" if v["artist"] else ""
    price = f" — ${v['value']:,}" if isinstance(v["value"], (int, float)) and v["value"] else ""
    svc = v["service_str"]
    # Adapt the sentence whether or not we know the specific services.
    project_phrase = f"a <strong>{svc}</strong> project" if svc else "a project"
    project_phrase2 = f"your {svc} project" if svc else "your project"

    if touch == 1:
        subject = "Your Freshly Baked quote's still warm"
        lead_line = (
            f"You started pricing out {project_phrase}"
            f"{for_artist}{price} with us — just wanted to make sure you got what you needed."
        )
        body = f"""
          <p style="color:#ccc">Hey {v['greet']},</p>
          <p style="color:#ccc">{lead_line}</p>
          <p style="color:#ccc">If you were close, you can pick up right where you left off:</p>
          <p style="text-align:center;margin:28px 0">
            <a href="{rates}" style="background:#D8E166;color:#222020;text-decoration:none;
               font-weight:bold;padding:14px 28px;border-radius:8px;display:inline-block">
               See my pricing</a>
          </p>
          <p style="color:#ccc">And if something's holding you back — budget, timeline, a question
             about the deal — just hit reply. I read these myself.</p>
          <p style="color:#ccc">— Alexander<br><span style="color:#888">Freshly Baked Studios · Brooklyn, NY</span></p>
        """
    else:
        subject = "Want me to take a listen?"
        body = f"""
          <p style="color:#ccc">Hey {v['greet']},</p>
          <p style="color:#ccc">Circling back one more time. If you've got a track that's close but not
             quite hitting, I'm happy to give it a quick listen and tell you exactly what I'd do with
             it — no charge, no strings. Sometimes it's one thing holding a record back.</p>
          <p style="color:#ccc">Just reply with a link and I'll take a pass. Or if you're ready to
             lock in {project_phrase2}{for_artist}, your quote's still here:</p>
          <p style="text-align:center;margin:28px 0">
            <a href="{rates}" style="background:#D8E166;color:#222020;text-decoration:none;
               font-weight:bold;padding:14px 28px;border-radius:8px;display:inline-block">
               Pick up my quote</a>
          </p>
          <p style="color:#ccc">— Alexander<br><span style="color:#888">Freshly Baked Studios · Brooklyn, NY</span></p>
        """

    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                max-width:560px;margin:0 auto;background:#141213;color:#eee;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:8px">
        <img src="https://storage.googleapis.com/fbs-static-assets/axd-logo.png"
             alt="Freshly Baked Studios" style="width:140px;height:auto;margin-bottom:8px">
      </div>
      {body}
      <p style="color:#555;font-size:11px;text-align:center;margin-top:28px;border-top:1px solid #2a2626;padding-top:16px">
        Freshly Baked Studios · Brooklyn, NY<br>
        You got this because you requested a quote on our site.
        <a href="{unsub}" style="color:#777">Unsubscribe</a>
      </p>
    </div>
    """
    return subject, html


def _send_email(to_email: str, subject: str, html: str) -> bool:
    api_key = os.getenv("SENDGRID_API_KEY")
    if not api_key:
        print("[nurture] SENDGRID_API_KEY not set — cannot send")
        return False
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, HtmlContent

    msg = Mail(
        from_email=NURTURE_FROM,
        to_emails=to_email,
        subject=subject,
        html_content=HtmlContent(html),
    )
    try:
        resp = SendGridAPIClient(api_key).send(msg)
        return resp.status_code in (200, 201, 202)
    except Exception as e:
        print(f"[nurture] send failed to {to_email}: {e}")
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _set_nurture_state(supabase, lead_id, patch: dict):
    """Merge a patch into deal_leads.metadata.nurture for one row."""
    row = supabase.table("deal_leads").select("metadata").eq("id", lead_id).limit(1).execute()
    meta = (row.data[0].get("metadata") or {}) if row.data else {}
    nur = meta.get("nurture") or {}
    nur.update(patch)
    meta["nurture"] = nur
    supabase.table("deal_leads").update({"metadata": meta}).eq("id", lead_id).execute()


def run_nurture(supabase, dry_run: bool = None) -> dict:
    """
    Select leads due for a follow-up and send (or preview) their email.

    dry_run defaults to (not NURTURE_ENABLED). When dry-running, no lead is
    emailed and no state is written — the owner gets a preview digest instead.
    """
    if dry_run is None:
        dry_run = not NURTURE_ENABLED
    now = _now()
    since = (now - LOOKBACK).isoformat()

    rows = (
        supabase.table("deal_leads").select("*").gte("created_at", since).execute().data
        or []
    )
    booked = {r["email"].lower() for r in rows if r.get("step") == "paid" and r.get("email")}

    # Earliest 'contact' row per email = the lead's first quote.
    contacts = sorted(
        [r for r in rows if r.get("step") == "contact" and r.get("email")],
        key=lambda r: r.get("created_at") or "",
    )
    by_email = {}
    for r in contacts:
        by_email.setdefault(r["email"].lower(), r)

    t1, t2 = [], []
    for email, r in by_email.items():
        if email in booked or _is_suppressed_email(email):
            continue
        nur = (r.get("metadata") or {}).get("nurture") or {}
        if nur.get("unsubscribed"):
            continue
        created = _parse(r.get("created_at"))
        if not created:
            continue
        if not nur.get("t1_sent_at"):
            if now - created >= TOUCH1_DELAY:
                t1.append(r)
        elif not nur.get("t2_sent_at"):
            t1_at = _parse(nur.get("t1_sent_at"))
            if t1_at and now - t1_at >= TOUCH2_DELAY:
                t2.append(r)

    summary = {
        "dry_run": dry_run,
        "enabled": NURTURE_ENABLED,
        "touch1_due": len(t1),
        "touch2_due": len(t2),
        "sent": 0,
        "targets": [],
    }

    if dry_run:
        # Preview only — email the owner the digest, touch nothing.
        _send_owner_preview(t1, t2)
        summary["targets"] = [
            {"email": r["email"], "touch": n}
            for n, group in ((1, t1), (2, t2))
            for r in group
        ]
        return summary

    for touch, group in ((1, t1), (2, t2)):
        for r in group:
            subject, html = build_touch(r, touch)
            if _send_email(r["email"], subject, html):
                key = "t1_sent_at" if touch == 1 else "t2_sent_at"
                _set_nurture_state(supabase, r["id"], {key: now.isoformat()})
                summary["sent"] += 1
    return summary


def _send_owner_preview(t1, t2):
    lines = []
    for n, group in ((1, t1), (2, t2)):
        for r in group:
            v = _lead_view(r)
            lines.append(f"  • Touch {n}: {v['email']} ({v['artist'] or 'no artist'}, {v['service_str']})")
    listing = "\n".join(lines) or "  (no leads due today)"
    sample_subject, sample_html = (build_touch(t1[0], 1) if t1 else build_touch(t2[0], 2) if t2 else ("(none)", "<p>No leads due.</p>"))
    html = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto">
      <h2>Lead nurture — PREVIEW (sending is OFF)</h2>
      <p>Set <code>NURTURE_ENABLED=true</code> to start sending these for real.</p>
      <p><b>{len(t1)}</b> due for touch 1, <b>{len(t2)}</b> for touch 2:</p>
      <pre style="background:#f4f4f4;padding:12px;border-radius:6px">{listing}</pre>
      <h3>Sample ({sample_subject}):</h3>
      {sample_html}
    </div>
    """
    _send_email(OWNER_EMAIL, "Lead nurture preview — sending is OFF", html)
