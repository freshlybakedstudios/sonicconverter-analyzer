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
    { "t1_sent_at": iso, "t2_sent_at": iso, "t3_sent_at": iso, "unsubscribed": bool }

Booking touches (pre-call reminder + no-show recovery) read the Cal.com
Postgres (CAL_DATABASE_URL) and use marker rows in deal_leads
(step='booking_touch', metadata={booking_uid, kind, mode}) for idempotency —
see run_booking_nurture().

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
NURTURE_REPLY_TO = os.getenv("NURTURE_REPLY_TO", "almgren@freshlybakedstudios.com")
# BCC every nurture send into the studio mailbox so outbound auto-replies are
# visible in Gmail (SendGrid API sends otherwise never touch the mailbox).
NURTURE_BCC = os.getenv("NURTURE_BCC", "almgren@freshlybakedstudios.com")
NURTURE_ENABLED = os.getenv("NURTURE_ENABLED", "false").lower() == "true"
_UNSUB_SECRET = os.getenv("NURTURE_UNSUB_SECRET", "fbs-nurture-unsub")

# Wait this long after a lead abandons before touch 1, and after touch 1 before touch 2.
# Touch 1 fires ~30 min after they leave the site — timely cart-abandonment nudge.
TOUCH1_DELAY = timedelta(minutes=30)
TOUCH2_DELAY = timedelta(days=4)
# Touch 3 = long-tail re-engage: 8 days after touch 2 ≈ day 12 of the sequence.
TOUCH3_DELAY = timedelta(days=8)
LOOKBACK = timedelta(days=30)  # don't chase leads older than this

SERVICE_LABELS = {
    "fullProduction": "Full Production",
    "mixing": "Mixing",
    "mastering": "Mastering",
    "dolbyAtmos": "Dolby Atmos",
}

# Mirrors the frontend's SERVICE_RATES / bulk tiers (constants.ts) so the
# owner digest can show the same math the lead saw. Standard tier only —
# premium (>1M listener) deals won't reconcile and fall back to value-only.
SERVICE_RATES = {
    "fullProduction": 3500,
    "mixing": 800,
    "mastering": 250,
    "dolbyAtmos": 800,  # 500 when bundled with mixing
}
ATMOS_BUNDLE_RATE = 500
BULK_TIERS = [(7, 0.20), (4, 0.15), (2, 0.10)]  # min tracks -> discount


def price_breakdown(service_keys, track_count):
    """Recompute the quote the same way preCalculateDeal does. Returns
    (total, human-readable breakdown string) or (None, None) if the
    services are unknown."""
    if not service_keys or not track_count:
        return None, None
    discount = next((d for m, d in BULK_TIERS if track_count >= m), 0)
    has_mixing = "mixing" in service_keys
    parts, per_track = [], 0
    for k in service_keys:
        rate = SERVICE_RATES.get(k)
        if rate is None:
            return None, None
        if k == "dolbyAtmos" and has_mixing:
            rate = ATMOS_BUNDLE_RATE
        # floor(x+0.5) = JS Math.round (Python's round() is banker's rounding
        # and turns $212.50 into $212, breaking parity with the site's quote)
        discounted = int(rate * (1 - discount) + 0.5)
        per_track += discounted
        parts.append(f"{SERVICE_LABELS.get(k, k)} ${discounted:,}")
    total = per_track * track_count
    disc_str = f", {int(discount * 100)}% bulk discount" if discount else ""
    breakdown = (
        f"{track_count} track{'s' if track_count != 1 else ''} × "
        f"${per_track:,}/track ({' + '.join(parts)}{disc_str}) = ${total:,}"
    )
    return total, breakdown

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
    """Return (subject, html) for touch 1, 2 or 3."""
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
    elif touch == 3:
        # Long-tail re-engage — social proof + easy door back in. NOTE: the
        # streams figure must stay "3.3B+" to match the site (the 12.3B Muso
        # number is disputed mislinks — never cite it).
        subject = "Before I close your file"
        body = f"""
          <p style="color:#ccc">Hey {v['greet']},</p>
          <p style="color:#ccc">Last note from me, promise. I know timing is everything with a record,
             so no pressure — just two things worth knowing before you decide who touches
             {project_phrase2}{for_artist}:</p>
          <p style="color:#ccc">Eight artists booked calls with the studio in the last 90 days, so the
             calendar does move. And the credits behind the desk sit at
             <strong>3.3B+ streams</strong> across projects I've worked on — your record
             would be in experienced hands.</p>
          <p style="color:#ccc">Your numbers are still saved if you want to pick things back up:</p>
          <p style="text-align:center;margin:28px 0">
            <a href="{rates}" style="background:#D8E166;color:#222020;text-decoration:none;
               font-weight:bold;padding:14px 28px;border-radius:8px;display:inline-block">
               Revisit my quote</a>
          </p>
          <p style="color:#ccc">And if it's easier to just talk it through, reply here — I read these
             myself.</p>
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
    from sendgrid.helpers.mail import Bcc, IpPoolName, Mail, HtmlContent, ReplyTo

    msg = Mail(
        from_email=NURTURE_FROM,
        to_emails=to_email,
        subject=subject,
        html_content=HtmlContent(html),
    )
    # deals@ is SEND-ONLY (no alias/forward exists — audited 2026-07-20: probes
    # to it never arrive anywhere). Every "just hit reply" reply bounced until
    # this ReplyTo. Replies must go to a real, monitored mailbox.
    msg.reply_to = ReplyTo(NURTURE_REPLY_TO, "Alexander Almgren")
    if NURTURE_BCC and NURTURE_BCC.lower() != to_email.lower():
        msg.add_bcc(Bcc(NURTURE_BCC))
    # Transactional/nurture stream rides its own dedicated IP (pool2) so cold
    # outreach (pool1) can never taint it.
    msg.ip_pool_name = IpPoolName("production_pool2")
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

    t1, t2, t3 = [], [], []
    for email, r in by_email.items():
        if email in booked or _is_suppressed_email(email):
            continue
        nur = (r.get("metadata") or {}).get("nurture") or {}
        if nur.get("unsubscribed"):
            continue
        # replied_at is set by nurture-reply-sync.js on the studio Mac when the
        # lead answers by email — a human conversation has started, stop the robot.
        if nur.get("replied_at"):
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
        elif not nur.get("t3_sent_at"):
            t2_at = _parse(nur.get("t2_sent_at"))
            if t2_at and now - t2_at >= TOUCH3_DELAY:
                t3.append(r)

    summary = {
        "dry_run": dry_run,
        "enabled": NURTURE_ENABLED,
        "touch1_due": len(t1),
        "touch2_due": len(t2),
        "touch3_due": len(t3),
        "sent": 0,
        "targets": [],
    }

    if dry_run:
        # Preview only — email the owner the digest, touch nothing.
        _send_owner_preview(t1, t2, t3)
        summary["targets"] = [
            {"email": r["email"], "touch": n}
            for n, group in ((1, t1), (2, t2), (3, t3))
            for r in group
        ]
        return summary

    for touch, group in ((1, t1), (2, t2), (3, t3)):
        for r in group:
            subject, html = build_touch(r, touch)
            if _send_email(r["email"], subject, html):
                _set_nurture_state(supabase, r["id"], {f"t{touch}_sent_at": now.isoformat()})
                summary["sent"] += 1
    return summary


# ---------------------------------------------------------------------------
# Daily owner digest — every new lead from the last 24h with enough context to
# send one personal line each. Automation warms; a human email closes.
# ---------------------------------------------------------------------------
DIGEST_HOUR_ET = 9  # send once daily, after 9am New York time


def _now_et():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now(timezone.utc) - timedelta(hours=4)


def run_daily_digest(supabase) -> dict:
    """Email the owner a digest of the last 24h of deal-calc leads, once per
    day after DIGEST_HOUR_ET. Independent of NURTURE_ENABLED — it never
    contacts a lead. Once-per-day guard = a step='digest_sent' marker row."""
    now_et = _now_et()
    if now_et.hour < DIGEST_HOUR_ET:
        return {"skipped": "before send hour"}
    today = now_et.strftime("%Y-%m-%d")

    marker = (
        supabase.table("deal_leads").select("id").eq("step", "digest_sent")
        .contains("metadata", {"date": today}).limit(1).execute()
    )
    if marker.data:
        return {"skipped": "already sent today"}

    since = (_now() - timedelta(hours=24)).isoformat()
    rows = (
        supabase.table("deal_leads").select("*").gte("created_at", since)
        .order("created_at").execute().data or []
    )

    paid, leads_by_email = [], {}
    for r in rows:
        email = (r.get("email") or "").lower()
        if not email or _is_suppressed_email(email):
            continue
        if r.get("step") == "paid":
            paid.append(r)
        elif r.get("step") in ("contact", "checkout_started"):
            # keep the most advanced row per email (checkout beats contact)
            prev = leads_by_email.get(email)
            if not prev or r.get("step") == "checkout_started":
                leads_by_email[email] = r

    # Drop leads who also paid in the window — they're in the wins section
    paid_emails = {(p.get("email") or "").lower() for p in paid}
    leads = [r for e, r in leads_by_email.items() if e not in paid_emails]

    if not leads and not paid:
        # still stamp the marker so we don't re-query all day
        supabase.table("deal_leads").insert({
            "name": "", "email": "digest@internal", "step": "digest_sent",
            "metadata": {"date": today, "leads": 0},
            "created_at": _now().isoformat(),
        }).execute()
        return {"skipped": "no leads in window"}

    def lead_block(r):
        v = _lead_view(r)
        meta = r.get("metadata") or {}
        val = f"${v['value']:,}" if isinstance(v["value"], (int, float)) and v["value"] else "value unknown"
        tracks = meta.get("track_count")
        # Show the full price math so the total explains itself
        # ("2 tracks × $945/track (Mixing $720 + Mastering $225, 10% bulk
        # discount) = $1,890" instead of a bare $1,890).
        calc_total, breakdown = price_breakdown(meta.get("services"), tracks)
        if breakdown and calc_total == v["value"]:
            price_line = breakdown
        else:
            # premium tier / legacy rows / rate changes — show what we know
            track_str = f" · {tracks} track{'s' if tracks != 1 else ''}" if tracks else ""
            price_line = f"{val}{track_str}"
        hot = r.get("step") == "checkout_started"
        nur = meta.get("nurture") or {}
        touches = ("t1" if nur.get("t1_sent_at") else "") + ("+t2" if nur.get("t2_sent_at") else "")
        opener_about = v["service_str"].lower() if v["service_str"] else "a project"
        # Display name: artist, then real name, then the email's local part —
        # never the "there" greeting fallback
        who = v["artist"] or (v["greet"] if v["greet"] != "there" else v["email"].split("@")[0])
        # Only add "for <artist>" when it isn't just repeating the greeting
        for_artist = f" for {v['artist']}" if v["artist"] and v["artist"] != v["greet"] else ""
        opener = (
            f"Hey {v['greet']} — saw you priced out {opener_about}"
            f"{for_artist}. "
            f"Send me your latest track and I'll take a quick listen — "
            f"I'll tell you exactly what I'd do with it."
        )
        return f"""
        <div style="border:1px solid #ddd;border-radius:8px;padding:14px;margin-bottom:12px{';border-color:#c00' if hot else ''}">
          <b>{who}</b> &lt;{v['email']}&gt;
          {'<span style="color:#c00;font-weight:bold"> · STARTED CHECKOUT — hottest</span>' if hot else ''}<br>
          {v['service_str'] or 'services unknown'} · {price_line}
          {f' · nurture sent: {touches}' if touches else ' · no nurture sent yet'}<br>
          <span style="color:#555;font-size:13px">Suggested opener (personalize + send from your inbox):</span><br>
          <em style="color:#333">{opener}</em>
        </div>"""

    def paid_block(p):
        meta = p.get("metadata") or {}
        amount = meta.get("amount_total")
        amt = f"${amount/100:,.0f}" if isinstance(amount, (int, float)) else "?"
        note = ""
        if meta.get("payment_kind") == "reserve":
            note = (f" — <b style='color:#c00'>slot reservation: balance of "
                    f"${meta.get('deal_total') or '?'} still due, send a Stripe payment link before delivery</b>")
        shown = meta.get("testimonials_shown")
        if shown:
            note += f" <span style='color:#777;font-size:12px'>(reviews shown: {shown})</span>"
        return f"<li><b>{p.get('name') or p.get('email')}</b> paid {amt}{note}</li>"

    wins_html = f"<h3>💰 Paid ({len(paid)})</h3><ul>{''.join(paid_block(p) for p in paid)}</ul>" if paid else ""
    leads_html = "".join(lead_block(r) for r in leads) or "<p>(none)</p>"
    html = f"""
    <div style="font-family:sans-serif;max-width:640px;margin:0 auto;color:#222">
      <h2>Deal-calc leads — last 24h</h2>
      {wins_html}
      <h3>🔥 Leads to personally follow up ({len(leads)})</h3>
      <p style="color:#555;font-size:13px">The automated nurture already ran or will —
      these are worth one personal line from you. Reply from your own inbox, not SendGrid.</p>
      {leads_html}
    </div>"""

    n_leads = len(leads)
    subject = f"☀️ {n_leads} lead{'s' if n_leads != 1 else ''} to follow up" + (f" · {len(paid)} paid 💰" if paid else "")
    sent = _send_email(OWNER_EMAIL, subject, html)
    if sent:
        supabase.table("deal_leads").insert({
            "name": "", "email": "digest@internal", "step": "digest_sent",
            "metadata": {"date": today, "leads": n_leads, "paid": len(paid)},
            "created_at": _now().isoformat(),
        }).execute()
    return {"sent": sent, "leads": n_leads, "paid": len(paid)}


def _send_owner_preview(t1, t2, t3=()):
    lines = []
    for n, group in ((1, t1), (2, t2), (3, t3)):
        for r in group:
            v = _lead_view(r)
            lines.append(f"  • Touch {n}: {v['email']} ({v['artist'] or 'no artist'}, {v['service_str']})")
    listing = "\n".join(lines) or "  (no leads due today)"
    first = next(((r, n) for n, group in ((1, t1), (2, t2), (3, t3)) for r in group), None)
    sample_subject, sample_html = build_touch(*first) if first else ("(none)", "<p>No leads due.</p>")
    # This preview fires on any dry run. Only claim "sending is OFF" when the
    # live flag really is off — a manual dry-run while live reads differently.
    if NURTURE_ENABLED:
        heading = "Lead nurture — DRY RUN (live sending is ON)"
        note = "This was a manual dry-run check. The scheduler is live and sends automatically; nothing was sent by this run."
        subject = "Lead nurture dry-run — live sending is ON"
    else:
        heading = "Lead nurture — PREVIEW (sending is OFF)"
        note = "Set <code>NURTURE_ENABLED=true</code> to start sending these for real."
        subject = "Lead nurture preview — sending is OFF"
    html = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto">
      <h2>{heading}</h2>
      <p>{note}</p>
      <p><b>{len(t1)}</b> due for touch 1, <b>{len(t2)}</b> for touch 2, <b>{len(t3)}</b> for touch 3:</p>
      <pre style="background:#f4f4f4;padding:12px;border-radius:6px">{listing}</pre>
      <h3>Sample ({sample_subject}):</h3>
      {sample_html}
    </div>
    """
    _send_email(OWNER_EMAIL, subject, html)


# ---------------------------------------------------------------------------
# Booking touches — pre-call reminder + no-show recovery.
#
# Bookings live in the Cal.com Postgres (CAL_DATABASE_URL). We never write to
# Cal's DB; idempotency markers are rows in OUR deal_leads table:
#     step='booking_touch', metadata={booking_uid, kind: precall|noshow, mode}
# Ships DISABLED: with BOOKING_NURTURE_ENABLED != 'true' the owner gets a
# one-time preview per due booking (stamped mode='preview') and no lead is
# emailed. Flip BOOKING_NURTURE_ENABLED=true to go live.
# ---------------------------------------------------------------------------
CAL_DATABASE_URL = os.getenv("CAL_DATABASE_URL", "")
BOOKING_NURTURE_ENABLED = os.getenv("BOOKING_NURTURE_ENABLED", "false").lower() == "true"
CAL_BOOKING_LINK = "https://cal.freshlybakedstudios.com/freshlybakedstudios/30min"

PRECALL_WINDOW = timedelta(hours=3)   # remind once the call is <= 3h out
# Skip the reminder when the slot was booked last-minute — Cal's own
# confirmation email is minutes old and a second email looks robotic.
PRECALL_MIN_NOTICE = timedelta(hours=3, minutes=30)
NOSHOW_AFTER = timedelta(hours=20)    # "next day" — rebook email fires 20h+ after the slot
NOSHOW_WINDOW = timedelta(hours=72)   # never chase calls older than 3 days (bounds first-deploy backlog)


def _cal_fetch_bookings():
    """Accepted bookings (one row per attendee) from 72h back to 30d ahead.
    The far-future rows exist only so no-show detection can see reschedules."""
    import psycopg2
    now = _now()
    lo = (now - NOSHOW_WINDOW).replace(tzinfo=None)
    hi = (now + timedelta(days=30)).replace(tzinfo=None)
    conn = psycopg2.connect(CAL_DATABASE_URL, connect_timeout=10)
    try:
        with conn.cursor() as cur:
            cur.execute(
                '''SELECT b.uid, b."startTime", b."createdAt", b.rescheduled,
                          b.metadata->>'videoCallUrl',
                          a.email, a.name, a."timeZone"
                     FROM "Booking" b
                     JOIN "Attendee" a ON a."bookingId" = b.id
                    WHERE b.status = 'accepted'
                      AND b."startTime" BETWEEN %s AND %s''',
                (lo, hi),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    for uid, start, created, resched, video, email, name, tz in rows:
        out.append({
            "uid": uid,
            # Cal stores naive UTC timestamps
            "start": start.replace(tzinfo=timezone.utc),
            "created": created.replace(tzinfo=timezone.utc) if created else None,
            "rescheduled": bool(resched),
            "video_url": video,
            "email": (email or "").strip().lower(),
            "name": (name or "").strip(),
            "tz": tz or "America/New_York",
        })
    return out


def _booking_markers(supabase):
    """Set of (booking_uid, kind, mode) already handled."""
    since = (_now() - timedelta(days=14)).isoformat()
    rows = (
        supabase.table("deal_leads").select("metadata")
        .eq("step", "booking_touch").gte("created_at", since).execute().data or []
    )
    out = set()
    for r in rows:
        m = r.get("metadata") or {}
        out.add((m.get("booking_uid"), m.get("kind"), m.get("mode")))
    return out


def _stamp_booking(supabase, b, kind, mode):
    supabase.table("deal_leads").insert({
        "name": b["name"], "email": b["email"], "step": "booking_touch",
        "metadata": {"booking_uid": b["uid"], "kind": kind, "mode": mode},
        "created_at": _now().isoformat(),
    }).execute()


def _lead_unsubscribed(supabase, email):
    try:
        rows = (
            supabase.table("deal_leads").select("metadata")
            .ilike("email", email).eq("step", "contact").execute().data or []
        )
        return any(
            ((r.get("metadata") or {}).get("nurture") or {}).get("unsubscribed")
            for r in rows
        )
    except Exception:
        return False


def _lead_paid_since(supabase, email, since_dt):
    """They paid after the call slot — the call clearly happened, skip recovery."""
    try:
        rows = (
            supabase.table("deal_leads").select("id").ilike("email", email)
            .eq("step", "paid").gte("created_at", since_dt.isoformat())
            .limit(1).execute().data
        )
        return bool(rows)
    except Exception:
        return False


def _fmt_local(dt_utc, tzname):
    """'4:30 PM (Chicago time)' in the attendee's own timezone."""
    try:
        from zoneinfo import ZoneInfo
        local = dt_utc.astimezone(ZoneInfo(tzname))
        city = tzname.split("/")[-1].replace("_", " ")
        return local.strftime("%I:%M %p").lstrip("0") + f" ({city} time)"
    except Exception:
        return dt_utc.strftime("%I:%M %p").lstrip("0") + " (UTC)"


def _booking_shell(body):
    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                max-width:560px;margin:0 auto;background:#141213;color:#eee;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:8px">
        <img src="https://storage.googleapis.com/fbs-static-assets/axd-logo.png"
             alt="Freshly Baked Studios" style="width:140px;height:auto;margin-bottom:8px">
      </div>
      {body}
    </div>
    """


def build_precall_email(b):
    """Reminder ~3h before the call: meet link + what to have ready.
    Transactional (tied to a booking they made) — no unsubscribe gate."""
    when = _fmt_local(b["start"], b["tz"])
    link = b["video_url"] or f"https://cal.freshlybakedstudios.com/booking/{b['uid']}"
    greet = b["name"].split()[0] if b["name"] else "there"
    subject = f"Talk soon — we're on at {when}"
    body = f"""
      <p style="color:#ccc">Hey {greet},</p>
      <p style="color:#ccc">Quick heads-up: our call is coming up at <strong>{when}</strong>.
         Here's your link when it's time:</p>
      <p style="text-align:center;margin:28px 0">
        <a href="{link}" style="background:#D8E166;color:#222020;text-decoration:none;
           font-weight:bold;padding:14px 28px;border-radius:8px;display:inline-block">
           Join the call</a>
      </p>
      <p style="color:#ccc">To get the most out of it, have handy:</p>
      <ul style="color:#ccc">
        <li>The track(s) you're working on — rough mixes, demos, or stems all work</li>
        <li>1–2 reference tracks (records whose sound you're chasing)</li>
        <li>A sense of where you want this record to land</li>
      </ul>
      <p style="color:#ccc">If the time stopped working, no stress — reply here or grab a
         new slot and we'll make it happen.</p>
      <p style="color:#ccc">— Alexander<br><span style="color:#888">Freshly Baked Studios · Brooklyn, NY</span></p>
      <p style="color:#555;font-size:11px;text-align:center;margin-top:28px;border-top:1px solid #2a2626;padding-top:16px">
        Freshly Baked Studios · Brooklyn, NY<br>
        You're getting this because you booked a call with us.
      </p>
    """
    return subject, _booking_shell(body)


def build_noshow_email(b):
    """Next-day friendly rebook after a missed call."""
    email = b["email"]
    unsub = f"{PUBLIC_API_BASE}/api/deal/nurture/unsubscribe?e={email}&t={unsub_token(email)}"
    greet = b["name"].split()[0] if b["name"] else "there"
    # "yesterday" only when it actually was — backlog catches can be up to 72h old
    ago = "yesterday" if (_now() - b["start"]) < timedelta(hours=36) else "the other day"
    subject = "We missed each other — let's find a new time"
    body = f"""
      <p style="color:#ccc">Hey {greet},</p>
      <p style="color:#ccc">Looks like our call {ago} didn't happen — totally fine,
         calendars do what calendars do.</p>
      <p style="color:#ccc">Still happy to talk through your record whenever works.
         Grab any slot that fits:</p>
      <p style="text-align:center;margin:28px 0">
        <a href="{CAL_BOOKING_LINK}" style="background:#D8E166;color:#222020;text-decoration:none;
           font-weight:bold;padding:14px 28px;border-radius:8px;display:inline-block">
           Pick a new time</a>
      </p>
      <p style="color:#ccc">Or if email's easier, just reply here — I read these myself.</p>
      <p style="color:#ccc">— Alexander<br><span style="color:#888">Freshly Baked Studios · Brooklyn, NY</span></p>
      <p style="color:#555;font-size:11px;text-align:center;margin-top:28px;border-top:1px solid #2a2626;padding-top:16px">
        Freshly Baked Studios · Brooklyn, NY<br>
        You got this because you booked a call with us.
        <a href="{unsub}" style="color:#777">Unsubscribe</a>
      </p>
    """
    return subject, _booking_shell(body)


def run_booking_nurture(supabase, dry_run: bool = None) -> dict:
    """Send (or preview) pre-call reminders and no-show recovery emails.

    Preview mode stamps mode='preview' markers so the owner sees each due
    booking exactly once, not on every 10-min poll. Live mode only checks
    mode='live' markers, so flipping the flag later still sends for
    already-previewed bookings that remain due.
    """
    if dry_run is None:
        dry_run = not BOOKING_NURTURE_ENABLED
    if not CAL_DATABASE_URL:
        return {"skipped": "CAL_DATABASE_URL not set"}
    now = _now()
    try:
        bookings = _cal_fetch_bookings()
    except Exception as e:
        return {"error": f"cal db unavailable: {e}"}

    mode = "preview" if dry_run else "live"
    markers = _booking_markers(supabase)
    starts_by_email = {}
    for b in bookings:
        starts_by_email.setdefault(b["email"], []).append(b["start"])

    precall, noshow = [], []
    for b in bookings:
        if _is_suppressed_email(b["email"]):
            continue

        def _handled(kind):
            return (b["uid"], kind, "live") in markers or (b["uid"], kind, mode) in markers

        delta = b["start"] - now
        if timedelta(0) < delta <= PRECALL_WINDOW:
            if _handled("precall"):
                continue
            if b["created"] and (b["start"] - b["created"]) < PRECALL_MIN_NOTICE:
                continue
            precall.append(b)
        elif NOSHOW_AFTER <= -delta <= NOSHOW_WINDOW:
            if _handled("noshow") or b["rescheduled"]:
                continue
            # A later accepted booking by the same email = they rebooked already.
            if any(s > b["start"] for s in starts_by_email.get(b["email"], [])):
                continue
            if _lead_unsubscribed(supabase, b["email"]):
                continue
            if _lead_paid_since(supabase, b["email"], b["start"]):
                continue
            noshow.append(b)

    summary = {
        "dry_run": dry_run,
        "enabled": BOOKING_NURTURE_ENABLED,
        "precall_due": len(precall),
        "noshow_due": len(noshow),
        "sent": 0,
        "targets": [
            {"email": b["email"], "kind": k, "booking": b["uid"]}
            for k, group in (("precall", precall), ("noshow", noshow))
            for b in group
        ],
    }

    if dry_run:
        if precall or noshow:
            _send_booking_preview(precall, noshow)
            for kind, group in (("precall", precall), ("noshow", noshow)):
                for b in group:
                    _stamp_booking(supabase, b, kind, "preview")
        return summary

    for kind, build, group in (
        ("precall", build_precall_email, precall),
        ("noshow", build_noshow_email, noshow),
    ):
        for b in group:
            subject, html = build(b)
            if _send_email(b["email"], subject, html):
                _stamp_booking(supabase, b, kind, "live")
                summary["sent"] += 1
    return summary


def _send_booking_preview(precall, noshow):
    def line(b, kind):
        when = b["start"].strftime("%a %b %d %H:%M UTC")
        return f"  • {kind}: {b['name'] or '?'} &lt;{b['email']}&gt; — call {when}"
    lines = [line(b, "pre-call reminder") for b in precall] + [line(b, "no-show rebook") for b in noshow]
    samples = ""
    if precall:
        s, h = build_precall_email(precall[0])
        samples += f"<h3>Sample pre-call ({s}):</h3>{h}"
    if noshow:
        s, h = build_noshow_email(noshow[0])
        samples += f"<h3>Sample no-show rebook ({s}):</h3>{h}"
    if BOOKING_NURTURE_ENABLED:
        b_heading = "Booking touches — DRY RUN (live sending is ON)"
        b_note = "This was a manual dry-run check. The scheduler is live and sends automatically; nothing was sent by this run."
        b_subject = "Booking touches dry-run — live sending is ON"
    else:
        b_heading = "Booking touches — PREVIEW (sending is OFF)"
        b_note = "Set <code>BOOKING_NURTURE_ENABLED=true</code> to send these for real."
        b_subject = "Booking touches preview — sending is OFF"
    html = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto">
      <h2>{b_heading}</h2>
      <p>{b_note}</p>
      <p><b>{len(precall)}</b> pre-call reminder(s) due, <b>{len(noshow)}</b> no-show rebook(s) due:</p>
      <pre style="background:#f4f4f4;padding:12px;border-radius:6px">{chr(10).join(lines)}</pre>
      <p style="color:#a00;font-size:13px"><b>No-show caveat:</b> "no-show" here means the call time
      passed, the booking is still 'accepted', and they haven't rebooked. Cal.com doesn't know who
      actually showed up — if someone below DID attend, cancel or mark their booking in Cal so the
      rebook email stays suppressed once this goes live.</p>
      {samples}
    </div>
    """
    _send_email(OWNER_EMAIL, b_subject, html)
