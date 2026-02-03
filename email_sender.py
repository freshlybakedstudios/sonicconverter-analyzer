"""
SendGrid email sender for analysis results.
Sends a styled HTML email with sonic breakdown, top matches, and CTA.
"""

import os
from typing import Dict, List

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, HtmlContent


BOOKING_URL = "https://freshlybakedstudios.com/book"

EMOTION_LABELS = {
    'power': 'Power',
    'nostalgia': 'Nostalgia',
    'tension': 'Tension',
    'aggressive': 'Aggression',
    'intense': 'Intensity',
    'dark': 'Darkness',
    'brooding': 'Brooding',
    'wonder': 'Wonder',
    'tenderness': 'Tenderness',
    'joyfulness': 'Joyfulness',
    'sadness': 'Sadness',
    'peacefulness': 'Peacefulness',
    'transcendence': 'Transcendence',
}


def _pct(val: float) -> str:
    return f"{val * 100:.0f}%"


def _build_eq_bar(label: str, value: float, max_val: float = 0.4) -> str:
    width = min(100, int(value / max_val * 100))
    return f"""
    <tr>
      <td style="color:#aaa;font-size:13px;width:90px;padding:3px 0">{label}</td>
      <td style="padding:3px 0">
        <div style="background:#1a1818;border-radius:4px;height:18px;width:100%">
          <div style="background:linear-gradient(90deg,#D8E166,#B5C851);height:18px;border-radius:4px;width:{width}%"></div>
        </div>
      </td>
      <td style="color:#ccc;font-size:12px;padding-left:8px;width:50px">{value:.1%}</td>
    </tr>"""


def _build_match_row(idx: int, match: Dict) -> str:
    sim = match.get('similarity', 0)
    name = match.get('name', 'Unknown')
    link_url = match.get('track_url') or match.get('spotify_url', '')
    conv = match.get('conversion_rate')
    conv_str = f'{conv:.1f}%' if conv is not None else '-'
    link = f'<a href="{link_url}" style="color:#B5C851;text-decoration:none;white-space:nowrap">{name}</a>' if link_url else f'<span style="white-space:nowrap">{name}</span>'
    return f"""
    <tr style="border-bottom:1px solid #3a3636">
      <td style="padding:6px 8px;color:#888;width:24px">{idx}</td>
      <td style="padding:6px 8px">{link}</td>
      <td style="padding:6px 8px;color:#D8E166;font-weight:bold;white-space:nowrap">{sim:.0%}</td>
      <td style="padding:6px 8px;color:#B0C936;font-weight:600;white-space:nowrap">{conv_str}</td>
    </tr>"""


def _build_conversion_section(user_profile: Dict) -> str:
    """Build the conversion comparison HTML block for the email."""
    conv_rate = user_profile.get('conversion_rate', 0)
    comp = user_profile.get('conversion_comparison', {})
    peer_median = comp.get('peer_median', 0)
    peer_top_25 = comp.get('peer_top_25', 0)
    return f'''
      <div style="background:#231f20;border:1px solid #3a3636;border-radius:8px;padding:20px;margin:0 0 16px">
        <h2 style="color:#fff;font-size:18px;margin:0 0 16px">Your Conversion Rate</h2>
        <table style="width:100%;border-collapse:collapse">
          <tr>
            <td style="text-align:center;padding:12px;background:#1a1818;border-radius:8px">
              <div style="font-size:22px;font-weight:700;color:#B5C851">{conv_rate:.1f}%</div>
              <div style="font-size:11px;color:#888;margin-top:4px">YOUR RATE</div>
            </td>
            <td style="width:12px"></td>
            <td style="text-align:center;padding:12px;background:#1a1818;border-radius:8px">
              <div style="font-size:22px;font-weight:700;color:#B5C851">{peer_median:.1f}%</div>
              <div style="font-size:11px;color:#888;margin-top:4px">PEER MEDIAN</div>
            </td>
            <td style="width:12px"></td>
            <td style="text-align:center;padding:12px;background:#1a1818;border-radius:8px">
              <div style="font-size:22px;font-weight:700;color:#B5C851">{peer_top_25:.1f}%</div>
              <div style="font-size:11px;color:#888;margin-top:4px">TOP 25%</div>
            </td>
          </tr>
        </table>
        <p style="font-size:12px;color:#666;text-align:center;margin-top:12px">Listener-to-follower conversion compared to your sonic peers</p>
      </div>'''


def send_results_email(name: str, email: str, analysis: Dict) -> bool:
    """
    Send the analysis results email via SendGrid.

    Parameters
    ----------
    name : str
        Recipient first name.
    email : str
        Recipient email address.
    analysis : dict
        Full analysis result from the /api/analyze endpoint, containing
        'features', 'matches', and 'recommendations'.

    Returns
    -------
    bool  True if sent successfully.
    """
    api_key = os.getenv('SENDGRID_API_KEY')
    if not api_key:
        print("SENDGRID_API_KEY not set — skipping email")
        return False

    features = analysis.get('features', {})
    matches = analysis.get('matches', [])
    recs = analysis.get('recommendations', [])
    genre_alignment = analysis.get('genre_alignment')
    user_profile = analysis.get('user_profile')

    # Build EQ bars
    eq_bands = [
        ('Sub', features.get('sub_ratio', 0)),
        ('Bass', features.get('bass_ratio', 0)),
        ('Low-Mid', features.get('low_mid_ratio', 0)),
        ('Mid', features.get('mid_ratio', 0)),
        ('Hi-Mid', features.get('high_mid_ratio', 0)),
        ('Presence', features.get('presence_ratio', 0)),
        ('Air', features.get('air_ratio', 0)),
    ]
    eq_html = ''.join(_build_eq_bar(l, v) for l, v in eq_bands)

    # Emotion tags
    emotions = features.get('emotion_summary', {}).get('emotions', [])
    if not emotions:
        emotions = []
        for i in range(1, 5):
            e = features.get(f'emotion_{i}')
            s = features.get(f'emotion_{i}_score', 0)
            if e and e != 'neutral':
                emotions.append((e, s))
    emotion_tags = ' '.join(
        f'<span style="display:inline-block;background:#D8E166;color:#1a1a2e;padding:4px 12px;'
        f'border-radius:20px;margin:2px;font-size:13px">'
        f'{EMOTION_LABELS.get(e, e.title())} {s:.0%}</span>'
        for e, s in emotions[:4]
    )

    # Matches table
    match_rows = ''.join(_build_match_row(i + 1, m) for i, m in enumerate(matches[:5]))

    # Recommendations
    rec_items = ''.join(
        f'<li style="padding:4px 0;color:#ccc">{r}</li>' for r in recs[:6]
    )

    bpm = features.get('bpm', 0)
    key = features.get('key', '?')
    scale = features.get('scale', '')
    lufs = features.get('lufs_integrated', 0)
    energy = features.get('energy', 0)

    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                max-width:600px;margin:0 auto;background:#141213;color:#eee;padding:32px;border-radius:12px">

      <div style="text-align:center;margin-bottom:24px">
        <img src="https://storage.googleapis.com/fbs-static-assets/axd-logo.png" alt="Freshly Baked Studios" style="width:180px;height:auto;margin-bottom:16px">
        <h1 style="color:#fff;margin:0;font-size:24px">Your Sonic Breakdown</h1>
        <p style="color:#888;margin:4px 0 0">Freshly Baked Studios</p>
      </div>

      <p style="color:#ccc">Hey {name},</p>
      <p style="color:#ccc">Here's the full analysis of your uploaded track.</p>

      <!-- Key Stats -->
      <table style="width:100%;margin:20px 0;border-collapse:collapse">
        <tr>
          <td style="text-align:center;padding:8px;width:25%">
            <div style="font-size:24px;color:#D8E166;font-weight:bold">{bpm:.0f}</div>
            <div style="font-size:11px;color:#888">BPM</div>
          </td>
          <td style="text-align:center;padding:8px;width:25%">
            <div style="font-size:24px;color:#D8E166;font-weight:bold">{key} {scale}</div>
            <div style="font-size:11px;color:#888">KEY</div>
          </td>
          <td style="text-align:center;padding:8px;width:25%">
            <div style="font-size:24px;color:#D8E166;font-weight:bold">{lufs:.1f}</div>
            <div style="font-size:11px;color:#888">LUFS</div>
          </td>
          <td style="text-align:center;padding:8px;width:25%">
            <div style="font-size:24px;color:#D8E166;font-weight:bold">{energy:.2f}</div>
            <div style="font-size:11px;color:#888">ENERGY</div>
          </td>
        </tr>
      </table>

      <!-- EQ Breakdown -->
      <h2 style="color:#fff;font-size:18px;margin:24px 0 12px">Frequency Balance</h2>
      <table style="width:100%;border-collapse:collapse">{eq_html}</table>

      <!-- Emotions -->
      <h2 style="color:#fff;font-size:18px;margin:24px 0 12px">Emotional Character</h2>
      <div style="margin-bottom:16px">{emotion_tags}</div>

      <!-- Genre Alignment -->
      {f'''
      <div style="background:#231f20;border:1px solid #D8E166;border-left:4px solid #D8E166;border-radius:8px;padding:16px 20px;margin:24px 0 16px">
        <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#888">Best Genre Fit</div>
        <div style="font-size:24px;font-weight:700;color:#B5C851;margin:4px 0">{genre_alignment["genre"]}</div>
        <div style="font-size:13px;color:#888">{genre_alignment["count"]} of {genre_alignment["total"]} matches ({genre_alignment["percentage"]:.0f}%)</div>
      </div>''' if genre_alignment else ''}

      <!-- Conversion Comparison -->
      {_build_conversion_section(user_profile) if user_profile and user_profile.get('conversion_rate') is not None else ''}

      <!-- Top Matches -->
      <h2 style="color:#fff;font-size:18px;margin:24px 0 12px">Similar Artists</h2>
      <table style="width:100%;border-collapse:collapse;color:#eee;font-size:14px">
        <tr style="border-bottom:2px solid #3a3636">
          <th style="text-align:left;padding:6px 8px;color:#888;width:24px">#</th>
          <th style="text-align:left;padding:6px 8px;color:#888">Artist</th>
          <th style="text-align:left;padding:6px 8px;color:#888">Match</th>
          <th style="text-align:left;padding:6px 8px;color:#888">Conv.</th>
        </tr>
        {match_rows}
      </table>

      <!-- Recommendations -->
      {"<h2 style='color:#fff;font-size:18px;margin:24px 0 12px'>Production Recommendations</h2><ul style='padding-left:20px;margin:0'>" + rec_items + "</ul>" if rec_items else ""}

      <!-- CTA -->
      <div style="text-align:center;margin:32px 0 16px">
        <a href="{BOOKING_URL}"
           style="display:inline-block;background:linear-gradient(135deg,#D8E166,#B5C851);
                  color:#1a1a2e;text-decoration:none;padding:14px 32px;border-radius:8px;
                  font-size:16px;font-weight:bold">
          Book a Free Mix Consultation
        </a>
      </div>

      <p style="color:#666;font-size:11px;text-align:center;margin-top:24px">
        Freshly Baked Studios &bull; freshlybakedstudios.com
      </p>
    </div>
    """

    message = Mail(
        from_email='analyzer@freshlybakedstudios.com',
        to_emails=email,
        subject=f'{name}, your sonic breakdown is ready',
        html_content=HtmlContent(html),
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print(f"Email sent to {email} — status {response.status_code}")
        return response.status_code in (200, 201, 202)
    except Exception as e:
        print(f"Email send failed: {e}")
        return False
