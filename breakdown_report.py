"""
Sonic breakdown PDF (2026-09-07).

Renders a four-page, print-ready report for one analysis_jobs row: the
sound, where the record stands, production notes, sonic neighbours. Same
numbers the web result shows, laid out for a manager to read on paper.

HTML is built here; app.py turns it into a PDF with WeasyPrint when the
libraries are present (Railway: pango/cairo via nixpacks) and otherwise
serves the HTML with print CSS so the browser's Save-as-PDF still works.
Replaces the html2canvas exporter, which froze in background tabs.
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import json
import os
import re
from statistics import median

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
FONT_DIR = os.path.join(STATIC_DIR, 'fonts')

BANDS = [('Sub', 'sub_ratio'), ('Bass', 'bass_ratio'), ('Low-Mid', 'low_mid_ratio'),
         ('Mid', 'mid_ratio'), ('Hi-Mid', 'high_mid_ratio'),
         ('Presence', 'presence_ratio'), ('Air', 'air_ratio')]
BAND_HZ = {'sub_ratio': '20–60 Hz', 'bass_ratio': '60–250 Hz', 'low_mid_ratio': '250–500 Hz',
           'mid_ratio': '500 Hz–2 kHz', 'high_mid_ratio': '2–4 kHz', 'presence_ratio': '4–8 kHz',
           'air_ratio': '8–20 kHz'}
DEVIATION_LABELS = {
    'dissonance': 'Dissonance', 'mid_ratio': 'Mids', 'air_ratio': 'High-end air',
    'spectral_flux': 'Spectral movement', 'onset_rate': 'Transient density',
    'beat_strength': 'Beat strength', 'lufs_integrated': 'Master loudness',
    'sub_ratio': 'Sub-bass', 'bass_ratio': 'Bass', 'loudness_range': 'Loudness range',
    'key_strength': 'Tonal centre', 'energy': 'Intensity', 'crest_factor': 'Peak dynamics',
    'dynamic_range': 'Dynamic range', 'danceability': 'Groove', 'brightness': 'Brightness',
    'presence_ratio': 'Presence', 'high_mid_ratio': 'Hi-mids', 'low_mid_ratio': 'Low-mids',
    'compression_amount': 'Compression', 'zcr': 'Edge', 'spectral_rolloff': 'Roll-off',
    'spectral_complexity': 'Spectral complexity', 'brightness_variance': 'Brightness variance',
}


def _j(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _esc(s):
    return _html.escape(str(s if s is not None else ''))


def _k(n):
    try:
        n = float(n)
    except Exception:
        return '—'
    if n >= 1e6:
        return f'{n / 1e6:.1f}M'
    if n >= 1e5:
        return f'{n / 1e3:.0f}K'
    if n >= 1e3:
        return f'{n / 1e3:.1f}K'
    return f'{n:.0f}'


def _money(n):
    try:
        n = float(n)
    except Exception:
        return '—'
    return f'${n / 1e3:.1f}K' if n >= 1000 else f'${n:.0f}'


# --- energy / compression / danceability labels (mirror app.js) -----------
def _energy_label(v):
    if v is None:
        return '—'
    return 'Intimate' if v < 0.05 else 'Laid-back' if v < 0.15 else 'Driving' if v < 0.35 else 'Intense'


def _compression_label(v):
    if v is None:
        return '—'
    return 'Open' if v < 0.4 else 'Balanced' if v < 0.7 else 'Dense'


def _dance_label(v):
    if v is None:
        return '—'
    return 'Freeform' if v < 0.8 else 'Steady' if v < 1.1 else 'Locked'


def _level_label(levels, v):
    if not levels or v is None:
        return '–'
    b = levels['bounds']
    return levels['labels'][0 if v < b[0] else 1 if v < b[1] else 2]


def _fmt(unit, v, levels=None):
    if v is None:
        return '—'
    if unit == 'level':
        return _level_label(levels, v)
    if unit == 'pct':
        return f'{v * 100:.1f}%'
    if unit == 'db':
        return f'{v:.1f} dB'
    if unit == 'rate':
        return f'{v:.1f} /s'
    return f'{v:.3f}' if abs(v) < 10 else f'{v:.0f}'


def _band(r):
    """Port of app.js recBand(): p25–p75 zone with the same tolerance rules."""
    you = r.get('you')
    p = r.get('percentiles') or {}
    if p:
        p25, p75 = p['p25'], p['p75']
    else:
        t = r.get('target_cohort') or 0
        sp = abs(t - (you or 0)) or (abs(t) * 0.1 + 1e-6)
        p25, p75 = t - sp * 0.15, t + sp * 0.15
    in_zone = you is not None and p25 <= you <= p75
    if not in_zone and you is not None:
        edge = p25 if you < p25 else p75
        u = r.get('unit_kind')
        if u == 'level':
            in_zone = _level_label(r.get('levels'), you) == _level_label(r.get('levels'), edge)
        else:
            d = abs(you - edge)
            if u == 'pct':
                in_zone = d < 0.005
            elif u == 'db':
                in_zone = d < 0.25
            elif u == 'rate':
                in_zone = d < 0.15
            else:
                in_zone = d < abs(edge) * 0.02
    return p25, p75, in_zone


def _zone_text(r, p25, p75):
    u = r.get('unit_kind')
    if u == 'level':
        a, b = _level_label(r.get('levels'), p25), _level_label(r.get('levels'), p75)
        return a if a == b else f'{a} – {b}'
    return f'{_fmt(u, p25)} – {_fmt(u, p75)}'.replace(' dB – ', ' – ').replace(' /s – ', ' – ')


def _move_text(r, p25, p75):
    """Headline move in real units, like the site: '≈ −1 dB', '+0.4 dB'."""
    you = r.get('you')
    u = r.get('unit_kind')
    if you is None:
        return ''
    edge = p25 if you < p25 else p75
    import math
    if u == 'pct' and you > 0 and edge > 0:
        db = 10 * math.log10(edge / you)
        return f'≈ {"+" if db > 0 else "−"}{abs(db):.1f} dB'
    if u == 'db':
        d = edge - you
        return f'≈ {"+" if d > 0 else "−"}{abs(d):.1f} dB'
    if u == 'rate':
        d = edge - you
        return f'≈ {"+" if d > 0 else "−"}{abs(d):.1f} /s'
    return ''


def _genre_tokens(s):
    if not s:
        return set()
    if isinstance(s, list):
        s = ', '.join(s)
    return {t.strip().lower() for t in str(s).split(',') if t.strip()}


def _first_genre(m):
    g = m.get('artist_genres') or m.get('track_genres') or m.get('primary_genre') or ''
    if isinstance(g, list):
        g = ', '.join(g)
    return str(g).split(',')[0].strip()


def build_breakdown_html(job: dict, prepared_for: str | None = None) -> str:
    rj = _j(job.get('result_json')) or {}
    f = rj.get('features') or _j(job.get('features')) or {}
    up = rj.get('user_profile') or {}
    src = rj.get('source') or {}
    tm = up.get('track_momentum') or {}
    cc = up.get('conversion_comparison') or {}
    so = up.get('sonic_originality') or {}
    quad = up.get('quadrant') or {}
    ranges = rj.get('recommendation_ranges') or []
    matches = rj.get('matches') or []
    flattery = rj.get('flattery_matches') or []
    related = [x.get('name') for x in (_j(job.get('related_artists')) or []) if x.get('name')]

    track = job.get('track_name') or src.get('track_name') or 'Untitled'
    artist = job.get('artist_name') or src.get('artist_name') or ''
    listeners = up.get('listeners') or src.get('artist_listeners')
    followers = up.get('followers')
    conv = up.get('conversion_rate')
    tier = (src.get('artist_tier') or rj.get('user_tier') or '').capitalize()
    peer_count = tm.get('peer_count') or cc.get('peer_pool_total') or len(rj.get('all_matches') or [])
    today = _dt.date.today().strftime('%B %-d, %Y')
    total_tracks = '274,000'

    # ---- page 1: sound ----
    lufs = f.get('lufs_integrated')
    lufs_est = f.get('lufs_integrated_est')
    key = f"{f.get('key') or '?'} {f.get('scale') or ''}".strip()
    tiles = [
        (f"{float(f.get('bpm') or 0):.0f}", 'BPM', ''),
        (_esc(key), 'Key', ''),
        (f"{lufs:.1f}" if isinstance(lufs, (int, float)) else '—', 'LUFS',
         f"integrated · est. {lufs_est:.1f} on streaming" if isinstance(lufs_est, (int, float)) else 'integrated'),
        (_energy_label(f.get('energy')), 'Energy', ''),
        (_compression_label(f.get('compression_amount')), 'Compression', ''),
        (_dance_label(f.get('danceability')), 'Danceability', ''),
    ]
    tiles_html = ''.join(
        f'<div class="tile"><b>{v}</b><span>{lab}</span>{("<small>" + _esc(sub) + "</small>") if sub else ""}</div>'
        for v, lab, sub in tiles)

    rmap = {r['feature']: r for r in ranges if r.get('feature')}
    out_bands = []
    rows = ''
    for name, key_ in BANDS:
        v = f.get(key_)
        if v is None:
            continue
        w = min(100.0, v / 0.30 * 100)
        zone_html, cls = '', ''
        r = rmap.get(key_)
        if r and r.get('percentiles'):
            p25, p75, in_zone = _band(r)
            zone_html = (f'<div class="zone" style="left:{p25 / 0.30 * 100:.1f}%;'
                         f'width:{max(0.5, (p75 - p25) / 0.30 * 100):.1f}%"></div>')
            if not in_zone:
                cls = ' out'
                out_bands.append((name, 'light' if v < p25 else 'heavy'))
        rows += (f'<div class="row"><div class="lab">{name}</div><div class="track">{zone_html}'
                 f'<div class="fill{cls}" style="width:{w:.1f}%"></div></div>'
                 f'<div class="val{cls}">{v * 100:.1f}%</div></div>')
    cohort_n = so.get('cohort_size') or (rmap.get('mid_ratio') or {}).get('agree', [0, 0])[1] if rmap else 0
    if out_bands:
        plural = {'Mid': 'mids are', 'Low-Mid': 'low-mids are', 'Hi-Mid': 'hi-mids are'}
        parts = [f'the {plural.get(n, n.lower() + " is")} {d}' for n, d in out_bands]
        band_note = (f'Share of spectral energy per band. Striped zones mark where the {cohort_n} highest-converting '
                     f'peers in this lane sit. {"One band lands" if len(out_bands) == 1 else str(len(out_bands)) + " bands land"} '
                     f'outside them: {", ".join(parts)}.')
    else:
        band_note = (f'Share of spectral energy per band. Striped zones mark where the {cohort_n} highest-converting '
                     f'peers in this lane sit. Every band lands inside them.')

    emos = []
    es = f.get('emotion_summary') or {}
    for e, s in (es.get('emotions') or [])[:4]:
        emos.append((e.capitalize(), round(float(s) * 100)))
    if not emos:
        for i in (1, 2, 3, 4):
            if f.get(f'emotion_{i}'):
                emos.append((str(f[f'emotion_{i}']).capitalize(), round(float(f.get(f'emotion_{i}_score') or 0) * 100)))
    emo_html = ''.join(f'<span class="chip">{_esc(e)}<b>{s}%</b></span>' for e, s in emos)
    emo_note = ''
    if len(emos) >= 2:
        emo_note = f'{emos[0][0]} and {emos[1][0].lower()} leading' + (f', {emos[2][0].lower()} right behind them.' if len(emos) > 2 else '.')

    # ---- page 2: where it stands ----
    perf = tm.get('composite_percentile')
    perf_pct = round(float(perf) * 100) if perf is not None else None
    perf_note = ('above average' if perf_pct and perf_pct >= 55 else 'around the median' if perf_pct and perf_pct >= 45 else 'below the median') if perf_pct is not None else ''
    pop_stats, cm_stats, pl_stats = tm.get('pop_stats') or {}, tm.get('cm_stats') or {}, tm.get('playlists_stats') or {}

    def _stat(k, sub, v):
        return f'<div class="stat"><span class="k">{k}<span class="s">{sub}</span></span><span class="v">{v}</span></div>'

    momentum = ''
    if perf_pct is not None:
        momentum += _stat('Performance percentile', f'vs sonic cohort · {perf_note}', perf_pct)
    if tm.get('scanned_cm_score') is not None:
        momentum += _stat('Chartmetric score', f"peer median {cm_stats.get('median', 0):.0f} · top 25% starts at {cm_stats.get('p75', 0):.0f}", f"{float(tm['scanned_cm_score']):.0f}")
    if tm.get('scanned_playlists') is not None:
        momentum += _stat('Playlist placements', f"peer median {pl_stats.get('median', 0):.0f} · top 25% starts at {pl_stats.get('p75', 0):.0f}", f"{int(tm['scanned_playlists']):,}")
    if tm.get('scanned_popularity') is not None:
        momentum += _stat('Spotify popularity', f"recency-weighted · peer median {pop_stats.get('median', 0):.0f}", f"{int(tm['scanned_popularity'])}")

    conv_html = ''
    if conv is not None:
        conv_html += _stat('Listeners becoming followers', 'this artist', f'{float(conv):.2f}%')
    if cc.get('peer_median') is not None:
        conv_html += _stat('Peer median', 'same sonic cohort', f"{float(cc['peer_median']):.2f}%")
    if cc.get('peer_top_25') is not None:
        conv_html += _stat('Peer top 25%', 'the artists winning in this lane', f"{float(cc['peer_top_25']):.2f}%")
    if up.get('fol_listener_ratio') is not None:
        ok = 'ok' if 0.1 <= float(up['fol_listener_ratio']) <= 1.0 else ''
        conv_html += _stat('Followers per listener', '0.1 – 1.0 is the healthy retention band', f"<span class='{ok}'>{float(up['fol_listener_ratio']):.2f}</span>")

    gap_html = ''
    if tm.get('gap_current_revenue') and tm.get('gap_target_revenue'):
        cur, tgt = tm['gap_current_revenue'], tm['gap_target_revenue']
        gap_html = f'''<div class="card"><h3 style="margin-top:0">What closing the gap is worth</h3><div class="grid2">
<div><div class="big">{_money(cur)}<span class="unit"> / year</span></div><div class="note">Estimated Spotify streaming revenue today, at {_k(listeners)} listeners. Streaming only.</div></div>
<div><div class="big">{_money(tgt)}<span class="unit"> / year</span></div><div class="note">What artists whose tracks sit in the top 25% of this cohort typically earn, at a median {_k(tm.get('gap_target_listeners'))} listeners. About +{_money(tm.get('gap_additional_revenue', tgt - cur))} a year of headroom on this one track's lane.</div></div>
</div></div>'''

    orig_html = ''
    if so.get('composite_score') is not None:
        score = so['composite_score']
        devs = ''
        for d in (so.get('top_deviations') or [])[:4]:
            z = float(d.get('z') or 0)
            devs += f'<div class="stat"><span class="k">{DEVIATION_LABELS.get(d.get("feature"), d.get("feature"))}</span><span class="v small">{"+" if z >= 0 else "−"}{abs(z):.2f}<span class="sig"> σ</span></span></div>'
        qlabel = quad.get('label') or ''
        if score < 45:
            read = ('The track is executing the genre playbook more than reinventing it, which means the gap to the '
                    'winners is production, not identity. That is the good version of this result: it is fixable in the mix.')
        elif score < 70:
            read = 'The track sits close to its cohort with a few real signatures of its own. The production notes sharpen what already sets it apart.'
        else:
            read = 'The track is genuinely distinct from its cohort. The production notes below are about translation, not conformity: keep the signature, close the gaps that cost reach.'
        orig_html = f'''<div class="card"><h3 style="margin-top:0">Sonic originality</h3><div class="grid2">
<div><div class="big">{score}<span class="unit"> / 100</span></div><div class="note">Distance from the cohort consensus. 50 is typical, 75 reads as distinct. {read}</div></div>
<div>{devs}<div class="note">Where the track departs from the consensus.</div></div></div></div>'''

    # ---- page 3: production notes ----
    adjust, nailing = [], []
    for r in ranges:
        if r.get('you') is None:
            continue
        p25, p75, in_zone = _band(r)
        agree = r.get('agree') or [0, 0]
        entry = {
            'domain': r.get('domain') or '', 'action': (r.get('action') or '').replace(' — ', ': '),
            'you': _fmt(r.get('unit_kind'), r['you'], r.get('levels')),
            'zone': _zone_text(r, p25, p75), 'move': _move_text(r, p25, p75) if not in_zone else '',
            'agree': f'{agree[0]} of {agree[1]} agree' if agree and agree[1] else '',
        }
        (nailing if in_zone else adjust).append(entry)
    adjust = adjust[:5]
    adj_html = ''.join(
        f'<div class="adj"><div class="dom">{_esc(a["domain"])}</div><div><b>{_esc(a["action"])}</b>'
        f'<div class="yt">You: <em>{_esc(a["you"])}</em> &nbsp;→&nbsp; Target zone: <em>{_esc(a["zone"])}</em>'
        f'{(" <span class=\"mv\">" + _esc(a["move"]) + "</span>") if a["move"] else ""}</div>'
        f'<div class="ag">{_esc(a["agree"])}</div></div></div>' for a in adjust)
    nail_html = ''.join(
        f'<li><b>{_esc(n["action"].split(":")[0])}</b> · {_esc(n["you"])} <span>{_esc(n["agree"])}</span></li>' for n in nailing[:8])
    if adjust:
        moves = [a['action'].split(':')[0].lower() for a in adjust[:3]]
        one_move = ('Take the top adjustments as one move: ' + ', '.join(moves[:-1]) + (', and ' if len(moves) > 1 else '') + moves[-1] +
                    '. The record keeps its character and gains what the converting records in this lane all have. It is a mix decision, not a rewrite.')
    else:
        one_move = 'Nothing on the production side is holding this record back; every measured choice already sits with the winners in its lane.'

    # ---- page 4: neighbours ----
    my_genres = _genre_tokens(src.get('artist_genres')) | _genre_tokens(src.get('track_genres'))
    def _overlap(m):
        return len((_genre_tokens(m.get('artist_genres')) | _genre_tokens(m.get('track_genres'))) & my_genres)
    # Same order the web result shows (backend ranking), top 8.
    peers = [m for m in matches if m.get('name')][:8]
    peer_rows = ''.join(
        f"<tr><td class='nm'>{_esc(m['name'])}{' <span class=\"tag\">audience match</span>' if m['name'] in related else ''}</td>"
        f"<td>{_esc(_first_genre(m)) or '—'}</td><td class='num'>{(m.get('similarity') or 0) * 100:.0f}%</td>"
        f"<td class='num'>{_k(m.get('listeners'))}</td><td class='num'>{float(m.get('conversion_rate') or 0):.2f}%</td></tr>"
        for m in peers)
    # Trajectory targets exactly as the site lists them: established artists in backend order.
    big = [x for x in flattery if x.get('name') and (x.get('listeners') or 0) >= 1e6][:3] \
        or [x for x in flattery if x.get('name')][:3]
    traj_rows = ''.join(
        f"<tr><td class='nm'>{_esc(x['name'])}</td><td class='num'>{_k(x.get('listeners'))}</td><td class='num'>{(x.get('similarity') or 0) * 100:.0f}%</td></tr>"
        for x in big)
    rel_html = ''.join(f'<span class="chip">{_esc(n)}</span>' for n in related[:8])
    audience_note = ''
    am = [m['name'] for m in peers if m['name'] in related]
    if am:
        audience_note = f'{_esc(am[0])} is also a confirmed audience overlap through Chartmetric related artists, not just a sonic match.'

    prepared = f'Prepared for {_esc(prepared_for)} · {today}' if prepared_for else f'{today}'
    fonts = f"""
@font-face{{font-family:'Londrina Solid';font-weight:400;src:url('fonts/LondrinaSolid-Regular.ttf') format('truetype')}}
@font-face{{font-family:'Londrina Solid';font-weight:900;src:url('fonts/LondrinaSolid-Black.ttf') format('truetype')}}
@font-face{{font-family:'Space Grotesk';font-weight:300 700;src:url('fonts/SpaceGrotesk[wght].ttf') format('truetype')}}
"""
    css = fonts + """
@page{size:letter;margin:0}
*{box-sizing:border-box}
html,body{margin:0;background:#171614;color:#DEE6B8;font-family:'Space Grotesk',Helvetica,Arial,sans-serif;font-size:11.5pt;line-height:1.45;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.pg{width:8.5in;height:11in;padding:.7in .75in .6in;position:relative;page-break-after:always;background:#171614;overflow:hidden}
.pg:last-child{page-break-after:auto}
.eyebrow{font-size:8.5pt;letter-spacing:.18em;text-transform:uppercase;color:#8D8F59}
h1{font-family:'Londrina Solid',Impact,sans-serif;font-weight:900;font-size:42pt;line-height:.95;margin:.12in 0 0;color:#D8E166}
h2{font-family:'Londrina Solid',Impact,sans-serif;font-weight:400;font-size:22pt;margin:0 0 .1in;color:#D8E166}
h3{font-size:9pt;letter-spacing:.14em;text-transform:uppercase;color:#8D8F59;margin:.22in 0 .08in;font-weight:700}
.artist{font-size:18pt;color:#DEE6B8;margin-top:.06in}
.meta{display:flex;gap:.28in;margin-top:.22in;padding-top:.16in;border-top:1px solid #33322c;flex-wrap:wrap}
.meta div{min-width:1.1in} .meta b{display:block;font-size:17pt;color:#fff;font-weight:700;line-height:1.1} .meta span{font-size:8pt;letter-spacing:.12em;text-transform:uppercase;color:#8D8F59}
.card{background:#201f1c;border:1px solid #2f2e29;border-radius:8px;padding:.16in .2in;margin-top:.14in}
.tiles{display:flex;gap:.08in} .tile{flex:1;background:#201f1c;border:1px solid #2f2e29;border-radius:8px;padding:.12in .06in;text-align:center} .tile b{display:block;font-family:'Londrina Solid',Impact,sans-serif;font-weight:400;font-size:19pt;color:#D8E166;line-height:1} .tile span{font-size:7.5pt;letter-spacing:.12em;text-transform:uppercase;color:#8D8F59;display:block;margin-top:.05in} .tile small{display:block;font-size:7pt;color:#6f7050;margin-top:.02in}
.row{display:flex;align-items:center;gap:.12in;margin:.055in 0} .lab{width:.85in;font-size:9.5pt;color:#BABC95} .track{position:relative;flex:1;height:.14in;background:#2a2925;border-radius:3px;overflow:hidden} .fill{position:absolute;left:0;top:0;bottom:0;background:#B0C936;border-radius:3px} .fill.out{background:#D8E166} .zone{position:absolute;top:0;bottom:0;background:repeating-linear-gradient(135deg,#3d3d2e 0 3px,#4a4a37 3px 6px)} .val{width:.55in;font-size:9.5pt;text-align:right;color:#BABC95} .val.out{color:#D8E166;font-weight:700}
.chips{display:flex;gap:.08in;flex-wrap:wrap} .chip{border:1px solid #4a4a37;border-radius:999px;padding:.04in .14in;font-size:9.5pt;color:#DEE6B8} .chip b{color:#D8E166;font-weight:700;margin-left:.06in}
.note{font-size:9.5pt;color:#8D8F59;margin-top:.08in}
.grid2{display:flex;gap:.14in} .grid2>*{flex:1;min-width:0}
.stat{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid #2f2e29;padding:.06in 0} .stat:last-child{border-bottom:0} .stat .k{color:#BABC95;font-size:10pt} .stat .v{font-family:'Londrina Solid',Impact,sans-serif;font-size:18pt;color:#D8E166;white-space:nowrap} .stat .v.small{font-size:13pt} .stat .s{font-size:8.5pt;color:#8D8F59;display:block} .sig{font-family:'Space Grotesk',Arial,sans-serif;font-size:11pt;font-weight:700}
.big{font-family:'Londrina Solid',Impact,sans-serif;font-size:30pt;color:#D8E166;line-height:1} .unit{font-size:11pt;color:#8D8F59;font-family:'Space Grotesk',Arial,sans-serif}
.adj{display:flex;gap:.14in;padding:.12in 0;border-bottom:1px solid #2f2e29} .adj:last-child{border-bottom:0} .adj .dom{width:1.25in;flex:none;font-size:8pt;letter-spacing:.12em;text-transform:uppercase;color:#8D8F59;padding-top:.04in} .adj b{display:block;font-size:12pt;color:#fff;font-weight:700} .adj .yt{margin:.04in 0;font-size:10pt} .adj .yt em{font-style:normal;color:#D8E166;font-weight:700} .adj .mv{color:#BABC95;font-size:9.5pt;margin-left:.06in} .adj .ag{font-size:8.5pt;color:#8D8F59}
table{width:100%;border-collapse:collapse;font-size:10pt} th{text-align:left;font-size:8pt;letter-spacing:.12em;text-transform:uppercase;color:#8D8F59;padding:.04in .06in;border-bottom:1px solid #3a3931;font-weight:700} td{padding:.055in .06in;border-bottom:1px solid #26251f} td.nm{color:#fff;font-weight:500} td.num,th.num{text-align:right} .tag{font-size:7.5pt;letter-spacing:.1em;text-transform:uppercase;color:#171614;background:#D8E166;border-radius:999px;padding:.01in .07in;margin-left:.06in;font-weight:700}
ul.nail{columns:2;column-gap:.3in;padding-left:.18in;margin:.04in 0 0;font-size:10pt} ul.nail li{margin:.03in 0;color:#BABC95} ul.nail li b{color:#DEE6B8;font-weight:500} ul.nail li span{color:#6f7050;font-size:8.5pt}
.foot{position:absolute;left:.75in;right:.75in;bottom:.4in;display:flex;justify-content:space-between;font-size:8pt;color:#6f7050;border-top:1px solid #2f2e29;padding-top:.08in}
.ok{color:#B0C936}
"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{_esc(track)} — Sonic Breakdown</title><style>{css}</style></head><body>
<div class="pg">
  <div class="eyebrow">Sonic breakdown · Freshly Baked Studios</div>
  <h1>{_esc(track)}</h1>
  <div class="artist">{_esc(artist)}</div>
  <div class="meta">
    <div><b>{_k(listeners)}</b><span>Monthly listeners</span></div>
    <div><b>{_k(followers)}</b><span>Followers</span></div>
    <div><b>{f'{float(conv):.2f}%' if conv is not None else '—'}</b><span>Listener → follower</span></div>
    <div><b>{_esc(tier) or '—'}</b><span>Tier</span></div>
    <div><b>{int(peer_count):,}</b><span>Sonic peers compared</span></div>
  </div>
  <h3>The sound</h3>
  <div class="tiles">{tiles_html}</div>
  <h3>Frequency balance</h3>
  <div class="card">{rows}<div class="note">{band_note}</div></div>
  <h3>Emotional character</h3>
  <div class="chips">{emo_html}</div>
  <div class="note">{emo_note}</div>
  <div class="foot"><span>{prepared}</span><span>Matched against a {total_tracks}-track sonic database</span><span>1 / 4</span></div>
</div>
<div class="pg">
  <h2>Where the record stands</h2>
  <div class="note" style="margin:0 0 .12in">Every figure below is live Spotify or Chartmetric data for this track against the {int(peer_count):,} tracks that sound most like it.</div>
  <div class="grid2">
    <div class="card"><h3 style="margin-top:0">Track momentum</h3>{momentum}</div>
    <div class="card"><h3 style="margin-top:0">Audience conversion</h3>{conv_html}</div>
  </div>
  {gap_html}
  {orig_html}
  <div class="foot"><span>{_esc(artist)} · {_esc(track)}</span><span>Freshly Baked Studios · Brooklyn</span><span>2 / 4</span></div>
</div>
<div class="pg">
  <h2>Production notes</h2>
  <div class="note" style="margin:0 0 .1in">Each target zone is where the highest-converting peers in this lane actually sit. "Agree" is how many of those winners share that choice.</div>
  <div class="card"><h3 style="margin-top:0">Adjustments to make</h3>{adj_html or '<div class="note">Nothing outside the zone.</div>'}</div>
  <div class="card"><h3 style="margin-top:0"><span class="ok">✓</span> Already in the zone</h3><ul class="nail">{nail_html}</ul></div>
  <div class="card"><h3 style="margin-top:0">Read as one move</h3><p style="margin:0;font-size:10.5pt">{_esc(one_move)}</p></div>
  <div class="foot"><span>{_esc(artist)} · {_esc(track)}</span><span>Freshly Baked Studios · Brooklyn</span><span>3 / 4</span></div>
</div>
<div class="pg">
  <h2>Sonic neighbours</h2>
  <div class="note" style="margin:0 0 .1in">Artists at the same tier whose records measure closest to this one. Conversion is each artist's own listener-to-follower rate, so the column shows what the same sound is doing for other people.</div>
  <div class="card"><table><thead><tr><th>Artist</th><th>Lane</th><th class="num">Sonic match</th><th class="num">Listeners</th><th class="num">Conversion</th></tr></thead><tbody>{peer_rows}</tbody></table>{('<div class="note">' + audience_note + '</div>') if audience_note else ''}</div>
  <div class="grid2" style="margin-top:.14in">
    <div class="card" style="margin-top:0"><h3 style="margin-top:0">Chartmetric related artists</h3><div class="chips">{rel_html or '<span class="note">Not available for this scan.</span>'}</div></div>
    <div class="card" style="margin-top:0"><h3 style="margin-top:0">Trajectory targets</h3><table><thead><tr><th>Artist, 1M+ listeners</th><th class="num">Listeners</th><th class="num">Match</th></tr></thead><tbody>{traj_rows}</tbody></table><div class="note">Highest sonic matches among established artists. Proof the sound scales.</div></div>
  </div>
  <div class="card" style="margin-top:.14in;padding:.12in .2in"><h3 style="margin-top:0">How this was measured</h3><p style="margin:0;font-size:9.5pt;color:#BABC95">Sixty-three audio features from the released master (loudness, dynamics, spectral balance, harmony, rhythm, an emotion model), compared against {total_tracks} analysed tracks and the live Spotify and Chartmetric numbers behind them. The cohort is the {cohort_n} tracks in this lane whose artists convert listeners into followers at the highest rate. Targets are where they sit, not where a formula says a record should be.</p></div>
  <div class="foot"><span>Alexander Almgren · almgren@freshlybakedstudios.com · freshlybakedstudios.com</span><span>4 / 4</span></div>
</div>
</body></html>"""


def render_pdf(html_str: str) -> bytes:
    """WeasyPrint render; raises ImportError/OSError when the libs are absent."""
    from weasyprint import HTML  # lazy: Railway has pango/cairo, dev boxes may not
    return HTML(string=html_str, base_url=STATIC_DIR + '/').write_pdf()


def safe_filename(artist: str, track: str) -> str:
    s = f"{artist} - {track} - Sonic Breakdown.pdf"
    return re.sub(r'[^\w\-\. ()&]+', '', s)[:120]
