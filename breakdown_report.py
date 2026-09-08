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



# ---------------------------------------------------------------------------
# v2 builder (2026-09-08, owner: "Sonic Quadrant, A&R pitch comparables, Where
# your track stands are missing; production recommendations as the ACTUAL graph
# that appears on the analyzer; a handful of similar artists"). Every component
# below is a port of the matching renderer in static/app.js so the PDF shows the
# same bars, the same quadrant plot, the same rec meters as the web result.
# ---------------------------------------------------------------------------
import math as _math

FEATURE_PRETTY = {
    'sub_ratio': 'Sub-bass', 'bass_ratio': 'Bass', 'low_mid_ratio': 'Low-mids', 'mid_ratio': 'Mids',
    'high_mid_ratio': 'High-mids', 'presence_ratio': 'Presence', 'air_ratio': 'Air',
    'brightness': 'Brightness (spectral centroid)', 'spectral_rolloff': 'Spectral rolloff',
    'brightness_variance': 'Brightness movement', 'energy': 'Energy', 'dynamic_range': 'Dynamic range',
    'loudness_range': 'Loudness range', 'lufs_integrated': 'Master loudness', 'compression_amount': 'Compression',
    'crest_factor': 'Crest factor', 'true_peak_dbfs': 'True peak', 'beat_strength': 'Beat strength',
    'onset_rate': 'Onset density', 'attack_time': 'Attack time', 'danceability': 'Danceability',
    'spectral_complexity': 'Spectral complexity', 'dissonance': 'Dissonance', 'key_strength': 'Tonal anchoring',
    'zcr': 'Brightness (ZCR)', 'spectral_flux': 'Spectral flux', 'harmonic_distortion': 'Harmonic distortion',
    'stereo_width': 'Stereo width', 'mid_side_ratio': 'Mid/side ratio', 'stereo_correlation': 'Stereo correlation',
}
FEATURE_DIRECTION = {
    'sub_ratio': ('heavier sub-bass than', 'lighter sub-bass than'), 'bass_ratio': ('heavier bass than', 'lighter bass than'),
    'low_mid_ratio': ('thicker low-mids than', 'cleaner low-mids than'), 'mid_ratio': ('more forward mids than', 'softer mids than'),
    'high_mid_ratio': ('more presence / edge than', 'softer upper-mids than'), 'presence_ratio': ('brighter presence than', 'darker presence than'),
    'air_ratio': ('more high-end air than', 'less high-end air than'), 'brightness': ('brighter spectral center than', 'darker spectral center than'),
    'spectral_rolloff': ('more high-frequency rolloff than', 'less high-frequency content than'),
    'brightness_variance': ('more brightness movement than', 'flatter brightness curve than'),
    'energy': ('higher energy than', 'more restrained than'), 'dynamic_range': ('more dynamic contrast than', 'flatter dynamics than'),
    'loudness_range': ('wider loudness variation than', 'tighter loudness than'), 'lufs_integrated': ('louder master than', 'quieter master than'),
    'compression_amount': ('more compressed than', 'more open / less compressed than'), 'crest_factor': ('punchier peaks than', 'flatter peaks than'),
    'true_peak_dbfs': ('higher peak level than', 'lower peak level than'), 'beat_strength': ('stronger beat than', 'softer beat than'),
    'onset_rate': ('denser percussion than', 'sparser percussion than'), 'attack_time': ('slower attacks than', 'sharper attacks than'),
    'danceability': ('more rhythmic pull than', 'looser groove than'), 'spectral_complexity': ('more spectral complexity than', 'simpler spectrum than'),
    'dissonance': ('more dissonant / edgy than', 'more consonant / clean than'), 'key_strength': ('more tonally anchored than', 'more tonally ambiguous than'),
    'zcr': ('brighter / noisier than', 'mellower / cleaner than'), 'spectral_flux': ('more spectral movement than', 'more static spectrum than'),
    'harmonic_distortion': ('more harmonic saturation than', 'cleaner harmonics than'), 'stereo_width': ('wider stereo image than', 'narrower stereo than'),
    'mid_side_ratio': ('more side energy than', 'more centered mix than'), 'stereo_correlation': ('more decorrelated stereo than', 'more correlated stereo than'),
}
EMOTION_LABELS = {'power': 'Power', 'nostalgia': 'Nostalgia', 'tension': 'Tension', 'aggressive': 'Aggression',
                  'intense': 'Intensity', 'dark': 'Darkness', 'brooding': 'Brooding', 'wonder': 'Wonder',
                  'tenderness': 'Tenderness', 'joyfulness': 'Joyfulness', 'sadness': 'Sadness',
                  'peacefulness': 'Peacefulness', 'transcendence': 'Transcendence'}


def _num(v):
    try:
        if v is None or (isinstance(v, float) and _math.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


# --- app.js ports: value formatting for the rec meters ----------------------
def _fmt_feat(kind, v):
    v = _num(v)
    if v is None:
        return '–'
    if kind == 'pct':
        return f'{v * 100:.1f}%'
    if kind == 'db':
        return f'{v:.1f} dB'
    if kind == 'lufs':
        return f'{v:.1f} LUFS'
    if kind == 'hz':
        return f'{round(v)} Hz'
    if kind == 'rate':
        return f'{v:.1f} /s'
    if kind == 'ms':
        return f'{v:.1f} ms'
    if kind == 'lu':
        return f'{v:.1f} LU'
    return f'{v:.3f}'


def _fmt_move(kind, you, target):
    you, target = _num(you), _num(target)
    if you is None or target is None:
        return ''
    def s(v, dp, unit):
        return ('+' if v >= 0 else '−') + f'{abs(v):.{dp}f}' + unit
    if kind == 'pct':
        if you > 0 and target > 0:
            return s(10 * _math.log10(target / you), 1, ' dB')
        return s((target - you) * 100, 1, ' pts')
    if kind in ('db', 'lufs'):
        return s(target - you, 1, ' dB')
    if kind == 'lu':
        return s(target - you, 1, ' LU')
    if kind == 'hz':
        return ('+' if target >= you else '−') + f'{abs(round(target - you))} Hz'
    if kind == 'rate':
        return s(target - you, 1, ' /s')
    if kind == 'ms':
        return s(target - you, 1, ' ms')
    return s(((target - you) / abs(you)) * 100, 0, '%') if you != 0 else ''


def _fmt_range(kind, a, b):
    A, B = _fmt_feat(kind, a), _fmt_feat(kind, b)
    m = re.match(r'^([\d.\-−]+)(.*)$', A)
    if m and B.endswith(m.group(2)):
        return m.group(1) + '–' + B
    return A + '–' + B


def _move_negligible(kind, you, edge):
    you, edge = _num(you), _num(edge)
    if you is None or edge is None:
        return False
    if kind == 'pct':
        return abs(10 * _math.log10(edge / you)) < 0.05 if (you > 0 and edge > 0) else abs((edge - you) * 100) < 0.05
    if kind in ('db', 'lufs', 'lu', 'rate', 'ms'):
        return abs(edge - you) < 0.05
    if kind == 'hz':
        return abs(edge - you) < 0.5
    return you != 0 and abs((edge - you) / abs(you)) * 100 < 1.5


def _rec_band(r):
    you = _num(r.get('you'))
    p = r.get('percentiles') or {}
    if p:
        p5, p25, p50, p75, p95 = (_num(p.get(k)) for k in ('p5', 'p25', 'p50', 'p75', 'p95'))
    else:
        t = _num(r.get('target_cohort')) or 0.0
        sp = abs(t - (you or 0)) or (abs(t) * 0.1 + 1e-6)
        p25, p75, p50 = t - sp * 0.15, t + sp * 0.15, t
        p5, p95 = t - sp * 0.6, t + sp * 0.6
    in_zone = you is not None and p25 <= you <= p75
    if not in_zone and you is not None:
        edge = p25 if you < p25 else p75
        if r.get('unit_kind') == 'level':
            in_zone = _level_label(r.get('levels'), you) == _level_label(r.get('levels'), edge)
        elif _move_negligible(r.get('unit_kind'), you, edge):
            in_zone = True
    return p5, p25, p50, p75, p95, in_zone



def _hatch_lines(x0, x1, h, color, opacity, step=7.0, width=2.2):
    """45° hatch strictly inside [x0, x1] × [0, h] as explicit lines: no CSS gradients, no patterns, so every
    PDF viewer draws it (repeating-linear-gradient was invisible in some viewers)."""
    out = []
    x = x0 - h
    while x < x1:
        ax, ay, bx, by = x, h, x + h, 0.0
        if ax < x0:
            ay = h - (x0 - ax); ax = x0
        if bx > x1:
            by = bx - x1; bx = x1
        if bx > ax:
            out.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-opacity="{opacity}" stroke-width="{width}"/>')
        x += step
    return ''.join(out)


def _meter_svg(width_in, zone_l, zone_w, median, dot, dot_cls, fill_to=None, opp=None, ticks=None, h=14.0):
    """One horizontal meter drawn as inline SVG (units = 1/100 in). Ports .rec-range-bar / .conv-bar."""
    W = width_in * 100.0
    px = lambda pct: pct / 100.0 * W
    top = 5.0
    s = [f'<svg width="{width_in:.2f}in" height="{(h + 10) / 100:.2f}in" viewBox="0 0 {W:.0f} {h + 10:.0f}" xmlns="http://www.w3.org/2000/svg" style="display:block;overflow:visible">']
    s.append(f'<defs><linearGradient id="cf" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#5a5a3a"/><stop offset="1" stop-color="#888899"/></linearGradient></defs>')
    s.append(f'<rect x="0" y="{top}" width="{W:.1f}" height="{h}" rx="{h / 2}" fill="#3a3636"/>')
    if fill_to is not None and fill_to > 0:
        s.append(f'<rect x="0" y="{top}" width="{px(fill_to):.1f}" height="{h}" rx="{h / 2}" fill="url(#cf)"/>')
    if opp and opp[1] > opp[0]:
        s.append(_hatch_lines(px(opp[0]), px(opp[1]), h, '#D8E166', 0.45).replace('y1="', f'y1="').replace('<line ', f'<g transform="translate(0,{top})"><line ').replace('/>', '/></g>') if False else f'<g transform="translate(0,{top})">' + _hatch_lines(px(opp[0]), px(opp[1]), h, '#D8E166', 0.45) + '</g>')
    if zone_w:
        s.append(f'<rect x="{px(zone_l):.1f}" y="{top}" width="{px(zone_w):.1f}" height="{h}" rx="3" fill="#D8E166" fill-opacity="0.16"/>')
        s.append(f'<g transform="translate(0,{top})">' + _hatch_lines(px(zone_l), px(zone_l + zone_w), h, '#D8E166', 0.55) + '</g>')
    for t in (ticks or []):
        s.append(f'<rect x="{px(t["pos"]) - 1:.1f}" y="{top}" width="2" height="{h}" fill="{"#D8E166" if t.get("target") else "#2a2628"}"/>')
    if median is not None:
        s.append(f'<rect x="{px(median) - 1:.1f}" y="{top}" width="2" height="{h}" fill="#B5C851"/>')
    if dot is not None:
        cx, cy, r = px(dot), top + h / 2, 9.0
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="#B5C851" stroke="#231f20" stroke-width="2"/>')
        if dot_cls == 'off-low':
            s.append(f'<polygon points="{cx + 13:.1f},{cy - 5:.1f} {cx + 20:.1f},{cy:.1f} {cx + 13:.1f},{cy + 5:.1f}" fill="#B5C851"/>')
        elif dot_cls == 'off-high':
            s.append(f'<polygon points="{cx - 13:.1f},{cy - 5:.1f} {cx - 20:.1f},{cy:.1f} {cx - 13:.1f},{cy + 5:.1f}" fill="#B5C851"/>')
    s.append('</svg>')
    return ''.join(s)


def _rec_row_html(r, band):
    you, kind = _num(r.get('you')), r.get('unit_kind')
    p5, p25, p50, p75, p95, in_zone = band
    pad = ((p95 - p5) or (abs(p95) * 0.1 + 1e-6)) * 0.06
    s_min, s_max = p5 - pad, p95 + pad
    def pos(v):
        return max(0.0, min(100.0, ((v - s_min) / (s_max - s_min)) * 100))
    zone_l = pos(p25)
    zone_w = max(pos(p75) - zone_l, 1.5)
    off_low, off_high = you < p5, you > p95
    dot = max(2.0, min(98.0, pos(you)))
    if in_zone:
        dot = max(zone_l, min(zone_l + zone_w, dot))
    dot_cls = ' off-low' if off_low else ' off-high' if off_high else ''
    edge = p25 if you < p25 else p75
    is_level = kind == 'level'
    you_str = _level_label(r.get('levels'), you) if is_level else _fmt_feat(kind, you)
    if is_level:
        lo, hi = _level_label(r.get('levels'), p25), _level_label(r.get('levels'), p75)
        zone_str = lo if lo == hi else f'{lo} – {hi}'
    else:
        zone_str = _fmt_range(kind, p25, p75)
    move = '✓ in the zone' if in_zone else ('aim for ' + _level_label(r.get('levels'), edge)) if is_level else (_fmt_move(kind, you, edge) + ' to land in')
    agree = r.get('agree') or [0, 0]
    return (f'<div class="rr"><div class="rr-head"><span class="rr-dom">{_esc(r.get("domain") or "")}</span>'
            f'<span class="rr-act">{_esc(r.get("action") or "")}</span><span class="rr-move{" inrange" if in_zone else ""}">{_esc(move)}</span></div>'
            f'<div class="rr-bar">{_meter_svg(6.55, zone_l, zone_w, pos(p50), dot, dot_cls.strip())}</div>'
            f'<div class="rr-leg"><span class="you">You <b>{_esc(you_str)}</b></span><span class="zone">Target zone <b>{_esc(zone_str)}</b></span>'
            f'<span class="ag">{agree[0]}/{agree[1]} agree</span></div></div>')


def _rec_ranges_html(ranges, f):
    """Port of renderRecRanges(): loudness row swapped to whole-track units when both sides have it."""
    you_int = _num(f.get('lufs_whole_track'))
    if you_int is None:
        you_int = _num(f.get('lufs_integrated_est'))
    adjust, strengths = [], []
    for r in ranges:
        if _num(r.get('you')) is None:
            continue
        rr = dict(r)
        if r.get('feature') == 'lufs_integrated' and you_int is not None and r.get('percentiles_est'):
            p = r['percentiles_est']
            action = r.get('action')
            ai = r.get('actions_integrated') or {}
            if you_int < _num(p.get('p25')) and ai.get('higher'):
                action = ai['higher']
            elif you_int > _num(p.get('p75')) and ai.get('lower'):
                action = ai['lower']
            rr.update({'you': you_int, 'percentiles': p, 'action': action, 'target_cohort': None, 'target_signature': None})
        band = _rec_band(rr)
        (strengths if band[5] else adjust).append(_rec_row_html(rr, band))
    html = ''
    if adjust:
        html += '<div class="rr-group">Adjustments to make</div>' + ''.join(adjust)
    if strengths:
        html += '<div class="rr-group strengths">✓ What you\'re already nailing</div>' + ''.join(strengths)
    return html or '<div class="rr-group">No strong consensus from your peer cohort.</div>', len(adjust), len(strengths)


# --- app.js ports: the 0–100 percentile bars ---------------------------------
def _bar_html(fill_pct, dot_pct, labels, ticks=(25, 50, 75, 99), min_gap=9, opp=None, tick_pos=None):
    """conv-bar with fill, ticks, dot and the collision-aware label row (site rule: higher priority wins)."""
    kept = []
    for l in sorted(labels, key=lambda x: -x['pr']):
        if all(abs(k['pos'] - l['pos']) >= min_gap for k in kept):
            kept.append(l)
    ticks_html = ''.join(f'<div class="tick{" target" if t.get("target") else ""}" style="left:{t["pos"]:.1f}%"></div>' for t in (tick_pos or [{'pos': p} for p in ticks]))
    opp_html = f'<div class="opp" style="left:{opp[0]:.1f}%;width:{max(0.0, opp[1] - opp[0]):.1f}%"></div>' if opp else ''
    labels_html = ''.join(f'<div class="lab{" you" if l.get("you") else ""}{" tgt" if l.get("tgt") else ""}" style="left:{l["pos"]:.1f}%"><span>{_esc(l["name"])}</span><span>{_esc(l["val"])}</span></div>' for l in kept)
    svg = _meter_svg(6.55, 0, 0, None, dot_pct if fill_pct is not None else None, '', fill_to=fill_pct, opp=opp,
                     ticks=(tick_pos or [{'pos': p} for p in ticks]))
    return f'<div class="cbar">{svg}</div><div class="cbar-labels">{labels_html}</div>'


def _pct_label(pct):
    if pct is None:
        return 'no peer data'
    p = round(pct * 100)
    return 'top 1%' if p >= 99 else 'top 10%' if p >= 90 else 'top 25%' if p >= 75 else 'above average' if p >= 50 else 'below average' if p >= 25 else 'bottom 25%'


def _fmt_num(n):
    n = _num(n)
    if n is None:
        return 'N/A'
    if 0 < n < 1:
        return f'{n:.2f}'
    if n >= 1000:
        return f'{n:,.0f}'
    return f'{round(n):,}'


def _listeners_str(n):
    n = _num(n)
    if not n:
        return '—'
    if n >= 1e6:
        return f'{n / 1e6:.1f}M monthly listeners'
    if n >= 1e3:
        return f'{n / 1e3:.0f}K monthly listeners'
    return f'{n:,.0f} monthly listeners'


def _quadrant_svg(tm, so, pitch, cloud):
    """Port of the Sonic Quadrant scatter: X = performance percentile, Y = originality, cuts at 75/75.
    Every style is an inline presentation attribute: WeasyPrint does not apply CSS classes inside inline SVG."""
    W, H, ML, MR, MT, MB = 640, 460, 60, 30, 30, 50
    iw, ih = W - ML - MR, H - MT - MB
    x = lambda v: ML + (v / 100.0) * iw
    y = lambda v: MT + ((100.0 - v) / 100.0) * ih
    FONT = "font-family=\"'Space Grotesk',Helvetica,Arial,sans-serif\""
    user_perf = round((_num(tm.get('composite_percentile')) or 0) * 100)
    user_orig = _num(so.get('composite_score')) or 0
    s = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block;background:#1d1c19;border-radius:6px">']
    for g in (25, 50, 75):
        s.append(f'<line x1="{x(g):.1f}" y1="{MT}" x2="{x(g):.1f}" y2="{MT + ih}" stroke="#ffffff" stroke-opacity="0.07" stroke-width="1"/>'
                 f'<line x1="{ML}" y1="{y(g):.1f}" x2="{ML + iw}" y2="{y(g):.1f}" stroke="#ffffff" stroke-opacity="0.07" stroke-width="1"/>')
    s.append(f'<line x1="{x(75):.1f}" y1="{MT}" x2="{x(75):.1f}" y2="{MT + ih}" stroke="#4ecdc4" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="4 4"/>'
             f'<line x1="{ML}" y1="{y(75):.1f}" x2="{ML + iw}" y2="{y(75):.1f}" stroke="#4ecdc4" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="4 4"/>')
    ql = f'font-size="11" font-weight="700" letter-spacing="0.5" {FONT}'
    s.append(f'<text x="{x(37.5):.1f}" y="{y(95):.1f}" text-anchor="middle" fill="#4ecdc4" fill-opacity="0.75" {ql}>AHEAD OF THE CURVE</text>')
    s.append(f'<text x="{x(99):.1f}" y="{y(95):.1f}" text-anchor="end" fill="#d4be8e" fill-opacity="0.9" {ql}>SIGNATURE OF SUCCESS</text>')
    s.append(f'<text x="{x(37.5):.1f}" y="{y(6):.1f}" text-anchor="middle" fill="#a89568" fill-opacity="0.85" {ql}>STUCK IN THE PACK</text>')
    s.append(f'<text x="{x(99):.1f}" y="{y(6):.1f}" text-anchor="end" fill="#4ecdc4" fill-opacity="0.75" {ql}>GENRE-PLAYBOOK WINNER</text>')
    s.append(f'<line x1="{ML}" y1="{MT + ih}" x2="{ML + iw}" y2="{MT + ih}" stroke="#ffffff" stroke-opacity="0.25" stroke-width="1"/>'
             f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + ih}" stroke="#ffffff" stroke-opacity="0.25" stroke-width="1"/>')
    al = f'font-size="12" fill="#888888" {FONT}'
    for t in (0, 25, 50, 75, 100):
        s.append(f'<text x="{x(t):.1f}" y="{MT + ih + 18}" text-anchor="middle" {al}>{t}</text><text x="{ML - 10}" y="{y(t) + 4:.1f}" text-anchor="end" {al}>{t}</text>')
    at = f'font-size="12" font-weight="600" fill="#aaaaaa" {FONT}'
    s.append(f'<text x="{ML + iw / 2:.1f}" y="{H - 10}" text-anchor="middle" {at}>Performance percentile →</text>')
    s.append(f'<text x="15" y="{MT + ih / 2:.1f}" text-anchor="middle" transform="rotate(-90 15 {MT + ih / 2:.1f})" {at}>Originality score →</text>')
    pitch = (pitch or [])[:5]
    names = {p.get('name') for p in pitch}
    for p in (cloud or []):
        if p.get('name') in names:
            continue
        s.append(f'<circle cx="{x(round((_num(p.get("perf_pct")) or 0) * 100)):.1f}" cy="{y(_num(p.get("orig_score")) or 0):.1f}" r="2.4" fill="#4ecdc4" fill-opacity="0.22"/>')
    for i, p in enumerate(pitch):
        px, py = x(round((_num(p.get('perf_pct')) or 0) * 100)), y(_num(p.get('orig_score')) or 0)
        s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#4ecdc4" fill-opacity="0.6" stroke="#4ecdc4" stroke-opacity="0.9" stroke-width="1"/>')
        dx, dy = ((8, 3), (-8, 3), (8, 15), (-8, -9), (8, -9))[i % 5]
        s.append(f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" text-anchor="{"start" if dx > 0 else "end"}" font-size="10" fill="#cccccc" {FONT}>{_esc(p.get("name") or "")}</text>')
    ux, uy = x(user_perf), y(user_orig)
    s.append(f'<circle cx="{ux:.1f}" cy="{uy:.1f}" r="10" fill="#d4be8e" stroke="#ffffff" stroke-width="2"/>'
             f'<text x="{ux + 14:.1f}" y="{uy + 4:.1f}" font-size="12" font-weight="700" fill="#d4be8e" {FONT}>You ({user_perf}, {round(user_orig)})</text></svg>')
    return ''.join(s)


def build_breakdown_html(job: dict, prepared_for: str | None = None) -> str:
    rj = _j(job.get('result_json')) or {}
    f = rj.get('features') or _j(job.get('features')) or {}
    up = rj.get('user_profile') or {}
    src = rj.get('source') or {}
    tm = up.get('track_momentum') or {}
    cc = up.get('conversion_comparison') or {}
    so = up.get('sonic_originality') or {}
    quad = up.get('quadrant') or {}
    pitch = up.get('pitch_comparables') or []
    cloud = up.get('cohort_scatter') or []
    ranges = rj.get('recommendation_ranges') or []
    matches = rj.get('matches') or []
    related = [x.get('name') for x in (_j(job.get('related_artists')) or []) if isinstance(x, dict) and x.get('name')]

    track = job.get('track_name') or src.get('track_name') or 'Untitled'
    artist = job.get('artist_name') or src.get('artist_name') or ''
    listeners = _num(up.get('listeners')) or _num(src.get('artist_listeners'))
    followers = _num(up.get('followers'))
    conv = _num(up.get('conversion_rate'))
    tier = (src.get('artist_tier') or rj.get('user_tier') or '').capitalize()
    peer_count = int(_num(tm.get('peer_count')) or _num(cc.get('peer_pool_total')) or len(rj.get('all_matches') or []) or 0)
    today = _dt.date.today().strftime('%B %-d, %Y')

    # ---- page 1: the sound (unchanged) --------------------------------------
    lufs, lufs_est = _num(f.get('lufs_integrated')), _num(f.get('lufs_integrated_est'))
    key = f"{f.get('key') or '?'} {f.get('scale') or ''}".strip()
    tiles = [(f"{_num(f.get('bpm')) or 0:.0f}", 'BPM', ''), (_esc(key), 'Key', ''),
             (f"{lufs:.1f}" if lufs is not None else '—', 'LUFS', f"integrated · est. {lufs_est:.1f} on streaming" if lufs_est is not None else 'integrated'),
             (_energy_label(_num(f.get('energy'))), 'Energy', ''), (_compression_label(_num(f.get('compression_amount'))), 'Compression', ''),
             (_dance_label(_num(f.get('danceability'))), 'Danceability', '')]
    tiles_html = ''.join(f'<div class="tile"><b>{v}</b><span>{lab}</span>{("<small>" + _esc(sub) + "</small>") if sub else ""}</div>' for v, lab, sub in tiles)
    rmap = {r['feature']: r for r in ranges if r.get('feature')}
    out_bands, rows = [], ''
    for name, key_ in BANDS:
        v = _num(f.get(key_))
        if v is None:
            continue
        w = min(100.0, v / 0.30 * 100)
        zone_html, cls = '', ''
        p25 = p75 = 0.0
        r = rmap.get(key_)
        if r and r.get('percentiles'):
            _, p25, _, p75, _, in_zone = _rec_band(r)
            if not in_zone:
                cls = ' out'
                out_bands.append((name, 'light' if v < p25 else 'heavy'))
        zl = (p25 / 0.30 * 100) if (r and r.get('percentiles')) else 0
        zw = max(0.5, (p75 - p25) / 0.30 * 100) if (r and r.get('percentiles')) else 0
        FW = 500.0
        tr = (f'<svg width="5.0in" height=".16in" viewBox="0 0 {FW:.0f} 16" xmlns="http://www.w3.org/2000/svg" style="display:block">'
              f'<rect x="0" y="1" width="{FW:.0f}" height="14" rx="3" fill="#2a2925"/>'
              f'<rect x="0" y="1" width="{min(100, w) / 100 * FW:.1f}" height="14" rx="3" fill="{"#D8E166" if cls else "#B5C851"}"/>'
              + (f'<rect x="{zl / 100 * FW:.1f}" y="1" width="{zw / 100 * FW:.1f}" height="14" fill="#D8E166" fill-opacity="0.18"/>'
                 f'<g transform="translate(0,1)">{_hatch_lines(zl / 100 * FW, (zl + zw) / 100 * FW, 14, "#D8E166", 0.7)}</g>' if zw else '')
              + '</svg>')
        rows += f'<div class="row"><div class="lab">{name}</div><div class="track">{tr}</div><div class="val{cls}">{v * 100:.1f}%</div></div>'
    cohort_n = so.get('cohort_size') or ((rmap.get('mid_ratio') or {}).get('agree') or [0, 0])[1]
    if out_bands:
        plural = {'Mid': 'mids are', 'Low-Mid': 'low-mids are', 'Hi-Mid': 'hi-mids are'}
        parts = [f'the {plural.get(n, n.lower() + " is")} {d}' for n, d in out_bands]
        band_note = (f'Share of spectral energy per band. Striped zones mark where the {cohort_n} highest-converting peers in this lane sit. '
                     f'{"One band lands" if len(out_bands) == 1 else str(len(out_bands)) + " bands land"} outside them: {", ".join(parts)}.')
    else:
        band_note = f'Share of spectral energy per band. Striped zones mark where the {cohort_n} highest-converting peers in this lane sit. Every band lands inside them.'
    emos = []
    es = f.get('emotion_summary') or {}
    for e, s_ in (es.get('emotions') or [])[:4]:
        emos.append((str(e).capitalize(), round(float(s_) * 100)))
    if not emos:
        for i in (1, 2, 3, 4):
            if f.get(f'emotion_{i}'):
                emos.append((str(f[f'emotion_{i}']).capitalize(), round((_num(f.get(f'emotion_{i}_score')) or 0) * 100)))
    emo_html = ''.join(f'<span class="chip">{_esc(e)}<b>{s_}%</b></span>' for e, s_ in emos)
    emo_note = (f'{emos[0][0]} and {emos[1][0].lower()} leading' + (f', {emos[2][0].lower()} right behind them.' if len(emos) > 2 else '.')) if len(emos) >= 2 else ''

    # ---- page 2: where your track stands (site's track-momentum panel + audience conversion) ----
    comp_pct = round((_num(tm.get('composite_percentile')) or 0) * 100)
    tm_html = ''
    if tm:
        labels = [dict(pos=25, name='p25', val='25', pr=1), dict(pos=50, name='Median', val='50', pr=2), dict(pos=75, name='Top 25%', val='75', pr=4),
                  dict(pos=99, name='Top 1%', val='99', pr=3), dict(pos=comp_pct, name='You', val=str(comp_pct), pr=5, you=True)]
        bar = _bar_html(comp_pct, comp_pct, labels)
        if comp_pct >= 90:
            summ = 'Your track is in the <b>top 10%</b> of its sonic cohort, performing better than nearly every track that sounds like it.'
        elif comp_pct >= 75:
            summ = 'Your track is in the <b>top 25%</b> of its sonic cohort, outperforming most tracks that sound like it.'
        elif comp_pct >= 50:
            summ = 'Your track is <b>above average</b> for its sonic cohort, doing better than most similar-sounding tracks.'
        elif comp_pct >= 25:
            summ = 'Your track is <b>below average</b> for its sonic cohort. There is headroom on the momentum side.'
        else:
            summ = 'Your track is in the <b>bottom 25%</b> of its sonic cohort. The biggest lifts here are playlist pitching and Spotify popularity growth.'
        def tm_row(label, sub, scanned, stats, pct):
            stats = stats or {}
            pct_s = (f'<span class="pct">{_pct_label(_num(pct))}</span> of {int(_num(stats.get("count")) or 0):,} sonic peers') if pct is not None else 'no comparable peer data'
            peer_s = f'<br>peer median {_fmt_num(stats.get("median"))} · top 25% {_fmt_num(stats.get("p75"))} · top 1% {_fmt_num(stats.get("p99"))}' if stats else ''
            val = _fmt_num(scanned) if scanned is not None else 'N/A'
            return f'<div class="tmr"><span class="tml">{label}<small>{sub}</small></span><span class="tmv{" na" if scanned is None else ""}">{val}</span><span class="tmp">{pct_s}{peer_s}</span></div>'
        rows_html = (tm_row('Spotify Popularity', 'Spotify’s 0–100 recency-weighted score', tm.get('scanned_popularity'), tm.get('pop_stats'), tm.get('percentile_popularity'))
                     + tm_row('Chartmetric Score', 'Multi-platform composite (0–100)', tm.get('scanned_cm_score'), tm.get('cm_stats'), tm.get('percentile_cm_score'))
                     + tm_row('Playlist Placements', 'Editorial + user playlists combined', tm.get('scanned_playlists'), tm.get('playlists_stats'), tm.get('percentile_playlists')))
        cur = _num(tm.get('gap_current_revenue')) or 0
        rate = _num(tm.get('revenue_per_listener'))
        tier_name, tgt_l, tgt_r, add = 'top 25%', _num(tm.get('gap_target_listeners')), _num(tm.get('gap_target_revenue')), _num(tm.get('gap_additional_revenue'))
        if comp_pct >= 75 and _num(tm.get('gap_target_listeners_t10')):
            tier_name, tgt_l, tgt_r, add = 'top 10%', _num(tm.get('gap_target_listeners_t10')), _num(tm.get('gap_target_revenue_t10')), _num(tm.get('gap_additional_revenue_t10'))
        gap = ''
        if add and add > 0 and tgt_l and rate:
            gap = (f'<div class="gap"><b>What “closing the gap” looks like:</b> tracks in the {tier_name} of this sonic cohort belong to artists with a median of <b>{tgt_l:,.0f} monthly listeners</b>, '
                   f'about ${tgt_r:,.0f} a year at ${rate:.2f} per listener, against ${cur:,.0f} today. Closing it is worth about <b>+${add:,.0f} a year</b>.'
                   f'<span class="gap-note">Peer-typical correlation from your actual sonic cohort, what artists with tracks at this level typically have. Not a personal forecast.</span></div>')
        elif comp_pct >= 75 and tgt_l and rate:
            gap = (f'<div class="gap"><b>You’re pacing the peer tier:</b> artists with tracks in the {tier_name} of this cohort sit at a median of <b>{tgt_l:,.0f} monthly listeners</b>, about ${tgt_r:,.0f} a year.'
                   f'<span class="gap-note">Peer-typical correlation from your actual sonic cohort. Not a personal forecast.</span></div>')
        tm_html = (f'<div class="panel-title">Where your track stands <span class="sub">vs {peer_count:,} tracks that sound like yours</span></div>'
                   f'<div class="tag">How this song is performing right now, compared to tracks that sound like yours.</div>{bar}<div class="summ">{summ}</div><div class="tmrows">{rows_html}</div>{gap}')

    conv_html = ''
    if conv is not None or cc.get('peer_median') is not None:
        cr = conv or 0.0
        p25, med, p75 = _num(cc.get('peer_bottom_25')) or 0, _num(cc.get('peer_median')) or 0, _num(cc.get('peer_top_25')) or 0
        p99 = min(_num(cc.get('peer_p99')) or p75 * 2, p75 * 3)
        at_top = cr >= p75
        target = p99 if (at_top and p99 > cr) else p75
        scale_max = max(p99, cr, target) * 1.05 or 1
        tp = lambda v: max(0.0, min(100.0, (v / scale_max) * 100))
        labels = [dict(pos=tp(p25), name='Bottom 25%', val=f'{p25:.2f}%', pr=1), dict(pos=tp(med), name='Median', val=f'{med:.2f}%', pr=2), dict(pos=tp(p75), name='Top 25%', val=f'{p75:.2f}%', pr=1)]
        ticks = [dict(pos=tp(p25)), dict(pos=tp(med)), dict(pos=tp(p75)), dict(pos=tp(p99))]
        opp = None
        if conv is not None:
            labels.append(dict(pos=tp(cr), name='You', val=f'{cr:.2f}%', pr=5, you=True))
        if target > cr:
            labels.append(dict(pos=tp(target), name='Top 1%' if at_top else 'Top 25%', val=f'{target:.2f}%', pr=4, tgt=True))
            ticks.append(dict(pos=tp(target), target=True))
            opp = (tp(cr), tp(target)) if conv is not None else None
        if not (at_top and target == p99):
            labels.append(dict(pos=tp(p99), name='Top 1%', val=f'{p99:.2f}%', pr=1))
        bar = _bar_html(tp(cr) if conv is not None else 0, tp(cr) if conv is not None else 0, labels, min_gap=8, opp=opp, tick_pos=ticks)
        fl, bucket = _num(up.get('fol_listener_ratio')), up.get('retention_bucket')
        small = listeners is not None and 0 < listeners < 500
        msg = ''
        if fl is not None and bucket:
            ratio = f'<b>{fl:.2f} followers per monthly listener</b>'
            if bucket == 'healthy':
                msg = f'Your ratio of {ratio} sits in the healthy retention band (0.1–1.0 per Chartlex / Chartmetric benchmarks). Above 0.1 correlates with 2–3× more Spotify Release Radar placement.'
            elif bucket == 'marginal':
                msg = f'Your ratio of {ratio} is in the marginal band, close to the threshold where retention becomes a concern. Healthy is above 0.1; below 0.067 indicates audience width without depth.'
            elif bucket == 'shallow':
                msg = f'Your ratio of {ratio} is below the shallow-audience threshold (0.067 per Chartlex). Your monthly listener count is growing faster than fan retention, width without depth.'
            else:
                msg = f'Your ratio of {ratio} is unusually high, typically seen on superstar accounts with massive cumulative followers, or on accounts whose monthly listener count has dropped off.'
        elif conv is not None:
            peer_n = int(_num(cc.get('peer_count')) or 0)
            fans = int(_num(up.get('additional_fans')) or 0)
            if fans > 0 and at_top:
                msg = f'You’re already in the <b>top 25%</b> of {peer_n:,} sonic peers. Reaching the top 1% ({target:.1f}%) would convert an estimated <b>{fans:,} more fans</b>.'
            elif fans > 0:
                msg = f'Across {peer_n:,} sonic peers, the top 25% convert at <b>{target:.1f}%</b>. Closing that gap means an estimated <b>{fans:,} additional fans</b>, about {round(fans / 12):,} a month.'
            else:
                msg = f'You’re converting at <b>{cr:.1f}%</b>, above the top 1% of {peer_n:,} sonic peers. Your listener-to-follower conversion is exceptional.'
        if small and msg:
            msg = f'<b>Small sample:</b> with only {listeners:,.0f} monthly listeners this ratio reflects a tiny audience, often friends and early supporters. ' + msg
        conv_html = f'<div class="panel-title" style="margin-top:.22in">Audience conversion <span class="sub">listeners becoming followers</span></div>{bar}<div class="summ">{msg}</div>'

    # ---- page 3: originality + sonic quadrant + comparables --------------------
    orig_html, quad_html, pitch_html = '', '', ''
    score = _num(so.get('composite_score'))
    if score is not None:
        sc = round(score)
        labels = [dict(pos=25, name='Low orig', val='25', pr=1), dict(pos=50, name='Typical', val='50', pr=2), dict(pos=75, name='Distinct', val='75', pr=3),
                  dict(pos=99, name='Singular', val='99', pr=2), dict(pos=sc, name='You', val=str(sc), pr=5, you=True)]
        bar = _bar_html(sc, sc, labels)
        if sc >= 75:
            summ = 'Your sound is in the <b>top 25% of sonically distinct tracks</b> within your cohort. The deviations below are your signature.'
        elif sc >= 50:
            summ = 'Your sound is <b>moderately distinct</b> from your sonic cohort: some signature features, mostly within consensus.'
        elif sc >= 25:
            summ = 'Your sound <b>mostly follows cohort consensus</b>. You’re executing the genre playbook more than reinventing it.'
        else:
            summ = 'Your sound <b>closely matches cohort consensus</b> on most dimensions. Strong commercial fit; low sonic differentiation.'
        qcls = 'signature' if quad.get('quadrant') == 'signature_of_success' else 'stuck' if quad.get('quadrant') == 'stuck_in_pack' else ''
        qbox = f'<div class="qbox {qcls}"><div class="ql">{_esc(quad.get("label") or "")}</div><div class="qm">{_esc(quad.get("message") or "")}</div></div>' if quad else ''
        devs = ''.join(f'<div class="orow"><span class="ol">{_esc(FEATURE_PRETTY.get(d.get("feature"), d.get("feature")))}</span><span class="oz">{"+" if (_num(d.get("z")) or 0) > 0 else ""}{(_num(d.get("z")) or 0):.2f}σ</span>'
                       f'<span class="oc"><i>{_esc((FEATURE_DIRECTION.get(d.get("feature")) or ("higher than", "lower than"))[0 if d.get("direction") == "high" else 1])}</i> cohort consensus · you {_esc(d.get("user_val"))} vs median {_esc(d.get("cohort_mean"))}</span></div>'
                       for d in (so.get('top_deviations') or [])[:4]) or '<div class="note">No strongly distinctive features: every dimension is within 1σ of your cohort consensus.</div>'
        fits = ''.join(f'<div class="orow"><span class="ol">{_esc(FEATURE_PRETTY.get(d.get("feature"), d.get("feature")))}</span><span class="oz fits">{"+" if (_num(d.get("z")) or 0) > 0 else ""}{(_num(d.get("z")) or 0):.2f}σ</span>'
                       f'<span class="oc">matches cohort · you {_esc(d.get("user_val"))} vs median {_esc(d.get("cohort_mean"))}</span></div>'
                       for d in (so.get('fits_consensus') or [])[:4]) or '<div class="note">No close-consensus dimensions.</div>'
        orig_html = (f'<div class="panel-title">Sonic originality</div>{qbox}{bar}<div class="summ">{summ}</div>'
                     f'<div class="grid2" style="margin-top:.1in"><div><h3 style="margin-top:0">Where your sound stands out</h3>{devs}</div><div><h3 style="margin-top:0">Where you match the consensus</h3>{fits}</div></div>')
    if tm and so:
        quad_html = f'<div class="panel-title" style="margin-top:.16in">Sonic quadrant <span class="sub">originality × performance, cuts at 75 / 75</span></div><div class="sqwrap">{_quadrant_svg(tm, so, pitch, cloud)}</div>'
    if pitch:
        prow = ''
        for i, p in enumerate(pitch[:5]):
            name = f'{p.get("name")} — {p.get("track_name")}' if p.get('track_name') else (p.get('name') or '')
            sim, perf = round((_num(p.get('similarity')) or 0) * 100), round((_num(p.get('perf_pct')) or 0) * 100)
            og = _num(p.get('orig_score')) or 0
            cm = f'{round(_num(p.get("cm_track_score")))}' if _num(p.get('cm_track_score')) is not None else 'N/A'
            if og >= 80 and perf >= 80:
                angle = 'Distinctive sonic profile AND scaled performance: clean proof your sound rewards distinctiveness in this lane.'
            elif og >= 80:
                angle = 'Strong sonic distinctiveness for the tier: proof your kind of sonic edge has commercial traction.'
            elif perf >= 80:
                angle = 'Top performer in your sonic neighborhood: a comparable on numbers, not just sound.'
            else:
                angle = 'Sits in your sonic lane with both distinctiveness and traction above the cohort floor.'
            pop = p.get('sp_track_popularity')
            pop = pop if pop is not None else '—'
            prow += (f'<div class="prow"><div class="pn">{i + 1}. {_esc(name)}</div><div class="pl">{_listeners_str(p.get("listeners"))}</div>'
                     f'<div class="pst"><span>{sim}% sonic match</span><span>Popularity {_esc(pop)}</span><span>CM {cm}</span><span>{int(_num(p.get("playlists_total")) or 0):,} playlists</span><span>Originality {round(og)}</span></div>'
                     f'<div class="pa">{angle}</div></div>')
        pitch_html = f'<div class="panel-title">A&amp;R pitch comparables <span class="sub">same tier, sonic peers in Signature of Success</span></div><div class="prows">{prow}</div>'

    # ---- page 4: production recommendations (the site's meters, verbatim port) ----
    rec_html, n_adj, n_ok = _rec_ranges_html(ranges, f)

    # ---- page 5: similar artists (site table, top 8) + related + method ----
    def _mp(m):
        return min(0.99, max(0.0, (_num(m.get('similarity')) or 0) + (_num(m.get('_tag_aff')) or 0)))
    def _genres(m):
        g = []
        for k in ('primary_genre', 'secondary_genre'):
            v = m.get(k)
            if v and str(v).lower() != 'unknown' and v not in g:
                g.append(v)
        for v in (m.get('artist_genres') or []):
            if v and v not in g:
                g.append(v)
        return ', '.join(g) if g else '-'
    peers = [m for m in matches if m.get('name')][:8]
    sim_rows = ''.join(
        f'<tr><td>{i + 1}</td><td class="nm">{_esc(m["name"])}{" <span class=\"tag\">audience match</span>" if m["name"] in related else ""}</td>'
        f'<td class="num">{_mp(m) * 100:.1f}%</td><td class="num">{(_num(m.get("conversion_rate")) or 0):.1f}%</td><td>{_esc(m.get("tier") or "-")}</td>'
        f'<td class="gn">{_esc(_genres(m))}</td><td class="em">{"".join("<span class=\"mini\">" + _esc(EMOTION_LABELS.get(e, e)) + "</span>" for e in [e for e in (m.get("emotions") or []) if e and e != "neutral"][:3])}</td></tr>'
        for i, m in enumerate(peers))
    rel_html = ''.join(f'<span class="chip">{_esc(n)}</span>' for n in related[:10])

    prepared = f'Prepared for {_esc(prepared_for)} · {today}' if prepared_for else today
    foot = lambda n: f'<div class="foot"><span>{_esc(artist)} · {_esc(track)}</span><span>Freshly Baked Studios · Brooklyn</span><span>{n} / 6</span></div>'
    fonts = ("@font-face{font-family:'Londrina Solid';font-weight:400;src:url('fonts/LondrinaSolid-Regular.ttf') format('truetype')}"
             "@font-face{font-family:'Londrina Solid';font-weight:900;src:url('fonts/LondrinaSolid-Black.ttf') format('truetype')}"
             "@font-face{font-family:'Space Grotesk';font-weight:300 700;src:url('fonts/SpaceGrotesk[wght].ttf') format('truetype')}")
    css = fonts + """
@page{size:letter;margin:0}
*{box-sizing:border-box}
html,body{margin:0;background:#171614;color:#DEE6B8;font-family:'Space Grotesk',Helvetica,Arial,sans-serif;font-size:11pt;line-height:1.42;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.pg{width:8.5in;height:11in;padding:.62in .7in .55in;position:relative;page-break-after:always;background:#171614;overflow:hidden}
.pg:last-child{page-break-after:auto}
.eyebrow{font-size:8.5pt;letter-spacing:.18em;text-transform:uppercase;color:#8D8F59}
h1{font-family:'Londrina Solid',Impact,sans-serif;font-weight:900;font-size:42pt;line-height:.95;margin:.12in 0 0;color:#D8E166}
h2{font-family:'Londrina Solid',Impact,sans-serif;font-weight:400;font-size:22pt;margin:0 0 .08in;color:#D8E166}
h3{font-size:8.5pt;letter-spacing:.14em;text-transform:uppercase;color:#8D8F59;margin:.2in 0 .07in;font-weight:700}
.artist{font-size:18pt;color:#DEE6B8;margin-top:.06in}
.meta{display:flex;gap:.28in;margin-top:.22in;padding-top:.16in;border-top:1px solid #33322c;flex-wrap:wrap}
.meta div{min-width:1.1in} .meta b{display:block;font-size:17pt;color:#fff;font-weight:700;line-height:1.1} .meta span{font-size:8pt;letter-spacing:.12em;text-transform:uppercase;color:#8D8F59}
.card{background:#201f1c;border:1px solid #2f2e29;border-radius:8px;padding:.14in .18in;margin-top:.12in}
.tiles{display:flex;gap:.08in} .tile{flex:1;background:#201f1c;border:1px solid #2f2e29;border-radius:8px;padding:.12in .06in;text-align:center} .tile b{display:block;font-family:'Londrina Solid',Impact,sans-serif;font-size:22pt;color:#D8E166;line-height:1} .tile span{display:block;font-size:8pt;letter-spacing:.12em;text-transform:uppercase;color:#8D8F59;margin-top:.04in} .tile small{display:block;font-size:7.5pt;color:#6f7050;margin-top:.02in}
.row{display:flex;align-items:center;gap:.12in;margin:.055in 0} .lab{width:.85in;font-size:9.5pt;color:#BABC95} .track{flex:1;height:.16in} .track .fill{position:absolute;left:0;top:0;bottom:0;background:#B5C851;border-radius:3px} .track .fill.out{background:#D8E166} .track .zone{position:absolute;top:0;bottom:0;background:repeating-linear-gradient(135deg,#D8E166 0,#D8E166 2px,transparent 2px,transparent 6px);opacity:.45} .val{width:.55in;text-align:right;font-size:9.5pt;color:#BABC95} .val.out{color:#D8E166;font-weight:700}
.chips{display:flex;gap:.08in;flex-wrap:wrap} .chip{border:1px solid #4a4a37;border-radius:999px;padding:.03in .13in;font-size:9.5pt;color:#DEE6B8} .chip b{color:#D8E166;font-weight:700;margin-left:.06in}
.note{font-size:9.5pt;color:#8D8F59;margin-top:.08in}
.grid2{display:flex;gap:.14in} .grid2>*{flex:1;min-width:0}
.panel-title{font-size:13pt;font-weight:700;color:#DEE6B8;margin:.04in 0 .02in} .panel-title .sub{font-size:9pt;font-weight:400;color:#8D8F59;margin-left:.08in}
.tag{font-size:9.5pt;color:#8D8F59;margin-bottom:.1in}
/* conv-bar port */
.cbar{margin-top:.04in}
.cbar .fill{position:absolute;left:0;top:0;height:.16in;border-radius:.08in 0 0 .08in;background:linear-gradient(90deg,#5a5a3a 0%,#888899 100%)}
.cbar .opp{position:absolute;top:0;height:.16in;background:repeating-linear-gradient(135deg,#D8E166 0,#D8E166 3px,transparent 3px,transparent 7px);opacity:.4}
.cbar .tick{position:absolute;top:0;width:2px;height:.16in;background:rgba(64,56,58,.9)} .cbar .tick.target{background:#D8E166}
.cbar .dot{position:absolute;top:-.02in;width:.2in;height:.2in;border-radius:50%;background:#B5C851;border:2px solid #201f1c;transform:translateX(-50%)}
.cbar-labels{position:relative;height:.34in;margin-top:.04in} .cbar-labels .lab{position:absolute;transform:translateX(-50%);font-size:8pt;color:#888899;line-height:1.25;text-align:center;width:.9in} .cbar-labels .lab span{display:block} .cbar-labels .lab.you{color:#B5C851;font-weight:700} .cbar-labels .lab.tgt{color:#D8E166}
.summ{font-size:10pt;color:#DEE6B8;margin:.04in 0 .08in} .summ b{color:#D8E166}
.tmrows{border-top:1px solid #2f2e29} .tmr{display:flex;align-items:flex-start;gap:.12in;padding:.06in 0;border-bottom:1px solid #2f2e29} .tml{width:1.7in;color:#ccc;font-weight:500;font-size:10pt} .tml small{display:block;color:#777;font-size:8pt;font-weight:400} .tmv{width:.8in;text-align:right;font-weight:700;color:#B5C851;font-size:14pt} .tmv.na{color:#666;font-weight:400;font-style:italic;font-size:10pt} .tmp{flex:1;color:#888;font-size:8.5pt;line-height:1.4} .tmp .pct{color:#B5C851;font-weight:600}
.gap{margin-top:.1in;background:rgba(78,205,196,.06);border-left:3px solid rgba(78,205,196,.5);padding:.08in .12in;border-radius:4px;font-size:9.5pt} .gap b{color:#4ecdc4} .gap-note{display:block;color:#8D8F59;font-size:8pt;margin-top:.04in}
/* originality + quadrant */
.qbox{background:rgba(78,205,196,.07);border:1px solid rgba(78,205,196,.25);border-radius:8px;padding:.1in .15in;margin:.06in 0 .08in} .qbox .ql{font-size:9pt;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#4ecdc4;margin-bottom:.03in} .qbox .qm{font-size:9.5pt;color:#DEE6B8} .qbox.signature .ql{color:#d4be8e} .qbox.stuck .ql{color:#a89568}
.orow{display:flex;gap:.08in;align-items:baseline;padding:.025in 0;border-bottom:1px solid #2f2e29;font-size:8.5pt} .ol{width:1.1in;color:#DEE6B8;font-weight:600} .oz{width:.5in;color:#D8E166;font-weight:700} .oz.fits{color:#B0C936} .oc{flex:1;color:#8D8F59} .oc i{color:#BABC95;font-style:normal}
.sqwrap{width:4.6in;margin:0 auto} .sq{width:100%;height:auto;display:block} .sq-grid{stroke:rgba(255,255,255,.06);stroke-width:1} .sq-quad{stroke:rgba(78,205,196,.5);stroke-width:1.5;stroke-dasharray:4 4} .sq-axis{stroke:rgba(255,255,255,.2);stroke-width:1} .sq-al{fill:#888;font-size:12px;font-family:'Space Grotesk',Helvetica,sans-serif} .sq-ql{fill:rgba(78,205,196,.7);font-size:11px;font-weight:700;letter-spacing:.5px;font-family:'Space Grotesk',Helvetica,sans-serif} .sq-ql.success{fill:rgba(212,190,142,.85)} .sq-ql.stuck{fill:rgba(168,149,104,.8)} .sq-cloud{fill:rgba(78,205,196,.22)} .sq-peer{fill:rgba(78,205,196,.6);stroke:rgba(78,205,196,.9);stroke-width:1} .sq-pl{fill:#ccc;font-size:10px;font-family:'Space Grotesk',Helvetica,sans-serif} .sq-user{fill:#d4be8e;stroke:#fff;stroke-width:2} .sq-ul{fill:#d4be8e;font-size:12px;font-weight:700;font-family:'Space Grotesk',Helvetica,sans-serif} .sq-at{fill:#aaa;font-size:12px;font-weight:600;font-family:'Space Grotesk',Helvetica,sans-serif}
/* pitch comparables */
.prows{margin-top:.02in} .prow{padding:.035in 0;border-bottom:1px solid rgba(255,255,255,.06)} .prow:last-child{border-bottom:0} .pn{font-size:10pt;font-weight:600;color:#B5C851;display:inline-block;width:5.2in} .pl{display:inline-block;width:1.7in;text-align:right;font-size:9pt;color:#aaa;vertical-align:top} .pst{margin-top:.03in;font-size:8pt} .pst span{display:inline-block;padding:1px 7px;background:rgba(78,205,196,.08);border:1px solid rgba(78,205,196,.2);border-radius:999px;margin-right:5px;color:#d4be8e} .pa{font-size:8.5pt;color:#d8d8d8;margin-top:0} .pa::before{content:"→ ";color:#4ecdc4;font-weight:700}
/* rec meters (rec-range port) */
.rr-group{font-size:8.5pt;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#888899;margin:.1in 0 .04in} .rr-group.strengths{color:#B0C936;margin-top:.14in}
.rr{margin:.06in 0 .1in} .rr-head{display:flex;align-items:baseline;gap:.08in;font-size:9.5pt;color:#e8e8f0;margin-bottom:.05in} .rr-dom{font-size:7pt;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#B5C851;background:rgba(216,225,102,.25);padding:1px 6px;border-radius:4px;white-space:nowrap} .rr-act{flex:1} .rr-move{white-space:nowrap;font-size:9pt;font-weight:700;color:#D8E166} .rr-move.inrange{color:#B0C936}
.rr-bar{margin-bottom:.03in} .rr-band{position:absolute;top:0;height:.14in;border-radius:3px;background:repeating-linear-gradient(135deg,#D8E166 0,#D8E166 3px,transparent 3px,transparent 7px);opacity:.5} .rr-edge{position:absolute;top:0;width:2px;height:.14in;background:#B5C851} .rr-dot{position:absolute;top:-.03in;width:.18in;height:.18in;border-radius:50%;background:#B5C851;border:2px solid #231f20;transform:translateX(-50%)}
.rr-dot.off-low::after,.rr-dot.off-high::after{content:'';position:absolute;top:4px;border:5px solid transparent} .rr-dot.off-low::after{left:15px;border-left-color:#B5C851} .rr-dot.off-high::after{right:15px;border-right-color:#B5C851}
.rr-leg{display:flex;gap:.16in;font-size:8.5pt;color:#888899} .rr-leg b{color:#e8e8f0;font-weight:600} .rr-leg .you b{color:#B5C851} .rr-leg .zone b{color:#D8E166}
/* tables */
table{width:100%;border-collapse:collapse;font-size:9pt} th{text-align:left;font-size:7.5pt;letter-spacing:.12em;text-transform:uppercase;color:#8D8F59;padding:.04in .05in;border-bottom:1px solid #3a3936} td{padding:.045in .05in;border-bottom:1px solid #2a2925;vertical-align:top} td.nm{color:#DEE6B8;font-weight:500} td.num{text-align:right;font-variant-numeric:tabular-nums} td.gn{color:#8D8F59;font-size:8pt} td.em{white-space:nowrap} td.em .mini{display:inline-block;border:1px solid #4a4a37;border-radius:999px;padding:0 5px;font-size:7.5pt;color:#BABC95;margin:1px 2px 1px 0} .tag{font-size:7pt;letter-spacing:.08em;text-transform:uppercase;color:#4ecdc4;margin-left:.06in}
.foot{position:absolute;left:.7in;right:.7in;bottom:.35in;display:flex;justify-content:space-between;font-size:8pt;color:#6f7050;border-top:1px solid #2f2e29;padding-top:.07in}
"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{_esc(track)} — Sonic Breakdown</title><style>{css}</style></head><body>
<div class="pg">
  <div class="eyebrow">Sonic breakdown · Freshly Baked Studios</div>
  <h1>{_esc(track)}</h1>
  <div class="artist">{_esc(artist)}</div>
  <div class="meta">
    <div><b>{_k(listeners)}</b><span>Monthly listeners</span></div>
    <div><b>{_k(followers)}</b><span>Followers</span></div>
    <div><b>{f'{conv:.2f}%' if conv is not None else '—'}</b><span>Listener → follower</span></div>
    <div><b>{_esc(tier) or '—'}</b><span>Tier</span></div>
    <div><b>{peer_count:,}</b><span>Sonic peers compared</span></div>
  </div>
  <h3>The sound</h3>
  <div class="tiles">{tiles_html}</div>
  <h3>Frequency balance</h3>
  <div class="card">{rows}<div class="note">{band_note}</div></div>
  <h3>Emotional character</h3>
  <div class="chips">{emo_html}</div>
  <div class="note">{emo_note}</div>
  <div class="foot"><span>{prepared}</span><span>Matched against a 300,000-track sonic database</span><span>1 / 6</span></div>
</div>
<div class="pg">
  <h2>Where you stand</h2>
  <div class="note" style="margin:0 0 .1in">This track’s momentum against its sonic peers’ tracks, plus artist-level context. Every signal is live Spotify or Chartmetric data.</div>
  <div class="card">{tm_html or '<div class="note">Track momentum was not available for this scan.</div>'}</div>
  <div class="card">{conv_html or '<div class="note">Audience conversion was not available for this scan.</div>'}</div>
  {foot(2)}
</div>
<div class="pg">
  <h2>Sonic originality</h2>
  <div class="card" style="margin-top:.04in">{orig_html or '<div class="note">Originality was not available for this scan.</div>'}</div>
  {foot(3)}
</div>
<div class="pg">
  <h2>Sonic quadrant and comparables</h2>
  <div class="card" style="margin-top:.04in">{quad_html or '<div class="note">Quadrant needs both momentum and originality.</div>'}</div>
  <div class="card">{pitch_html or '<div class="note">No pitch comparables for this scan.</div>'}</div>
  {foot(4)}
</div>
<div class="pg">
  <h2>Similar artists</h2>
  <div class="card" style="margin-top:.04in"><div class="panel-title">Closest records <span class="sub">from the 300,000-track universe, same order as the analyzer</span></div>
  <table><thead><tr><th>#</th><th>Artist</th><th class="num">Match</th><th class="num">Conversion</th><th>Tier</th><th>Genre</th><th>Emotions</th></tr></thead><tbody>{sim_rows}</tbody></table></div>
  <div class="grid2"><div class="card"><h3 style="margin-top:0">Chartmetric related artists</h3><div class="chips">{rel_html or '<span class="note">Not available for this scan.</span>'}</div></div>
  <div class="card"><h3 style="margin-top:0">How this was measured</h3><p style="margin:0;font-size:8.5pt;color:#BABC95">Sixty-three audio features from the record itself (loudness, spectrum, dynamics, rhythm, tonality, stereo) matched against the Freshly Baked Studios universe of 300,000+ measured tracks. Peers are same-tier artists whose records measure closest; target zones come from the highest-converting peers in the lane. Performance is Spotify popularity, Chartmetric score and playlist reach, ranked against those peers. Not a forecast.</p></div></div>
  {foot(5)}
</div>
<div class="pg">
  <h2>Production recommendations</h2>
  <div class="note" style="margin:0 0 .06in">Each meter is the same one on the analyzer: the striped band is where the highest-converting peers in this lane sit (25th to 75th percentile), the thin line is their median, the dot is this record. {n_adj} adjustment{'s' if n_adj != 1 else ''}, {n_ok} already in the zone.</div>
  <div class="card">{rec_html}</div>
  <div class="foot"><span>Alexander Almgren · almgren@freshlybakedstudios.com · freshlybakedstudios.com</span><span>6 / 6</span></div>
</div>
</body></html>"""


_NIX_LIBS = [('*-glib-2.*', 'libglib-2.0.so.0'), ('*-glib-2.*', 'libgobject-2.0.so.0'),
             ('*-harfbuzz-*', 'libharfbuzz.so.0'), ('*-fontconfig-2.*', 'libfontconfig.so.1'),
             ('*-cairo-1.*', 'libcairo.so.2'), ('*-pango-1.*', 'libpango-1.0.so.0'),
             ('*-pango-1.*', 'libpangoft2-1.0.so.0'), ('*-pango-1.*', 'libpangocairo-1.0.so.0')]
_preloaded = False


def _preload_nix_libs():
    """Railway/nixpacks puts pango, cairo, glib etc. in /nix/store but not on the
    loader path, so WeasyPrint's dlopen-by-name fails. Loading each library by
    absolute path (RTLD_GLOBAL) first makes the by-name lookups resolve to the
    already-loaded copies; their own deps resolve through nix RUNPATHs.
    Verified in the container 2026-09-07. No-op where /nix/store is absent."""
    global _preloaded
    if _preloaded or not os.path.isdir('/nix/store'):
        return
    import ctypes
    import glob

    def newest(pat):
        ds = [d for d in glob.glob('/nix/store/' + pat) if not d.endswith('.drv') and os.path.isdir(d + '/lib')]
        return sorted(ds)[-1] if ds else None
    for pat, so in _NIX_LIBS:
        d = newest(pat)
        if not d:
            continue
        try:
            ctypes.CDLL(os.path.join(d, 'lib', so), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    # fontconfig wants a config file; the nix package ships one, else write a minimal one.
    if not os.environ.get('FONTCONFIG_FILE'):
        confs = sorted(glob.glob('/nix/store/*-fontconfig-*/etc/fonts/fonts.conf'))
        if confs:
            os.environ['FONTCONFIG_FILE'] = confs[-1]
        else:
            conf = '/tmp/fbs-fonts.conf'
            with open(conf, 'w') as fh:
                fh.write('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>'
                         f'<dir>{FONT_DIR}</dir><cachedir>/tmp/fbs-fc-cache</cachedir></fontconfig>')
            os.environ['FONTCONFIG_FILE'] = conf
    _preloaded = True


def render_pdf(html_str: str) -> bytes:
    """WeasyPrint render; raises ImportError/OSError when the libs are absent."""
    _preload_nix_libs()
    from weasyprint import HTML  # lazy: Railway has pango/cairo, dev boxes may not
    return HTML(string=html_str, base_url=STATIC_DIR + '/').write_pdf()


def safe_filename(artist: str, track: str) -> str:
    s = f"{artist} - {track} - Sonic Breakdown.pdf"
    return re.sub(r'[^\w\-\. ()&]+', '', s)[:120]
