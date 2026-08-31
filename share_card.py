"""Shareable analyzer result card — branded PNG built from a completed scan.

Same technique as the pie-ad end cards (Website2/ad-creative/pie-doodle-frames/
build_inspector_card.py): compose an SVG with Londrina Solid vector text over the
site bg.png texture with the pie logo, then rasterize with rsvg-convert. No PIL —
text auto-fit uses a pre-measured char-width table (card_assets/londrina_widths.json,
advances at 1000pt measured from the bundled TTFs).

Formats: "story" 1080x1920 (IG story, primary) and "og" 1200x630 (landscape).
"""
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ASSETS = HERE / "card_assets"
FONT_DIR = HERE / "fonts"

ANALYZER_URL = "analyze.freshlybakedstudios.com"

# Pie palette straight from the site component (animatedPie.tsx) — matches the
# ad end cards. lime-base frozen at the pulse's warm midpoint.
PIE_CLS = """
.cls-1 { fill: #9fcc3b; } .cls-2 { fill: #cdd382; } .cls-3 { fill: #d6e14d; }
.cls-4 { fill: #bcc48c; } .cls-5 { fill: #8c8f51; } .cls-6 { fill: #271a2e; }
.path-trace { fill: #0A0A0A; }
.lime-base { fill: #bad144 !important; }
"""

LIME = "#D8E166"
CREAM = "#DFE5BC"
OLIVE = "#8D8F59"
INK = "#0A0A0A"
BG = "#222020"

_widths = json.loads((ASSETS / "londrina_widths.json").read_text())
# Codepoints Londrina Solid can actually draw (measured from the bundled TTF).
# Railway has no fallback fonts, so anything outside this set would render as
# a tofu box — transliterate the common cases and drop the rest.
_covered = set(json.loads((ASSETS / "londrina_cover.json").read_text()))
_TRANSLIT = {"≈": "~", "—": "-", "–": "-", "‘": "'", "’": "'",
             "“": '"', "”": '"', " ": " "}
_pie_inner = (ASSETS / "pie_inner.svg").read_text()
_bg_b64 = base64.b64encode((HERE / "static" / "bg.png").read_bytes()).decode()

_fontconfig_file = None


def _ensure_fontconfig() -> str:
    """Point rsvg/pango at the bundled Londrina TTFs regardless of host setup."""
    global _fontconfig_file
    if _fontconfig_file and Path(_fontconfig_file).exists():
        return _fontconfig_file
    cache = Path(tempfile.gettempdir()) / "fbs-fontconfig-cache"
    cache.mkdir(exist_ok=True)
    conf = Path(tempfile.gettempdir()) / "fbs-share-fonts.conf"
    conf.write_text(f"""<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>{FONT_DIR}</dir>
  <dir>/usr/share/fonts</dir>
  <dir>/usr/local/share/fonts</dir>
  <dir>/System/Library/Fonts</dir>
  <dir>/Library/Fonts</dir>
  <cachedir>{cache}</cachedir>
</fontconfig>
""")
    _fontconfig_file = str(conf)
    return _fontconfig_file


def renderer_available() -> bool:
    return shutil.which("rsvg-convert") is not None


def _text_width(text: str, size: float, weight: str = "Black") -> float:
    tbl = _widths.get(weight) or _widths["Regular"]
    avg = tbl.get("n", 500)
    return sum(tbl.get(ch, avg) for ch in text) / 1000.0 * size


def _fit(text: str, ceiling: float, max_w: float, weight: str = "Black") -> float:
    """Largest font size <= ceiling whose rendered width fits max_w."""
    if not text:
        return ceiling
    w = _text_width(text, ceiling, weight)
    if w <= max_w:
        return ceiling
    return max(18.0, ceiling * max_w / w)


def _xesc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _stat_list(d: dict) -> list:
    v = d.get("stats")
    return [s for s in v if isinstance(s, dict)] if isinstance(v, list) else []


def _emotion_list(d: dict) -> list:
    v = d.get("emotions")
    return [e for e in v if isinstance(e, str)] if isinstance(v, list) else []


def _clean(s, cap=60) -> str:
    """Strip control chars and undrawable glyphs, collapse whitespace, cap length."""
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(s or ""))
    s = "".join(_TRANSLIT.get(ch, ch) for ch in s)
    s = "".join(ch for ch in s if ord(ch) in _covered)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > cap:
        s = s[:cap - 1].rstrip() + "…"
    return s


def _pie(scale: float, x: float, y: float) -> str:
    return (f'<g transform="translate({x} {y}) scale({scale})">'
            f'<svg viewBox="0 0 1903.99 1417.09" width="1904" height="1417" '
            f'overflow="visible">{_pie_inner}</svg></g>')


def _defs(w: int, h: int) -> str:
    return (f'<defs><pattern id="bgtex" patternUnits="userSpaceOnUse" width="420" height="412">'
            f'<image href="data:image/png;base64,{_bg_b64}" width="420" height="412"/></pattern></defs>'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>'
            f'<rect width="{w}" height="{h}" fill="url(#bgtex)"/>')


def _stat_block(x, y, w, h, value, label, sub=""):
    """One stat tile: big lime value, small cream label, optional subtitle."""
    vsize = _fit(value, 74, w - 44)
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{INK}" opacity="0.42"/>',
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="none" stroke="{LIME}" stroke-opacity="0.35" stroke-width="3"/>',
        f'<text x="{x + w / 2}" y="{y + 76}" text-anchor="middle" font-family="Londrina Solid" '
        f'font-weight="900" font-size="{vsize:.1f}" fill="{LIME}">{_xesc(value)}</text>',
        f'<text x="{x + w / 2}" y="{y + 118}" text-anchor="middle" font-family="Londrina Solid" '
        f'font-weight="400" font-size="30" letter-spacing="2" fill="{CREAM}">{_xesc(label)}</text>',
    ]
    if sub:
        parts.append(
            f'<text x="{x + w / 2}" y="{y + 150}" text-anchor="middle" font-family="Londrina Solid" '
            f'font-weight="400" font-size="24" fill="{OLIVE}">{_xesc(sub)}</text>')
    return "".join(parts)


def build_story_svg(d: dict) -> str:
    """1080x1920 IG-story card from sanitized card data."""
    W, H = 1080, 1920
    track = _clean(d.get("track"), 40) or "MY LATEST TRACK"
    artist = _clean(d.get("artist"), 40)
    stats = _stat_list(d)
    genre = _clean(d.get("genre"), 32)
    genre_detail = _clean(d.get("genre_detail"), 64)
    emotions = [_clean(e, 22) for e in _emotion_list(d)[:3] if _clean(e, 22)]

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
         f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
         f'<style>{PIE_CLS}</style>{_defs(W, H)}']

    # Brand row
    s.append(f'<text x="540" y="108" text-anchor="middle" font-family="Londrina Solid" '
             f'font-weight="400" font-size="40" letter-spacing="6" fill="{OLIVE}">FRESHLY BAKED STUDIOS</text>')
    s.append(f'<text x="540" y="196" text-anchor="middle" font-family="Londrina Solid" '
             f'font-weight="900" font-size="64" fill="{CREAM}">SONG ANALYZER REPORT</text>')

    # Track title (lime, punchy, slight tilt) + artist
    tsize = _fit(track.upper(), 108, 940)
    asize = _fit(("BY " + artist).upper(), 52, 900)
    s.append(f'<g transform="rotate(-2 540 320)">'
             f'<text x="540" y="330" text-anchor="middle" font-family="Londrina Solid" font-weight="900" '
             f'font-size="{tsize:.1f}" fill="{INK}" stroke="{INK}" stroke-width="22" stroke-linejoin="round">{_xesc(track.upper())}</text>'
             f'<text x="540" y="330" text-anchor="middle" font-family="Londrina Solid" font-weight="900" '
             f'font-size="{tsize:.1f}" fill="{LIME}">{_xesc(track.upper())}</text></g>')
    if artist:
        s.append(f'<text x="540" y="404" text-anchor="middle" font-family="Londrina Solid" '
                 f'font-weight="400" font-size="{asize:.1f}" fill="{CREAM}">{_xesc(("BY " + artist).upper())}</text>')

    # Stats grid: 2 cols x 3 rows
    gx, gy, gw, gh, gap = 70, 470, 460, 168, 20
    for i, st in enumerate(stats[:6]):
        col, row = i % 2, i // 2
        s.append(_stat_block(gx + col * (gw + gap), gy + row * (gh + gap), gw, gh,
                             _clean(st.get("value"), 18), _clean(st.get("label"), 16),
                             _clean(st.get("sub"), 26)))
    grid_bottom = gy + 3 * gh + 2 * gap  # 1044

    y = grid_bottom + 78
    # Sonic lane callout
    if genre:
        lane = genre.upper()
        lsize = _fit("SONIC LANE:  " + lane, 58, 940)
        # Manual two-segment centering: Londrina's NBSP glyph is zero-width in
        # rsvg, so a single <text> with tspans loses the gap after the colon.
        left = "SONIC LANE:"
        gap = 0.35 * lsize
        lw, rw = _text_width(left, lsize), _text_width(lane, lsize)
        x0 = 540 - (lw + gap + rw) / 2
        s.append(f'<text x="{x0:.0f}" y="{y}" font-family="Londrina Solid" font-weight="900" '
                 f'font-size="{lsize:.1f}" fill="{CREAM}">{_xesc(left)}</text>'
                 f'<text x="{x0 + lw + gap:.0f}" y="{y}" font-family="Londrina Solid" font-weight="900" '
                 f'font-size="{lsize:.1f}" fill="{LIME}">{_xesc(lane)}</text>')
        y += 52
        if genre_detail:
            dsize = _fit(genre_detail, 34, 940, "Regular")
            s.append(f'<text x="540" y="{y}" text-anchor="middle" font-family="Londrina Solid" '
                     f'font-weight="400" font-size="{dsize:.1f}" fill="{OLIVE}">{_xesc(genre_detail)}</text>')
            y += 46
    # Emotion chips
    if emotions:
        y += 24
        chip_h, pad, gap_c = 62, 30, 18
        chips = [(e.upper(), _text_width(e.upper(), 34) + 2 * pad) for e in emotions]
        total = sum(w for _, w in chips) + gap_c * (len(chips) - 1)
        cx = (1080 - total) / 2
        for label, wch in chips:
            s.append(f'<rect x="{cx:.0f}" y="{y}" width="{wch:.0f}" height="{chip_h}" rx="{chip_h / 2}" '
                     f'fill="{INK}" opacity="0.42"/>'
                     f'<rect x="{cx:.0f}" y="{y}" width="{wch:.0f}" height="{chip_h}" rx="{chip_h / 2}" '
                     f'fill="none" stroke="{CREAM}" stroke-opacity="0.45" stroke-width="3"/>'
                     f'<text x="{cx + wch / 2:.0f}" y="{y + 43}" text-anchor="middle" font-family="Londrina Solid" '
                     f'font-weight="900" font-size="34" fill="{CREAM}">{_xesc(label)}</text>')
            cx += wch + gap_c

    # Pie hero, seated in the band between the chips and the CTA
    s.append(_pie(0.27, 280, 1290))

    # CTA
    s.append(f'<text x="540" y="1712" text-anchor="middle" font-family="Londrina Solid" font-weight="900" '
             f'font-size="54" fill="{CREAM}">SCAN YOUR SONG FREE</text>')
    s.append(f'<g transform="rotate(2 540 1790)">'
             f'<rect x="150" y="1740" width="780" height="96" rx="16" fill="{LIME}" stroke="{INK}" stroke-width="9"/>'
             f'<text x="540" y="1805" text-anchor="middle" font-family="Londrina Solid" font-weight="900" '
             f'font-size="46" fill="{INK}">{ANALYZER_URL}</text></g>')
    s.append(f'<text x="540" y="1888" text-anchor="middle" font-family="Londrina Solid" font-weight="400" '
             f'font-size="28" fill="{OLIVE}">Multi-platinum production, mixing and mastering. 3.3B+ streams.</text>')
    s.append('</svg>')
    return "".join(s)


def build_og_svg(d: dict) -> str:
    """1200x630 landscape card: pie left, track + key stats right."""
    W, H = 1200, 630
    track = _clean(d.get("track"), 34) or "MY LATEST TRACK"
    artist = _clean(d.get("artist"), 34)
    stats = _stat_list(d)[:3]

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
         f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
         f'<style>{PIE_CLS}</style>{_defs(W, H)}']
    s.append(_pie(0.26, -40, 100))
    tx = 470
    s.append(f'<text x="{tx}" y="92" font-family="Londrina Solid" font-weight="400" '
             f'font-size="30" letter-spacing="4" fill="{OLIVE}">FRESHLY BAKED STUDIOS</text>')
    s.append(f'<text x="{tx}" y="152" font-family="Londrina Solid" font-weight="900" '
             f'font-size="48" fill="{CREAM}">SONG ANALYZER REPORT</text>')
    tsize = _fit(track.upper(), 76, 680)
    s.append(f'<g transform="rotate(-2 {tx} 230)">'
             f'<text x="{tx}" y="242" font-family="Londrina Solid" font-weight="900" font-size="{tsize:.1f}" '
             f'fill="{INK}" stroke="{INK}" stroke-width="16" stroke-linejoin="round">{_xesc(track.upper())}</text>'
             f'<text x="{tx}" y="242" font-family="Londrina Solid" font-weight="900" font-size="{tsize:.1f}" '
             f'fill="{LIME}">{_xesc(track.upper())}</text></g>')
    if artist:
        asize = _fit(("BY " + artist).upper(), 38, 680)
        s.append(f'<text x="{tx}" y="298" font-family="Londrina Solid" font-weight="400" '
                 f'font-size="{asize:.1f}" fill="{CREAM}">{_xesc(("BY " + artist).upper())}</text>')
    bx = tx
    for st in stats:
        s.append(_stat_block(bx, 340, 220, 150, _clean(st.get("value"), 12), _clean(st.get("label"), 14)))
        bx += 240
    s.append(f'<g transform="rotate(1 800 560)">'
             f'<rect x="{tx}" y="522" width="660" height="76" rx="14" fill="{LIME}" stroke="{INK}" stroke-width="8"/>'
             f'<text x="{tx + 330}" y="574" text-anchor="middle" font-family="Londrina Solid" font-weight="900" '
             f'font-size="38" fill="{INK}">{ANALYZER_URL}</text></g>')
    s.append('</svg>')
    return "".join(s)


def render_card_png(card_data: dict, fmt: str = "story") -> bytes:
    """Build the SVG for `fmt` and rasterize to PNG bytes via rsvg-convert."""
    if fmt == "og":
        svg, w, h = build_og_svg(card_data), 1200, 630
    else:
        svg, w, h = build_story_svg(card_data), 1080, 1920
    env = dict(os.environ, FONTCONFIG_FILE=_ensure_fontconfig())
    proc = subprocess.run(
        ["rsvg-convert", "-w", str(w), "-h", str(h)],
        input=svg.encode("utf-8"), capture_output=True, env=env, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"rsvg-convert failed: {proc.stderr.decode()[:400]}")
    return proc.stdout


if __name__ == "__main__":
    sample = {
        "track": "Midnight Getaway",
        "artist": "The Paper Kites",
        "stats": [
            {"value": "124", "label": "BPM"},
            {"value": "F# minor", "label": "KEY"},
            {"value": "-9.8", "label": "LUFS", "sub": "Integrated"},
            {"value": "Driving", "label": "ENERGY"},
            {"value": "Punchy", "label": "COMPRESSION"},
            {"value": "Groovy", "label": "DANCEABILITY"},
        ],
        "genre": "Indie Rock",
        "genre_detail": "43 of 50 closest sonic matches (86%) share this lane",
        "emotions": ["Power", "Joyful activation", "Tension"],
    }
    Path("/tmp/share_card_story.png").write_bytes(render_card_png(sample, "story"))
    Path("/tmp/share_card_og.png").write_bytes(render_card_png(sample, "og"))
    print("wrote /tmp/share_card_story.png and /tmp/share_card_og.png")
