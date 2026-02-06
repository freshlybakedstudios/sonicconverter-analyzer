"""
FastAPI server for the SonicConverter web analyzer.
Preloads the 1.2 GB GEMS universe cache at startup, then serves:
  POST /api/register  — capture lead (name + email)
  POST /api/analyze   — upload audio, return sonic breakdown + matches
Static files served from ./static/
"""

import math
import os
import secrets
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from supabase import create_client

load_dotenv()

from audio_analyzer import extract_features
from chartmetric_lookup import lookup_artist_by_spotify
from email_sender import send_results_email
from track_matcher import TrackMatcher, _genre_families

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
matcher = TrackMatcher()
supabase = None
access_tokens: dict = {}  # token -> {name, email, created_at}


# ---------------------------------------------------------------------------
# Lifespan: preload cache
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase
    print("Starting up...")

    # Try to load cache if it exists
    try:
        matcher.load_cache()
    except FileNotFoundError:
        print("⚠️  Cache not found - run build_gems_universe_cache.py or wait for background build")

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY')
    if url and key:
        supabase = create_client(url, key)
    else:
        print("⚠️  SUPABASE_URL/SUPABASE_SERVICE_KEY not set - some features disabled")

    print("Ready.")
    yield
    print("Shutting down.")


app = FastAPI(title="SonicConverter Analyzer", lifespan=lifespan)

# CORS — allow the Firebase-hosted frontend + localhost dev + Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://freshlybakedstudios.com",
        "https://www.freshlybakedstudios.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https://.*\.up\.railway\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check for Railway
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "cache_loaded": matcher.cache is not None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All production features we can compare (matches the consensus analyzer)
PRODUCTION_FEATURES = [
    'sub_ratio', 'bass_ratio', 'low_mid_ratio', 'mid_ratio',
    'high_mid_ratio', 'presence_ratio', 'air_ratio',
    'brightness', 'spectral_rolloff', 'brightness_variance',
    'energy', 'beat_strength', 'onset_rate', 'attack_time', 'danceability',
    'lufs_integrated', 'dynamic_range', 'loudness_range', 'crest_factor',
    'compression_amount',
    'spectral_complexity', 'dissonance', 'key_strength', 'zcr',
    'spectral_flux',
]

FEATURE_DESCRIPTIONS = {
    'sub_ratio': {
        'higher': 'Strengthen sub-bass foundation — adding presence in the 20-60 Hz range for depth that\'s felt, not heard',
        'lower': 'Tighten sub-bass response — easing the 20-60 Hz range to clean up low-end weight',
        'domain': 'EQ / Spectrum', 'unit': '%',
    },
    'bass_ratio': {
        'higher': 'Enhance low-end weight — boosting the 60-250 Hz range for tighter kick and fuller bass',
        'lower': 'Lighten low-end energy — reducing the 60-250 Hz range for more space and definition',
        'domain': 'EQ / Spectrum', 'unit': '%',
    },
    'low_mid_ratio': {
        'higher': 'Add body and warmth — gently lifting the 250-500 Hz range for fuller tone',
        'lower': 'Clear low-mid buildup — easing the 250-500 Hz range to reduce muddiness',
        'domain': 'EQ / Spectrum', 'unit': '%',
    },
    'mid_ratio': {
        'higher': 'Bring focus to the mids — emphasizing 500 Hz-2 kHz for presence and detail',
        'lower': 'Soften mid presence — reducing 500 Hz-2 kHz for a smoother, more open sound',
        'domain': 'EQ / Spectrum', 'unit': '%',
    },
    'high_mid_ratio': {
        'higher': 'Add articulation — boosting 2-5 kHz for definition and clarity',
        'lower': 'Soften harshness — easing 2-5 kHz to reduce edge and listener fatigue',
        'domain': 'EQ / Spectrum', 'unit': '%',
    },
    'presence_ratio': {
        'higher': 'Open vocal clarity — adding lift around 5-8 kHz for intelligibility and sheen',
        'lower': 'Smooth upper mids — reducing 5-8 kHz to mellow brightness and sibilance',
        'domain': 'EQ / Spectrum', 'unit': '%',
    },
    'air_ratio': {
        'higher': 'Add high-end air — boosting 8-20 kHz for shimmer and openness',
        'lower': 'Reduce high-end air — trimming 8-20 kHz for a softer, more vintage character',
        'domain': 'EQ / Spectrum', 'unit': '%',
    },
    'brightness': {
        'higher': 'Brighten the mix — enhancing upper frequencies for clarity and air',
        'lower': 'Darken the tone — reducing high content for a smoother, more subdued sound',
        'domain': 'EQ / Spectrum', 'unit': 'Hz',
    },
    'energy': {
        'higher': 'Lift overall intensity — creating a more energetic, forward mix',
        'lower': 'Dial back intensity — allowing more space and dynamic breathing room',
        'domain': 'Energy / Movement', 'unit': '',
    },
    'lufs_integrated': {
        'higher': 'Push overall loudness — raising master level for a competitive finish',
        'lower': 'Lower playback loudness — keeping more headroom and dynamic range',
        'domain': 'Dynamics', 'unit': 'LUFS',
    },
    'dynamic_range': {
        'higher': 'Open up dynamic contrast — widening the loudness range for more movement',
        'lower': 'Tighten dynamics — narrowing the loudness range for a denser, more consistent sound',
        'domain': 'Dynamics', 'unit': 'dB',
    },
    'compression_amount': {
        'higher': 'Increase glue — adding compression for a tighter, more cohesive mix',
        'lower': 'Restore openness — easing compression to regain punch and dynamic life',
        'domain': 'Dynamics', 'unit': '',
    },
    'crest_factor': {
        'higher': 'Increase transient punch — allowing peaks to stand out more clearly',
        'lower': 'Smooth peak dynamics — reducing crest factor for controlled loudness',
        'domain': 'Dynamics', 'unit': '',
    },
    'attack_time': {
        'higher': 'Soften transients — slowing the attack for a smoother onset',
        'lower': 'Sharpen transients — speeding the attack for punchier, more immediate hits',
        'domain': 'Dynamics', 'unit': 'ms',
    },
    'beat_strength': {
        'higher': 'Strengthen the rhythmic grid — more defined kick and snare for drive',
        'lower': 'Loosen the beat — softening rhythmic emphasis for a more organic feel',
        'domain': 'Rhythm / Transients', 'unit': '',
    },
    'onset_rate': {
        'higher': 'Add transient drive — more percussive hits per second for rhythmic density',
        'lower': 'Refine transient shape — softening edges so rhythm breathes naturally',
        'domain': 'Rhythm / Transients', 'unit': '/s',
    },
    'danceability': {
        'higher': 'Tighten the groove — more consistent timing and rhythmic pull',
        'lower': 'Loosen the pocket — allowing more swing and rhythmic freedom',
        'domain': 'Rhythm / Transients', 'unit': '',
    },
    'dissonance': {
        'higher': 'Add harmonic tension — introducing subtle dissonance for energy and edge',
        'lower': 'Smooth harmonic texture — reducing dissonance for purity and warmth',
        'domain': 'Tonal Character', 'unit': '',
    },
    'key_strength': {
        'higher': 'Strengthen tonal center — clearer key for harmonic stability',
        'lower': 'Blur tonal center — more ambiguity for atmospheric texture',
        'domain': 'Tonal Character', 'unit': '',
    },
    'spectral_flux': {
        'higher': 'Add motion and sparkle — more spectral change for an evolving, lively mix',
        'lower': 'Reduce spectral motion — stabilizing tone for smoother consistency',
        'domain': 'Energy / Movement', 'unit': '',
    },
    'zcr': {
        'higher': 'Increase grit and presence — more edge and texture',
        'lower': 'Smooth timbre — for a cleaner, rounder tone',
        'domain': 'Tonal Character', 'unit': '',
    },
    'loudness_range': {
        'higher': 'Widen loudness variation — more dynamic contrast between sections',
        'lower': 'Tighten loudness variation — more consistent level across sections',
        'domain': 'Dynamics', 'unit': 'LU',
    },
}

# Features to compare (all the ones we extract + match against)
COMPARISON_FEATURES = [
    'sub_ratio', 'bass_ratio', 'low_mid_ratio', 'mid_ratio',
    'high_mid_ratio', 'presence_ratio', 'air_ratio', 'brightness',
    'energy', 'lufs_integrated', 'dynamic_range', 'compression_amount',
    'crest_factor', 'attack_time', 'beat_strength', 'onset_rate',
    'danceability', 'dissonance', 'key_strength', 'spectral_flux',
    'zcr', 'loudness_range',
]


def _find_consensus(features: dict, high_converter_gems: list) -> list:
    """
    Find consensus patterns: for each production feature, check whether
    most high converters are higher or lower than the user. If 70%+ agree
    on direction, that's a consensus recommendation.

    Same method as the production outreach pipeline.
    """
    if not high_converter_gems:
        return []

    target_bpm = float(features.get('bpm', 120))
    time_based = {'attack_time', 'onset_rate'}
    consensus_results = []

    for feat in PRODUCTION_FEATURES:
        user_val = features.get(feat)
        if user_val is None:
            continue
        try:
            user_val = float(user_val)
        except (TypeError, ValueError):
            continue

        directions = []
        values = []

        for gems_row in high_converter_gems:
            converter_val = gems_row.get(feat)
            if converter_val is None:
                continue
            try:
                converter_val = float(converter_val)
            except (TypeError, ValueError):
                continue

            # Normalize time-based features by BPM ratio
            if feat in time_based:
                converter_bpm = float(gems_row.get('bpm', 120) or 120)
                if converter_bpm > 0:
                    converter_val = converter_val * (target_bpm / converter_bpm)

            values.append(converter_val)

            # Meaningful difference threshold: >5% or >0.01
            threshold = max(abs(user_val) * 0.05, 0.01) if user_val != 0 else 0.01
            if abs(converter_val - user_val) > threshold:
                directions.append('higher' if converter_val > user_val else 'lower')

        if not directions or len(values) < 3:
            continue

        higher_count = directions.count('higher')
        lower_count = directions.count('lower')
        pool_size = len(high_converter_gems)
        max_count = max(higher_count, lower_count)

        # Consensus: 50%+ of the full pool must agree on direction
        if max_count / pool_size >= 0.50:
            converter_avg = sum(values) / len(values)
            direction = 'higher' if higher_count > lower_count else 'lower'
            consensus_results.append({
                'feature': feat,
                'direction': direction,
                'user_val': user_val,
                'converter_avg': converter_avg,
                'count': max_count,
                'total': pool_size,
                'consensus_pct': max_count / pool_size * 100,
            })

    # Sort by consensus strength (strongest agreement first)
    consensus_results.sort(key=lambda c: (-c['consensus_pct'], -c['count']))
    return consensus_results


# Features that are energy ratios (show % and dB delta)
RATIO_FEATURES = {
    'sub_ratio', 'bass_ratio', 'low_mid_ratio', 'mid_ratio',
    'high_mid_ratio', 'presence_ratio', 'air_ratio',
}
# Features already in dB
DB_FEATURES = {'lufs_integrated', 'dynamic_range', 'loudness_range', 'crest_factor'}


def _format_rec(feat: str, consensus: dict) -> str:
    """Format a single consensus recommendation into a human-readable string."""
    desc = FEATURE_DESCRIPTIONS.get(feat)
    if not desc:
        return None

    direction = consensus['direction']
    action = desc.get(direction, '')
    if not action:
        return None

    domain = desc.get('domain', '')
    unit = desc.get('unit', '')
    user_val = consensus['user_val']
    peer_val = consensus['converter_avg']
    count = consensus['count']
    total = consensus['total']

    # Format values + compute dB delta where applicable
    delta_str = ''

    if feat in RATIO_FEATURES:
        user_str = f"{user_val * 100:.1f}%"
        peer_str = f"{peer_val * 100:.1f}%"
        if user_val > 0 and peer_val > 0:
            db_delta = 20 * math.log10(peer_val / user_val)
            if abs(db_delta) >= 0.05:
                delta_str = f" ({db_delta:+.1f} dB)"
    elif feat in DB_FEATURES:
        db_delta = peer_val - user_val
        user_str = f"{user_val:.1f} dB"
        peer_str = f"{peer_val:.1f} dB"
        if abs(db_delta) >= 0.05:
            delta_str = f" ({db_delta:+.1f} dB)"
    elif unit == 'LUFS':
        db_delta = peer_val - user_val
        user_str = f"{user_val:.1f} LUFS"
        peer_str = f"{peer_val:.1f} LUFS"
        if abs(db_delta) >= 0.05:
            delta_str = f" ({db_delta:+.1f} dB)"
    elif unit == 'Hz':
        user_str = f"{user_val:.0f} Hz"
        peer_str = f"{peer_val:.0f} Hz"
    elif unit in ('ms', '/s', 'LU'):
        user_str = f"{user_val:.1f} {unit}"
        peer_str = f"{peer_val:.1f} {unit}"
    else:
        user_str = f"{user_val:.3f}"
        peer_str = f"{peer_val:.3f}"
        # Compute ~X% delta for dimensionless features
        if user_val != 0:
            pct_delta = ((peer_val - user_val) / abs(user_val)) * 100
            if abs(pct_delta) >= 1:
                delta_str = f" (~{abs(pct_delta):.0f}%)"

    return (
        f"[{domain}] {action}{delta_str}\n"
        f"{count}/{total} top converters agree"
    )


def _generate_recommendations(features: dict, matches: list,
                               gems_by_artist: dict = None,
                               genre_alignment: dict = None,
                               user_profile: dict = None) -> list:
    """
    Generate consensus-based production recommendations by comparing the user's
    track against the highest-converting matched artists.

    Same approach as the production outreach pipeline:
    1. Take matched artists, sort by conversion rate
    2. Use the top converters as the benchmark cohort
    3. For each of ~25 production features, check directional consensus (70%+ agree)
    4. Surface the strongest consensus patterns across 5 domains
    """
    recs = []

    if gems_by_artist and matches:
        # Get user's emotions for overlap filtering
        user_emotions = {
            features.get('emotion_1'), features.get('emotion_2'),
            features.get('emotion_3'), features.get('emotion_4'),
        } - {None, ''}

        # Filter to matches with conversion data + emotion overlap (2+ shared)
        with_conversion = []
        for m in matches:
            if not m.get('conversion_rate') or m['conversion_rate'] <= 0:
                continue
            m_emotions = set(m.get('emotions', [])) - {None, '', 'neutral'}
            if len(user_emotions & m_emotions) >= 2:
                with_conversion.append(m)

        # Fallback: if too few with emotion overlap, use all with conversion data
        if len(with_conversion) < 4:
            with_conversion = [m for m in matches if m.get('conversion_rate') and m['conversion_rate'] > 0]

        MIN_CONSENSUS_POOL = 10
        if len(with_conversion) >= MIN_CONSENSUS_POOL:
            conversions = sorted([m['conversion_rate'] for m in with_conversion])
            # Try top 25% first
            t25 = conversions[int(len(conversions) * 0.75)]
            top_25 = [m for m in with_conversion if m['conversion_rate'] >= t25]
            if len(top_25) >= MIN_CONSENSUS_POOL:
                high_converter_ids = [str(m['artist_id']) for m in top_25]
            else:
                # Expand to top 50%
                t50 = conversions[int(len(conversions) * 0.50)]
                top_50 = [m for m in with_conversion if m['conversion_rate'] >= t50]
                if len(top_50) >= MIN_CONSENSUS_POOL:
                    high_converter_ids = [str(m['artist_id']) for m in top_50]
                else:
                    # Use all
                    high_converter_ids = [str(m['artist_id']) for m in with_conversion]
        else:
            high_converter_ids = [str(m['artist_id']) for m in with_conversion]

        # Get GEMS rows for the high converters
        high_converter_gems = [
            gems_by_artist[aid] for aid in high_converter_ids
            if aid in gems_by_artist
        ]

        print(f"  Recs: {len(with_conversion)} with conversion, {len(high_converter_ids)} high converter IDs, {len(high_converter_gems)} with GEMS data, {len(gems_by_artist)} total in gems_by_artist")

        if len(high_converter_gems) < 3:
            high_converter_gems = list(gems_by_artist.values())
            print(f"  Recs: fell back to all {len(high_converter_gems)} GEMS entries")

        # Find consensus patterns (same method as pipeline)
        consensus = _find_consensus(features, high_converter_gems)

        # Take top recs, max 2 per domain for variety
        domain_count = {}
        for c in consensus:
            if len(recs) >= 8:
                break
            feat = c['feature']
            desc = FEATURE_DESCRIPTIONS.get(feat)
            if not desc:
                continue
            domain = desc.get('domain', '')
            if domain_count.get(domain, 0) >= 2:
                continue

            text = _format_rec(feat, c)
            if text:
                recs.append(text)
                domain_count[domain] = domain_count.get(domain, 0) + 1

    # --- Conversion-based rec ---
    if user_profile:
        user_conv = user_profile.get('conversion_rate')
        conv_comparison = user_profile.get('conversion_comparison', {})
        peer_median = conv_comparison.get('peer_median')
        if user_conv is not None and peer_median is not None and user_conv < peer_median:
            recs.append(
                f"Your listener-to-follower conversion ({user_conv:.1f}%) is below the median "
                f"of your sonic peers ({peer_median:.1f}%). Optimizing your Spotify profile, "
                f"sonic profile, release strategy, and playlist pitching could help close this gap."
            )

    # --- Genre alignment rec ---
    if genre_alignment and genre_alignment.get('genre'):
        g = genre_alignment['genre']
        pct = genre_alignment.get('percentage', 0)
        recs.append(
            f"Your sound most aligns with {g} ({pct:.0f}% of your top matches). "
            f"Consider referencing top {g} producers for mix decisions and playlist targeting."
        )

    return recs


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# Same tier ranges as emotion_conversion_analyzer.py
TIER_RANGES = {
    'micro': (0, 5_000),
    'emerging': (5_000, 10_000),
    'mid': (10_000, 50_000),
    'rising': (50_000, 100_000),
    'established': (100_000, 1_000_000),
    'superstar': (1_000_000, float('inf')),
}


def _listeners_to_tier(listeners: int) -> str:
    for tier, (lo, hi) in TIER_RANGES.items():
        if lo <= listeners < hi:
            return tier
    return 'unknown'


@app.post("/api/register")
async def register(
    name: str = Form(...),
    email: str = Form(...),
    spotify_url: Optional[str] = Form(None),
    monthly_listeners: Optional[int] = Form(None),
):
    """Capture lead and return an access token for the analyze endpoint."""
    if not name or not email:
        raise HTTPException(400, "Name and email are required")

    # Store lead in Supabase (core columns only; spotify_url/monthly_listeners
    # require ALTER TABLE — see README)
    row = {
        'name': name,
        'email': email,
        'created_at': datetime.utcnow().isoformat(),
        'analysis_count': 0,
    }
    try:
        supabase.table('analyzer_leads').insert(row).execute()
    except Exception as e:
        print(f"Supabase insert error (continuing): {e}")
    # Try to update with optional columns (may fail if columns don't exist yet)
    if spotify_url or monthly_listeners is not None:
        extra = {}
        if spotify_url:
            extra['spotify_url'] = spotify_url
        if monthly_listeners is not None:
            extra['monthly_listeners'] = monthly_listeners
        try:
            supabase.table('analyzer_leads').update(extra).eq('email', email).execute()
        except Exception:
            pass  # columns don't exist yet — fine, data is in-memory token

    token = secrets.token_urlsafe(32)
    access_tokens[token] = {
        'name': name,
        'email': email,
        'spotify_url': spotify_url or '',
        'monthly_listeners': monthly_listeners,
        'created_at': time.time(),
    }
    return {"token": token, "name": name}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    token: str = Form(...),
    genre: Optional[str] = Form(None),
):
    """
    Accept an uploaded audio file, run analysis, return results.
    Also sends results email.
    """
    # Validate token
    lead = access_tokens.get(token)
    if not lead:
        raise HTTPException(401, "Invalid or expired token. Please register first.")

    # Validate file type
    filename = file.filename or ''
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ('mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac'):
        raise HTTPException(400, f"Unsupported file type: .{ext}. Please upload mp3 or wav.")

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
    try:
        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:  # 50 MB limit
            raise HTTPException(400, "File too large (max 50 MB)")
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        # Extract features
        print(f"Analyzing: {filename} ({len(contents) / 1024 / 1024:.1f} MB)")
        t0 = time.time()
        features = extract_features(tmp_path, genre_hint=genre or '')
        t_features = time.time() - t0
        print(f"  Feature extraction: {t_features:.1f}s")

        # Save original dropdown selection before it gets overwritten
        dropdown_genre = genre or ''

        # If user provided Spotify URL, pull their genre from cache for matching
        spotify_url = lead.get('spotify_url', '')
        spotify_base = spotify_url.split('?')[0].rstrip('/') if spotify_url else ''
        cached_artist_data = None
        cm_data = None
        user_code2 = None
        user_pronoun = None
        if spotify_base:
            for aid, adata in matcher._artists.items():
                cache_url = (adata.get('spotify_url') or '').split('?')[0].rstrip('/')
                if cache_url == spotify_base:
                    cached_artist_data = (aid, adata)
                    # Use cached genres if available, fall back to dropdown
                    cached_genres = adata.get('genres', '')
                    if cached_genres:
                        print(f"  Using cached genres: {cached_genres} (ignoring dropdown: {genre or 'empty'})")
                        genre = cached_genres
                    else:
                        print(f"  No cached genres — keeping dropdown: {genre or 'empty'}")
                    # Get user's country code and pronouns
                    user_code2 = adata.get('code2', '')
                    if user_code2:
                        print(f"  User country: {user_code2}")
                    user_pronoun = adata.get('pronoun_title', '')
                    if user_pronoun:
                        print(f"  User pronoun: {user_pronoun}")
                    break

        # If not in cache, try Chartmetric real-time lookup
        if not cached_artist_data and spotify_base:
            print(f"  Cache miss for {spotify_base} — trying Chartmetric lookup...")
            t_cm = time.time()
            cm_data = lookup_artist_by_spotify(spotify_base)
            t_cm = time.time() - t_cm
            if cm_data:
                print(f"  CM lookup: {cm_data['name']} — {cm_data['genres']} "
                      f"({cm_data['tier']}, {cm_data['listeners']:.0f} listeners) "
                      f"[{t_cm:.1f}s]")
                # Use CM genres if available, fall back to dropdown
                if cm_data['genres']:
                    print(f"  Using CM genres: {cm_data['genres']} (ignoring dropdown: {genre or 'empty'})")
                    genre = cm_data['genres']
                else:
                    print(f"  No CM genres — keeping dropdown: {genre or 'empty'}")
                # Extract pronoun from CM metadata
                cm_meta = cm_data.get('_meta', {})
                if cm_meta and cm_meta.get('pronoun_title'):
                    user_pronoun = cm_meta['pronoun_title']
                    print(f"  User pronoun (CM): {user_pronoun}")
            else:
                print(f"  CM lookup: no result [{t_cm:.1f}s]")

        # Find matches — get extra so we can filter by tier
        user_monthly = lead.get('monthly_listeners')
        # CM lookup can provide more accurate listener count
        if cm_data and cm_data['listeners'] > 0:
            user_monthly = cm_data['listeners']
        user_tier = _listeners_to_tier(user_monthly) if user_monthly else 'micro'  # Default to lowest tier
        fetch_n = 500  # Always fetch enough for family filtering

        # Pure sonic matching — no genre penalty
        t0 = time.time()
        all_matches = matcher.find_matches(features, genre_hint='', top_n=fetch_n, threshold=0.55)
        t_match = time.time() - t0

        # Dropdown boost: only applies when dropdown is COUNTER to artist's genres
        # (i.e., user is intentionally exploring a different lane)
        DROPDOWN_BOOST = 0.03
        artist_families = _genre_families(genre or '')
        dropdown_families = _genre_families(dropdown_genre) if dropdown_genre else set()
        dropdown_is_counter = dropdown_families and not (dropdown_families & artist_families)

        if dropdown_genre and dropdown_is_counter:
            print(f"  Dropdown '{dropdown_genre}' is COUNTER to artist — applying {DROPDOWN_BOOST*100:.0f}% boost")
            dropdown_lower = dropdown_genre.lower()
            for m in all_matches:
                match_genres = set()
                for field in ('primary_genre', 'secondary_genre'):
                    g = (m.get(field) or '').strip().lower()
                    if g:
                        match_genres.add(g)
                for g in m.get('artist_genres', []):
                    if g:
                        match_genres.add(g.lower())
                # Check if dropdown appears in any genre string
                if any(dropdown_lower in g for g in match_genres):
                    m['similarity'] = m.get('similarity', 0) + DROPDOWN_BOOST
            # Re-sort by boosted similarity
            all_matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        elif dropdown_genre:
            print(f"  Dropdown '{dropdown_genre}' ALIGNS with artist — no boost")

        # Save unfiltered matches for flattery (uses looser filtering)
        all_matches_unfiltered = list(all_matches)

        # Filter out matches with genre families the artist doesn't have
        # Uses the full genre taxonomy from genre_mapping.json
        # (e.g., electronic artists won't appear for pop/rock artists)
        # This strict filtering applies to Similar Artists only
        user_families = _genre_families(genre or '')

        # Dropdown expands allowed families - user knows their track's lane
        if dropdown_genre:
            dropdown_families = _genre_families(dropdown_genre)
            user_families = user_families | dropdown_families
            print(f"  Dropdown '{dropdown_genre}' adds families: {dropdown_families}")

        print(f"  Allowed families: {user_families}")

        def has_foreign_families(m):
            # Collect all genre strings from the match
            cand_genres = []
            for field in ('primary_genre', 'secondary_genre'):
                g = (m.get(field) or '').strip()
                if g:
                    cand_genres.append(g)
            for g in m.get('artist_genres', []):
                if g:
                    cand_genres.append(g)
            # Get families for this candidate
            cand_families = _genre_families(*cand_genres)
            # If candidate has families that user doesn't have, it's foreign
            foreign = cand_families - user_families
            # Allow through if no foreign families, or if user has no families (no filtering)
            return bool(foreign) and bool(user_families)

        pre_filter_count = len(all_matches)
        all_matches = [m for m in all_matches if not has_foreign_families(m)]
        print(f"  Family filter: {pre_filter_count} → {len(all_matches)} matches")

        # Country boost: same-region artists get a small boost
        COUNTRY_BOOST = 0.02
        if user_code2:
            boosted_count = 0
            for m in all_matches:
                match_code2 = m.get('code2', '')
                if match_code2 and match_code2 == user_code2:
                    m['similarity'] = m.get('similarity', 0) + COUNTRY_BOOST
                    boosted_count += 1
            if boosted_count > 0:
                all_matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                print(f"  Country boost: {boosted_count} matches from {user_code2} boosted +{COUNTRY_BOOST*100:.0f}%")

        # Debug: show top 40 matches with genre families
        print(f"  Top 40 matches (before tier filter):")
        for i, m in enumerate(all_matches[:40]):
            artist = m.get('name') or 'Unknown'
            sim = m.get('similarity') or 0
            artist_genres = m.get('artist_genres') or []
            primary = m.get('primary_genre') or ''
            genres = ', '.join(artist_genres[:3]) if artist_genres else primary or '-'
            fams = _genre_families(primary, *artist_genres)
            print(f"    {i+1}. {artist[:20]:20} {sim*100:.1f}% | {genres[:30]:30} | fams: {fams}")

        MIN_PEER_MATCHES = 10
        tier_order = list(TIER_RANGES.keys())

        # Filter to user's tier, widening if too few (same as pipeline)
        if user_tier:
            same_tier = [m for m in all_matches if m.get('tier') == user_tier]
            tiers_used = {user_tier}

            if len(same_tier) < MIN_PEER_MATCHES and user_tier in tier_order:
                target_idx = tier_order.index(user_tier)
                for radius in range(1, len(tier_order)):
                    neighbors = []
                    if target_idx - radius >= 0:
                        neighbors.append(tier_order[target_idx - radius])
                    if target_idx + radius < len(tier_order):
                        neighbors.append(tier_order[target_idx + radius])
                    tiers_used.update(neighbors)
                    same_tier = [m for m in all_matches if m.get('tier') in tiers_used]
                    if len(same_tier) >= MIN_PEER_MATCHES:
                        break

            matches = same_tier[:20]
            used = ', '.join(t for t in tier_order if t in tiers_used)
            print(f"  Matching: {t_match:.1f}s — {len(all_matches)} total, {len(matches)} in tier(s): {used}")
        else:
            matches = all_matches[:20]
            print(f"  Matching: {t_match:.1f}s — {len(matches)} matches found")

        # Flattery matches — higher-tier artists using family-based genre filtering
        # Same approach as Similar Artists: must share at least one genre family
        flattery_matches = []
        if user_tier and all_matches_unfiltered:
            tier_order_map = {t: i for i, t in enumerate(TIER_RANGES.keys())}
            user_tier_num = tier_order_map.get(user_tier, -1)

            # Use family-based filtering (same as Similar Artists)
            user_families = _genre_families(genre or '')
            print(f"  Flattery: user families = {user_families}")

            flattery_candidates = []
            seen_artists = set()
            dropdown_lower = dropdown_genre.lower() if dropdown_genre else ''

            for m in all_matches_unfiltered:
                cand_tier_num = tier_order_map.get(m.get('tier', 'unknown'), -1)
                if cand_tier_num <= user_tier_num:
                    continue

                aid = m.get('artist_id')
                if aid in seen_artists:
                    continue

                # Family-based filtering with exclusive family check
                cand_genre_parts = []
                for field in ('primary_genre', 'secondary_genre'):
                    g = (m.get(field) or '').strip()
                    if g:
                        cand_genre_parts.append(g)
                for g in m.get('artist_genres', []):
                    if g:
                        cand_genre_parts.append(g)
                cand_genre_str = ', '.join(cand_genre_parts)
                cand_families = _genre_families(cand_genre_str)

                # Must share at least one family
                if not (user_families & cand_families):
                    continue

                # Exclusive families: if candidate has these and user doesn't, filter out
                # These are strong genre identities that shouldn't cross-contaminate
                EXCLUSIVE_FAMILIES = {'electronic', 'metal', 'hip-hop', 'country', 'classical', 'jazz', 'latin', 'k-pop'}
                foreign_exclusive = (cand_families & EXCLUSIVE_FAMILIES) - user_families
                if foreign_exclusive:
                    continue  # Has exclusive family user doesn't have

                seen_artists.add(aid)

                # Genre boost for flattery scoring
                genre_boost = 0
                if dropdown_lower:
                    if any(dropdown_lower in g for g in cand_genre_parts):
                        genre_boost = 0.02

                # Pronoun boost: same pronoun gets a meaningful boost
                pronoun_boost = 0
                cand_pronoun = m.get('pronoun_title', '')
                if user_pronoun and cand_pronoun and cand_pronoun == user_pronoun:
                    pronoun_boost = 0.035  # 3.5% boost for matching pronouns

                total_boost = genre_boost + pronoun_boost
                flattery_candidates.append((cand_tier_num, m.get('similarity', 0) + total_boost, m, cand_pronoun))

            # Sort by tier (highest first), then boosted similarity
            flattery_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

            # Debug: show top candidates with pronoun info
            if user_pronoun:
                print(f"  Flattery candidates (top 10):")
                for i, (tier_num, score, m, cand_pronoun) in enumerate(flattery_candidates[:10]):
                    name = m.get('artist_name') or 'Unknown'
                    tier = m.get('tier', '?')
                    pron_match = "✓" if cand_pronoun == user_pronoun else ""
                    print(f"    {i+1}. {name[:20]:<20} {tier:<12} {score:.1%} {cand_pronoun or '?':<10} {pron_match}")

            for _, _, m, _ in flattery_candidates[:3]:
                flattery_matches.append(m)

        if flattery_matches:
            print(f"  Flattery: {len(flattery_matches)} trajectory targets found")

        # Genre alignment: count across ALL genre fields per match
        genre_alignment = None
        if matches:
            genre_counts: dict = {}
            for m in matches:
                # Collect all genres for this match
                all_genres = set()
                for field in ('primary_genre', 'secondary_genre'):
                    g = (m.get(field) or '').strip()
                    if g and g.lower() != 'unknown':
                        all_genres.add(g)
                for g in m.get('artist_genres', []):
                    if g and g.lower() != 'unknown':
                        all_genres.add(g)
                for g in all_genres:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
            if genre_counts:
                top_genre = max(genre_counts, key=genre_counts.get)
                genre_alignment = {
                    'genre': top_genre,
                    'count': genre_counts[top_genre],
                    'total': len(matches),
                    'percentage': genre_counts[top_genre] / len(matches) * 100,
                }

        # User profile: build conversion comparison from matched artists
        user_profile = None
        u_listeners = float(lead.get('monthly_listeners') or 0)
        u_followers = 0
        u_conversion = None

        # Reuse cached artist data from Spotify URL lookup (done before matching)
        if cached_artist_data:
            aid, adata = cached_artist_data
            tier_data = matcher._tiers.get(str(aid), {})
            u_listeners = float(tier_data.get('listeners') or 0) or u_listeners
            u_followers = float(tier_data.get('followers') or 0)
            if u_listeners > 0 and u_followers > 0:
                u_conversion = round((u_followers * 0.1) / (u_listeners * 4.3) * 100, 2)
            print(f"  User profile: {adata.get('name')}, listeners={u_listeners}, followers={u_followers}, conversion={u_conversion}")
        elif cm_data:
            # Use Chartmetric data for user profile
            u_listeners = float(cm_data['listeners'] or 0) or u_listeners
            u_followers = float(cm_data['followers'] or 0)
            u_conversion = cm_data.get('conversion_rate')
            print(f"  User profile (CM): {cm_data['name']}, listeners={u_listeners}, followers={u_followers}, conversion={u_conversion}")
        else:
            print(f"  User profile: no Spotify match (url={'yes' if spotify_base else 'no'}, monthly_listeners={u_listeners})")

        # Build conversion comparison vs matched artists (works with or without Spotify URL)
        if u_listeners > 0 and matches:
            match_conversions = [
                m['conversion_rate'] for m in matches
                if m.get('conversion_rate') is not None
            ]
            conv_comparison = {}
            if match_conversions:
                sorted_conv = sorted(match_conversions)
                conv_comparison = {
                    'peer_median': round(sorted_conv[len(sorted_conv) // 2], 2),
                    'peer_top_25': round(sorted_conv[int(len(sorted_conv) * 0.75)], 2),
                    'peer_count': len(sorted_conv),
                }

            # Fan gap: how many additional followers if you hit top 25%
            additional_fans = 0
            additional_revenue = 0
            peer_top_25 = conv_comparison.get('peer_top_25', 0)
            if peer_top_25 > 0 and u_listeners > 0:
                top25_followers_target = int(round((peer_top_25 / 100) * u_listeners * 4.3 / 0.1))
                current_followers = u_followers if u_followers > 0 else 0
                additional_fans = max(int(round(top25_followers_target - current_followers)), 0)
                additional_revenue = additional_fans * 25

            user_profile = {
                'name': lead.get('name', 'Artist'),
                'listeners': u_listeners,
                'followers': u_followers,
                'conversion_rate': u_conversion,
                'conversion_comparison': conv_comparison,
                'additional_fans': additional_fans,
                'additional_revenue': additional_revenue,
            }

        # Get raw GEMS data for matched artists (for consensus comparison)
        # Use the ISRC from each match directly — these ISRCs came from GEMS
        gems_by_artist = {}
        for m in matches:
            isrc = m.get('isrc')
            aid = str(m['artist_id'])
            if isrc and isrc in matcher._gems_by_isrc and aid not in gems_by_artist:
                gems_by_artist[aid] = matcher._gems_by_isrc[isrc]
        print(f"  GEMS lookup: {len(gems_by_artist)} of {len(matches)} matches have GEMS data")

        # Generate recommendations
        recs = _generate_recommendations(features, matches,
                                          gems_by_artist=gems_by_artist,
                                          genre_alignment=genre_alignment,
                                          user_profile=user_profile)

        # Build response
        result = {
            'features': {
                'bpm': features.get('bpm', 0),
                'key': features.get('key', '?'),
                'scale': features.get('scale', ''),
                'lufs_integrated': features.get('lufs_integrated', 0),
                'energy': features.get('energy', 0),
                'dynamic_range': features.get('dynamic_range', 0),
                'compression_amount': features.get('compression_amount', 0),
                'brightness': features.get('brightness', 0),
                'danceability': features.get('danceability', 0),
                'beat_strength': features.get('beat_strength', 0),
                'sub_ratio': features.get('sub_ratio', 0),
                'bass_ratio': features.get('bass_ratio', 0),
                'low_mid_ratio': features.get('low_mid_ratio', 0),
                'mid_ratio': features.get('mid_ratio', 0),
                'high_mid_ratio': features.get('high_mid_ratio', 0),
                'presence_ratio': features.get('presence_ratio', 0),
                'air_ratio': features.get('air_ratio', 0),
                'emotion_summary': features.get('emotion_summary', {}),
                'emotion_1': features.get('emotion_1', ''),
                'emotion_1_score': features.get('emotion_1_score', 0),
                'emotion_2': features.get('emotion_2', ''),
                'emotion_2_score': features.get('emotion_2_score', 0),
                'emotion_3': features.get('emotion_3', ''),
                'emotion_3_score': features.get('emotion_3_score', 0),
                'emotion_4': features.get('emotion_4', ''),
                'emotion_4_score': features.get('emotion_4_score', 0),
            },
            'matches': matches,
            'recommendations': recs,
            'genre_alignment': genre_alignment,
            'timing': {
                'feature_extraction_s': round(t_features, 2),
                'matching_s': round(t_match, 2),
            },
        }

        if user_profile:
            result['user_profile'] = user_profile

        if flattery_matches:
            result['flattery_matches'] = flattery_matches

        # Send email (non-blocking — don't fail the request if email fails)
        try:
            send_results_email(lead['name'], lead['email'], result)
        except Exception as e:
            print(f"Email send error (non-fatal): {e}")

        # Update analysis count in Supabase
        try:
            supabase.table('analyzer_leads').update({
                'analysis_count': supabase.rpc('increment_analysis_count', {'p_email': lead['email']}).execute() if False else 1,
                'last_analysis_at': datetime.utcnow().isoformat(),
            }).eq('email', lead['email']).execute()
        except Exception:
            pass

        return result

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Static files + SPA fallback
# ---------------------------------------------------------------------------
static_dir = Path(__file__).resolve().parent / 'static'
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
@app.get("/analyzer")
async def serve_index():
    index = static_dir / 'index.html'
    if index.exists():
        return FileResponse(str(index))
    return {"message": "SonicConverter API is running. Put index.html in static/"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run(
        'app:app',
        host='0.0.0.0',
        port=8000,
        reload=False,
    )
