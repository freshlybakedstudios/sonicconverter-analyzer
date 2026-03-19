"""
FastAPI server for the SonicConverter web analyzer.
Preloads the 1.2 GB GEMS universe cache at startup, then serves:
  POST /api/register  — capture lead (name + email)
  POST /api/analyze   — upload audio, return sonic breakdown + matches
Static files served from ./static/
"""

import asyncio
import csv
import io
import json
import math
import os
import requests
import secrets
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from supabase import create_client

load_dotenv()

from audio_analyzer import extract_features
from chartmetric_lookup import (
    lookup_artist_by_spotify,
    get_cm_token,
    fetch_listener_history,
    fetch_artist_events,
    _resolve_isrc_to_cm_track_id,
    _fetch_track_playlists_structured,
    _fetch_related_artists,
    _extract_track_credits,
    _fetch_curator_contact,
    _upsert_gems_features,
    _lookup_gems_features,
)
from email_sender import send_results_email
from job_manager import JobManager
from track_matcher import TrackMatcher, _genre_families

# ---------------------------------------------------------------------------
# Pushover notifications
# ---------------------------------------------------------------------------
PUSHOVER_USER_KEY = os.getenv('PUSHOVER_USER_KEY', 'usbh1c1xzp7ooasfu8vvwbqmgiiipp')
PUSHOVER_API_TOKEN = os.getenv('PUSHOVER_API_TOKEN', 'azq6g5x1rfzg9sykm6xscw8ypn4m1y')

def send_pushover_notification(title: str, message: str):
    """Send a push notification via Pushover."""
    try:
        requests.post(
            'https://api.pushover.net/1/messages.json',
            data={
                'token': PUSHOVER_API_TOKEN,
                'user': PUSHOVER_USER_KEY,
                'title': title,
                'message': message,
            },
            timeout=10,
        )
    except Exception as e:
        print(f"Pushover notification failed: {e}")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
matcher = TrackMatcher()
supabase = None
access_tokens: dict = {}  # token -> {name, email, created_at}
job_mgr = JobManager()
enrichment_pool = ThreadPoolExecutor(max_workers=3)
# SSE subscribers: job_id -> list of asyncio.Queue
sse_subscribers: dict = {}
_event_loop = None  # Set when the server starts

# ---------------------------------------------------------------------------
# Enrichment pause/resume: prioritize user-facing CM calls over background enrichment
# ---------------------------------------------------------------------------
_enrichment_gate = threading.Event()
_enrichment_gate.set()  # Start open (enrichment can run)

def _pause_enrichment():
    """Pause background enrichment so user-facing CM calls get priority."""
    if _enrichment_gate.is_set():
        print("  ⏸️  Pausing enrichment for user-facing CM calls")
    _enrichment_gate.clear()

def _resume_enrichment():
    """Resume background enrichment after user-facing CM calls complete."""
    if not _enrichment_gate.is_set():
        print("  ▶️  Resuming enrichment")
    _enrichment_gate.set()

# ---------------------------------------------------------------------------
# Resource switching: track web user activity, notify local Mac to pause/resume
# ---------------------------------------------------------------------------
_last_api_activity = 0.0          # timestamp of last user-facing API request
_user_was_active = False          # was a user active in the previous check?
_IDLE_TIMEOUT = 300               # 5 minutes of no requests = idle
_LOCAL_WEBHOOK_URL = os.getenv(
    'LOCAL_PIPELINE_WEBHOOK',
    'http://localhost:7890/webhook/activity'
)

def _notify_local_pipeline(event: str):
    """Fire-and-forget notification to local Mac pipeline manager."""
    try:
        requests.post(
            _LOCAL_WEBHOOK_URL,
            json={'event': event, 'timestamp': time.time()},
            timeout=3,
        )
    except Exception:
        pass  # Local machine might not be reachable from Railway

def _check_idle_and_notify():
    """Background thread: polls activity and notifies local pipeline."""
    global _user_was_active
    while True:
        time.sleep(30)  # check every 30 seconds
        now = time.time()
        is_active = (now - _last_api_activity) < _IDLE_TIMEOUT

        if is_active and not _user_was_active:
            # User just became active
            print("🌐 Web user active — notifying local pipeline to pause")
            _notify_local_pipeline('user_active')
            _user_was_active = True
        elif not is_active and _user_was_active:
            # User went idle
            print("💤 Web user idle — notifying local pipeline to resume")
            _notify_local_pipeline('user_idle')
            _user_was_active = False


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

    global _event_loop
    _event_loop = asyncio.get_event_loop()

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY')
    if url and key:
        supabase = create_client(url, key)
        job_mgr.set_supabase(supabase)
    else:
        print("⚠️  SUPABASE_URL/SUPABASE_SERVICE_KEY not set - some features disabled")

    # Start resource-switching activity monitor
    activity_thread = threading.Thread(target=_check_idle_and_notify, daemon=True)
    activity_thread.start()
    print("✅ Resource-switching activity monitor started")

    print("Ready.")
    yield
    print("Shutting down.")
    enrichment_pool.shutdown(wait=False)


app = FastAPI(title="SonicConverter Analyzer", lifespan=lifespan)

# CORS — allow the Firebase-hosted frontend + localhost dev + Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://freshlybakedstudios.com",
        "https://www.freshlybakedstudios.com",
        "https://analyze.freshlybakedstudios.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3000",
        "http://localhost:3002",
        "http://192.168.1.169:3002",
    ],
    allow_origin_regex=r"https://.*\.up\.railway\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: track user-facing API activity for resource switching
# ---------------------------------------------------------------------------
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class ActivityTrackingMiddleware(BaseHTTPMiddleware):
    """Record timestamps of user-facing API requests."""
    # Paths that indicate a real user (not health checks or internal)
    USER_PATHS = ('/api/analyze', '/api/deal/', '/api/register')

    async def dispatch(self, request: Request, call_next):
        global _last_api_activity
        path = request.url.path
        if any(path.startswith(p) for p in self.USER_PATHS):
            _last_api_activity = time.time()
        return await call_next(request)

app.add_middleware(ActivityTrackingMiddleware)


# ---------------------------------------------------------------------------
# Health check for Railway
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "cache_loaded": matcher.cache is not None}

@app.get("/api/pipeline/status")
async def pipeline_status():
    """Check if a web user is currently active (for local pipeline manager)."""
    now = time.time()
    idle_seconds = now - _last_api_activity if _last_api_activity > 0 else -1
    is_active = _last_api_activity > 0 and idle_seconds < _IDLE_TIMEOUT
    return {
        "user_active": is_active,
        "idle_seconds": round(idle_seconds, 1) if idle_seconds >= 0 else None,
        "idle_timeout": _IDLE_TIMEOUT,
        "last_activity": datetime.fromtimestamp(_last_api_activity).isoformat() if _last_api_activity > 0 else None,
    }


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
        f"[{domain}] {action}\n"
        f"You: {user_str} → Target: {peer_str}{delta_str} | {count}/{total} agree"
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

    # Store lead in Supabase with retry logic for connection issues
    row = {
        'name': name,
        'email': email,
        'created_at': datetime.utcnow().isoformat(),
        'analysis_count': 0,
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            supabase.table('analyzer_leads').insert(row).execute()
            print(f"Lead saved: {email}")
            # Send push notification
            send_pushover_notification(
                "New Analyzer Lead",
                f"{name}\n{email}"
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                print(f"Supabase insert retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"Supabase insert FAILED after {max_retries} attempts: {e} — lead: {email}")
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
    print(f"Lead saved: {email}, spotify_url={'yes: ' + spotify_url if spotify_url else 'none'}, listeners={monthly_listeners}")
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

        # Pause enrichment so user-facing CM calls get priority
        _pause_enrichment()

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
        fetch_n = 20000  # Good coverage without excessive processing

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

            matches = same_tier  # Return ALL matches, paginated on frontend
            used = ', '.join(t for t in tier_order if t in tiers_used)
            print(f"  Matching: {t_match:.1f}s — {len(all_matches)} total, {len(matches)} in tier(s): {used}")
        else:
            matches = list(all_matches)  # Return ALL matches
            print(f"  Matching: {t_match:.1f}s — {len(matches)} matches found")

        total_match_count = len(matches)

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
                EXCLUSIVE_FAMILIES = {'electronic', 'metal', 'hip-hop', 'country', 'classical', 'jazz', 'latin'}
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

            # Sort by tier (highest first), then sonic similarity as tiebreaker
            flattery_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

            # Debug: show top candidates with pronoun info
            if flattery_candidates:
                print(f"  Flattery candidates (top 10):")
                for i, (tier_num, score, m, cand_pronoun) in enumerate(flattery_candidates[:10]):
                    name = m.get('name') or 'Unknown'
                    tier = m.get('tier', '?')
                    # Compute families same way as filter does
                    cand_genre_parts = []
                    for field in ('primary_genre', 'secondary_genre'):
                        g = (m.get(field) or '').strip()
                        if g:
                            cand_genre_parts.append(g)
                    for g in m.get('artist_genres', []):
                        if g:
                            cand_genre_parts.append(g)
                    cand_families = _genre_families(', '.join(cand_genre_parts))
                    shared = user_families & cand_families
                    print(f"    {i+1}. {name[:20]:<20} {tier:<12} {score:.1%} | shared: {shared}")

            for _, _, m, _ in flattery_candidates[:20]:
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
        MAX_REASONABLE_CONVERSION = 15.0
        if u_listeners > 0 and matches:
            match_conversions = [
                m['conversion_rate'] for m in matches
                if m.get('conversion_rate') is not None
                and 0 < m['conversion_rate'] <= MAX_REASONABLE_CONVERSION
            ]
            conv_comparison = {}
            if match_conversions:
                sorted_conv = sorted(match_conversions)
                n = len(sorted_conv)
                conv_comparison = {
                    'peer_median': round(sorted_conv[n // 2], 2),
                    'peer_top_25': round(sorted_conv[int(n * 0.75)], 2),
                    'peer_bottom_25': round(sorted_conv[n // 4], 2),
                    'peer_p99': round(sorted_conv[min(n - 1, int(n * 0.99))], 2),
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

        # Create background enrichment job
        job_id = job_mgr.create_job(token, features, matches, all_matches=matches)

        # Get user's CM artist ID for related artists lookup
        user_cm_id = None
        if cm_data:
            user_cm_id = cm_data.get('cm_id')
        elif cached_artist_data:
            aid, adata = cached_artist_data
            user_cm_id = adata.get('cm_id') or adata.get('id')

        # Build response
        result = {
            'job_id': job_id,
            'total_match_count': total_match_count,
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
            'all_matches': all_matches[:5000],
            'total_all_matches': len(all_matches),
            'user_tier': user_tier or '',
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

        # Kick off background enrichment: displayed matches FIRST, then wider pool
        # This ensures every match the user sees gets playlist data
        displayed_ids = {str(m.get('artist_id', '')) for m in matches}
        wider_pool = sorted(all_matches, key=lambda m: m.get('similarity', 0), reverse=True)
        wider_extra = [m for m in wider_pool if str(m.get('artist_id', '')) not in displayed_ids]
        enrichment_matches = matches + wider_extra
        enrichment_matches = enrichment_matches[:200]  # Cap total
        # Resume enrichment gate — user-facing CM calls are done
        _resume_enrichment()
        enrichment_pool.submit(
            _run_background_enrichment,
            job_id, enrichment_matches, user_cm_id,
        )

        return result

    finally:
        # Ensure enrichment gate is open even on error paths
        _resume_enrichment()
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
def _sse_publish(job_id: str, event: str, data: dict):
    """Publish an SSE event to all subscribers for this job (thread-safe)."""
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    queues = sse_subscribers.get(job_id, [])
    for q in queues:
        try:
            if _event_loop and _event_loop.is_running():
                _event_loop.call_soon_threadsafe(q.put_nowait, payload)
            else:
                q.put_nowait(payload)
        except Exception:
            pass


def _resolve_curator_id_by_playlist(spotify_playlist_id: str, cm_playlist_id: str = '') -> int | None:
    """Look up cm_curator_id — first from editorial_playlists table, then CM API."""
    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')

    # Step 1: Check editorial_playlists table
    if supa_url and supa_key and spotify_playlist_id:
        try:
            headers = {
                'apikey': supa_key,
                'Authorization': f'Bearer {supa_key}',
                'Accept': 'application/json',
            }
            resp = requests.get(
                f"{supa_url}/rest/v1/editorial_playlists"
                f"?playlist_id=eq.{spotify_playlist_id}"
                f"&select=cm_curator_id&limit=1",
                headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                rows = resp.json()
                if rows and rows[0].get('cm_curator_id'):
                    return rows[0]['cm_curator_id']
        except Exception as e:
            print(f"Playlist→curator DB lookup failed for {spotify_playlist_id}: {e}")

    return None


def _resolve_curator_id_by_name(curator_name: str) -> int | None:
    """Look up cm_curator_id from our curators table by name."""
    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')
    if not supa_url or not supa_key or not curator_name:
        return None
    try:
        headers = {
            'apikey': supa_key,
            'Authorization': f'Bearer {supa_key}',
            'Accept': 'application/json',
        }
        resp = requests.get(
            f"{supa_url}/rest/v1/curators"
            f"?curator_name=eq.{requests.utils.quote(curator_name)}"
            f"&select=cm_curator_id&limit=1",
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            rows = resp.json()
            if rows:
                return rows[0].get('cm_curator_id')
    except Exception as e:
        print(f"Curator name lookup failed for '{curator_name}': {e}")
    return None


def _lookup_curator_local(cm_curator_id: int) -> dict | None:
    """Look up curator from our local curators table (143K+ records)."""
    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')
    if not supa_url or not supa_key:
        return None
    try:
        project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
        headers = {
            'apikey': supa_key,
            'Authorization': f'Bearer {supa_key}',
            'Accept': 'application/json',
        }
        resp = requests.get(
            f"{supa_url}/rest/v1/curators"
            f"?cm_curator_id=eq.{cm_curator_id}"
            f"&select=cm_curator_id,curator_name,submission_email,"
            f"instagram_url,facebook_url,website_url,twitter_url,"
            f"groover_url,submithub_url,spotify_url,total_followers",
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            rows = resp.json()
            if rows:
                row = rows[0]
                return {
                    'email': row.get('submission_email') or '',
                    'instagram_url': row.get('instagram_url') or '',
                    'facebook_url': row.get('facebook_url') or '',
                    'website_url': row.get('website_url') or '',
                    'twitter_url': row.get('twitter_url') or '',
                    'groover_url': row.get('groover_url') or '',
                    'submithub_url': row.get('submithub_url') or '',
                    'spotify_url': row.get('spotify_url') or '',
                    'total_followers': row.get('total_followers') or 0,
                }
    except Exception as e:
        print(f"Local curator lookup failed for {cm_curator_id}: {e}")
    return None


def _upsert_curator(cm_curator_id: int, curator_data: dict):
    """Upsert curator contact info into our curators table for future lookups."""
    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')
    if not supa_url or not supa_key or not cm_curator_id:
        return
    try:
        headers = {
            'apikey': supa_key,
            'Authorization': f'Bearer {supa_key}',
            'Content-Type': 'application/json',
            'Prefer': 'resolution=merge-duplicates',
        }
        payload = {'cm_curator_id': cm_curator_id}
        for field in ('submission_email', 'instagram_url', 'facebook_url',
                      'website_url', 'twitter_url', 'groover_url',
                      'submithub_url', 'curator_name'):
            val = curator_data.get(field)
            if val:
                payload[field] = val
        # Map 'email' to 'submission_email' if present
        if curator_data.get('email') and 'submission_email' not in payload:
            payload['submission_email'] = curator_data['email']
        requests.post(
            f"{supa_url}/rest/v1/curators",
            json=payload, headers=headers, timeout=10,
        )
    except Exception as e:
        print(f"Curator upsert failed for {cm_curator_id}: {e}")


def _update_curator_cache(cm_curator_id: int, curator_info: dict):
    """Write scraped email back to curator cache so we don't re-scrape."""
    try:
        supa_url = os.getenv('SUPABASE_URL')
        supa_key = os.getenv('SUPABASE_SERVICE_KEY')
        if not supa_url or not supa_key:
            return
        from datetime import timezone
        project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
        headers = {
            'apikey': supa_key,
            'Authorization': f"Bearer {supa_key}",
            'Content-Type': 'application/json',
            'Prefer': 'resolution=merge-duplicates,return=representation',
        }
        # Merge email into existing cached data
        cache_data = {k: v for k, v in curator_info.items()
                      if k not in ('name', 'cm_curator_id', 'playlist_name',
                                   'playlist_link', 'followers')}
        payload = {
            'curator_key': str(cm_curator_id),
            'contact_data': json.dumps(cache_data),
            'fetched_at': datetime.now(timezone.utc).isoformat(),
        }
        requests.post(
            f"{supa_url}/rest/v1/curator_contacts_cache",
            json=payload, headers=headers, timeout=10,
        )
    except Exception as e:
        print(f"Curator cache update failed for {cm_curator_id}: {e}")


def _compute_playlist_score(sonic_similarity: float, followers: int,
                            is_editorial: bool, is_current: bool,
                            confidence_boost: float = 0.0,
                            added_at: str = '') -> float:
    """Playlist ranking — recency weighted heavily over follower count."""
    import math as _math
    from datetime import datetime, timezone

    follower_score = _math.log10(max(followers, 1)) / 6.0
    editorial_bonus = 1.0 if is_editorial else 0.0
    recency_bonus = 1.0 if is_current else 0.3

    # Freshness boost based on when track was added to playlist
    freshness_bonus = 0.0
    if added_at:
        try:
            added = datetime.fromisoformat(added_at.replace('Z', '+00:00'))
            days_ago = (datetime.now(timezone.utc) - added).days
            if days_ago <= 30:
                freshness_bonus = 1.0   # Added in last month — very active
            elif days_ago <= 90:
                freshness_bonus = 0.7   # Last 3 months — still active
            elif days_ago <= 180:
                freshness_bonus = 0.4   # Last 6 months — somewhat active
            elif days_ago <= 365:
                freshness_bonus = 0.1   # Last year — borderline
            # Older than 1 year = 0.0
        except (ValueError, TypeError):
            pass

    return (
        0.25 * sonic_similarity +
        0.15 * confidence_boost +
        0.10 * min(follower_score, 1.0) +
        0.05 * editorial_bonus +
        0.10 * recency_bonus +
        0.35 * freshness_bonus        # Recency is king
    )


# ---------------------------------------------------------------------------
# Background enrichment
# ---------------------------------------------------------------------------
def _run_background_enrichment(job_id: str, matches: list, user_cm_id: int = None):
    """
    Runs in a thread. Enriches matches with:
    1. CM Related Artists (fast, 1 API call) → confidence signal
    2. Playlists per match (slow, 2-3 calls each) → streamed via SSE
    3. Track credits → piggybacks on playlist resolution
    4. Curator emails → runs after playlists
    """
    try:
        refresh_token = os.getenv('REFRESH_TOKEN')
        if not refresh_token:
            print(f"Enrichment [{job_id[:8]}]: No REFRESH_TOKEN, skipping")
            job_mgr.update_job(job_id, status='complete')
            _sse_publish(job_id, 'complete', {'status': 'complete'})
            return

        token = get_cm_token(refresh_token)
        if not token:
            print(f"Enrichment [{job_id[:8]}]: Failed to get CM token")
            job_mgr.update_job(job_id, status='complete')
            _sse_publish(job_id, 'complete', {'status': 'complete'})
            return

        # --- Phase 3: Related Artists (fast, 1 API call) ---
        related_set = set()
        if user_cm_id:
            try:
                related = _fetch_related_artists(token, user_cm_id, limit=50)
                if related:
                    for ra in related:
                        name = (ra.get('name') or '').lower().strip()
                        if name:
                            related_set.add(name)
                        sp_url = ra.get('spotify_url') or ''
                        if sp_url:
                            related_set.add(sp_url.split('?')[0].rstrip('/'))
                    print(f"Enrichment [{job_id[:8]}]: {len(related_set)} related artists found")
                    job_mgr.update_job(job_id,
                                       related_artists=related,
                                       progress={'related_artists': 'done'})
                    _sse_publish(job_id, 'related_artists', {
                        'related': related,
                        'count': len(related),
                    })
            except Exception as e:
                print(f"Enrichment [{job_id[:8]}]: Related artists failed: {e}")
        job_mgr.update_job(job_id, progress={'related_artists': 'done'})

        # Build confidence map — which matches are also CM-related?
        confidence_map = {}
        for m in matches:
            match_name = (m.get('name') or '').lower().strip()
            match_sp = (m.get('spotify_url') or '').split('?')[0].rstrip('/')
            is_double = match_name in related_set or (match_sp and match_sp in related_set)
            if is_double:
                key = str(m.get('artist_id', m.get('name', '')))
                confidence_map[key] = 'double_validated'
        if confidence_map:
            job_mgr.update_job(job_id, confidence_map=confidence_map)
            _sse_publish(job_id, 'confidence', {'confidence_map': confidence_map})
            print(f"Enrichment [{job_id[:8]}]: {len(confidence_map)} double-validated matches")

        # --- Phase 2: Playlists + Credits + Curators (batched, 10 artists at a time) ---
        sorted_matches = sorted(matches, key=lambda m: m.get('similarity', 0), reverse=True)
        total = len(sorted_matches)
        all_playlists = []
        seen_curator_ids = set()
        curator_count = 0
        BATCH_SIZE = 10

        _sse_publish(job_id, 'enrichment_progress', {
            'batch': 0, 'total_batches': (total + BATCH_SIZE - 1) // BATCH_SIZE,
            'curators_found': 0, 'phase': 'playlists',
        })

        for batch_start in range(0, total, BATCH_SIZE):
            # Brief pause if user-facing CM calls need priority (2s max to prevent blocking)
            if not _enrichment_gate.wait(timeout=2):
                _enrichment_gate.set()  # Force open — never block enrichment
                _enrichment_gate.set()
            batch = sorted_matches[batch_start:batch_start + BATCH_SIZE]
            batch_playlists = []

            # --- Batch step A: Playlists + Credits for this batch ---
            for idx_in_batch, m in enumerate(batch):
                idx = batch_start + idx_in_batch
                isrc = m.get('isrc') or ''
                match_key = str(m.get('artist_id', m.get('name', '')))
                artist_name = m.get('name', '?')
                tier = m.get('tier', '?')
                print(f"Enrichment [{job_id[:8]}]: [{idx+1}/{total}] {artist_name} (tier={tier}, key={match_key}, isrc={isrc[:12] if isrc else 'none'})")
                if not isrc:
                    print(f"Enrichment [{job_id[:8]}]:   -> SKIP (no isrc)")
                    job_mgr.update_job(job_id, progress={'playlists': f'{idx+1}/{total}'})
                    continue

                try:
                    cm_track_id = _resolve_isrc_to_cm_track_id(token, isrc)
                    if not cm_track_id:
                        print(f"Enrichment [{job_id[:8]}]:   -> SKIP (no cm_track_id for {isrc})")
                        job_mgr.update_job(job_id, progress={'playlists': f'{idx+1}/{total}'})
                        continue

                    playlists = _fetch_track_playlists_structured(
                        token, cm_track_id,
                        isrc=isrc,
                        artist_name=m.get('name', ''),
                        track_name=m.get('track_name', ''),
                    )

                    try:
                        credits = _extract_track_credits(token, cm_track_id)
                        if credits and (credits.get('producers') or credits.get('writers')):
                            job_mgr.update_job(job_id, credits={match_key: credits})
                            _sse_publish(job_id, 'credits', {
                                'match_key': match_key,
                                'artist_name': m.get('name', ''),
                                'credits': credits,
                            })
                    except Exception as e:
                        print(f"Enrichment [{job_id[:8]}]: Credits failed for {isrc}: {e}")

                    print(f"Enrichment [{job_id[:8]}]:   -> {len(playlists) if playlists else 0} playlists for {artist_name}")
                    if playlists:
                        similarity = m.get('similarity', 0)
                        conf_boost = 0.5 if match_key in confidence_map else 0.0
                        for pl in playlists:
                            pl['sonic_match'] = m.get('name', '')
                            pl['track_name'] = m.get('track_name', '')
                            pl['sonic_similarity'] = similarity
                            pl['score'] = _compute_playlist_score(
                                similarity,
                                pl.get('followers', 0),
                                pl.get('editorial', False),
                                pl.get('status') == 'current',
                                confidence_boost=conf_boost,
                                added_at=pl.get('added_at', ''),
                            )
                            pl['double_validated'] = match_key in confidence_map
                        batch_playlists.extend(playlists)
                        all_playlists.extend(playlists)

                        job_mgr.update_job(job_id, playlists={match_key: playlists})
                        _sse_publish(job_id, 'playlists', {
                            'match_key': match_key,
                            'artist_name': m.get('name', ''),
                            'playlists': playlists,
                            'progress': f'{idx+1}/{total}',
                        })

                    job_mgr.update_job(job_id, progress={'playlists': f'{idx+1}/{total}'})

                except Exception as e:
                    print(f"Enrichment [{job_id[:8]}]: Playlist failed for {isrc}: {e}")
                    job_mgr.update_job(job_id, progress={'playlists': f'{idx+1}/{total}'})

            # --- Batch step B: Curator contacts for this batch's playlists ---
            batch_curators = []
            for pl in batch_playlists:
                cm_cid = pl.get('cm_curator_id')
                curator_name = pl.get('curator_name', '')
                # Try to resolve curator ID from editorial_playlists table
                if not cm_cid:
                    spotify_pid = pl.get('playlist_id', '')
                    cm_pid = pl.get('cm_playlist_id', '')
                    if spotify_pid:
                        cm_cid = _resolve_curator_id_by_playlist(spotify_pid, cm_pid)
                        if cm_cid:
                            pl['cm_curator_id'] = cm_cid
                # Fallback: resolve by curator name
                if not cm_cid and curator_name:
                    cm_cid = _resolve_curator_id_by_name(curator_name)
                    if cm_cid:
                        pl['cm_curator_id'] = cm_cid
                ckey = str(cm_cid) if cm_cid else curator_name
                if not ckey or ckey in seen_curator_ids:
                    continue
                # Skip Spotify editorial (not a real curator you can contact)
                spotify_uid = pl.get('spotify_user_id', '')
                if spotify_uid == 'spotify' or curator_name.lower() == 'spotify':
                    continue
                seen_curator_ids.add(ckey)
                batch_curators.append({
                    'name': curator_name,
                    'cm_curator_id': cm_cid,
                    'playlist_name': pl.get('name', ''),
                    'playlist_link': pl.get('link', ''),
                    'followers': pl.get('followers', 0),
                    'spotify_user_id': spotify_uid,
                    'sonic_match': pl.get('sonic_match', ''),
                    'track_name': pl.get('track_name', ''),
                })

            if batch_curators:
                print(f"Enrichment [{job_id[:8]}]: Batch {batch_start//BATCH_SIZE+1} — resolving {len(batch_curators)} curators...")

            for curator_info in batch_curators:
                try:
                    cm_cid = curator_info.get('cm_curator_id')

                    # Step 1: Check our local curators table (143K+ records, instant)
                    local = None
                    if cm_cid:
                        local = _lookup_curator_local(cm_cid)
                        if local:
                            curator_info.update({k: v for k, v in local.items() if v})
                            print(f"Enrichment [{job_id[:8]}]: LOCAL curator {cm_cid} ({curator_info['name']}): "
                                  f"email={'yes' if local.get('email') else 'no'}, "
                                  f"ig={'yes' if local.get('instagram_url') else 'no'}, "
                                  f"fb={'yes' if local.get('facebook_url') else 'no'}, "
                                  f"web={'yes' if local.get('website_url') else 'no'}")

                    # Step 2: CM API fallback (if no contact info from local table)
                    has_any_local = (curator_info.get('email') or
                                    curator_info.get('instagram_url') or
                                    curator_info.get('facebook_url') or
                                    curator_info.get('website_url'))
                    if cm_cid and not has_any_local:
                        contact = _fetch_curator_contact(token, cm_cid)
                        if contact:
                            curator_info.update(contact)
                            _upsert_curator(cm_cid, contact)
                            print(f"Enrichment [{job_id[:8]}]: CM API curator {cm_cid} ({curator_info['name']}): "
                                  f"email={'yes' if contact.get('email') else 'no'}, "
                                  f"ig={'yes' if contact.get('instagram_url') else 'no'}, "
                                  f"fb={'yes' if contact.get('facebook_url') else 'no'}, "
                                  f"web={'yes' if contact.get('website_url') else 'no'}")

                    # Step 3: Check name-based cache (for curators without CM ID)
                    if not curator_info.get('email') and not cm_cid:
                        try:
                            supa_url = os.getenv('SUPABASE_URL')
                            supa_key = os.getenv('SUPABASE_SERVICE_KEY')
                            if supa_url and supa_key:
                                cache_resp = requests.get(
                                    f"{supa_url}/rest/v1/curator_contacts_cache"
                                    f"?curator_key=eq.{requests.utils.quote(curator_info.get('name', ''))}",
                                    headers={'apikey': supa_key, 'Authorization': f'Bearer {supa_key}'},
                                    timeout=10,
                                )
                                if cache_resp.status_code == 200 and cache_resp.json():
                                    cached = json.loads(cache_resp.json()[0].get('contact_data', '{}'))
                                    if cached.get('email'):
                                        curator_info['email'] = cached['email']
                                        curator_info['email_source'] = cached.get('email_source', 'cached')
                                        print(f"Enrichment [{job_id[:8]}]: Cache hit for '{curator_info['name']}': {cached['email']}")
                        except Exception:
                            pass

                    # Step 4: Scraper fallback (IG → FB → website chain)
                    if not curator_info.get('email'):
                        ig = curator_info.get('instagram_url', '')
                        fb = curator_info.get('facebook_url', '')
                        web = curator_info.get('website_url', '')
                        if ig or fb or web:
                            print(f"Enrichment [{job_id[:8]}]: Scraper trying '{curator_info['name']}' — ig={bool(ig)}, fb={bool(fb)}, web={bool(web)}")
                            try:
                                from curator_scraper import scrape_curator_emails
                                scrape_result = scrape_curator_emails(
                                    curator_info['name'],
                                    instagram_url=ig,
                                    facebook_url=fb,
                                    website_url=web,
                                )
                                if scrape_result and scrape_result.get('email'):
                                    curator_info['email'] = scrape_result['email']
                                    curator_info['email_source'] = scrape_result.get('email_source', 'scraper')
                                    print(f"Enrichment [{job_id[:8]}]: Scraper found email for '{curator_info['name']}': {scrape_result['email']}")
                                    if cm_cid:
                                        _update_curator_cache(cm_cid, curator_info)
                                        _upsert_curator(cm_cid, curator_info)
                                    else:
                                        # No CM ID — store by name so we don't re-scrape
                                        _update_curator_cache(curator_info.get('name', 'unknown'), curator_info)
                                else:
                                    print(f"Enrichment [{job_id[:8]}]: Scraper no email for '{curator_info['name']}'")
                            except ImportError:
                                print(f"Enrichment [{job_id[:8]}]: curator_scraper not available")
                            except Exception as e:
                                print(f"Enrichment [{job_id[:8]}]: Scraper error for '{curator_info.get('name', '?')}': {e}")

                    has_contact = (curator_info.get('email') or
                                   curator_info.get('instagram_url') or
                                   curator_info.get('facebook_url') or
                                   curator_info.get('website_url') or
                                   curator_info.get('twitter_url') or
                                   curator_info.get('groover_url') or
                                   curator_info.get('submithub_url') or
                                   curator_info.get('submission_url'))
                    if not has_contact:
                        print(f"Enrichment [{job_id[:8]}]: No contact info for curator '{curator_info.get('name', '?')}' (cm_id={cm_cid}) — skipped")
                    if has_contact:
                        curator_count += 1
                        job_mgr.update_job(job_id,
                                           curator_emails={curator_info['name']: curator_info})
                        _sse_publish(job_id, 'curator_emails', {
                            'curator': curator_info,
                            'progress': f'{curator_count} curators',
                        })
                except Exception as e:
                    print(f"Enrichment [{job_id[:8]}]: Curator failed for {curator_info.get('name', '?')}: {e}")

            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Enrichment [{job_id[:8]}]: Batch {batch_num}/{total_batches} done — {curator_count} curators so far")
            _sse_publish(job_id, 'enrichment_progress', {
                'batch': batch_num,
                'total_batches': total_batches,
                'curators_found': curator_count,
            })

        # Dedupe playlists by playlist_id, keeping highest-scored entry
        seen_pl_ids = {}
        for pl in all_playlists:
            pid = pl.get('playlist_id', '')
            if not pid:
                continue
            existing = seen_pl_ids.get(pid)
            if not existing or (pl.get('score', 0) > existing.get('score', 0)):
                seen_pl_ids[pid] = pl
        deduped_playlists = sorted(seen_pl_ids.values(),
                                    key=lambda p: p.get('score', 0), reverse=True)

        job_mgr.update_job(job_id, all_playlists=deduped_playlists)
        _sse_publish(job_id, 'all_playlists', {
            'playlists': deduped_playlists,
            'total': len(deduped_playlists),
        })
        print(f"Enrichment [{job_id[:8]}]: {len(deduped_playlists)} unique playlists, {curator_count} curators with contact info")

        # Done
        job_mgr.update_job(job_id, status='complete')
        _sse_publish(job_id, 'complete', {'status': 'complete'})
        print(f"Enrichment [{job_id[:8]}]: Complete")
        # Notify resource-switcher that we're done — local scripts can resume
        _notify_local_pipeline('user_idle')

    except Exception as e:
        print(f"Enrichment [{job_id[:8]}]: Fatal error: {e}")
        job_mgr.update_job(job_id, status='error')
        _sse_publish(job_id, 'error', {'error': str(e)})
        _notify_local_pipeline('user_idle')


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------
@app.get("/api/analysis/{job_id}/stream")
async def stream_enrichment(job_id: str):
    """SSE endpoint for progressive enrichment updates."""
    job = job_mgr.get_job_state(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    queue = asyncio.Queue()
    if job_id not in sse_subscribers:
        sse_subscribers[job_id] = []
    sse_subscribers[job_id].append(queue)

    async def event_generator():
        try:
            # Send current state first (catch-up for late joiners)
            state = job_mgr.get_job_state(job_id)
            if state:
                if state.get('related_artists'):
                    yield f"event: related_artists\ndata: {json.dumps({'related': state['related_artists'], 'count': len(state['related_artists'])})}\n\n"
                if state.get('confidence_map'):
                    yield f"event: confidence\ndata: {json.dumps({'confidence_map': state['confidence_map']})}\n\n"
                if state.get('playlists'):
                    for mk, pls in state['playlists'].items():
                        yield f"event: playlists\ndata: {json.dumps({'match_key': mk, 'playlists': pls})}\n\n"
                if state.get('credits'):
                    for mk, creds in state['credits'].items():
                        yield f"event: credits\ndata: {json.dumps({'match_key': mk, 'credits': creds})}\n\n"
                if state.get('all_playlists'):
                    yield f"event: all_playlists\ndata: {json.dumps({'playlists': state['all_playlists'], 'total': len(state['all_playlists'])})}\n\n"
                if state.get('curator_emails'):
                    for name, info in state['curator_emails'].items():
                        yield f"event: curator_emails\ndata: {json.dumps({'curator': info})}\n\n"
                if state.get('status') == 'complete':
                    yield f"event: complete\ndata: {json.dumps({'status': 'complete'})}\n\n"
                    return

            # Stream live updates
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    yield msg
                    if '"status": "complete"' in msg or '"status": "error"' in msg:
                        break
                except asyncio.TimeoutError:
                    yield f": keepalive\n\n"
        finally:
            # Cleanup subscriber
            if job_id in sse_subscribers:
                try:
                    sse_subscribers[job_id].remove(queue)
                except ValueError:
                    pass
                if not sse_subscribers[job_id]:
                    del sse_subscribers[job_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# CSV export endpoint
# ---------------------------------------------------------------------------
@app.get("/api/analysis/{job_id}/csv")
async def export_csv(job_id: str):
    """Download playlists + curator contacts as CSV."""
    state = job_mgr.get_job_state(job_id)
    if not state:
        raise HTTPException(404, "Job not found")

    # Flatten all playlists from every match into one list, dedup by playlist_id
    all_playlists = []
    seen_pids = set()
    playlists_dict = state.get('playlists') or {}
    for match_key, pls in playlists_dict.items():
        for pl in pls:
            pid = pl.get('playlist_id', '')
            if pid and pid in seen_pids:
                continue
            if pid:
                seen_pids.add(pid)
            all_playlists.append(pl)
    all_playlists.sort(key=lambda p: p.get('score', 0), reverse=True)

    # Build curator lookup by name
    curator_map = state.get('curator_emails') or {}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Playlist', 'Playlist URL', 'Followers', 'Type', 'Status',
        'Added/Updated', 'Score', 'Sonic Match', 'Similarity',
        'Curator', 'Email', 'Instagram', 'Facebook', 'Website',
        'Groover', 'SubmitHub', 'Submission URL',
    ])

    for pl in all_playlists:
        curator_name = pl.get('curator_name', '')
        contact = curator_map.get(curator_name, {})

        writer.writerow([
            pl.get('name', ''),
            pl.get('link', ''),
            pl.get('followers', 0),
            'Editorial' if pl.get('editorial') else 'Indie',
            pl.get('status', ''),
            pl.get('added_at') or pl.get('last_updated', ''),
            f"{pl.get('score', 0):.3f}",
            pl.get('sonic_match', ''),
            f"{pl.get('sonic_similarity', 0):.1%}",
            curator_name,
            contact.get('email', ''),
            contact.get('instagram_url', ''),
            contact.get('facebook_url', ''),
            contact.get('website_url', ''),
            contact.get('groover_url', ''),
            contact.get('submithub_url', ''),
            contact.get('submission_url', ''),
        ])

    csv_bytes = output.getvalue().encode('utf-8')
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="playlists_{job_id[:8]}.csv"'},
    )


# ---------------------------------------------------------------------------
# Spotify URL analysis endpoint
# ---------------------------------------------------------------------------
@app.post("/api/analyze-url")
async def analyze_url(
    spotify_url: str = Form(...),
    token: str = Form(...),
    genre: Optional[str] = Form(None),
):
    """
    Analyze a Spotify track by URL.
    Tries: 1) Mac worker pickup (30s), 2) Spotify preview fallback.
    """
    lead = access_tokens.get(token)
    if not lead:
        raise HTTPException(401, "Invalid or expired token. Please register first.")

    # Validate Spotify URL
    if 'open.spotify.com/track/' not in spotify_url and 'spotify:track:' not in spotify_url:
        raise HTTPException(400, "Please provide a valid Spotify track URL")

    # Extract track ID
    if 'spotify:track:' in spotify_url:
        track_id = spotify_url.split('spotify:track:')[1].split('?')[0]
    else:
        track_id = spotify_url.split('track/')[1].split('?')[0].split('/')[0]

    # Job ID created later — only set to pending_features if we actually need Mac worker

    # Try to get Spotify preview URL + artist info via Spotify Web API
    preview_url = None
    sp_token = os.getenv('SPOTIFY_CLIENT_ID')
    sp_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    track_name = ''
    artist_name = ''
    artist_spotify_url = ''
    track_isrc = ''
    cm_data = None
    user_cm_id = None

    if sp_token and sp_secret:
        try:
            auth_resp = requests.post(
                'https://accounts.spotify.com/api/token',
                data={'grant_type': 'client_credentials'},
                auth=(sp_token, sp_secret),
                timeout=10,
            )
            if auth_resp.status_code == 200:
                sp_bearer = auth_resp.json()['access_token']
                track_resp = requests.get(
                    f'https://api.spotify.com/v1/tracks/{track_id}',
                    headers={'Authorization': f'Bearer {sp_bearer}'},
                    timeout=10,
                )
                if track_resp.status_code == 200:
                    track_data = track_resp.json()
                    preview_url = track_data.get('preview_url')
                    track_name = track_data.get('name', '')
                    track_isrc = (track_data.get('external_ids') or {}).get('isrc', '')
                    artists = track_data.get('artists', [])
                    artist_name = ', '.join(a['name'] for a in artists)
                    # Get primary artist's Spotify URL for CM lookup
                    if artists:
                        ext_urls = artists[0].get('external_urls', {})
                        artist_spotify_url = ext_urls.get('spotify', '')
        except Exception as e:
            print(f"Spotify API failed: {e}")

    # Look up track's artist in Chartmetric for genres + related artists
    # NOTE: Use track's artist for genre/CM ID, but keep user's own tier from registration
    # Pause enrichment so user-facing CM calls get priority
    _pause_enrichment()
    track_artist_cm_data = None
    if artist_spotify_url:
        print(f"  URL analysis: looking up track artist {artist_name} via CM...")
        track_artist_cm_data = lookup_artist_by_spotify(artist_spotify_url)
        if track_artist_cm_data:
            user_cm_id = track_artist_cm_data.get('cm_id')
            cm_genres = track_artist_cm_data.get('genres', '')
            if cm_genres:
                print(f"  URL analysis: CM genres = {cm_genres} (overriding dropdown '{genre or 'none'}')")
                genre = cm_genres
            # Only set listeners if the user doesn't already have them from registration
            if not lead.get('monthly_listeners') and track_artist_cm_data.get('listeners'):
                lead['monthly_listeners'] = track_artist_cm_data['listeners']
            print(f"  URL analysis: {track_artist_cm_data['name']} — {track_artist_cm_data.get('tier', '?')} tier, "
                  f"{track_artist_cm_data.get('listeners', 0):.0f} listeners")
        else:
            print(f"  URL analysis: CM lookup returned nothing for {artist_spotify_url}")

    # On Railway (production), use cached features if available — Railway has a 30s proxy timeout
    # that kills the connection before the Mac worker can finish (90s).
    # On localhost, always scan fresh for accurate production recs.
    features = None
    is_local = os.getenv('RAILWAY_ENVIRONMENT') is None

    if not is_local and track_isrc:
        features = _lookup_gems_features(track_isrc)
        if features:
            print(f"  URL analysis: using cached features for ISRC {track_isrc}")

    # 1) Try Spotify preview (immediate)
    if not features and preview_url:
        print(f"  URL analysis: downloading preview for {track_id}...")
        try:
            preview_resp = requests.get(preview_url, timeout=30)
            if preview_resp.status_code == 200 and len(preview_resp.content) > 1000:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                tmp.write(preview_resp.content)
                tmp.flush()
                tmp.close()
                try:
                    features = extract_features(tmp.name, genre_hint=genre or '')
                    print(f"  URL analysis: features extracted from preview ({len(preview_resp.content) // 1024}KB)")
                finally:
                    os.unlink(tmp.name)
        except Exception as e:
            print(f"  URL analysis: preview download/analysis failed: {e}")
    elif not features:
        print(f"  URL analysis: no preview_url available for {track_id}")

    # Create job for enrichment tracking (needed regardless of feature source)
    if not features:
        # Create as pending_features — Mac worker will pick this up
        job_id = job_mgr.create_job(token, {}, [])
        job_mgr.update_job(job_id, status='pending_features',
                           spotify_url=spotify_url)
        # CRITICAL: Pause local scripts BEFORE Mac worker captures audio
        # GEMS uses Spotify playback — if it's running it will contaminate the capture
        global _last_api_activity
        _last_api_activity = time.time()
        _notify_local_pipeline('user_active')
        print(f"  URL analysis: waiting for Mac worker (up to 90s)...")
        deadline = time.time() + 90

        def _poll_supabase_for_features():
            """Synchronous Supabase poll — runs in thread to avoid blocking event loop."""
            if not job_mgr._supabase:
                return None
            try:
                resp = job_mgr._supabase.table('analysis_jobs').select('status,features').eq('id', job_id).execute()
                if resp.data:
                    row = resp.data[0]
                    if row.get('status') == 'features_ready':
                        f = row.get('features', {})
                        if isinstance(f, str):
                            f = json.loads(f)
                        return f
            except Exception as e:
                print(f"  URL analysis: poll error: {e}")
            return None

        loop = asyncio.get_event_loop()
        while time.time() < deadline:
            result = await loop.run_in_executor(None, _poll_supabase_for_features)
            if result:
                features = result
                print(f"  URL analysis: features received from Mac worker")
                break
            await asyncio.sleep(3)

    if not features:
        job_mgr.update_job(job_id, status='error')
        raise HTTPException(
            503,
            "Could not analyze this track. Spotify preview is not available for this track "
            "and the local audio worker is offline. Try uploading the audio file instead."
        )

    # We have features — now run the normal matching pipeline
    job_mgr.update_job(job_id, status='matching', features=features)

    # Run emotion detection on features (same as file upload path)
    from audio_analyzer import _emotion_detector
    emo = _emotion_detector.detect(features, genre or '')
    top_emo = emo.get('emotions', [])
    for i in range(4):
        if i < len(top_emo):
            features[f'emotion_{i+1}'] = top_emo[i][0]
            features[f'emotion_{i+1}_score'] = top_emo[i][1]
        else:
            features[f'emotion_{i+1}'] = 'neutral'
            features[f'emotion_{i+1}_score'] = 0.0
    features['emotion_summary'] = emo

    # Find matches (same logic as /api/analyze)
    # Use user's own tier if they registered with an artist URL,
    # otherwise fall back to the track artist's tier
    user_monthly = lead.get('monthly_listeners')
    if not user_monthly and track_artist_cm_data:
        user_monthly = track_artist_cm_data.get('listeners', 0)
    user_tier = _listeners_to_tier(user_monthly) if user_monthly else 'micro'
    fetch_n = 20000

    all_found = matcher.find_matches(features, genre_hint=genre or '', top_n=fetch_n, threshold=0.55)

    # Apply genre family filtering
    user_families = _genre_families(genre or '')

    def has_foreign(m):
        cand_genres = []
        for field in ('primary_genre', 'secondary_genre'):
            g = (m.get(field) or '').strip()
            if g:
                cand_genres.append(g)
        for g in m.get('artist_genres', []):
            if g:
                cand_genres.append(g)
        cand_families = _genre_families(*cand_genres)
        foreign = cand_families - user_families
        return bool(foreign) and bool(user_families)

    # Exclude self-matches (the artist being analyzed)
    exclude_artist_id = None
    if track_artist_cm_data:
        exclude_artist_id = str(track_artist_cm_data.get('cm_id', ''))
    if exclude_artist_id:
        all_found = [m for m in all_found if str(m.get('artist_id', '')) != exclude_artist_id]

    # Save unfiltered matches for flattery (uses looser filtering)
    all_matches_unfiltered = list(all_found)

    all_found = [m for m in all_found if not has_foreign(m)]

    # Tier filtering for display table
    MIN_PEER_MATCHES = 10
    tier_order = list(TIER_RANGES.keys())
    if user_tier:
        same_tier = [m for m in all_found if m.get('tier') == user_tier]
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
                same_tier = [m for m in all_found if m.get('tier') in tiers_used]
                if len(same_tier) >= MIN_PEER_MATCHES:
                    break
        found_matches = same_tier
    else:
        found_matches = list(all_found)

    total_match_count = len(found_matches)

    # Production recommendations (consensus comparison)
    gems_by_artist = {}
    for m in found_matches:
        isrc = m.get('isrc')
        aid = str(m.get('artist_id', ''))
        if isrc and isrc in matcher._gems_by_isrc and aid not in gems_by_artist:
            gems_by_artist[aid] = matcher._gems_by_isrc[isrc]
    recs = _generate_recommendations(features, found_matches,
                                      gems_by_artist=gems_by_artist)

    # Flattery matches — higher-tier artists from unfiltered pool (before genre filter)
    flattery_matches = []
    if user_tier and all_matches_unfiltered:
        tier_order_map = {t: i for i, t in enumerate(TIER_RANGES.keys())}
        user_tier_num = tier_order_map.get(user_tier, 0)
        flattery_candidates = []
        seen_artists = set()
        for m in all_matches_unfiltered:
            cand_tier = m.get('tier', '')
            cand_tier_num = tier_order_map.get(cand_tier, 0)
            if cand_tier_num <= user_tier_num:
                continue
            aid = m.get('artist_id')
            if aid in seen_artists:
                continue
            seen_artists.add(aid)
            total_boost = 0
            cand_genres = []
            for g in m.get('artist_genres', []):
                if g:
                    cand_genres.append(g)
            primary = (m.get('primary_genre') or '').strip()
            if primary:
                cand_genres.append(primary)
            cand_families = _genre_families(*cand_genres)
            if user_families and cand_families:
                shared = cand_families & user_families
                if not shared:
                    continue  # No genre overlap, skip
                # Exclusive families: strong genre identities that shouldn't cross-contaminate
                EXCLUSIVE_FAMILIES = {'electronic', 'metal', 'hip-hop', 'country', 'classical', 'jazz', 'latin'}
                foreign_exclusive = (cand_families & EXCLUSIVE_FAMILIES) - user_families
                if foreign_exclusive:
                    continue
                total_boost += 0.05 * len(shared)
            cand_pronoun = m.get('pronoun_title', 'They')
            flattery_candidates.append((cand_tier_num, m.get('similarity', 0) + total_boost, m, cand_pronoun))

        # Sort by tier (highest first), then sonic similarity as tiebreaker
        flattery_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        if flattery_candidates:
            print(f"  Flattery: {len(flattery_candidates)} total candidates from {len(all_matches_unfiltered)} unfiltered pool")
            print(f"  Flattery: user families = {user_families}, user tier = {user_tier} (num={user_tier_num})")
            for i, (tn, sc, m, _) in enumerate(flattery_candidates[:10]):
                print(f"    {i+1}. {m.get('name','?')[:25]:<25} tier={m.get('tier','?'):<12} sim={m.get('similarity',0):.1%} boosted={sc:.1%}")

        for _, _, m, _ in flattery_candidates[:20]:
            flattery_matches.append(m)

    new_job_id = job_mgr.create_job(token, features, found_matches)

    # Build user_profile for "Where You Stand" conversion comparison
    user_profile = None
    u_listeners = 0
    u_followers = 0
    u_conversion = None

    # Get user's artist data from registration spotify_url or CM lookup
    reg_spotify = lead.get('spotify_url', '')
    reg_base = reg_spotify.split('?')[0].rstrip('/') if reg_spotify else ''
    if reg_base:
        # Try GEMS cache first
        for aid, adata in matcher._artists.items():
            cache_url = (adata.get('spotify_url') or '').split('?')[0].rstrip('/')
            if cache_url == reg_base:
                u_listeners = float(adata.get('listeners', 0) or 0)
                u_followers = float(adata.get('followers', 0) or 0)
                u_conversion = adata.get('conversion_rate')
                print(f"  User profile (cache): {adata.get('name')}, listeners={u_listeners}, followers={u_followers}, conversion={u_conversion}")
                break
        # Fall back to CM data if cache had zeros
        if not u_listeners and track_artist_cm_data:
            u_listeners = float(track_artist_cm_data.get('listeners', 0) or 0)
            u_followers = float(track_artist_cm_data.get('followers', 0) or 0)
            u_conversion = track_artist_cm_data.get('conversion_rate')
            print(f"  User profile (CM): {track_artist_cm_data['name']}, listeners={u_listeners}, followers={u_followers}, conversion={u_conversion}")
    # Also try using track_artist_cm_data if no reg URL but we have CM data
    elif track_artist_cm_data:
        u_listeners = float(track_artist_cm_data.get('listeners', 0) or 0)
        u_followers = float(track_artist_cm_data.get('followers', 0) or 0)
        u_conversion = track_artist_cm_data.get('conversion_rate')
        print(f"  User profile (CM fallback): {track_artist_cm_data['name']}, listeners={u_listeners}, followers={u_followers}, conversion={u_conversion}")

    if u_listeners > 0 and all_found:
        # Filter out anomalous conversion rates (retired artists, bad data)
        MAX_REASONABLE_CONVERSION = 15.0
        match_conversions = [
            m['conversion_rate'] for m in all_found
            if m.get('conversion_rate') is not None
            and 0 < m['conversion_rate'] <= MAX_REASONABLE_CONVERSION
        ]
        conv_comparison = {}
        if match_conversions:
            sorted_conv = sorted(match_conversions)
            n = len(sorted_conv)
            conv_comparison = {
                'peer_median': round(sorted_conv[n // 2], 2),
                'peer_top_25': round(sorted_conv[int(n * 0.75)], 2),
                'peer_bottom_25': round(sorted_conv[n // 4], 2),
                'peer_p99': round(sorted_conv[min(n - 1, int(n * 0.99))], 2),
                'peer_count': len(sorted_conv),
            }

        additional_fans = 0
        additional_revenue = 0
        peer_top_25 = conv_comparison.get('peer_top_25', 0)
        peer_p99 = conv_comparison.get('peer_p99', 0)
        # Target p75 if below it, otherwise target p99
        target_rate = peer_top_25
        if u_conversion is not None and u_conversion >= peer_top_25 and peer_p99 > u_conversion:
            target_rate = peer_p99
        if target_rate > 0 and u_listeners > 0:
            target_followers = int(round((target_rate / 100) * u_listeners * 4.3 / 0.1))
            current_followers = u_followers if u_followers > 0 else 0
            additional_fans = max(int(round(target_followers - current_followers)), 0)
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
        print(f"  User profile built: conversion={u_conversion}, fans_gap={additional_fans}, peers={conv_comparison.get('peer_count', 0)}")

    print(f"  URL analysis: {len(found_matches)} tier-filtered matches, "
          f"{len(all_found)} total genre-filtered for enrichment, "
          f"{len(flattery_matches)} flattery, {len(recs)} recs")

    result = {
        'job_id': new_job_id,
        'total_match_count': total_match_count,
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
        'matches': found_matches,
        'all_matches': all_found[:5000],
        'total_all_matches': len(all_found),
        'user_tier': user_tier or '',
        'recommendations': recs,
        'source': {
            'type': 'spotify_url',
            'track_name': track_name,
            'artist_name': artist_name,
            'artist_genres': genre or '',
            'artist_tier': track_artist_cm_data.get('tier', '') if track_artist_cm_data else '',
            'artist_listeners': track_artist_cm_data.get('listeners', 0) if track_artist_cm_data else 0,
            'preview_used': features is not None and preview_url is not None,
        },
        'timing': {},
    }

    if flattery_matches:
        result['flattery_matches'] = flattery_matches

    if user_profile:
        result['user_profile'] = user_profile

    # Upsert analyzed track's audio features into gems_complete_analysis
    # (Artist upsert already handled by lookup_artist_by_spotify above)
    if track_isrc and features:
        enrichment_pool.submit(_upsert_gems_features, track_isrc, features, genre or '')

    # Kick off background enrichment: displayed matches FIRST, then wider pool
    # This ensures every match the user sees gets playlist data
    displayed_ids = {str(m.get('artist_id', '')) for m in found_matches}
    wider_pool = sorted(all_found, key=lambda m: m.get('similarity', 0), reverse=True)
    wider_extra = [m for m in wider_pool if str(m.get('artist_id', '')) not in displayed_ids]
    enrichment_matches = found_matches + wider_extra
    enrichment_matches = enrichment_matches[:200]  # Cap total
    # Resume enrichment gate — user-facing CM calls are done
    _resume_enrichment()
    enrichment_pool.submit(
        _run_background_enrichment,
        new_job_id, enrichment_matches, user_cm_id,
    )

    return result


# ---------------------------------------------------------------------------
# Deal Calculator: artist lookup endpoint
# ---------------------------------------------------------------------------

@app.post("/api/deal/lookup")
async def deal_lookup(
    spotify_url: str = Form(...),
):
    """
    Look up an artist by Spotify URL and return metrics, peer comparison,
    and sonic gap data for the deal calculator frontend.
    """
    if not spotify_url or 'spotify.com' not in spotify_url:
        raise HTTPException(400, "Valid Spotify URL required")

    # 1. Look up artist via existing function (upserts fire automatically)
    artist_data = lookup_artist_by_spotify(spotify_url)
    if not artist_data:
        raise HTTPException(404, "Artist not found on Spotify/Chartmetric")

    listeners = artist_data.get('listeners', 0) or 0
    followers = artist_data.get('followers', 0) or 0
    tier = artist_data.get('tier') or _listeners_to_tier(int(listeners))
    conversion_rate = artist_data.get('conversion_rate', 0) or 0

    # 2. Peer comparison: filter GEMS cache by tier
    peer_comparison = None
    if matcher.cache and matcher._tiers:
        tier_conversion_rates = []
        for artist_id, tier_data in matcher._tiers.items():
            a_listeners = int(float(tier_data.get('listeners', 0) or 0))
            a_tier = _listeners_to_tier(a_listeners)
            if a_tier == tier:
                a_followers = int(float(tier_data.get('followers', 0) or 0))
                if a_listeners > 0:
                    cr = (a_followers * 0.1) / (a_listeners * 4.3) * 100
                    if 0 < cr < 100:
                        tier_conversion_rates.append(cr)

        if tier_conversion_rates:
            tier_conversion_rates.sort()
            n = len(tier_conversion_rates)
            peer_comparison = {
                'median_conversion': tier_conversion_rates[n // 2],
                'p25_conversion': tier_conversion_rates[n // 4],
                'p75_conversion': tier_conversion_rates[3 * n // 4],
                'p95_conversion': tier_conversion_rates[min(n - 1, int(n * 0.95))],
                'p99_conversion': tier_conversion_rates[min(n - 1, int(n * 0.99))],
                'tier_count': n,
            }

    # 3. Sonic analysis: run the same pipeline as /api/analyze-url
    #    Get features (from GEMS cache or Spotify preview), match against GEMS universe,
    #    compute sonic gap + conversion opportunity from matched higher-converting artists.
    sonic_gap = None
    conversion_opportunity = None
    top_track = artist_data.get('top_track')
    features = None

    if top_track and top_track.get('isrc'):
        isrc = top_track['isrc']
        # Try cached features first
        features = _lookup_gems_features(isrc)
        if features:
            print(f"Deal lookup: using cached GEMS features for ISRC {isrc}")
        else:
            print(f"Deal lookup: no GEMS cache for ISRC {isrc}")

    # If no cached features, use Mac worker to play + record via Spotify desktop
    if not features and top_track and top_track.get('spotify_url'):
        track_url = top_track['spotify_url']
        print(f"Deal lookup: no cached features, creating job for Mac worker — {track_url}")
        # Pause local scripts before Mac worker captures audio
        _notify_local_pipeline('user_active')

        try:
            # Create a Supabase job so Mac worker picks it up
            deal_job_id = job_mgr.create_job('deal-lookup', {}, [])
            job_mgr.update_job(deal_job_id, status='pending_features', spotify_url=track_url)
            print(f"Deal lookup: created job {deal_job_id}, waiting for Mac worker (up to 90s)...")

            def _poll_deal_features():
                if not job_mgr._supabase:
                    return None
                try:
                    resp = job_mgr._supabase.table('analysis_jobs').select('status,features').eq('id', deal_job_id).execute()
                    if resp.data:
                        row = resp.data[0]
                        if row.get('status') == 'features_ready':
                            f = row.get('features', {})
                            if isinstance(f, str):
                                f = json.loads(f)
                            return f
                except Exception as e:
                    print(f"Deal lookup: poll error: {e}")
                return None

            loop = asyncio.get_event_loop()
            deadline = time.time() + 90
            while time.time() < deadline:
                result = await loop.run_in_executor(None, _poll_deal_features)
                if result:
                    features = result
                    print(f"Deal lookup: features received from Mac worker")
                    # Cache for next time
                    if top_track.get('isrc'):
                        genres_str = artist_data.get('genres', '')
                        _upsert_gems_features(top_track['isrc'], features, genre=genres_str)
                    break
                await asyncio.sleep(3)

            if not features:
                print(f"Deal lookup: Mac worker timed out after 90s")
        except Exception as e:
            print(f"Deal lookup: Mac worker job failed: {e}")

    # If we have features, run sonic matching to get conversion opportunity + gap
    print(f"Deal lookup: features={'yes' if features else 'NO'}, gems_list={len(matcher._gems_list) if matcher._gems_list else 0}")
    if features and matcher._gems_list:
        genres_str = artist_data.get('genres', '')

        # Run emotion detection
        try:
            from audio_analyzer import _emotion_detector
            emo = _emotion_detector.detect(features, genres_str)
            top_emo = emo.get('emotions', [])
            for i in range(4):
                if i < len(top_emo):
                    features[f'emotion_{i+1}'] = top_emo[i][0]
                    features[f'emotion_{i+1}_score'] = top_emo[i][1]
                else:
                    features[f'emotion_{i+1}'] = 'neutral'
                    features[f'emotion_{i+1}_score'] = 0.0
        except Exception as e:
            print(f"Deal lookup: emotion detection failed: {e}")

        # Match against GEMS universe
        try:
            all_matches = matcher.find_matches(features, genre_hint=genres_str, top_n=5000, threshold=0.55)
            print(f"Deal lookup: {len(all_matches)} sonic matches found")

            if all_matches:
                # Sonic gap: consensus vs high-converting matches
                high_converter_matches = [
                    m for m in all_matches
                    if m.get('conversion_rate') and float(m['conversion_rate']) > (conversion_rate * 1.2)
                ]
                if high_converter_matches:
                    # Get GEMS rows for high converters
                    high_converter_gems = []
                    for m in high_converter_matches[:50]:
                        isrc = m.get('isrc')
                        if isrc and isrc in matcher._gems_by_isrc:
                            high_converter_gems.append(matcher._gems_by_isrc[isrc])

                    if high_converter_gems:
                        consensus = _find_consensus(features, high_converter_gems)
                        sonic_gap = [
                            {
                                'feature': c['feature'],
                                'direction': c['direction'],
                                'description': f"Consider {'increasing' if c['direction'] == 'higher' else 'decreasing'} {c['feature'].replace('_', ' ')}"
                            }
                            for c in consensus[:5]
                        ]

                # Conversion opportunity from sonically matched artists IN THE SAME TIER
                match_conversions = [
                    m['conversion_rate'] for m in all_matches
                    if m.get('conversion_rate') is not None and m['conversion_rate'] > 0
                    and m.get('tier') == tier
                ]
                print(f"Deal lookup: {len(match_conversions)} same-tier sonic matches with conversion data (tier={tier})")
                # Always override tier_count with sonic peer count when we have sonic matches
                if match_conversions and peer_comparison:
                    peer_comparison['tier_count'] = len(match_conversions)
                if match_conversions and conversion_rate and conversion_rate > 0 and listeners > 0:
                    sorted_conv = sorted(match_conversions)
                    peer_top_25 = sorted_conv[int(len(sorted_conv) * 0.75)]

                    # Use p75 as target for below-p75, then p95, then p99
                    if peer_top_25 > conversion_rate:
                        target_cr = peer_top_25
                    elif len(sorted_conv) >= 10:
                        p95 = sorted_conv[min(len(sorted_conv) - 1, int(len(sorted_conv) * 0.95))]
                        p99 = sorted_conv[min(len(sorted_conv) - 1, int(len(sorted_conv) * 0.99))]
                        if p95 > conversion_rate:
                            target_cr = p95
                        elif p99 > conversion_rate:
                            target_cr = p99
                        else:
                            target_cr = 0
                    else:
                        target_cr = 0

                    if target_cr > conversion_rate:
                        target_followers = int(round((target_cr / 100) * listeners * 4.3 / 0.1))
                        current_followers = int(followers) if followers > 0 else 0
                        additional_fans = max(int(round(target_followers - current_followers)), 0)
                        additional_revenue = additional_fans * 25  # ~$25/fan/year (streaming + merch + tickets)

                        conversion_opportunity = {
                            'additional_fans': additional_fans,
                            'additional_revenue': additional_revenue,
                            'target_conversion': round(target_cr, 2),
                            'sonic_peer_count': len(match_conversions),
                        }
                        print(f"Deal lookup: sonic conversion opportunity — "
                              f"current {conversion_rate:.2f}% → target {target_cr:.2f}%, "
                              f"+{additional_fans} fans, +${additional_revenue}")
        except Exception as e:
            print(f"Deal lookup: sonic matching failed: {e}")
            import traceback; traceback.print_exc()

    # Fallback: tier-based conversion opportunity if sonic analysis didn't produce one
    if not conversion_opportunity and peer_comparison and conversion_rate and conversion_rate > 0 and listeners > 0:
        # Use p75 for below-p75 artists, p95 for above-p75
        if peer_comparison['p75_conversion'] > conversion_rate:
            target_cr = peer_comparison['p75_conversion']
        elif peer_comparison.get('p95_conversion', 0) > conversion_rate:
            target_cr = peer_comparison['p95_conversion']
        elif peer_comparison.get('p99_conversion', 0) > conversion_rate:
            target_cr = peer_comparison['p99_conversion']
        else:
            target_cr = 0  # above p99 with peer data — no fake numbers

        if target_cr > conversion_rate:
            current_followers_equiv = conversion_rate * listeners * 4.3 / (0.1 * 100)
            target_followers_equiv = target_cr * listeners * 4.3 / (0.1 * 100)
            additional_fans = max(int(target_followers_equiv - current_followers_equiv), 0)
            additional_revenue = additional_fans * 25  # $25/fan/year
            conversion_opportunity = {
                'additional_fans': additional_fans,
                'additional_revenue': additional_revenue,
                'target_conversion': round(target_cr, 2),
            }

    # 4. Build metrics object for risk assessment
    meta = artist_data.get('_meta', {})
    stats = meta.get('cm_statistics', {}) if isinstance(meta, dict) else {}
    metrics = {
        'career_stage': artist_data.get('career_stage'),
        'spotify_monthly_listeners': listeners,
        'instagram_followers': stats.get('ins_followers'),
        'instagram_engagement_rate': None,
        'tiktok_followers': stats.get('tiktok_followers'),
        'youtube_subscribers': stats.get('ycs_subscribers'),
        'shazam_count': stats.get('shazam_count'),
        'spotify_playlist_reach': stats.get('sp_playlist_total_reach'),
        'spotify_popularity': stats.get('sp_popularity'),
    }

    catalog_size = artist_data.get('catalog_size', 20)

    # 4b. Cross-platform revenue multiplier (ported from find-deal-ready.js)
    platform_multiplier = 1.0
    if stats:
        if int(float(stats.get('num_am_playlists') or 0)) > 0:
            platform_multiplier += 0.45   # Apple Music ~15% market
        if int(float(stats.get('num_az_playlists') or 0)) > 0:
            platform_multiplier += 0.35   # Amazon ~13% market
        if int(float(stats.get('youtube_monthly_video_views') or 0)) > 10000:
            platform_multiplier += 0.25   # YouTube Music ~10%
        if int(float(stats.get('pandora_listeners_28_day') or 0)) > 0:
            platform_multiplier += 0.15   # Pandora, mostly US
        if int(float(stats.get('deezer_fans') or 0)) > 0:
            platform_multiplier += 0.10   # Deezer ~2% market
        if int(float(stats.get('soundcloud_plays') or 0)) > 0:
            platform_multiplier += 0.08   # SoundCloud, small
        if int(float(stats.get('shazam_count') or 0)) > 500:
            platform_multiplier += 0.05   # Discoverability signal
    platform_multiplier = min(platform_multiplier, 3.0)
    print(f"Deal lookup: platform multiplier = {platform_multiplier:.2f}x for {artist_data.get('name')}")

    # 5. Fetch historical data — use Supabase snapshots (all platforms) + CM API fallback (Spotify only)
    listener_history = []
    platform_history = {}  # All platform snapshots from Supabase
    cm_id = artist_data.get('cm_id')
    token = None

    # 5a. Try Supabase artists_history first (has ALL platform data already collected)
    artist_cm_id = str(cm_id) if cm_id else None
    if artist_cm_id and supabase:
        try:
            result = supabase.table('artists_history') \
                .select('snapshot_date,sp_monthly_listeners,sp_followers,sp_popularity,sp_playlist_total_reach,'
                        'tiktok_followers,tiktok_likes,tiktok_track_posts,tiktok_top_video_views,'
                        'ycs_subscribers,ycs_views,youtube_monthly_video_views,'
                        'ins_followers,shazam_count,deezer_fans,soundcloud_followers,soundcloud_plays,'
                        'pandora_listeners_28_day,facebook_followers,cm_artist_score,'
                        'boomplay_streams,boomplay_favorites,genius_views,songkick_fans') \
                .eq('artist_id', artist_cm_id) \
                .order('snapshot_date', desc=False) \
                .execute()

            if result.data:
                snapshots = result.data
                print(f"Deal lookup: {len(snapshots)} Supabase snapshots for {artist_data.get('name')}")

                # Build per-platform history arrays from snapshots
                PLATFORM_FIELDS = {
                    'spotify_listeners': 'sp_monthly_listeners',
                    'spotify_followers': 'sp_followers',
                    'spotify_popularity': 'sp_popularity',
                    'spotify_playlist_reach': 'sp_playlist_total_reach',
                    'tiktok_followers': 'tiktok_followers',
                    'tiktok_likes': 'tiktok_likes',
                    'tiktok_track_posts': 'tiktok_track_posts',
                    'tiktok_top_video_views': 'tiktok_top_video_views',
                    'youtube_subscribers': 'ycs_subscribers',
                    'youtube_views': 'ycs_views',
                    'youtube_monthly_views': 'youtube_monthly_video_views',
                    'instagram_followers': 'ins_followers',
                    'shazam_count': 'shazam_count',
                    'deezer_fans': 'deezer_fans',
                    'soundcloud_followers': 'soundcloud_followers',
                    'soundcloud_plays': 'soundcloud_plays',
                    'pandora_listeners': 'pandora_listeners_28_day',
                    'facebook_followers': 'facebook_followers',
                    'cm_artist_score': 'cm_artist_score',
                    'boomplay_streams': 'boomplay_streams',
                    'genius_views': 'genius_views',
                    'songkick_fans': 'songkick_fans',
                }

                for output_key, db_field in PLATFORM_FIELDS.items():
                    history = []
                    for snap in snapshots:
                        val = snap.get(db_field)
                        date = snap.get('snapshot_date', '')[:10]
                        if val and date and float(val) > 0:
                            history.append({'date': date, 'value': int(float(val))})
                    if history:
                        platform_history[output_key] = history

                # Also build listener_history for backward compatibility
                if 'spotify_listeners' in platform_history:
                    listener_history = [
                        {'date': p['date'], 'listeners': p['value']}
                        for p in platform_history['spotify_listeners']
                    ]

                print(f"Deal lookup: platform history keys = {list(platform_history.keys())}")
        except Exception as e:
            print(f"Deal lookup: Supabase history fetch failed: {e}")

    # 5b. Fallback to Chartmetric API for Spotify listeners if Supabase had nothing
    if not listener_history and cm_id:
        try:
            refresh_token = os.getenv('REFRESH_TOKEN')
            if refresh_token:
                token = get_cm_token(refresh_token)
                if token:
                    listener_history = fetch_listener_history(token, cm_id)
                    print(f"Deal lookup: {len(listener_history)} CM API data points for {artist_data.get('name')}")
                    # Also add to platform_history
                    if listener_history:
                        platform_history['spotify_listeners'] = [
                            {'date': p['date'], 'value': p['listeners']}
                            for p in listener_history
                        ]
        except Exception as e:
            print(f"Deal lookup: CM listener history fetch failed: {e}")

    # Fetch past + future events from Chartmetric to estimate annual touring activity
    upcoming_events = None
    # Ensure we have a CM token for events (may not be set if Supabase had listener data)
    if cm_id and not token:
        try:
            refresh_token = os.getenv('REFRESH_TOKEN')
            if refresh_token:
                token = get_cm_token(refresh_token)
        except Exception:
            pass
    if cm_id and token:
        try:
            print(f"Deal lookup: fetching events for cm_id={cm_id} ({artist_data.get('name')})")
            raw_events = fetch_artist_events(token, cm_id, lookback_days=730, lookahead_days=365)
            print(f"Deal lookup: got {len(raw_events)} raw events for cm_id={cm_id}")
            if raw_events:
                from datetime import timedelta
                now = datetime.now()
                today_iso = now.strftime('%Y-%m-%d')
                d30 = (now + timedelta(days=30)).strftime('%Y-%m-%d')
                d90 = (now + timedelta(days=90)).strftime('%Y-%m-%d')

                # Parse dates and compute time span for annualization
                dates = [e.get('start_date', '')[:10] for e in raw_events if e.get('start_date')]
                dates = [d for d in dates if d]
                future_events = [e for e in raw_events if (e.get('start_date') or '') >= today_iso]

                # Annualize: shows_per_year = total events / span in years
                if dates:
                    earliest = min(dates)
                    latest = max(dates)
                    span_days = (datetime.strptime(latest, '%Y-%m-%d') - datetime.strptime(earliest, '%Y-%m-%d')).days
                    span_years = max(span_days / 365.25, 0.25)  # floor at 3 months
                    shows_per_year = round(len(raw_events) / span_years)
                else:
                    shows_per_year = len(raw_events)

                capacities = [e['venue_capacity'] for e in raw_events if e.get('venue_capacity')]
                # Use low_price (face value) and cap at $500 to avoid VIP/resale skew
                TICKET_PRICE_CAP = 500
                prices = []
                for e in raw_events:
                    low = e.get('low_price')
                    high = e.get('high_price')
                    if low and high:
                        prices.append(min(low, TICKET_PRICE_CAP))  # use face value, not avg with VIP
                    elif low:
                        prices.append(min(low, TICKET_PRICE_CAP))
                    elif high:
                        prices.append(min(high, TICKET_PRICE_CAP))

                # Use median to resist outliers
                median_price = 0
                if prices:
                    sorted_prices = sorted(prices)
                    mid = len(sorted_prices) // 2
                    if len(sorted_prices) % 2 == 0:
                        median_price = round((sorted_prices[mid - 1] + sorted_prices[mid]) / 2, 2)
                    else:
                        median_price = round(sorted_prices[mid], 2)

                upcoming_events = {
                    'total_shows': shows_per_year,  # annualized
                    'raw_event_count': len(raw_events),
                    'future_event_count': len(future_events),
                    'next_30_days': len([e for e in future_events if (e.get('start_date') or '')[:10] <= d30]),
                    'next_90_days': len([e for e in future_events if (e.get('start_date') or '')[:10] <= d90]),
                    'headliner_pct': round(len([e for e in raw_events if e.get('is_headliner')]) / len(raw_events) * 100),
                    'avg_venue_capacity': round(sum(capacities) / len(capacities)) if capacities else 0,
                    'median_venue_capacity': sorted(capacities)[len(capacities) // 2] if capacities else 0,
                    'avg_ticket_price': median_price,  # median of face-value-capped prices
                    'countries': list(set(e.get('code2', '') for e in raw_events if e.get('code2'))),
                }
                print(f"Deal lookup: {len(raw_events)} events ({shows_per_year}/yr annualized) for {artist_data.get('name')}")
        except Exception as e:
            print(f"Deal lookup: event fetch failed: {e}")

    # Send push notification
    send_pushover_notification(
        "Deal Calculator Lookup",
        f"{artist_data.get('name', 'Unknown')} | {tier} | {int(listeners):,} listeners"
    )

    # Extract image URL from Chartmetric metadata
    image_url = None
    _meta = artist_data.get('_meta') or {}
    image_url = _meta.get('image_url') or _meta.get('cover_url') or None

    return {
        'name': artist_data.get('name', ''),
        'genres': artist_data.get('genres', ''),
        'career_stage': artist_data.get('career_stage', ''),
        'listeners': listeners,
        'followers': followers,
        'tier': tier,
        'conversion_rate': conversion_rate,
        'catalog_size': catalog_size,
        'top_track': top_track,
        'peer_comparison': peer_comparison,
        'sonic_gap': sonic_gap,
        'conversion_opportunity': conversion_opportunity,
        'metrics': metrics,
        'listener_history': listener_history,
        'platform_history': platform_history,
        'platform_multiplier': round(platform_multiplier, 2),
        'upcoming_events': upcoming_events,
        'image_url': image_url,
    }


# ---------------------------------------------------------------------------
# Deal Calculator: lead capture
# ---------------------------------------------------------------------------

@app.post("/api/deal/lead")
async def deal_lead_capture(data: dict):
    """
    Save contact info + optional artist data for lead tracking.
    Fires a Pushover notification.
    """
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    step = data.get('step', 'unknown')

    if not email:
        raise HTTPException(400, "Email required")

    # Save to Supabase
    if supabase:
        try:
            row = {
                'name': name,
                'email': email,
                'step': step,
                'artist_name': data.get('artist_name'),
                'created_at': datetime.utcnow().isoformat(),
            }
            supabase.table('deal_leads').insert(row).execute()
        except Exception as e:
            print(f"Deal lead save error: {e}")

    # Push notification
    send_pushover_notification(
        "Deal Calculator Lead",
        f"{name}\n{email}\nStep: {step}"
    )

    return {"ok": True}


# ---------------------------------------------------------------------------
# Deal Calculator: Stripe checkout
# ---------------------------------------------------------------------------

import stripe as stripe_lib

stripe_lib.api_key = os.getenv('STRIPE_SECRET_KEY', '')

FRONTEND_URL = os.getenv('FRONTEND_URL', 'https://freshlybakedstudios.com')


@app.post("/api/deal/checkout")
async def deal_checkout(data: dict):
    """
    Create a Stripe Checkout Session for a deal deposit or full payment.
    Returns the hosted checkout URL.
    """
    if not stripe_lib.api_key:
        raise HTTPException(500, "Payment system not configured")

    email = data.get('email', '')
    name = data.get('name', '')
    artist_name = data.get('artist_name') or 'Unknown Artist'
    services = data.get('services', [])
    track_count = data.get('track_count', 1)
    total = data.get('total', 0)
    deposit_amount = data.get('deposit_amount', 0)
    split_percent = data.get('split_percent', 0)
    contract_text = data.get('contract_text', '')
    contract_agreed = data.get('contract_agreed', False)

    if deposit_amount <= 0:
        raise HTTPException(400, "Invalid deposit amount")

    if not contract_agreed:
        raise HTTPException(400, "Contract agreement is required")

    # Build line item description
    service_names = ', '.join(s.get('label', '') for s in services)
    description = f"{service_names} — {track_count} track{'s' if track_count > 1 else ''}"
    if split_percent > 0:
        description += f" | {split_percent}% backend"

    is_deposit = deposit_amount < total

    try:
        session = stripe_lib.checkout.Session.create(
            mode='payment',
            customer_email=email,
            payment_method_types=['card', 'affirm', 'klarna'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': f"{'50% Deposit' if is_deposit else 'Full Payment'} — Freshly Baked Studios",
                        'description': description,
                    },
                    'unit_amount': deposit_amount * 100,  # Stripe uses cents
                },
                'quantity': 1,
            }],
            metadata={
                'artist_name': artist_name,
                'customer_name': name,
                'services': service_names,
                'track_count': str(track_count),
                'total': str(total),
                'deposit': str(deposit_amount),
                'split_percent': str(split_percent),
                'contract_agreed': 'true',
            },
            success_url=f"{FRONTEND_URL}/rates?step=confirmed&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/rates?step=book",
        )

        # Save to Supabase (including contract text and agreement timestamp)
        if supabase:
            try:
                supabase.table('deal_leads').insert({
                    'name': name,
                    'email': email,
                    'step': 'checkout_started',
                    'artist_name': artist_name,
                    'metadata': {
                        'session_id': session.id,
                        'deposit_amount': deposit_amount,
                        'total': total,
                        'services': service_names,
                        'contract_agreed': True,
                        'contract_agreed_at': datetime.utcnow().isoformat(),
                        'contract_text': contract_text,
                    },
                    'created_at': datetime.utcnow().isoformat(),
                }).execute()
            except Exception as e:
                print(f"Deal checkout save error: {e}")

        # Push notification
        send_pushover_notification(
            "Deal Checkout Started!",
            f"{name} ({artist_name})\n${deposit_amount:,} of ${total:,}\n{service_names}"
        )

        return {"url": session.url, "session_id": session.id}

    except stripe_lib.error.StripeError as e:
        raise HTTPException(400, str(e))


@app.get("/api/deal/checkout/status")
async def deal_checkout_status(session_id: str):
    """
    Verify a Stripe Checkout Session after payment redirect.
    """
    if not stripe_lib.api_key:
        raise HTTPException(500, "Payment system not configured")

    try:
        session = stripe_lib.checkout.Session.retrieve(session_id)

        # On successful payment: notify + email contract
        print(f"Checkout status: session.status={session.status} payment_status={session.payment_status}")
        if session.status == 'complete':
            customer_email = session.customer_details.email if session.customer_details else ''
            customer_name = session.metadata.get('customer_name', '')
            print(f"Payment complete: email={customer_email} name={customer_name}")

            try:
                send_pushover_notification(
                    "PAYMENT RECEIVED!",
                    f"{session.customer_details.name or customer_email}\n"
                    f"${session.amount_total / 100:,.0f}\n"
                    f"Session: {session_id[:8]}..."
                )
            except Exception as e:
                print(f"Deal payment notification error: {e}")

            # Email the contract to the artist
            if customer_email and supabase:
                try:
                    # Fetch the contract text from the lead record
                    result = supabase.table('deal_leads').select('metadata').eq(
                        'email', customer_email
                    ).eq('step', 'checkout_started').order(
                        'created_at', desc=True
                    ).limit(1).execute()

                    print(f"Contract lookup: found={bool(result.data)} rows={len(result.data) if result.data else 0}")
                    if result.data:
                        has_contract = bool(result.data[0].get('metadata', {}).get('contract_text'))
                        print(f"Contract text present: {has_contract}")

                    if result.data and result.data[0].get('metadata', {}).get('contract_text'):
                        contract_text = result.data[0]['metadata']['contract_text']
                        sent = _send_contract_email(customer_name or customer_email, customer_email, contract_text)
                        print(f"Contract email sent: {sent}")
                    else:
                        print(f"No contract text found for {customer_email}")
                except Exception as e:
                    print(f"Contract email error: {e}")
            else:
                print(f"Skipping contract email: email={customer_email} supabase={bool(supabase)}")

        return {
            "status": session.status,
            "customer_email": session.customer_details.email if session.customer_details else '',
            "amount_total": session.amount_total,
        }

    except stripe_lib.error.StripeError as e:
        raise HTTPException(400, str(e))


def _send_contract_email(name: str, email: str, contract_text: str) -> bool:
    """Send the signed contract to the artist via SendGrid."""
    api_key = os.getenv('SENDGRID_API_KEY')
    if not api_key:
        print("SENDGRID_API_KEY not set — skipping contract email")
        return False

    # Convert contract text to HTML (preserve formatting)
    contract_html = contract_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')

    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                max-width:700px;margin:0 auto;background:#141213;color:#eee;padding:32px;border-radius:12px">

      <div style="text-align:center;margin-bottom:24px">
        <img src="https://storage.googleapis.com/fbs-static-assets/axd-logo.png" alt="Freshly Baked Studios" style="width:180px;height:auto;margin-bottom:16px">
        <h1 style="color:#fff;margin:0;font-size:24px">Your Production Agreement</h1>
        <p style="color:#888;margin:4px 0 0">Freshly Baked Studios</p>
      </div>

      <p style="color:#ccc">Hey {name},</p>
      <p style="color:#ccc">Thank you for your payment! Below is a copy of the Music Production Agreement you agreed to. Please save this for your records.</p>
      <p style="color:#ccc;font-size:13px">A formal version will be sent for signature once we collect your legal details during onboarding.</p>

      <div style="background:#1a1818;border:1px solid #3a3636;border-radius:8px;padding:24px;margin:24px 0;font-size:12px;line-height:1.6;color:#ccc;font-family:monospace">
        {contract_html}
      </div>

      <p style="color:#666;font-size:11px;text-align:center;margin-top:24px">
        Freshly Baked Studios &bull; freshlybakedstudios.com
      </p>
    </div>
    """

    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, HtmlContent

    message = Mail(
        from_email='deals@freshlybakedstudios.com',
        to_emails=email,
        subject=f'{name}, your production agreement — Freshly Baked Studios',
        html_content=HtmlContent(html),
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print(f"Contract email sent to {email} — status {response.status_code}")
        return response.status_code in (200, 201, 202)
    except Exception as e:
        print(f"Contract email send failed: {e}")
        return False


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
