"""
FastAPI server for the Sonic Analyzer web app.
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

import bcrypt
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
    invalidate_cm_token,
    fetch_listener_history,
    fetch_artist_events,
    _resolve_isrc_to_cm_track_id,
    _fetch_track_playlists_structured,
    _fetch_related_artists,
    _extract_track_credits,
    _fetch_curator_contact,
    _upsert_gems_features,
    _lookup_gems_features,
    fetch_track_momentum,
    lookup_track_genre,
    _resolve_genre_ids,
)
from email_sender import send_results_email
from job_manager import JobManager
from track_matcher import (TrackMatcher, _genre_families, match_in_lane,
                           candidate_lane_families)

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
# Telegram chat-widget bridge — values set as Railway env vars (never in code/git)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_OWNER_CHAT_ID = os.getenv('TELEGRAM_OWNER_CHAT_ID', '')
TELEGRAM_WEBHOOK_SECRET = os.getenv('TELEGRAM_WEBHOOK_SECRET', '')
CHAT_BACKEND_URL = os.getenv('BACKEND_PUBLIC_URL', 'https://analyze.freshlybakedstudios.com')
access_tokens: dict = {}  # DEPRECATED — kept for backwards compat during migration
NDA_VERSION = "2026-03-v1"
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

    # Self-register the Telegram webhook so the chat widget works without any
    # manual setWebhook step. No-op if the bot token/secret aren't configured.
    if TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET:
        try:
            hook_url = (
                f"{CHAT_BACKEND_URL}/api/chat/telegram-webhook"
                f"?secret={TELEGRAM_WEBHOOK_SECRET}"
            )
            requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
                params={'url': hook_url, 'allowed_updates': '["message"]'},
                timeout=10,
            )
            print("✅ Telegram chat webhook registered")
        except Exception as e:
            print(f"⚠️  Telegram webhook registration failed: {e}")

    # Start resource-switching activity monitor
    activity_thread = threading.Thread(target=_check_idle_and_notify, daemon=True)
    activity_thread.start()
    print("✅ Resource-switching activity monitor started")

    print("Ready.")
    yield
    print("Shutting down.")
    enrichment_pool.shutdown(wait=False)


app = FastAPI(title="Sonic Analyzer", lifespan=lifespan)

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
    USER_PATHS = ('/api/analyze', '/api/deal/', '/api/register', '/api/signup', '/api/login', '/api/analysis/')

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
    """Check if a web user is currently active (for local pipeline manager).

    Reports active if there's recent API activity OR if any enrichment jobs
    are still running in Supabase (survives deploys/restarts).
    """
    now = time.time()
    idle_seconds = now - _last_api_activity if _last_api_activity > 0 else -1
    is_active = _last_api_activity > 0 and idle_seconds < _IDLE_TIMEOUT

    # Check if anyone is connected via SSE (active tab watching results)
    has_sse = bool(sse_subscribers)
    if has_sse:
        is_active = True

    # Also check Supabase for in-progress enrichment jobs — this survives deploys
    enrichment_active = False
    if not is_active and supabase:
        try:
            from datetime import timezone, timedelta
            stale_cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
            resp = supabase.table('analysis_jobs').select('id').in_(
                'status', ['enriching', 'matching', 'pending_features', 'capturing']
            ).gte('updated_at', stale_cutoff).limit(1).execute()
            if resp.data:
                enrichment_active = True
                is_active = True
        except Exception:
            pass

    return {
        "user_active": is_active,
        "enrichment_active": enrichment_active,
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
    'harmonic_distortion', 'stereo_width', 'mid_side_ratio',
    'stereo_correlation', 'true_peak_dbfs',
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
        # Peak-to-RMS DR meter — NOT the EBU R128 LRA below (which uses
        # "loudness range" / "loudness variation" wording). Keeping the
        # two distinct in the UI so a rec card never reads "narrow the
        # loudness range" + "tighten the loudness variation" as if they
        # were two takes on the same measurement.
        'higher': 'Open up peak dynamics — widening the dynamic range for more movement',
        'lower': 'Tighten peak dynamics — narrowing the dynamic range for a denser, more consistent sound',
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
        'domain': 'Dynamics', 'unit': 'dB',
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
        'lower': 'Ease rhythmic rigidity — less quantized, more human groove',
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
        'higher': 'Widen loudness variation (EBU R128) — more dynamic contrast between sections',
        'lower': 'Tighten loudness variation (EBU R128) — more consistent level across sections',
        'domain': 'Dynamics', 'unit': 'LU',
    },
    'harmonic_distortion': {
        'higher': 'Add harmonic saturation — introducing warmth and character through subtle distortion',
        'lower': 'Reduce harmonic distortion — cleaning up the signal for more transparency',
        'domain': 'Tonal Character', 'unit': '',
    },
    'stereo_width': {
        'higher': 'Widen the stereo image — creating more spatial separation and immersion',
        'lower': 'Narrow the stereo image — focusing the mix toward the center for punch and mono compatibility',
        'domain': 'Stereo / Spatial', 'unit': '',
    },
    'mid_side_ratio': {
        'higher': 'Increase center focus — more energy in the mid channel for a solid core',
        'lower': 'Increase side presence — more energy in the side channel for width and ambience',
        'domain': 'Stereo / Spatial', 'unit': '',
    },
    'stereo_correlation': {
        'higher': 'Improve mono compatibility — tighter phase correlation between left and right',
        'lower': 'Widen phase relationship — more decorrelation for a wider perceived image',
        'domain': 'Stereo / Spatial', 'unit': '',
    },
    'true_peak_dbfs': {
        'higher': 'Raise true peak ceiling — allowing more headroom usage',
        'lower': 'Lower true peak — more headroom margin for codec transparency',
        'domain': 'Dynamics', 'unit': 'dBFS',
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
    'harmonic_distortion', 'stereo_width', 'mid_side_ratio',
    'stereo_correlation', 'true_peak_dbfs',
]


def _find_consensus(features: dict, high_converter_gems: list) -> list:
    """
    Find consensus patterns: for each production feature, check whether
    most high converters are higher or lower than the user. If 50%+ agree
    on direction, that's a consensus recommendation.

    Same method as the production outreach pipeline.
    """
    if not high_converter_gems:
        return []

    target_bpm = float(features.get('bpm', 120))
    time_based = {'attack_time', 'onset_rate', 'beat_strength', 'danceability'}
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

    # Filter out recommendations where displayed values would be identical
    filtered = []
    for c in consensus_results:
        feat = c['feature']
        uv = c['user_val']
        cv = c['converter_avg']
        if feat in RATIO_FEATURES:
            if f"{uv*100:.1f}" == f"{cv*100:.1f}":
                continue
        elif feat in DB_FEATURES:
            if f"{uv:.1f}" == f"{cv:.1f}":
                continue
        else:
            if f"{uv:.3f}" == f"{cv:.3f}":
                continue
        filtered.append(c)
    consensus_results = filtered

    return consensus_results


# Features that are energy ratios (show % and dB delta)
RATIO_FEATURES = {
    'sub_ratio', 'bass_ratio', 'low_mid_ratio', 'mid_ratio',
    'high_mid_ratio', 'presence_ratio', 'air_ratio',
}
# Features already in dB
DB_FEATURES = {'lufs_integrated', 'dynamic_range', 'loudness_range', 'crest_factor', 'true_peak_dbfs'}


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

    # Skip crest factor recs when target is unrealistically low (<6 dB)
    # GEMS 4-second samples bias toward compressed moments, making targets too low
    if feat == 'crest_factor' and peer_val < 6.0:
        return None
    # Skip dynamic_range recs when target is below p5 of GEMS universe (<15 dB)
    # 4-second samples understate true dynamic range — don't recommend unrealistic targets
    if feat == 'dynamic_range' and direction == 'lower' and peer_val < 15.0:
        return None
    count = consensus['count']
    total = consensus['total']

    # Format values + compute dB delta where applicable
    delta_str = ''

    if feat in RATIO_FEATURES:
        user_str = f"{user_val * 100:.1f}%"
        peer_str = f"{peer_val * 100:.1f}%"
        if user_val > 0 and peer_val > 0:
            db_delta = 10 * math.log10(peer_val / user_val)
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
                               user_profile: dict = None) -> tuple:
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
            # Composite track-level cohort score:
            #   0.45 sp_track_popularity (Spotify popularity, recency-weighted, save-rate adjacent)
            #   0.25 cm_track_score      (Chartmetric composite track momentum)
            #   0.20 total_playlists     (editorial + user, curator+fan distribution)
            #   0.10 conversion_rate     (artist-level listener->follower stickiness)
            # When cm_track_score is null on a row, its 0.25 redistributes to pop and playlist:
            #   0.60 pop + 0.25 playlist + 0.15 conversion
            import bisect

            def _pct_rank(value, sorted_pool):
                """Percentile rank in [0,1]. None or empty pool -> 0.5 (neutral)."""
                if value is None or not sorted_pool:
                    return 0.5
                return bisect.bisect_left(sorted_pool, value) / len(sorted_pool)

            def _safe_int(v):
                try:
                    return int(v) if v is not None else 0
                except (TypeError, ValueError):
                    return 0

            # Build sorted pools once for percentile lookups across this match set
            pop_pool = sorted([m['sp_track_popularity'] for m in with_conversion
                               if m.get('sp_track_popularity') is not None])
            cm_pool = sorted([m['cm_track_score'] for m in with_conversion
                              if m.get('cm_track_score') is not None])
            plist_pool = sorted([_safe_int(m.get('editorial_playlists')) + _safe_int(m.get('user_playlists'))
                                 for m in with_conversion])
            conv_pool = sorted([m['conversion_rate'] for m in with_conversion
                                if m.get('conversion_rate') is not None])

            def _composite_score(m):
                pop_pct = _pct_rank(m.get('sp_track_popularity'), pop_pool)
                cm_pct = (_pct_rank(m.get('cm_track_score'), cm_pool)
                          if m.get('cm_track_score') is not None else None)
                plist_total = _safe_int(m.get('editorial_playlists')) + _safe_int(m.get('user_playlists'))
                plist_pct = _pct_rank(plist_total, plist_pool)
                conv_pct = _pct_rank(m.get('conversion_rate'), conv_pool)
                if cm_pct is not None:
                    return 0.45 * pop_pct + 0.25 * cm_pct + 0.20 * plist_pct + 0.10 * conv_pct
                return 0.60 * pop_pct + 0.25 * plist_pct + 0.15 * conv_pct

            # Score every match, preserve the same percentile-based fallback chain
            scored = [(_composite_score(m), m) for m in with_conversion]
            scores_sorted = sorted(s for s, _ in scored)
            t25 = scores_sorted[int(len(scores_sorted) * 0.75)]
            top_25 = [m for s, m in scored if s >= t25]
            if len(top_25) >= MIN_CONSENSUS_POOL:
                high_converter_ids = [str(m['artist_id']) for m in top_25]
            else:
                # Expand to top 50%
                t50 = scores_sorted[int(len(scores_sorted) * 0.50)]
                top_50 = [m for s, m in scored if s >= t50]
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

    # Return both recs and the cohort so the caller can run originality math
    # against the same cohort that produced these recommendations.
    return recs, locals().get('high_converter_gems', [])


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


# --- Trajectory-target market weighting -------------------------------------
# Non-Anglophone-market pop/regional tags. These routinely score high *sonic*
# matches against bright Western pop (shared production aesthetic), but a US/UK
# pitch deck shouldn't open with four J-pop superstars stacked on top. We do NOT
# drop or band them — they're real matches and a strong one can still rank high.
# We just apply a SLIGHT similarity penalty so same-market peers interleave and
# break up the run. Tuning is two dials: edit the token tuple (add
# 'latin'/'spanish' to nudge those too, drop a token to let that market rank
# natively) and NON_NATIVE_TRAJECTORY_PENALTY (bigger = pushes them down harder;
# 0 = off). Auto-disabled when the user is themselves a non-native artist.
NON_NATIVE_TRAJECTORY_PENALTY = 0.03  # ~ erases the typical foreign sonic edge so they interleave
NON_NATIVE_TRAJECTORY_TOKENS = (
    # East / SE Asian markets
    'j-pop', 'j-rock', 'j-rap', 'japanese', 'anime', 'city pop',
    'k-pop', 'k-rap', 'k-rock', 'korean',
    'c-pop', 'cantopop', 'mandopop', 'chinese', 'taiwanese', 'hong kong',
    'thai', 'vietnamese', 'indonesian', 'malaysian', 'opm', 'pinoy',
    # Non-Anglophone European / other markets
    'italian', 'french', 'german', 'greek', 'turkish', 'russian',
    'hebrew', 'israeli', 'arabic', 'hindi', 'bollywood', 'persian',
)


def _is_non_native_market(*genre_strings) -> bool:
    blob = ' '.join(g for g in genre_strings if g).lower()
    return any(tok in blob for tok in NON_NATIVE_TRAJECTORY_TOKENS)


def _cand_non_native(m: dict) -> bool:
    parts = [m.get('primary_genre') or '', m.get('secondary_genre') or '']
    parts += [g for g in (m.get('artist_genres') or []) if g]
    return _is_non_native_market(*parts)


# ---------------------------------------------------------------------------
# Track-level momentum panel — empirical peer comparison for the scanned track.
# Compares the scanned track's track-level signals (popularity / cm_score /
# playlists) against the same metrics in the sonic peer pool. Also computes a
# revenue gap projection by finding the median artist monthly_listeners among
# peer tracks at composite-momentum p75 and applying the documented
# $0.13/listener/year rate (Loud & Clear 2025). Empirical lookup, not a model.
# ---------------------------------------------------------------------------
REVENUE_PER_LISTENER_PER_YEAR = 0.13  # Loud & Clear 2025 mid-tier per-listener annualized
# All-in revenue per *converted fan* per year (streaming + merch + tickets).
# $5 is the top of the defensible range; the old $25 couldn't survive
# manager/lawyer scrutiny. Single source of truth — used by both the deal
# calculator lookup and the main sonic analyzer so the two never diverge.
REVENUE_PER_FAN_PER_YEAR = 5

def _safe_int(v):
    try: return int(v) if v is not None else 0
    except (TypeError, ValueError): return 0


def _pct_rank(value, sorted_pool):
    """Percentile rank in [0,1]. None value or empty pool returns None."""
    if value is None or not sorted_pool:
        return None
    import bisect
    return round(bisect.bisect_left(sorted_pool, value) / len(sorted_pool), 3)


def _build_track_momentum(scanned_track: dict, peer_matches: list, user_listeners: float) -> dict | None:
    """Build the track_momentum block for the user_profile response.

    Returns None if there's not enough data to compute anything useful.
    """
    if not scanned_track or not peer_matches:
        return None

    # Guard: brand-new releases / unindexed tracks have all-empty momentum
    # signals (Spotify popularity 0 or null, no CM track score, zero playlist
    # placements). Rendering the panel with zeros across the board gives the
    # artist a misleading "bottom 25% / Stuck in the Pack" diagnosis when the
    # actual cause is "the data isn't in our universe yet". Hide cleanly so
    # the artist-level Where You Stand, Sonic Originality, and Pitch
    # Comparables panels still render unaffected.
    _scan_pop = scanned_track.get('sp_track_popularity')
    _scan_cm = scanned_track.get('cm_track_score')
    _scan_pl = _safe_int(scanned_track.get('editorial_playlists')) + _safe_int(scanned_track.get('user_playlists'))
    if (_scan_pop is None or _scan_pop == 0) and _scan_cm is None and _scan_pl == 0:
        return None

    # Pools for each momentum dimension
    pop_pool = sorted([p['sp_track_popularity'] for p in peer_matches
                       if p.get('sp_track_popularity') is not None])
    cm_pool = sorted([p['cm_track_score'] for p in peer_matches
                      if p.get('cm_track_score') is not None])
    pl_pool = sorted([_safe_int(p.get('editorial_playlists')) + _safe_int(p.get('user_playlists'))
                      for p in peer_matches])

    def _stats(pool):
        if not pool: return None
        n = len(pool)
        return {
            'median': pool[n // 2],
            'p75': pool[int(n * 0.75)],
            'p99': pool[min(n - 1, int(n * 0.99))],
            'count': n,
        }

    # Scanned track's values
    scanned_pop = scanned_track.get('sp_track_popularity')
    scanned_cm = scanned_track.get('cm_track_score')
    scanned_pl = _safe_int(scanned_track.get('editorial_playlists')) + _safe_int(scanned_track.get('user_playlists'))

    pop_pct = _pct_rank(scanned_pop, pop_pool)
    cm_pct = _pct_rank(scanned_cm, cm_pool)
    pl_pct = _pct_rank(scanned_pl, pl_pool)

    # Composite percentile — same weighting as the cohort filter, renormalized
    # without the artist-level conversion signal (this panel is track-level only).
    # 0.50 popularity + 0.30 cm_score + 0.20 playlists. cm_score null: redistribute.
    def _composite_for(pop_value, cm_value, pl_value):
        p = _pct_rank(pop_value, pop_pool)
        c = _pct_rank(cm_value, cm_pool) if cm_value is not None else None
        pl = _pct_rank(pl_value, pl_pool)
        if p is None: p = 0.5
        if pl is None: pl = 0.5
        if c is not None:
            return 0.50 * p + 0.30 * c + 0.20 * pl
        return 0.714 * p + 0.286 * pl  # 0.50 + 0.30 → 0.80, renormalized

    composite_pct = round(_composite_for(scanned_pop, scanned_cm, scanned_pl), 3)

    # --- Revenue gap projection ---
    # Find peer tracks at composite p75+, get median artist listeners, apply rate.
    peer_with_listeners = [
        (
            _composite_for(p.get('sp_track_popularity'), p.get('cm_track_score'),
                           _safe_int(p.get('editorial_playlists')) + _safe_int(p.get('user_playlists'))),
            float(p.get('listeners') or 0),
        )
        for p in peer_matches
        if p.get('listeners') and float(p.get('listeners') or 0) > 0
    ]
    if peer_with_listeners:
        peer_with_listeners.sort(key=lambda x: x[0], reverse=True)
        top25_cut = max(int(len(peer_with_listeners) * 0.25), 10)
        top25_listeners = sorted([l for _, l in peer_with_listeners[:top25_cut]])
        target_listeners = top25_listeners[len(top25_listeners) // 2]
    else:
        target_listeners = None

    current_revenue = round(user_listeners * REVENUE_PER_LISTENER_PER_YEAR) if user_listeners > 0 else 0
    if target_listeners and target_listeners > user_listeners:
        target_revenue = round(target_listeners * REVENUE_PER_LISTENER_PER_YEAR)
        gap_revenue = target_revenue - current_revenue
    else:
        target_revenue = None
        gap_revenue = 0

    return {
        # Scanned track absolute values
        'scanned_popularity': scanned_pop,
        'scanned_cm_score': scanned_cm,
        'scanned_playlists': scanned_pl,
        # Peer pool stats per dimension
        'pop_stats': _stats(pop_pool),
        'cm_stats': _stats(cm_pool),
        'playlists_stats': _stats(pl_pool),
        # Percentiles
        'percentile_popularity': pop_pct,
        'percentile_cm_score': cm_pct,
        'percentile_playlists': pl_pct,
        'composite_percentile': composite_pct,
        'peer_count': len(peer_matches),
        # Revenue gap
        'gap_target_listeners': int(round(target_listeners)) if target_listeners else None,
        'gap_current_revenue': current_revenue,
        'gap_target_revenue': target_revenue,
        'gap_additional_revenue': gap_revenue,
        'revenue_per_listener': REVENUE_PER_LISTENER_PER_YEAR,
    }


# ---------------------------------------------------------------------------
# Sonic originality — distance from cohort centroid in z-normalized audio
# feature space. Uses the SAME high_converter_gems cohort that drives the
# production recommendations (top 25% composite within sonic peers), so the
# orig × perf quadrant story has a coherent reference: distance from "what's
# currently winning in your sound space".
#
# Same math as the matcher's similarity scoring, inverted. Same feature
# weights so the score emphasizes the dimensions the matcher cares about
# (frequency_spectrum 38%, dynamics ~15%, rhythm/transients ~12%).
# ---------------------------------------------------------------------------

# Weights mirror track_matcher.py:425-470 (audio-only, normalized).
# Frequency spectrum is 0.38 total spread across 7 sub-bands; rest are
# direct from the matcher. Emotion (0.07) and genre (0.08) excluded —
# originality is audio-distance only.
ORIGINALITY_WEIGHTS = {
    # Frequency spectrum (7 bands, 0.38 / 7 ≈ 0.054 each)
    'sub_ratio': 0.054, 'bass_ratio': 0.054, 'low_mid_ratio': 0.054,
    'mid_ratio': 0.054, 'high_mid_ratio': 0.054, 'presence_ratio': 0.054,
    'air_ratio': 0.054,
    # Brightness / spectral shape
    'brightness': 0.025, 'spectral_rolloff': 0.025, 'brightness_variance': 0.015,
    # Dynamics
    'energy': 0.060, 'dynamic_range': 0.030, 'loudness_range': 0.020,
    'lufs_integrated': 0.050, 'compression_amount': 0.010, 'crest_factor': 0.010,
    'true_peak_dbfs': 0.015,
    # Rhythm / transients
    'beat_strength': 0.050, 'onset_rate': 0.030, 'attack_time': 0.020,
    'danceability': 0.020,
    # Tonal character
    'spectral_complexity': 0.020, 'dissonance': 0.050, 'key_strength': 0.030,
    'zcr': 0.020, 'spectral_flux': 0.015, 'harmonic_distortion': 0.015,
    # Stereo imaging
    'stereo_width': 0.020, 'mid_side_ratio': 0.015, 'stereo_correlation': 0.015,
}
# Sanity: weights sum to ~1.0 across all 30 PRODUCTION_FEATURES dimensions.


# Plain-English directional labels per feature (signed deviation).
# Higher (+z) and lower (-z) get different framings — e.g. "louder than
# cohort" vs "quieter than cohort". Optional descriptions for the panel.
ORIGINALITY_DIRECTION_LABELS = {
    # Frequency spectrum
    'sub_ratio':           {'high': 'heavier sub-bass than',         'low': 'lighter sub-bass than'},
    'bass_ratio':          {'high': 'heavier bass than',              'low': 'lighter bass than'},
    'low_mid_ratio':       {'high': 'thicker low-mids than',          'low': 'cleaner low-mids than'},
    'mid_ratio':           {'high': 'more forward mids than',         'low': 'softer mids than'},
    'high_mid_ratio':      {'high': 'more presence / edge than',      'low': 'softer upper-mids than'},
    'presence_ratio':      {'high': 'brighter presence than',         'low': 'darker presence than'},
    'air_ratio':           {'high': 'more high-end air than',         'low': 'less high-end air than'},
    # Brightness / spectral shape
    'brightness':          {'high': 'brighter spectral center than',  'low': 'darker spectral center than'},
    'spectral_rolloff':    {'high': 'more high-frequency rolloff than','low': 'less high-frequency content than'},
    'brightness_variance': {'high': 'more brightness movement than',  'low': 'flatter brightness curve than'},
    # Dynamics
    'energy':              {'high': 'higher energy than',             'low': 'more restrained than'},
    'dynamic_range':       {'high': 'more dynamic contrast than',     'low': 'flatter dynamics than'},
    'loudness_range':      {'high': 'wider loudness variation than',  'low': 'tighter loudness than'},
    'lufs_integrated':     {'high': 'louder master than',             'low': 'quieter master than'},
    'compression_amount':  {'high': 'more compressed than',           'low': 'more open / less compressed than'},
    'crest_factor':        {'high': 'punchier peaks than',            'low': 'flatter peaks than'},
    'true_peak_dbfs':      {'high': 'higher peak level than',         'low': 'lower peak level than'},
    # Rhythm / transients
    'beat_strength':       {'high': 'stronger beat than',             'low': 'softer beat than'},
    'onset_rate':          {'high': 'denser percussion than',         'low': 'sparser percussion than'},
    'attack_time':         {'high': 'slower attacks than',            'low': 'sharper attacks than'},
    'danceability':        {'high': 'more rhythmic pull than',        'low': 'looser groove than'},
    # Tonal character
    'spectral_complexity': {'high': 'more spectral complexity than',  'low': 'simpler spectrum than'},
    'dissonance':          {'high': 'more dissonant / edgy than',     'low': 'more consonant / clean than'},
    'key_strength':        {'high': 'more tonally anchored than',     'low': 'more tonally ambiguous than'},
    'zcr':                 {'high': 'brighter / noisier than',        'low': 'mellower / cleaner than'},
    'spectral_flux':       {'high': 'more spectral movement than',    'low': 'more static spectrum than'},
    'harmonic_distortion': {'high': 'more harmonic saturation than',  'low': 'cleaner harmonics than'},
    # Stereo imaging
    'stereo_width':        {'high': 'wider stereo image than',        'low': 'narrower stereo than'},
    'mid_side_ratio':      {'high': 'more side energy than',          'low': 'more centered mix than'},
    'stereo_correlation':  {'high': 'more decorrelated stereo than',  'low': 'more correlated stereo than'},
}


def _compute_originality(features: dict, high_converter_gems: list) -> dict | None:
    """Sonic originality: weighted Euclidean distance from cohort centroid in
    z-normalized audio-feature space. Cohort is the same top-25% composite
    high-converter pool that drives production recommendations — so the
    originality × performance quadrant story has a consistent reference.

    Returns {composite_score (0-100), top_deviations, fits_consensus,
    cohort_size} or None if cohort is too small or no usable features.
    """
    if not high_converter_gems or len(high_converter_gems) < 10:
        return None

    target_bpm = float(features.get('bpm', 120))
    time_based = {'attack_time', 'onset_rate', 'beat_strength', 'danceability'}

    per_feature = []
    for feat in PRODUCTION_FEATURES:
        if feat not in ORIGINALITY_WEIGHTS:
            continue
        user_val = features.get(feat)
        if user_val is None:
            continue
        try:
            user_val = float(user_val)
        except (TypeError, ValueError):
            continue

        values = []
        for gems_row in high_converter_gems:
            v = gems_row.get(feat)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            # BPM-normalize time-based features (mirrors _find_consensus)
            if feat in time_based:
                cb = float(gems_row.get('bpm', 120) or 120)
                if cb > 0:
                    v = v * (target_bpm / cb)
            values.append(v)

        if len(values) < 5:
            continue

        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = var ** 0.5
        if std == 0:
            continue

        z = (user_val - mean) / std
        per_feature.append({
            'feature': feat,
            'cohort_mean': round(mean, 4),
            'cohort_std': round(std, 4),
            'user_val': round(user_val, 4),
            'z': round(z, 3),
            'abs_z': round(abs(z), 3),
            'direction': 'high' if z > 0 else 'low',
        })

    if not per_feature:
        return None

    # Composite originality: sqrt of weighted sum of z^2.
    distance_sq = sum(
        ORIGINALITY_WEIGHTS.get(f['feature'], 0.02) * (f['z'] ** 2)
        for f in per_feature
    )
    distance = distance_sq ** 0.5

    # Map distance → 0-100 score. Calibration: d=0 → 0, d≈1.5 → 60, d≈3 → 85,
    # asymptotes to 100. Soft sigmoid (1 - exp(-d/1.5)) * 100.
    import math
    composite_score = max(0, min(100, round(100 * (1 - math.exp(-distance / 1.5)))))

    # Sort by |z| descending — features where the track stands out most
    sorted_desc = sorted(per_feature, key=lambda x: -x['abs_z'])
    top_deviations = [f for f in sorted_desc if f['abs_z'] >= 1.0][:6]

    # Features where the track sits near cohort consensus (|z| < 0.5)
    sorted_asc = sorted(per_feature, key=lambda x: x['abs_z'])
    fits_consensus = [f for f in sorted_asc if f['abs_z'] < 0.5][:5]

    return {
        'composite_score': composite_score,
        'cohort_size': len(high_converter_gems),
        'distance': round(distance, 3),
        'top_deviations': top_deviations,
        'fits_consensus': fits_consensus,
    }


def _classify_quadrant(originality_score, performance_percentile):
    """Cross-reference originality and track-momentum performance percentiles
    into one of four quadrants. Returns {quadrant, label, message}.

    Thresholds: p75 cuts. High orig = composite_score >= 75. High perf =
    composite_percentile >= 75. Strict by design — only genuinely distinctive
    AND genuinely winning tracks land in "signature of success".
    """
    if originality_score is None or performance_percentile is None:
        return None
    orig_high = originality_score >= 75
    perf_high = performance_percentile >= 0.75
    if orig_high and perf_high:
        return {
            'quadrant': 'signature_of_success',
            'label': 'Signature of Success',
            'message': "You're distinct AND winning — your sound's deviation from cohort consensus IS your edge. The production recommendations below are informational, not prescriptive: adopting them risks closing the very gap that's working for you.",
        }
    if not orig_high and perf_high:
        return {
            'quadrant': 'genre_playbook_winner',
            'label': 'Genre-Playbook Winner',
            'message': "You sound like your cohort AND you're winning. You're executing the genre consensus well. Production recommendations are useful as fine-tuning, not redirection.",
        }
    if orig_high and not perf_high:
        return {
            'quadrant': 'ahead_of_the_curve',
            'label': 'Ahead of the Curve, or Off the Curve',
            'message': "You're distinct but not landing yet. Two possibilities: innovation ahead of audience, or a mix/production gap pushing you outside what's working. Cross-check your top deviations below — if they're on dissonance / dynamic_range / compression, investigate whether those are intentional choices or technical issues worth tightening.",
        }
    return {
        'quadrant': 'stuck_in_pack',
        'label': 'Stuck in the Pack',
        'message': "You sound like your cohort but aren't getting their results. The production recommendations below are genuinely actionable here — closing the gap is the move, since distinctiveness isn't currently the differentiator either.",
    }


# ---------------------------------------------------------------------------
# Pitch Comparables — top N tier-matched, sonic-peer artists who themselves
# land in the Signature of Success quadrant (high performance + high
# originality). Pool is `found_matches` (already same-tier and genre-family
# compatible per the matcher). For each candidate we compute the same
# performance composite + originality distance the user is scored on, then
# rank by combined score. Output is the artist-level proof-of-concept list
# an artist can put in front of an A&R rep.
# ---------------------------------------------------------------------------
PITCH_COMPARABLES_MIN_SIMILARITY = 0.70  # Tighter than matcher's 0.55 floor —
# pitch comparables should be HIGH sonic similarity, not borderline matches
# that survive only because of one shared genre family tag. Filter applies to
# both the pitch comparables list AND the cohort scatter cloud.

PITCH_COMPARABLES_MIN_GENRE_ALIGNMENT = 0.30  # Fraction of the candidate's
# individual genre tags whose families overlap with the user's. The 0.67
# original was tuned for broad lanes (rock/electronic) — for narrow lanes
# like breaks, even legit jungle artists have mixed tags
# (drum & bass + dance + electronic + uk + …) and the breaks share lands ~0.40,
# so 0.67 emptied the quadrant. 0.30 keeps the legit narrow-lane candidates
# while still rejecting "touched the lane on one tag" drift.

PITCH_COMPARABLES_MIN_PRIMARY_SHARE = 0.25  # Fraction of the candidate's
# tags that must resolve to the user's PRIMARY (most-frequent) genre family.
# Same narrow-lane rationale as above — breaks-primary artists carry many
# electronic/dance tags so the share runs ~0.40; 0.25 lets the real
# candidates through while still requiring a meaningful primary-lane presence.


def _primary_genre_family(genre_string: str) -> str | None:
    """Most frequent family across the user's genre tags. Ties broken by
    insertion order. Returns None if no families resolve."""
    if not genre_string:
        return None
    counts = {}
    for tag in genre_string.split(','):
        tag = tag.strip()
        if not tag:
            continue
        for fam in _genre_families(tag):
            counts[fam] = counts.get(fam, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _candidate_primary_family_share(candidate: dict, user_primary_family: str) -> float:
    """Fraction of the candidate's individual genre tags that resolve to
    user_primary_family. 1.0 = candidate is purely in user's dominant lane."""
    if not user_primary_family:
        return 1.0
    cand_tags = []
    for field in ('primary_genre', 'secondary_genre'):
        g = (candidate.get(field) or '').strip()
        if g:
            cand_tags.append(g)
    for g in (candidate.get('artist_genres') or []):
        if g and isinstance(g, str) and g.strip():
            cand_tags.append(g.strip())
    for g in (candidate.get('track_genres') or []):
        if g and isinstance(g, str) and g.strip():
            cand_tags.append(g.strip())
    if not cand_tags:
        return 1.0
    in_primary, total = 0, 0
    for tag in cand_tags:
        fams = _genre_families(tag)
        if not fams:
            continue
        total += 1
        if user_primary_family in fams:
            in_primary += 1
    if total == 0:
        return 1.0
    return in_primary / total


def _genre_alignment_fraction(user_families: set, candidate: dict) -> float:
    """Fraction of the candidate's individual genre tags whose families
    overlap with user_families. 1.0 = all tags in user's lane; 0.0 = none."""
    if not user_families:
        return 1.0
    cand_tags = []
    for field in ('primary_genre', 'secondary_genre'):
        g = (candidate.get(field) or '').strip()
        if g:
            cand_tags.append(g)
    for g in (candidate.get('artist_genres') or []):
        if g and isinstance(g, str) and g.strip():
            cand_tags.append(g.strip())
    for g in (candidate.get('track_genres') or []):
        if g and isinstance(g, str) and g.strip():
            cand_tags.append(g.strip())
    if not cand_tags:
        return 1.0
    aligned, total = 0, 0
    for tag in cand_tags:
        tag_fams = _genre_families(tag)
        if not tag_fams:
            continue
        total += 1
        if tag_fams & user_families:
            aligned += 1
    if total == 0:
        return 1.0
    return aligned / total

def _compute_pitch_comparables(found_matches: list, high_converter_gems: list,
                                gems_by_isrc: dict, user_families: set = None,
                                user_primary_family: str = None,
                                n: int = 5) -> list:
    """Returns up to N candidates with name, listeners, similarity, performance
    percentile, originality score, plus a pitch_angle string. Empty if pool
    is too thin or no candidates qualify.
    """
    import bisect, math
    if not found_matches or not high_converter_gems or not gems_by_isrc:
        return []
    # Tighten similarity floor — pitch comparables should be high-similarity peers
    found_matches = [m for m in found_matches
                     if (m.get('similarity') or 0) >= PITCH_COMPARABLES_MIN_SIMILARITY]
    # Stricter genre alignment — candidate must be majority in the user's lane,
    # not just share one family tag among many. Catches the hybrid-genre
    # passes (e.g. "uk alternative + uk hip-hop/rap" showing up as rock).
    if user_families:
        found_matches = [m for m in found_matches
                         if _genre_alignment_fraction(user_families, m) >= PITCH_COMPARABLES_MIN_GENRE_ALIGNMENT]
    # Tighter still — for hybrid users (e.g. rock + hip-hop), the alignment
    # filter passes any candidate touching either family. This filter rejects
    # candidates whose tags aren't majority in the user's DOMINANT family.
    if user_primary_family:
        found_matches = [m for m in found_matches
                         if _candidate_primary_family_share(m, user_primary_family) >= PITCH_COMPARABLES_MIN_PRIMARY_SHARE]
    if len(found_matches) < 5 or len(high_converter_gems) < 10:
        return []

    # Cohort centroid in z-space (same construction _compute_originality uses)
    centroid, stds = {}, {}
    for feat in PRODUCTION_FEATURES:
        if feat not in ORIGINALITY_WEIGHTS:
            continue
        vals = []
        for f in high_converter_gems:
            v = f.get(feat)
            if v is None: continue
            try: vals.append(float(v))
            except (TypeError, ValueError): continue
        if len(vals) < 10: continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        if std > 0:
            centroid[feat] = mean
            stds[feat] = std

    # Performance composite — percentile within found_matches
    pop_pool = sorted([x['sp_track_popularity'] for x in found_matches if x.get('sp_track_popularity') is not None])
    cm_pool  = sorted([x['cm_track_score']      for x in found_matches if x.get('cm_track_score')      is not None])
    pl_pool  = sorted([_safe_int(x.get('editorial_playlists')) + _safe_int(x.get('user_playlists')) for x in found_matches])

    def _perf(x):
        pop = _pct_rank(x.get('sp_track_popularity'), pop_pool)
        cm  = _pct_rank(x.get('cm_track_score'), cm_pool) if x.get('cm_track_score') is not None else None
        pl  = _pct_rank(_safe_int(x.get('editorial_playlists')) + _safe_int(x.get('user_playlists')), pl_pool)
        if pop is None: pop = 0.5
        if pl is None: pl = 0.5
        if cm is not None: return 0.50 * pop + 0.30 * cm + 0.20 * pl
        return 0.714 * pop + 0.286 * pl

    def _orig(isrc):
        if not isrc: return None
        f = gems_by_isrc.get(isrc)
        if not f: return None
        dist_sq, cnt = 0.0, 0
        for feat, w in ORIGINALITY_WEIGHTS.items():
            if feat not in centroid: continue
            v = f.get(feat)
            if v is None: continue
            try: z = (float(v) - centroid[feat]) / stds[feat]
            except (TypeError, ValueError, ZeroDivisionError): continue
            dist_sq += w * z * z
            cnt += 1
        if cnt < 5: return None
        return round(100 * (1 - math.exp(-(dist_sq ** 0.5) / 1.5)))

    scored = []
    for x in found_matches:
        p = _perf(x)
        o = _orig(x.get('isrc'))
        if o is None: continue
        scored.append({
            'name': x.get('name'),
            'spotify_url': x.get('spotify_url'),       # artist profile URL (kept for backwards compat)
            'track_url': x.get('track_url'),           # specific track URL — what the A&R link points to
            'track_name': x.get('track_name'),
            'tier': x.get('tier'),
            'listeners': int(float(x.get('listeners') or 0)),
            'followers': int(float(x.get('followers') or 0)),
            'similarity': round(x.get('similarity') or 0, 3),
            'sp_track_popularity': x.get('sp_track_popularity'),
            'cm_track_score': x.get('cm_track_score'),
            'playlists_total': _safe_int(x.get('editorial_playlists')) + _safe_int(x.get('user_playlists')),
            'perf_pct': round(p, 3),
            'orig_score': o,
        })
    if not scored: return []

    # Tiered candidate selection — pitch comparables are defined as
    # "artists who sound like you AND are winning by being distinctive",
    # which is literally the Signature of Success quadrant. Prefer SoS
    # peers; fall back progressively if the cohort doesn't have enough.
    #
    # Tier 1: Signature of Success — perf ≥ p75 AND orig ≥ 75
    # Tier 2: "Winning OR distinctive" — perf ≥ p75 OR orig ≥ 75
    # Tier 3: Soft p60 floors on both axes (original fallback)
    # Tier 4: Full scored pool (last resort)
    sos_peers = [c for c in scored if c['perf_pct'] >= 0.75 and c['orig_score'] >= 75]
    if len(sos_peers) >= n:
        qualified = sos_peers
    else:
        winning_or_distinctive = [c for c in scored if c['perf_pct'] >= 0.75 or c['orig_score'] >= 75]
        if len(winning_or_distinctive) >= n:
            qualified = winning_or_distinctive
        else:
            perf_sorted = sorted(c['perf_pct'] for c in scored)
            orig_sorted = sorted(c['orig_score'] for c in scored)
            p_floor = perf_sorted[int(len(perf_sorted) * 0.60)] if perf_sorted else 0
            o_floor = orig_sorted[int(len(orig_sorted) * 0.60)] if orig_sorted else 0
            soft_qualified = [c for c in scored if c['perf_pct'] >= p_floor and c['orig_score'] >= o_floor]
            qualified = soft_qualified if len(soft_qualified) >= n else scored

    # Reweight combined score to favor perf + orig over raw similarity.
    # Pitch comparables are A&R proof points — "moving the needle" matters
    # at least as much as sonic similarity. Bump perf + orig from 0.30 each
    # to 0.35 each; drop similarity from 0.40 to 0.30.
    for c in qualified:
        c['combined_score'] = round(
            0.30 * c['similarity']
            + 0.35 * c['perf_pct']
            + 0.35 * (c['orig_score'] / 100),
            3,
        )
    qualified.sort(key=lambda c: -c['combined_score'])
    return qualified[:n]


def _compute_cohort_scatter(found_matches: list, high_converter_gems: list,
                             gems_by_isrc: dict, user_families: set = None,
                             user_primary_family: str = None) -> list:
    """Lightweight per-peer scatter data for the Sonic Quadrant visualization.
    Returns list of {name, perf_pct, orig_score} for every same-tier peer
    where we can compute both axes. Frontend plots these as a dim cohort
    cloud behind the user's dot and the pitch-comparable labels.

    Reuses the same centroid + performance pools the pitch-comparables
    function builds, but returns the FULL scored set (not just the top 5).

    Filter scope deliberately LOOSER than the pitch-comparables list:
    the cloud's purpose is to show the user where their broad same-tier
    cohort lands on the orig × perf grid — the labeled picks are the
    strict A&R-quality subset rendered on top. Applying the strict
    sim/alignment/primary-share filters here would erase the texture
    that gives the chart context.
    """
    import bisect, math
    if not found_matches or not high_converter_gems or not gems_by_isrc:
        return []
    # No extra filtering — use the full same-tier pool the matcher returned.
    # user_families / user_primary_family params kept for API compatibility
    # but intentionally unused here.

    # Build cohort centroid (same as _compute_pitch_comparables, kept
    # local to avoid coupling)
    centroid, stds = {}, {}
    for feat in PRODUCTION_FEATURES:
        if feat not in ORIGINALITY_WEIGHTS:
            continue
        vals = []
        for f in high_converter_gems:
            v = f.get(feat)
            if v is None: continue
            try: vals.append(float(v))
            except (TypeError, ValueError): continue
        if len(vals) < 10: continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        if std > 0:
            centroid[feat] = mean
            stds[feat] = std

    pop_pool = sorted([x['sp_track_popularity'] for x in found_matches if x.get('sp_track_popularity') is not None])
    cm_pool  = sorted([x['cm_track_score']      for x in found_matches if x.get('cm_track_score')      is not None])
    pl_pool  = sorted([_safe_int(x.get('editorial_playlists')) + _safe_int(x.get('user_playlists')) for x in found_matches])

    def _perf(x):
        pop = _pct_rank(x.get('sp_track_popularity'), pop_pool)
        cm  = _pct_rank(x.get('cm_track_score'), cm_pool) if x.get('cm_track_score') is not None else None
        pl  = _pct_rank(_safe_int(x.get('editorial_playlists')) + _safe_int(x.get('user_playlists')), pl_pool)
        if pop is None: pop = 0.5
        if pl is None: pl = 0.5
        if cm is not None: return 0.50 * pop + 0.30 * cm + 0.20 * pl
        return 0.714 * pop + 0.286 * pl

    def _orig(isrc):
        if not isrc: return None
        f = gems_by_isrc.get(isrc)
        if not f: return None
        dist_sq, cnt = 0.0, 0
        for feat, w in ORIGINALITY_WEIGHTS.items():
            if feat not in centroid: continue
            v = f.get(feat)
            if v is None: continue
            try: z = (float(v) - centroid[feat]) / stds[feat]
            except (TypeError, ValueError, ZeroDivisionError): continue
            dist_sq += w * z * z
            cnt += 1
        if cnt < 5: return None
        return round(100 * (1 - math.exp(-(dist_sq ** 0.5) / 1.5)))

    scatter = []
    for x in found_matches:
        p = _perf(x)
        o = _orig(x.get('isrc'))
        if o is None: continue
        scatter.append({
            'name': x.get('name'),
            'perf_pct': round(p, 3),
            'orig_score': o,
        })
    return scatter


def _find_signature_consensus(features: dict, high_converter_gems: list) -> list:
    """Production-rec consensus against the signature-of-success subset within
    the cohort (peers who are themselves high-orig + high-perf).

    The default _find_consensus averages over the whole top-25%-composite
    cohort. That biases toward genre-consensus advice ("compress more, fit
    the middle"). This variant first filters the cohort to peers whose own
    audio features deviate strongly from the cohort centroid (top 50% by
    weighted z-distance), then runs consensus against that subset.

    Result: recs that reflect "what cohort winners who ALSO deviate from
    the genre norm actually do" — recs that don't punish originality.
    """
    signature_peers = _signature_subset(high_converter_gems)
    if not signature_peers:
        return []
    # Run the existing consensus math against this tighter reference set
    return _find_consensus(features, signature_peers)


def _generate_signature_recommendations(features: dict, high_converter_gems: list) -> list:
    """Mirror of _generate_recommendations' consensus-formatting step, run
    against signature-of-success peers instead of the full cohort. Returns
    formatted rec strings ready for the frontend."""
    consensus = _find_signature_consensus(features, high_converter_gems)
    if not consensus:
        return []

    recs = []
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
    return recs


def _signature_subset(high_converter_gems: list) -> list:
    """The 'signature-of-success' subset of the high-converter cohort: peers
    whose own audio features deviate most from the cohort centroid (top 50% by
    ORIGINALITY_WEIGHTS-weighted z-distance) — winners who ALSO break the genre
    norm. Returns [] if the cohort is too thin.

    Shared by _find_signature_consensus and _generate_recommendation_ranges so
    the 'signature' edge is computed identically everywhere.
    """
    if not high_converter_gems or len(high_converter_gems) < 6:
        return []

    # Build centroid + std per feature from the full cohort
    centroid, stds = {}, {}
    for feat in PRODUCTION_FEATURES:
        if feat not in ORIGINALITY_WEIGHTS:
            continue
        vals = []
        for f in high_converter_gems:
            v = f.get(feat)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if len(vals) < 5:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        if std > 0:
            centroid[feat] = mean
            stds[feat] = std

    if not centroid:
        return []

    # Per-peer weighted z-distance from cohort centroid
    peer_distances = []
    for peer in high_converter_gems:
        dist_sq, cnt = 0.0, 0
        for feat, w in ORIGINALITY_WEIGHTS.items():
            if feat not in centroid:
                continue
            v = peer.get(feat)
            if v is None:
                continue
            try:
                z = (float(v) - centroid[feat]) / stds[feat]
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            dist_sq += w * z * z
            cnt += 1
        if cnt >= 5:
            peer_distances.append((dist_sq ** 0.5, peer))

    if len(peer_distances) < 4:
        return []

    # Top 50% by originality = the signature-of-success peer subset
    peer_distances.sort(key=lambda x: -x[0])
    cutoff = max(int(len(peer_distances) * 0.50), 4)
    return [peer for _, peer in peer_distances[:cutoff]]


def _feature_avg(features: dict, gems_list: list, feat: str):
    """BPM-normalized average of one production feature across a gems list.
    Mirrors the normalization in _find_consensus so targets are comparable."""
    if not gems_list:
        return None
    target_bpm = float(features.get('bpm', 120) or 120)
    time_based = {'attack_time', 'onset_rate', 'beat_strength', 'danceability'}
    vals = []
    for g in gems_list:
        v = g.get(feat)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if feat in time_based:
            cb = float(g.get('bpm', 120) or 120)
            if cb > 0:
                v = v * (target_bpm / cb)
        vals.append(v)
    return sum(vals) / len(vals) if vals else None


def _feature_percentiles(features: dict, gems_list: list, feat: str,
                         ps=(5, 25, 50, 75, 95)):
    """BPM-normalized percentile distribution of one feature across a gems list.
    Lets the frontend anchor the range bar to where peers ACTUALLY sit, so the
    visual distance reflects real adjustment effort (a 1 dB move looks like a
    1 dB move). Returns {'p5':..,'p25':..,'p50':..,'p75':..,'p95':..} or None."""
    if not gems_list:
        return None
    target_bpm = float(features.get('bpm', 120) or 120)
    time_based = {'attack_time', 'onset_rate', 'beat_strength', 'danceability'}
    vals = []
    for g in gems_list:
        v = g.get(feat)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if feat in time_based:
            cb = float(g.get('bpm', 120) or 120)
            if cb > 0:
                v = v * (target_bpm / cb)
        vals.append(v)
    if len(vals) < 4:
        return None
    vals.sort()
    n = len(vals)

    def _pct(p):
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    return {f'p{p}': round(_pct(p), 6) for p in ps}


def _rec_unit_kind(feat: str, unit: str) -> str:
    """Frontend formatting hint for a production-feature value."""
    if feat in RATIO_FEATURES:
        return 'pct'
    if feat in DB_FEATURES:
        return 'db'
    if unit == 'LUFS':
        return 'lufs'
    if unit == 'Hz':
        return 'hz'
    if unit == '/s':
        return 'rate'
    if unit == 'ms':
        return 'ms'
    if unit == 'LU':
        return 'lu'
    if unit == 'dBFS':
        return 'db'
    return 'raw'


# Chunk→whole-track Integrated model (Model C), fitted on 73 full tracks
# simulated through the GEMS 10s-window pipeline (the universe's basis).
# LOO-validated: MAE 0.72 dB, ~90% within ±1.5 dB. Missing predictors impute
# universe medians. Display-only ("est.") — NEVER feeds matching or storage.
_INT_EST_COEF = {'lufs': 0.959, 'corr': -3.771, 'crest': -0.050, 'dr': -0.004, 'lra': 0.035, 'const': 5.621}
_INT_EST_MEDIANS = {'corr': 0.828, 'crest': 10.62, 'dr': 19.78, 'lra': 2.84}


def _est_integrated_from_gems_row(row: dict):
    """Estimate a peer track's true whole-track Integrated LUFS from its
    stored chunk features. Returns None when the row can't support it."""
    try:
        lufs = row.get('lufs_integrated')
        if lufs is None:
            return None
        lufs = float(lufs)
        if not (-60.0 < lufs < 0.0):
            return None

        def _p(key, med, lo, hi):
            v = row.get(key)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return med
            return v if (np.isfinite(v) and lo <= v <= hi) else med

        c = _INT_EST_COEF
        m = _INT_EST_MEDIANS
        est = (c['lufs'] * lufs
               + c['corr'] * _p('stereo_correlation', m['corr'], -1.0, 1.0)
               + c['crest'] * _p('crest_factor', m['crest'], 0.0, 40.0)
               + c['dr'] * _p('dynamic_range', m['dr'], 0.0, 60.0)
               + c['lra'] * _p('loudness_range', m['lra'], 0.0, 30.0)
               + c['const'])
        return float(min(max(est, -40.0), -2.0))
    except Exception:
        return None


def _generate_recommendation_ranges(features: dict, high_converter_gems: list) -> list:
    """Amalgamate the two production-rec cohorts into per-feature TARGET RANGES.

    For each feature the full high-converter cohort agrees on (>=50% directional
    consensus, via _find_consensus), compute two targets:
      - target_cohort    = full cohort average     (genre consensus / 'safe' edge)
      - target_signature = signature-subset average (distinctive winners / 'edge')
    The band between them is the artist's mastering/production wiggle room.
    Returns structured objects the frontend renders as range meters. The two
    legacy string lists (recommendations / signature_recommendations) are the
    two edges of these same bands.
    """
    if not high_converter_gems:
        return []

    cohort_consensus = _find_consensus(features, high_converter_gems)
    if not cohort_consensus:
        return []

    signature_peers = _signature_subset(high_converter_gems)

    ranges = []
    domain_count = {}
    for c in cohort_consensus:
        # Emit up to 12 (max 2/domain). The frontend splits these into
        # "Adjustments" (you outside the zone) vs "doing right" (you inside it),
        # so we surface a few more than the old 8 to keep both groups populated.
        if len(ranges) >= 12:
            break
        feat = c['feature']
        desc = FEATURE_DESCRIPTIONS.get(feat)
        if not desc:
            continue
        domain = desc.get('domain', '')
        if domain_count.get(domain, 0) >= 2:
            continue

        direction = c['direction']
        action = desc.get(direction, '')
        if not action:
            continue

        cohort_target = c['converter_avg']
        # Same realism guards as _format_rec — GEMS 4s samples bias these low.
        if feat == 'crest_factor' and cohort_target < 6.0:
            continue
        if feat == 'dynamic_range' and direction == 'lower' and cohort_target < 15.0:
            continue

        sig_target = _feature_avg(features, signature_peers, feat) if signature_peers else None

        row_entry = {
            'feature': feat,
            'domain': domain,
            'action': action,
            'unit_kind': _rec_unit_kind(feat, desc.get('unit', '')),
            'you': round(c['user_val'], 6),
            'target_cohort': round(cohort_target, 6),
            'target_signature': round(sig_target, 6) if sig_target is not None else None,
            # Where peers actually sit — anchors the bar to a realistic scale and
            # gives the striped p25–p75 "aim here" zone.
            'percentiles': _feature_percentiles(features, high_converter_gems, feat),
            'agree': [c['count'], c['total']],
            'direction': direction,
        }
        # Loudness only: peer zone converted to whole-track Integrated (est.)
        # via Model C, so the frontend can render the row in mastering-meter
        # units when the user side has an Integrated value (measured or est.).
        if feat == 'lufs_integrated':
            est_vals = [v for v in (_est_integrated_from_gems_row(g) for g in high_converter_gems) if v is not None]
            if len(est_vals) >= 8:
                row_entry['percentiles_est'] = {
                    p: round(float(np.percentile(est_vals, q)), 2)
                    for p, q in (('p5', 5), ('p25', 25), ('p50', 50), ('p75', 75), ('p95', 95))
                }
                # Both direction texts: `action`/`direction` above came from the
                # CHUNK comparison, which can point the wrong way once the row
                # renders in Integrated currency (user's chunk↔integrated offset
                # differs from peers'). The frontend re-picks by zone side.
                row_entry['actions_integrated'] = {
                    'higher': desc.get('higher', ''),
                    'lower': desc.get('lower', ''),
                }
        ranges.append(row_entry)
        domain_count[domain] = domain_count.get(domain, 0) + 1

    return ranges


def _generate_full_target_ranges(features: dict, high_converter_gems: list) -> list:
    """Target ranges for EVERY production feature — the plugin/export variant.

    _generate_recommendation_ranges caps output at 12 (2/domain, consensus
    only) as a display choice for the web UI. The mastering-meter plugin wants
    the full picture: one range object per PRODUCTION_FEATURE with the peer
    percentile band and both targets. Consensus fields (action/direction/agree)
    are populated when the cohort agrees, null otherwise — same realism guards
    as the curated list apply to those fields only, never to the band itself.
    """
    if not high_converter_gems:
        return []

    consensus_by_feat = {c['feature']: c for c in _find_consensus(features, high_converter_gems)}
    signature_peers = _signature_subset(high_converter_gems)

    ranges = []
    for feat in PRODUCTION_FEATURES:
        user_val = features.get(feat)
        try:
            user_val = float(user_val)
        except (TypeError, ValueError):
            continue

        percentiles = _feature_percentiles(features, high_converter_gems, feat)
        cohort_target = _feature_avg(features, high_converter_gems, feat)
        if percentiles is None or cohort_target is None:
            continue

        desc = FEATURE_DESCRIPTIONS.get(feat, {})
        sig_target = _feature_avg(features, signature_peers, feat) if signature_peers else None

        c = consensus_by_feat.get(feat)
        if c:
            # Same realism guards as _format_rec / _generate_recommendation_ranges
            if feat == 'crest_factor' and c['converter_avg'] < 6.0:
                c = None
            elif feat == 'dynamic_range' and c['direction'] == 'lower' and c['converter_avg'] < 15.0:
                c = None

        ranges.append({
            'feature': feat,
            'domain': desc.get('domain', ''),
            'action': desc.get(c['direction'], '') if c else None,
            'unit_kind': _rec_unit_kind(feat, desc.get('unit', '')),
            'you': round(user_val, 6),
            'target_cohort': round(cohort_target, 6),
            'target_signature': round(sig_target, 6) if sig_target is not None else None,
            'percentiles': percentiles,
            'agree': [c['count'], c['total']] if c else None,
            'direction': c['direction'] if c else None,
        })

    return ranges


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def _validate_session(token: str) -> dict:
    """Validate session token from Supabase. Returns user dict or raises 401.

    Also checks legacy in-memory tokens for backwards compat.
    """
    # Check legacy in-memory tokens first (will be removed after migration)
    legacy = access_tokens.get(token)
    if legacy:
        return legacy

    if not supabase:
        raise HTTPException(401, "Service unavailable")

    resp = supabase.table('sessions').select('user_id, expires_at').eq('token', token).execute()
    if not resp.data:
        raise HTTPException(401, "Invalid or expired session. Please log in.")

    session = resp.data[0]
    from datetime import timezone
    expires = datetime.fromisoformat(session['expires_at'].replace('Z', '+00:00'))
    if expires < datetime.now(timezone.utc):
        supabase.table('sessions').delete().eq('token', token).execute()
        raise HTTPException(401, "Session expired. Please log in again.")

    user_resp = supabase.table('users').select('*').eq('id', session['user_id']).execute()
    if not user_resp.data:
        raise HTTPException(401, "User not found.")

    return user_resp.data[0]


def _create_session(user_id: str) -> str:
    """Create a new session token in Supabase. Returns the token."""
    token = secrets.token_urlsafe(32)
    from datetime import timezone, timedelta
    expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    supabase.table('sessions').insert({
        'token': token,
        'user_id': user_id,
        'expires_at': expires,
    }).execute()
    return token


def _check_scan_cap(user: dict):
    """Check if user can scan. Raises 403 if at limit. Does NOT decrement."""
    if 'id' not in user:
        return  # Legacy tokens have no cap
    if user.get('scans_used', 0) >= user.get('max_scans', 3):
        raise HTTPException(403, "Scan limit reached. Contact us for more scans.")


def _use_scan(user: dict):
    """Decrement a scan from the user's cap. Call AFTER successful analysis."""
    if 'id' not in user:
        return
    try:
        supabase.rpc('use_scan', {'p_user_id': user['id']}).execute()
    except Exception as e:
        print(f"Scan decrement failed: {e}")


def _send_reset_email(email: str, reset_token: str) -> bool:
    """Send password reset email via SendGrid."""
    api_key = os.getenv('SENDGRID_API_KEY')
    if not api_key:
        print("SENDGRID_API_KEY not set — skipping reset email")
        return False

    reset_url = f"https://analyze.freshlybakedstudios.com/?reset={reset_token}"

    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                max-width:500px;margin:0 auto;background:#141213;color:#eee;padding:32px;border-radius:12px">
      <h2 style="color:#fff;margin:0 0 16px">Reset Your Password</h2>
      <p style="color:#ccc">Click the button below to reset your Sonic Analyzer password. This link expires in 1 hour.</p>
      <div style="text-align:center;margin:24px 0">
        <a href="{reset_url}" style="background:#4ecdc4;color:#000;padding:12px 32px;border-radius:8px;text-decoration:none;font-weight:600;display:inline-block">Reset Password</a>
      </div>
      <p style="color:#666;font-size:12px">If you didn't request this, you can ignore this email.</p>
      <p style="color:#444;font-size:11px;margin-top:24px">Freshly Baked Studios &bull; freshlybakedstudios.com</p>
    </div>
    """

    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, HtmlContent
    message = Mail(
        from_email='noreply@freshlybakedstudios.com',
        to_emails=email,
        subject='Reset your Sonic Analyzer password',
        html_content=HtmlContent(html),
    )
    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print(f"Reset email sent to {email} — status {response.status_code}")
        return response.status_code in (200, 201, 202)
    except Exception as e:
        print(f"Reset email failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.post("/api/signup")
async def signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    nda_agreed: str = Form("false"),
    spotify_url: Optional[str] = Form(None),
    monthly_listeners: Optional[int] = Form(None),
):
    """Create a new user account with NDA acceptance."""
    if not name or not email or not password:
        raise HTTPException(400, "Name, email, and password are required")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if nda_agreed.lower() != 'true':
        raise HTTPException(400, "You must agree to the Terms of Use & Non-Disclosure Agreement.")

    # Check if email already exists
    existing = supabase.table('users').select('id').eq('email', email.lower().strip()).execute()
    if existing.data:
        raise HTTPException(409, "An account with this email already exists. Please log in.")

    # Hash password
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    from datetime import timezone
    now = datetime.now(timezone.utc).isoformat()

    # Create user
    user_row = {
        'name': name,
        'email': email.lower().strip(),
        'password_hash': password_hash,
        'nda_agreed_at': now,
        'nda_version': NDA_VERSION,
        'spotify_url': spotify_url or None,
        'monthly_listeners': monthly_listeners,
        'max_scans': 3,
        'scans_used': 0,
    }

    try:
        resp = supabase.table('users').insert(user_row).execute()
        user = resp.data[0]
    except Exception as e:
        print(f"User creation failed: {e}")
        raise HTTPException(500, "Could not create account. Please try again.")

    # Create session
    token = _create_session(user['id'])

    # Also save to analyzer_leads for lead tracking
    try:
        supabase.table('analyzer_leads').insert({
            'name': name,
            'email': email.lower().strip(),
            'created_at': now,
            'analysis_count': 0,
        }).execute()
    except Exception:
        pass  # duplicate or missing columns — fine

    send_pushover_notification(
        "New Sonic Analyzer Signup",
        f"{name}\n{email}\nNDA: {NDA_VERSION}"
    )
    print(f"Signup: {email} (NDA {NDA_VERSION})")

    return {
        "token": token,
        "name": name,
        "scans_remaining": user['max_scans'],
    }


@app.post("/api/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
):
    """Log in with email and password."""
    if not email or not password:
        raise HTTPException(400, "Email and password are required")

    resp = supabase.table('users').select('*').eq('email', email.lower().strip()).execute()
    if not resp.data:
        raise HTTPException(401, "Invalid email or password.")

    user = resp.data[0]

    if not bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
        raise HTTPException(401, "Invalid email or password.")

    if not user.get('nda_agreed_at'):
        raise HTTPException(403, "NDA acceptance required. Please sign up again.")

    token = _create_session(user['id'])
    print(f"Login: {email}")

    return {
        "token": token,
        "name": user['name'],
        "scans_remaining": user['max_scans'] - user['scans_used'],
    }


@app.post("/api/logout")
async def logout(token: str = Form(...)):
    """Log out — destroy session, stop enrichment, resume pipelines."""
    if supabase:
        try:
            # Mark any active enrichment/matching jobs for this token as stale
            supabase.table('analysis_jobs').update({
                'status': 'stale'
            }).eq('token', token).in_('status', ['enriching', 'matching']).execute()
        except Exception:
            pass
        try:
            supabase.table('sessions').delete().eq('token', token).execute()
        except Exception:
            pass
    access_tokens.pop(token, None)
    # Reset activity timestamp so status endpoint reports idle
    global _last_api_activity
    _last_api_activity = 0
    # Signal resource-switcher that no user is active — resume GEMS/discovery
    _notify_local_pipeline('user_idle')
    return {"ok": True}


@app.get("/api/me")
async def me(token: str):
    """Check if session is valid and return user info."""
    try:
        user = _validate_session(token)
    except HTTPException:
        raise HTTPException(401, "Invalid session")

    # Legacy token
    if 'id' not in user:
        return {"name": user.get('name', ''), "email": user.get('email', ''), "scans_remaining": 999}

    return {
        "name": user['name'],
        "email": user['email'],
        "scans_remaining": user['max_scans'] - user['scans_used'],
        "scans_used": user['scans_used'],
        "max_scans": user['max_scans'],
    }


@app.get("/api/queue-status")
async def queue_status(job_id: Optional[str] = None):
    """Check how many jobs are in the audio capture queue."""
    if not supabase:
        return {"queue_length": 0, "position": 0}
    try:
        q = supabase.table('analysis_jobs').select('id,created_at').in_(
            'status', ['pending_features', 'capturing']
        ).order('created_at').execute()
        queue = q.data or []
        position = 0
        if job_id:
            for i, j in enumerate(queue):
                if j['id'] == job_id:
                    position = i + 1
                    break
        return {"queue_length": len(queue), "position": position}
    except Exception:
        return {"queue_length": 0, "position": 0}


@app.post("/api/forgot-password")
async def forgot_password(email: str = Form(...)):
    """Send a password reset link. Always returns ok (don't leak email existence)."""
    if supabase:
        resp = supabase.table('users').select('id').eq('email', email.lower().strip()).execute()
        if resp.data:
            reset_token = secrets.token_urlsafe(32)
            from datetime import timezone, timedelta
            supabase.table('password_reset_tokens').insert({
                'token': reset_token,
                'user_id': resp.data[0]['id'],
                'expires_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            }).execute()
            _send_reset_email(email.lower().strip(), reset_token)
    return {"ok": True}


@app.post("/api/reset-password")
async def reset_password(
    token: str = Form(...),
    new_password: str = Form(...),
):
    """Reset password using a reset token."""
    if len(new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    resp = supabase.table('password_reset_tokens').select('*').eq('token', token).eq('used', False).execute()
    if not resp.data:
        raise HTTPException(400, "Invalid or expired reset link.")

    reset = resp.data[0]
    from datetime import timezone
    expires = datetime.fromisoformat(reset['expires_at'].replace('Z', '+00:00'))
    if expires < datetime.now(timezone.utc):
        raise HTTPException(400, "Reset link has expired. Please request a new one.")

    # Update password
    password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    supabase.table('users').update({
        'password_hash': password_hash,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }).eq('id', reset['user_id']).execute()

    # Mark token as used
    supabase.table('password_reset_tokens').update({'used': True}).eq('token', token).execute()

    # Kill all sessions for this user (force re-login)
    supabase.table('sessions').delete().eq('user_id', reset['user_id']).execute()

    print(f"Password reset completed for user {reset['user_id']}")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Legacy register endpoint (backwards compat — will be removed)
# ---------------------------------------------------------------------------
@app.post("/api/register")
async def register(
    name: str = Form(...),
    email: str = Form(...),
    spotify_url: Optional[str] = Form(None),
    monthly_listeners: Optional[int] = Form(None),
):
    """DEPRECATED — use /api/signup. Kept for cached frontends."""
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
    artist_spotify_url: Optional[str] = Form(None),
):
    """
    Accept an uploaded audio file, run analysis, return results.
    Also sends results email.
    """
    # Validate token
    lead = _validate_session(token)
    _check_scan_cap(lead)

    # Validate file type
    filename = file.filename or ''
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ('mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac', 'aiff', 'aif'):
        raise HTTPException(400, f"Unsupported file type: .{ext}. Please upload mp3, wav, or aiff.")

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
    try:
        contents = await file.read()
        if len(contents) > 100 * 1024 * 1024:  # 100 MB limit
            raise HTTPException(400, "File too large (max 100 MB)")
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

        # Save original dropdown selection — drives the LANE (heavy weight).
        dropdown_genre = (genre or '').strip()

        # Pause enrichment so user-facing CM calls get priority
        _pause_enrichment()

        # `artist_genre` carries the CM artist tags (separate from dropdown).
        # Used for in_lane CF checks only, NEVER widens the dropdown-defined
        # lane (per user's "heavy weight on dropdown" requirement).
        artist_genre = ''
        cached_artist_data = None
        cm_data = None
        user_code2 = None
        user_pronoun = None

        # Priority 1: form-provided artist Spotify URL (specific to the uploaded
        # track — typically when the user is analyzing a track by someone other
        # than the artist on their account). Drives `artist_genre` for the gate's
        # CF signal. Does NOT influence tier/listeners (those still come from
        # the user's registered account).
        provided_artist_url = (artist_spotify_url or '').split('?')[0].rstrip('/')
        track_artist_cm = None
        if provided_artist_url:
            print(f"  Upload: form-provided artist URL = {provided_artist_url}")
            t_cm = time.time()
            track_artist_cm = lookup_artist_by_spotify(provided_artist_url)
            t_cm = time.time() - t_cm
            if track_artist_cm:
                artist_genre = track_artist_cm.get('genres', '') or ''
                print(f"  Upload: CM (form-artist) = {track_artist_cm.get('name')} — "
                      f"genres={artist_genre[:120]} [{t_cm:.1f}s]")
            else:
                print(f"  Upload: form-artist CM lookup empty [{t_cm:.1f}s]")

        # Priority 2: user's registered Spotify URL (existing behavior — still
        # gives us user_code2 / user_pronoun for boosts, even when the form
        # artist URL took over the genre signal).
        spotify_url = lead.get('spotify_url', '')
        spotify_base = spotify_url.split('?')[0].rstrip('/') if spotify_url else ''
        if spotify_base:
            for aid, adata in matcher._artists.items():
                cache_url = (adata.get('spotify_url') or '').split('?')[0].rstrip('/')
                if cache_url == spotify_base:
                    cached_artist_data = (aid, adata)
                    cached_genres = adata.get('genres', '')
                    if not artist_genre and cached_genres:
                        artist_genre = cached_genres
                        print(f"  Upload: registered-artist genres (cache) = {cached_genres[:120]}")
                    user_code2 = adata.get('code2', '')
                    if user_code2:
                        print(f"  User country: {user_code2}")
                    user_pronoun = adata.get('pronoun_title', '')
                    if user_pronoun:
                        print(f"  User pronoun: {user_pronoun}")
                    break

        if not cached_artist_data and spotify_base and not provided_artist_url:
            print(f"  Cache miss for {spotify_base} — trying Chartmetric lookup...")
            t_cm = time.time()
            cm_data = lookup_artist_by_spotify(spotify_base)
            t_cm = time.time() - t_cm
            if cm_data:
                print(f"  CM lookup: {cm_data['name']} — {cm_data['genres']} "
                      f"({cm_data['tier']}, {cm_data['listeners']:.0f} listeners) "
                      f"[{t_cm:.1f}s]")
                if not artist_genre and cm_data['genres']:
                    artist_genre = cm_data['genres']
                cm_meta = cm_data.get('_meta', {})
                if cm_meta and cm_meta.get('pronoun_title'):
                    user_pronoun = cm_meta['pronoun_title']
                    print(f"  User pronoun (CM): {user_pronoun}")
            else:
                print(f"  CM lookup: no result [{t_cm:.1f}s]")

        # The matcher's `genre_hint` is empty (pure sonic match) — the lane is
        # applied as a post-filter in_lane below. So `genre` doesn't need to
        # carry artist-soup data anymore; keep it as the dropdown value (the
        # heavy-weight lane signal). Match logging keeps "genre=" readable.
        genre = dropdown_genre

        # Find matches — get extra so we can filter by tier
        user_monthly = lead.get('monthly_listeners')
        # CM lookup can provide more accurate listener count. The form-provided
        # track artist (Priority 1) wins: when scanning a client's track, the
        # tier — and everything downstream of it (tier-filtered matches,
        # trajectory targets, pitch comparables, peer pools) — must be THEIR
        # tier, not the account's.
        if track_artist_cm and float(track_artist_cm.get('listeners') or 0) > 0:
            user_monthly = float(track_artist_cm['listeners'])
        elif cm_data and cm_data['listeners'] > 0:
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

        # Lane resolution — HEAVY WEIGHT on the dropdown (per user's request).
        # Dropdown family ONLY defines the lane. The artist's CM genres (from
        # form-provided URL or registered account) supplement the gate's CF
        # check but never widen the lane itself. When no dropdown was picked,
        # fall back to the artist's first two resolvable families (positional,
        # same approach as the URL path).
        if dropdown_genre:
            user_families = _genre_families(dropdown_genre)
        else:
            artist_parts = [g.strip() for g in (artist_genre or '').split(',') if g.strip()]
            seen, tags = set(), []
            for g in artist_parts:
                lg = g.lower()
                if lg in seen:
                    continue
                if _genre_families(g):
                    tags.append(g); seen.add(lg)
                    if len(tags) >= 2:
                        break
            user_families = _genre_families(*tags) if tags else set()

        print(f"  Upload lane (heavy-weight dropdown): {user_families} | "
              f"dropdown='{dropdown_genre}' | artist_genre='{artist_genre[:80]}'")

        # Canonical gate — shared single source of truth in track_matcher.py
        # (match_in_lane / in_lane_families). Rules: sparse-data drop,
        # cf-overlap with lane, EXCLUSIVE_FAMILIES foreign drop (incl. reggae;
        # 'electronic' deliberately NOT exclusive), electronic-subgenre
        # identity split, rock-cluster contamination drop for pure electronic-
        # subgenre lanes, primary-foreign drop with artist-soup bypass.
        in_lane = match_in_lane

        pre_filter_count = len(all_matches)
        lane_filtered = [m for m in all_matches if in_lane(m, user_families)]
        print(f"  Family filter (upload, new gate): {pre_filter_count} → {len(lane_filtered)} matches")

        # Safety valve (parity with the URL path's track∪artist relax): if the
        # dropdown lane starves the pool, widen to dropdown ∪ artist families.
        # Widen-only — cannot change results when the lane is healthy.
        MIN_AFTER_LANE_FILTER = 25
        _av_parts = [g.strip() for g in (artist_genre or '').split(',') if g.strip()]
        artist_families = _genre_families(*_av_parts) if _av_parts else set()
        if len(lane_filtered) < MIN_AFTER_LANE_FILTER and (artist_families - user_families):
            relaxed = user_families | artist_families
            lane_filtered = [m for m in all_matches if in_lane(m, relaxed)]
            print(f"  Upload lane relax: dropdown lane starved (<{MIN_AFTER_LANE_FILTER}); "
                  f"relaxed to dropdown∪artist → {len(lane_filtered)}")
        all_matches = lane_filtered

        # Country boost: same-region artists get a small boost. Targets the
        # SCANNED artist's market when a form artist URL was provided (client
        # scans), falling back to the account's country.
        COUNTRY_BOOST = 0.02
        boost_code2 = None
        if track_artist_cm:
            boost_code2 = (track_artist_cm.get('_meta') or {}).get('code2')
        boost_code2 = boost_code2 or user_code2
        if boost_code2:
            boosted_count = 0
            for m in all_matches:
                match_code2 = m.get('code2', '')
                if match_code2 and match_code2 == boost_code2:
                    m['similarity'] = m.get('similarity', 0) + COUNTRY_BOOST
                    boosted_count += 1
            if boosted_count > 0:
                all_matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                print(f"  Country boost: {boosted_count} matches from {boost_code2} boosted +{COUNTRY_BOOST*100:.0f}%")

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
            # Market banding: only demote foreign-market targets when the USER
            # is an Anglophone-market artist themselves (else it's not "foreign").
            user_non_native = _is_non_native_market(genre or '')
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

                # Same gate as Similar Artists (in_lane above) — keeps the
                # trajectory pool consistent with what's surfacing in the table.
                if not in_lane(m, user_families):
                    continue
                # Sparse-data drop already enforced by in_lane(); explicit sparse
                # check here keeps the trajectory-specific log readable.
                cand_genre_parts = []
                for field in ('primary_genre', 'secondary_genre'):
                    g = (m.get(field) or '').strip()
                    if g:
                        cand_genre_parts.append(g)
                for g in m.get('artist_genres', []):
                    if g:
                        cand_genre_parts.append(g)
                cand_families = _genre_families(*cand_genre_parts)

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
                # Slight market penalty: nudge foreign-market targets down so they
                # interleave with same-market peers instead of stacking on top.
                nn_penalty = NON_NATIVE_TRAJECTORY_PENALTY if (not user_non_native and _is_non_native_market(*cand_genre_parts)) else 0.0
                score = m.get('similarity', 0) + total_boost - nn_penalty
                flattery_candidates.append((cand_tier_num, score, m, cand_pronoun))

            # Sort by tier (highest first), then market-weighted sonic similarity.
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

        # Priority chain for user_profile data:
        #   1. Form-provided Artist Spotify URL — the actual artist of the
        #      uploaded track. Wins over registered account when the user
        #      is analyzing someone else's release. Without this, Where You
        #      Stand / Originality / Pitch Comparables / Revenue / Save Rate
        #      all skip rendering because user_profile stays None.
        #   2. Registered account data from cache.
        #   3. CM live lookup of registered URL.
        if track_artist_cm:
            u_listeners = float(track_artist_cm.get('listeners') or 0) or u_listeners
            u_followers = float(track_artist_cm.get('followers') or 0)
            u_conversion = track_artist_cm.get('conversion_rate')
            print(f"  User profile (form-artist): {track_artist_cm.get('name')}, "
                  f"listeners={u_listeners}, followers={u_followers}, conversion={u_conversion}")
        elif cached_artist_data:
            aid, adata = cached_artist_data
            tier_data = matcher._tiers.get(str(aid), {})
            u_listeners = float(tier_data.get('listeners') or 0) or u_listeners
            u_followers = float(tier_data.get('followers') or 0)
            if u_listeners > 0 and u_followers > 0:
                u_conversion = round((u_followers * 0.1) / (u_listeners * 4.3) * 100, 2)
            print(f"  User profile (registered cache): {adata.get('name')}, "
                  f"listeners={u_listeners}, followers={u_followers}, conversion={u_conversion}")
        elif cm_data:
            u_listeners = float(cm_data['listeners'] or 0) or u_listeners
            u_followers = float(cm_data['followers'] or 0)
            u_conversion = cm_data.get('conversion_rate')
            print(f"  User profile (registered CM live): {cm_data['name']}, "
                  f"listeners={u_listeners}, followers={u_followers}, conversion={u_conversion}")
        else:
            print(f"  User profile: no source (form_artist={'yes' if provided_artist_url else 'no'}, "
                  f"registered_url={'yes' if spotify_base else 'no'}, "
                  f"monthly_listeners={u_listeners})")

        # Cap user conversion at 15% — same sanity bound applied to peer pool.
        # Prevents stale-follower outliers (e.g. 1.1K followers / 61 listeners)
        # from showing as "top 1%" when the underlying ratio is noise.
        if u_conversion is not None and u_conversion > 15.0:
            u_conversion = 15.0

        # Build conversion comparison vs matched artists (works with or without Spotify URL)
        MAX_REASONABLE_CONVERSION = 15.0
        # Raw followers/listener ratio — the industry-standard metric (Chartlex,
        # Chartmetric). See memory/reference_conversion_metrics_research.md for
        # benchmark sources. conversion_rate * (4.3/0.1) / 100 is mathematically
        # equivalent, but the raw ratio has documented bucketing thresholds.
        MAX_REASONABLE_RATIO = MAX_REASONABLE_CONVERSION / 2.326  # ≈ 6.45
        u_fol_listener_ratio = None
        if u_listeners > 0 and u_followers > 0:
            u_fol_listener_ratio = round(u_followers / u_listeners, 3)
        if u_fol_listener_ratio is not None and u_fol_listener_ratio > MAX_REASONABLE_RATIO:
            u_fol_listener_ratio = MAX_REASONABLE_RATIO

        if u_listeners > 0 and matches:
            match_conversions = [
                m['conversion_rate'] for m in matches
                if m.get('conversion_rate') is not None
                and 0 < m['conversion_rate'] <= MAX_REASONABLE_CONVERSION
            ]
            # Raw ratio peer pool — same artists, just expressed as fol/listener
            match_ratios = []
            for m in matches:
                ml = m.get('listeners') or 0
                mf = m.get('followers') or 0
                if ml > 0 and mf > 0:
                    r = mf / ml
                    if 0 < r <= MAX_REASONABLE_RATIO:
                        match_ratios.append(r)
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
                    'peer_pool_total': len(matches),
                }
                if match_ratios:
                    sorted_r = sorted(match_ratios)
                    rn = len(sorted_r)
                    conv_comparison['peer_ratio_median']    = round(sorted_r[rn // 2], 3)
                    conv_comparison['peer_ratio_top_25']    = round(sorted_r[int(rn * 0.75)], 3)
                    conv_comparison['peer_ratio_bottom_25'] = round(sorted_r[rn // 4], 3)
                    conv_comparison['peer_ratio_p99']       = round(sorted_r[min(rn - 1, int(rn * 0.99))], 3)

            # Fan gap: how many additional followers if you hit top 25%
            additional_fans = 0
            additional_revenue = 0
            peer_top_25 = conv_comparison.get('peer_top_25', 0)
            if peer_top_25 > 0 and u_listeners > 0:
                top25_followers_target = int(round((peer_top_25 / 100) * u_listeners * 4.3 / 0.1))
                current_followers = u_followers if u_followers > 0 else 0
                additional_fans = max(int(round(top25_followers_target - current_followers)), 0)
                additional_revenue = additional_fans * REVENUE_PER_FAN_PER_YEAR

            # Estimated save rate range from follower conversion (proxy)
            # Based on industry benchmarks: Playlist Push, LoudLab, Chartlex
            conv = u_conversion or 0.5
            if conv < 0.5:
                save_low, save_high = 2.0, 4.0
            elif conv < 1.0:
                save_low, save_high = 3.0, 5.0
            elif conv < 2.0:
                save_low, save_high = 4.0, 7.0
            elif conv < 3.0:
                save_low, save_high = 6.0, 10.0
            elif conv < 5.0:
                save_low, save_high = 8.0, 15.0
            else:
                save_low, save_high = 12.0, 20.0

            # Retention bucket from documented benchmarks (Chartlex):
            #   > 1.0   = stale-or-superstar (more followers than monthly listeners)
            #   0.1-1.0 = healthy retention band
            #   0.067-0.1 = marginal
            #   < 0.067 = shallow (width without depth)
            retention_bucket = None
            if u_fol_listener_ratio is not None:
                if u_fol_listener_ratio > 1.0:
                    retention_bucket = 'stale_or_superstar'
                elif u_fol_listener_ratio >= 0.1:
                    retention_bucket = 'healthy'
                elif u_fol_listener_ratio >= 0.067:
                    retention_bucket = 'marginal'
                else:
                    retention_bucket = 'shallow'

            # Track momentum: file-upload path doesn't have an ISRC for the scanned
            # track, so we can't do a track-level lookup here. Stays None — frontend
            # falls back to the artist-level Where You Stand display.
            user_profile = {
                'name': lead.get('name', 'Artist'),
                'listeners': u_listeners,
                'followers': u_followers,
                'conversion_rate': u_conversion,
                'fol_listener_ratio': u_fol_listener_ratio,
                'retention_bucket': retention_bucket,
                'estimated_save_rate_low': save_low,
                'estimated_save_rate_high': save_high,
                'conversion_comparison': conv_comparison,
                'additional_fans': additional_fans,
                'additional_revenue': additional_revenue,
                'track_momentum': None,
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

        # Generate recommendations + capture the cohort that produced them.
        # The same cohort feeds the sonic-originality computation below so the
        # originality × performance quadrant story stays internally consistent.
        recs, high_converter_gems = _generate_recommendations(features, matches,
                                          gems_by_artist=gems_by_artist,
                                          genre_alignment=genre_alignment,
                                          user_profile=user_profile)

        # Sonic originality — distance from cohort centroid in z-space.
        # Same cohort as production recs, flipped framing ("what makes you
        # distinct" vs "what to change to fit"). user_profile gets a
        # 'sonic_originality' and 'quadrant' block when available.
        sonic_originality = _compute_originality(features, high_converter_gems)
        if user_profile is not None and sonic_originality:
            user_profile['sonic_originality'] = sonic_originality
            tm = user_profile.get('track_momentum') or {}
            user_profile['quadrant'] = _classify_quadrant(
                sonic_originality.get('composite_score'),
                tm.get('composite_percentile'),
            )

        # Pitch comparables — A&R-ready list of same-tier, sonically similar
        # artists who are themselves in the Signature of Success quadrant.
        # Pool is `matches` (already tier-filtered + genre-family OK).
        if user_profile is not None:
            # Reuse the genre families computed for the matcher's family
            # filter — same source of truth (user's detected/declared genres).
            _file_upload_user_fams = _genre_families(genre or '')
            _file_upload_user_primary = _primary_genre_family(genre or '')
            user_profile['pitch_comparables'] = _compute_pitch_comparables(
                matches, high_converter_gems, matcher._gems_by_isrc,
                user_families=_file_upload_user_fams,
                user_primary_family=_file_upload_user_primary,
            )
            # Cohort scatter — every same-tier peer with both axes computed,
            # for the Sonic Quadrant background cloud.
            user_profile['cohort_scatter'] = _compute_cohort_scatter(
                matches, high_converter_gems, matcher._gems_by_isrc,
                user_families=_file_upload_user_fams,
                user_primary_family=_file_upload_user_primary,
            )

        # Signature production recommendations — same math as production recs
        # but against the signature-of-success peer subset (top 50% by
        # originality within the high-converter cohort). Surfaced as a
        # parallel list so the user can compare "fit-the-cohort" vs
        # "match the winners-who-also-deviate" framings.
        signature_recs = _generate_signature_recommendations(features, high_converter_gems)

        # Create background enrichment job
        job_id = job_mgr.create_job(token, features, matches, all_matches=matches)

        # Get user's CM artist ID for related-artists lookup (used by the
        # background enrichment job for the "related artists" reverse lookup).
        # Same priority chain as user_profile: form-provided artist URL wins
        # so enrichment uses the actual uploaded track's artist.
        user_cm_id = None
        if track_artist_cm:
            user_cm_id = track_artist_cm.get('cm_id')
        elif cm_data:
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
                # Whole-track BS.1770 (stereo) — display-only mastering readout;
                # upload path only, absent on Spotify scans (12s captures).
                'lufs_whole_track': features.get('lufs_whole_track'),
                'lra_whole_track': features.get('lra_whole_track'),
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
            'signature_recommendations': signature_recs,
            'recommendation_ranges': _generate_recommendation_ranges(features, high_converter_gems),
            'full_target_ranges': _generate_full_target_ranges(features, high_converter_gems),
            'genre_alignment': genre_alignment,
            'timing': {
                'feature_extraction_s': round(t_features, 2),
                'matching_s': round(t_match, 2),
            },
        }

        if user_profile:
            result['user_profile'] = user_profile

        # Include source info for artist card. The form-provided track artist
        # (Priority 1) OWNS the card when present — its CM lookup already
        # drives the lane genres and the user_profile numbers, so the card
        # must show the same artist (name, genres, tier), not the account.
        if track_artist_cm:
            _ta_genres = (track_artist_cm.get('genres') or '').strip()
            if _ta_genres.lower() in ('others', 'unknown'):
                _ta_genres = ''  # CM's untagged placeholder — show nothing, not junk
            result['source'] = {
                'type': 'file_upload',
                'artist_name': track_artist_cm.get('name', ''),
                'artist_genres': _ta_genres,
                'artist_tier': track_artist_cm.get('tier', '') or user_tier or '',
                'artist_listeners': float(track_artist_cm.get('listeners', 0) or 0),
            }
        elif cm_data:
            result['source'] = {
                'type': 'file_upload',
                'artist_name': cm_data.get('name', ''),
                'artist_genres': cm_data.get('genres', ''),
                'artist_tier': cm_data.get('tier', ''),
                'artist_listeners': cm_data.get('listeners', 0),
            }
        elif cached_artist_data:
            aid, adata = cached_artist_data
            result['source'] = {
                'type': 'file_upload',
                'artist_name': adata.get('name', ''),
                'artist_genres': adata.get('genres', ''),
                'artist_tier': user_tier or '',
                'artist_listeners': float(adata.get('listeners', 0) or 0),
            }

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

        _use_scan(lead)
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
    global _last_api_activity
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

        _no_subscriber_since = None  # Track when subscribers disappeared

        for batch_start in range(0, total, BATCH_SIZE):
            # Stop enrichment if no one is listening (tab closed) — 30s grace for reconnects
            if job_id not in sse_subscribers or not sse_subscribers[job_id]:
                if _no_subscriber_since is None:
                    _no_subscriber_since = time.time()
                    print(f"Enrichment [{job_id[:8]}]: No SSE subscribers — waiting 30s for reconnect")
                elif time.time() - _no_subscriber_since > 30:
                    print(f"Enrichment [{job_id[:8]}]: No SSE subscribers for 30s — stopping (tab closed)")
                    job_mgr.update_job(job_id, status='stale')
                    _notify_local_pipeline('user_idle')
                    return
            else:
                _no_subscriber_since = None

            # Keep activity alive so resource-switcher doesn't resume GEMS mid-enrichment
            _last_api_activity = time.time()

            # Brief pause if user-facing CM calls need priority (2s max to prevent blocking)
            if not _enrichment_gate.wait(timeout=2):
                _enrichment_gate.set()  # Force open — never block enrichment
                _enrichment_gate.set()
            # Refresh CM token each batch — prevents 401s when token expires mid-enrichment
            token = get_cm_token(refresh_token)
            if not token:
                print(f"Enrichment [{job_id[:8]}]: CM token refresh failed at batch {batch_start // BATCH_SIZE + 1}")
                break

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
                    # On 401, force token refresh and retry once
                    if '401' in str(e):
                        print(f"Enrichment [{job_id[:8]}]: 401 for {isrc} — refreshing CM token")
                        invalidate_cm_token()
                        token = get_cm_token(refresh_token)
                        if token:
                            try:
                                cm_track_id = _resolve_isrc_to_cm_track_id(token, isrc)
                                if cm_track_id:
                                    playlists = _fetch_track_playlists_structured(
                                        token, cm_track_id, isrc=isrc,
                                        artist_name=m.get('name', ''),
                                        track_name=m.get('track_name', ''),
                                    )
                                    print(f"Enrichment [{job_id[:8]}]:   -> retry OK, {len(playlists) if playlists else 0} playlists")
                                    if playlists:
                                        similarity = m.get('similarity', 0)
                                        conf_boost = 0.5 if match_key in confidence_map else 0.0
                                        for pl in playlists:
                                            pl['sonic_match'] = m.get('name', '')
                                            pl['track_name'] = m.get('track_name', '')
                                            pl['sonic_similarity'] = similarity
                                            pl['score'] = _compute_playlist_score(
                                                similarity, pl.get('followers', 0), conf_boost)
                                        batch_playlists.extend(playlists)
                            except Exception as retry_e:
                                print(f"Enrichment [{job_id[:8]}]: Retry also failed for {isrc}: {retry_e}")
                    else:
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
                    'sonic_match_pct': pl.get('sonic_similarity', 0.80),
                })

            if batch_curators:
                print(f"Enrichment [{job_id[:8]}]: Batch {batch_start//BATCH_SIZE+1} — resolving {len(batch_curators)} curators...")

            curators_checked = 0
            for curator_info in batch_curators:
                try:
                    curators_checked += 1

                    # Check for tab closed every 10 curators
                    if curators_checked % 10 == 0:
                        if job_id not in sse_subscribers or not sse_subscribers[job_id]:
                            if _no_subscriber_since is None:
                                _no_subscriber_since = time.time()
                            elif time.time() - _no_subscriber_since > 30:
                                print(f"Enrichment [{job_id[:8]}]: No SSE subscribers for 30s (during curator resolve) — stopping")
                                job_mgr.update_job(job_id, status='stale')
                                _notify_local_pipeline('user_idle')
                                return
                        else:
                            _no_subscriber_since = None

                    cm_cid = curator_info.get('cm_curator_id')
                    # Live status so UI doesn't look frozen
                    _sse_publish(job_id, 'enrichment_progress', {
                        'batch': batch_start // BATCH_SIZE + 1,
                        'total_batches': (total + BATCH_SIZE - 1) // BATCH_SIZE,
                        'curators_found': curator_count,
                        'checking': curator_info.get('name', ''),
                    })

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
            progress_data = {
                'batch': batch_num,
                'total_batches': total_batches,
                'curators_found': curator_count,
            }
            _sse_publish(job_id, 'enrichment_progress', progress_data)
            # Store for SSE catch-up on reconnect
            job_mgr.update_job(job_id, progress=progress_data)

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

        # Campaign Forecast — predict streams from playlist pitching
        if curator_count > 0:
            forecast_curators = []
            total_reach = 0
            total_expected_streams = 0

            for name, curator in (job_mgr.get_job_state(job_id) or {}).get('curator_emails', {}).items():
                followers = curator.get('followers', 0) or 0
                sonic_match = curator.get('sonic_match_pct', 0.80)

                # Base acceptance rate and cost by contact method
                if curator.get('submithub_url'):
                    base_rate = 0.12
                    method = 'SubmitHub'
                    cost = 2.0
                elif curator.get('groover_url'):
                    base_rate = 0.20
                    method = 'Groover'
                    cost = 2.0
                elif curator.get('email'):
                    base_rate = 0.07
                    method = 'Email'
                    cost = 0.0
                elif curator.get('submission_url'):
                    base_rate = 0.12
                    method = 'Submission'
                    cost = 0.0
                elif curator.get('instagram_url'):
                    base_rate = 0.04
                    method = 'Instagram'
                    cost = 0.0
                elif curator.get('facebook_url'):
                    base_rate = 0.04
                    method = 'Facebook'
                    cost = 0.0
                else:
                    base_rate = 0.05
                    method = 'Other'
                    cost = 0.0

                # Targeting bonus — our pitches come with sonic match + reference artist
                targeting_multiplier = min(2.5, 1.0 + max(0, (sonic_match - 0.70)) * 3)
                acceptance_rate = min(0.60, base_rate * targeting_multiplier)

                # Stream rate based on playlist activity
                # (We don't have freshness per curator easily, use conservative estimate)
                stream_rate = 0.03  # 3% of followers = streams

                expected_streams = followers * stream_rate * acceptance_rate
                total_reach += followers
                total_expected_streams += expected_streams

                forecast_curators.append({
                    'name': curator.get('name', name),
                    'playlist_name': curator.get('playlist_name', ''),
                    'followers': followers,
                    'method': method,
                    'cost': cost,
                    'acceptance_rate': round(acceptance_rate * 100, 1),
                    'expected_streams': round(expected_streams),
                })

            # Sort by expected streams descending (best playlists first)
            forecast_curators.sort(key=lambda x: x['expected_streams'], reverse=True)

            # Expected placements
            total_acceptance = sum(c['acceptance_rate'] / 100 for c in forecast_curators)
            placements_low = max(1, int(total_acceptance * 0.7))
            placements_high = max(placements_low, int(total_acceptance * 1.3))

            # Estimated playlist streams — based on which curators actually accept
            # Best case: your top N placements land (biggest playlists)
            # Worst case: your bottom N placements land (smallest playlists)
            # Each curator's streams = followers × 3% stream rate (already calculated)
            per_curator_streams = [c['expected_streams'] / max(c['acceptance_rate'] / 100, 0.01)
                                   for c in forecast_curators]
            # Sort descending for best case, ascending for worst case
            per_curator_streams_desc = sorted(per_curator_streams, reverse=True)
            per_curator_streams_asc = sorted(per_curator_streams)

            # Best case: top N placements by stream potential
            playlist_streams_high = int(sum(per_curator_streams_desc[:placements_high]))
            # Worst case: bottom N placements by stream potential
            playlist_streams_low = int(sum(per_curator_streams_asc[:placements_low]))

            # Algorithmic bonus (if save rate > 5%, 1.5-3x additional streams)
            algo_low = int(playlist_streams_low * 1.5)
            algo_high = int(playlist_streams_high * 3.0)

            # Total combined streams (playlist + algorithmic)
            total_streams_low = playlist_streams_low + algo_low
            total_streams_high = playlist_streams_high + algo_high

            # Revenue estimate ($0.004 per stream) — covers combined streams
            revenue_low = round(total_streams_low * 0.004, 2)
            revenue_high = round(total_streams_high * 0.004, 2)

            # New followers (0.1% of total combined streams)
            followers_low = max(1, int(total_streams_low * 0.001))
            followers_high = int(total_streams_high * 0.001)

            # Cost breakdown by method
            cost_by_method = {}
            for c in forecast_curators:
                m = c['method']
                if m not in cost_by_method:
                    cost_by_method[m] = {'count': 0, 'cost': 0}
                cost_by_method[m]['count'] += 1
                cost_by_method[m]['cost'] += c['cost']
            total_cost = sum(v['cost'] for v in cost_by_method.values())

            # Ensure low <= high for all ranges
            playlist_streams_low, playlist_streams_high = min(playlist_streams_low, playlist_streams_high), max(playlist_streams_low, playlist_streams_high)
            algo_low, algo_high = min(algo_low, algo_high), max(algo_low, algo_high)
            total_streams_low, total_streams_high = min(total_streams_low, total_streams_high), max(total_streams_low, total_streams_high)
            followers_low, followers_high = min(followers_low, followers_high), max(followers_low, followers_high)
            revenue_low, revenue_high = min(revenue_low, revenue_high), max(revenue_low, revenue_high)

            # Cost per stream (use midpoint of stream range)
            mid_streams = max((playlist_streams_low + playlist_streams_high) / 2, 1)
            cost_per_stream = round(total_cost / mid_streams, 4)

            # ROI
            net_roi_low = revenue_low - total_cost
            net_roi_high = revenue_high - total_cost

            forecast = {
                'curator_count': curator_count,
                'total_reach': total_reach,
                'placements_low': placements_low,
                'placements_high': placements_high,
                'streams_low': playlist_streams_low,
                'streams_high': playlist_streams_high,
                'total_streams_low': total_streams_low,
                'total_streams_high': total_streams_high,
                'algo_streams_low': algo_low,
                'algo_streams_high': algo_high,
                'new_followers_low': followers_low,
                'new_followers_high': followers_high,
                'revenue_low': revenue_low,
                'revenue_high': revenue_high,
                'total_cost': total_cost,
                'cost_by_method': cost_by_method,
                'cost_per_stream': cost_per_stream,
                'net_roi_low': net_roi_low,
                'net_roi_high': net_roi_high,
                'top_curators': forecast_curators[:5],
            }

            _sse_publish(job_id, 'campaign_forecast', forecast)
            job_mgr.update_job(job_id, campaign_forecast=forecast)

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
                if state.get('progress') and isinstance(state['progress'], dict) and state['progress'].get('batch'):
                    yield f"event: enrichment_progress\ndata: {json.dumps(state['progress'])}\n\n"
                if state.get('all_playlists'):
                    yield f"event: all_playlists\ndata: {json.dumps({'playlists': state['all_playlists'], 'total': len(state['all_playlists'])})}\n\n"
                if state.get('curator_emails'):
                    for name, info in state['curator_emails'].items():
                        yield f"event: curator_emails\ndata: {json.dumps({'curator': info})}\n\n"
                if state.get('campaign_forecast'):
                    yield f"event: campaign_forecast\ndata: {json.dumps(state['campaign_forecast'])}\n\n"
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
    lead = _validate_session(token)
    _check_scan_cap(lead)

    # Validate Spotify URL
    if 'open.spotify.com/track/' not in spotify_url and 'spotify:track:' not in spotify_url:
        raise HTTPException(400, "Please provide a valid Spotify track URL")

    # Extract track ID
    if 'spotify:track:' in spotify_url:
        track_id = spotify_url.split('spotify:track:')[1].split('?')[0]
    else:
        track_id = spotify_url.split('track/')[1].split('?')[0].split('/')[0]

    job_id = None  # Created later only if Mac worker is needed

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

    # Try primary Spotify credentials, fall back to backup on 429
    sp_creds = [(sp_token, sp_secret)]
    sp_backup_id = os.getenv('SPOTIFY_CLIENT_ID_BACKUP')
    sp_backup_secret = os.getenv('SPOTIFY_CLIENT_SECRET_BACKUP')
    if sp_backup_id and sp_backup_secret:
        sp_creds.append((sp_backup_id, sp_backup_secret))

    for cred_id, cred_secret in sp_creds:
        if not cred_id or not cred_secret:
            continue
        try:
            auth_resp = requests.post(
                'https://accounts.spotify.com/api/token',
                data={'grant_type': 'client_credentials'},
                auth=(cred_id, cred_secret),
                timeout=10,
            )
            if auth_resp.status_code == 200:
                sp_bearer = auth_resp.json()['access_token']
                track_resp = requests.get(
                    f'https://api.spotify.com/v1/tracks/{track_id}',
                    headers={'Authorization': f'Bearer {sp_bearer}'},
                    timeout=10,
                )
                if track_resp.status_code == 429:
                    print(f"  URL analysis: Spotify 429 on track lookup, trying backup credentials")
                    continue
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
                    break
        except Exception as e:
            print(f"Spotify API failed: {e}")

    # Look up track's artist in Chartmetric for genres + related artists
    # NOTE: Use track's artist for genre/CM ID, but keep user's own tier from registration
    # Pause enrichment so user-facing CM calls get priority
    _pause_enrichment()
    track_artist_cm_data = None
    dropdown_genre = (genre or '').strip()  # explicit user selection — strongest intent
    artist_genre = ''                       # artist-level CM genres (back-catalog union)
    if artist_spotify_url:
        print(f"  URL analysis: looking up track artist {artist_name} via CM...")
        track_artist_cm_data = lookup_artist_by_spotify(artist_spotify_url)
        if track_artist_cm_data:
            user_cm_id = track_artist_cm_data.get('cm_id')
            cm_genres = track_artist_cm_data.get('genres', '')
            if cm_genres:
                artist_genre = cm_genres
                # Sanity check: if user selected a genre from dropdown, only trust CM
                # genres if they share at least one family with the user's selection.
                # CM sometimes returns garbage genres (e.g. hip-hop for death metal artists).
                if dropdown_genre:
                    user_fams = _genre_families(dropdown_genre)
                    cm_fams = _genre_families(cm_genres)
                    if user_fams and cm_fams and not (user_fams & cm_fams):
                        # CM genres incompatible with the user's pick — distrust them.
                        cm_primary = cm_genres.split(',')[0].strip()
                        if _genre_families(cm_primary) & user_fams:
                            artist_genre = cm_primary
                            print(f"  URL analysis: CM genres suspect, artist genre = primary only: {artist_genre}")
                        else:
                            artist_genre = dropdown_genre
                            print(f"  URL analysis: CM genres suspect ({cm_genres}), artist genre = dropdown: {artist_genre}")
                    else:
                        print(f"  URL analysis: artist genres = {cm_genres}")
                else:
                    print(f"  URL analysis: artist genres = {cm_genres} (no dropdown set)")
            # Only set listeners if the user doesn't already have them from registration
            if not lead.get('monthly_listeners') and track_artist_cm_data.get('listeners'):
                lead['monthly_listeners'] = track_artist_cm_data['listeners']
            print(f"  URL analysis: {track_artist_cm_data['name']} — {track_artist_cm_data.get('tier', '?')} tier, "
                  f"{track_artist_cm_data.get('listeners', 0):.0f} listeners")
        else:
            print(f"  URL analysis: CM lookup returned nothing for {artist_spotify_url}")

    # Track-level genre — split into two sources for lane vs display:
    #
    # `track_genre` drives the LANE resolver — must be stable + deterministic
    # across re-scans, so we use gems_complete_analysis (a CM snapshot from
    # when the track was first analyzed). Same result every scan, no surprise
    # lane shifts. Fallback: universe tracks table → live CM (if gems missing).
    #
    # `track_genre_display` is what the UI renders — we want this to reflect
    # *current* CM tags, since CM grows its taxonomy over time (e.g. Mr Twin
    # Sister's "Meet the Frownies" gems stored just 2 tags from years ago;
    # CM now has 44). So we hit live CM whenever the gems snapshot is sparse
    # (<3 tags), and overlay the richer list for display only. Lane logic
    # is unaffected.
    track_genre = ''
    track_genre_display = ''
    if track_isrc:
        _tg_refresh = os.getenv('REFRESH_TOKEN')
        _tg_tok = get_cm_token(_tg_refresh) if _tg_refresh else None
        gems_row = matcher._gems_by_isrc.get(track_isrc) or {}
        if gems_row:
            parts = []
            p = (gems_row.get('primary_genre') or '').strip()
            s = (gems_row.get('secondary_genre') or '').strip()
            if p:
                parts.append(p)
            if s:
                parts.append(s)
            track_genre = ', '.join(parts)
            if track_genre:
                print(f"  URL analysis: track genres (gems) = {track_genre[:120]}")
        if not track_genre:
            universe_row = matcher._tracks.get(track_isrc) or {}
            if (universe_row.get('track_genres') or '').strip():
                track_genre = _resolve_genre_ids(universe_row['track_genres'].strip(), _tg_tok)
                print(f"  URL analysis: track genres (universe) = {track_genre or '(unresolved)'}")
        # Live CM call for display enrichment when the lane-side snapshot is
        # sparse (also serves as last-resort lane source when gems/universe
        # gave us nothing). One extra CM call per scan (~1s) — acceptable.
        lane_tag_count = len([g for g in (track_genre or '').split(',') if g.strip()])
        if _tg_tok and (lane_tag_count < 3):
            try:
                cm_live = lookup_track_genre(_tg_tok, track_isrc) or ''
                # CM's "Others" placeholder is uninformative — drop it.
                if cm_live and cm_live.strip().lower() != 'others':
                    track_genre_display = cm_live
                    print(f"  URL analysis: track genres (CM live, display) = {cm_live[:120]}")
                if not track_genre and cm_live:
                    track_genre = cm_live  # use CM as last-resort lane source too
            except Exception as e:
                print(f"  URL analysis: track genre fetch failed for {track_isrc}: {e}")
        if not track_genre_display:
            track_genre_display = track_genre

    # Matching genre signal (the if-statement): explicit dropdown wins; else the
    # track's own genre IF it resolves to a real family; else fall back to the
    # artist's FIRST 1-2 genres (their dominant lane) — never the whole
    # back-catalog union, and never an empty lane (an empty lane would disable
    # the family filter entirely, which floods results across all genres).
    # Lane = the user's first TWO resolvable tags (CM positions 1 + 2 =
    # primary + first secondary, the two positions CM explicitly tags). Both
    # treated as equally authoritative identity signals. Position 3+ is left
    # out of the gate — CM doesn't document that ordering, and our policy is
    # "use the reliable positions, lean on sonic for the rest."
    #
    # When the CM track soup is absurdly long (>5 tags) we sanity-check against
    # the artist genres first — only tags BOTH sides agree on carry signal.
    if dropdown_genre:
        genre = dropdown_genre
    else:
        track_parts = [g.strip() for g in (track_genre or '').split(',') if g.strip()]
        artist_parts = [g.strip() for g in (artist_genre or '').split(',') if g.strip()]
        if len(track_parts) > 5 and artist_parts:
            artist_lc = {g.lower() for g in artist_parts}
            candidate = [g for g in track_parts if g.lower() in artist_lc]
        else:
            candidate = track_parts
        seen, tags = set(), []
        for g in candidate:
            lg = g.lower()
            if lg in seen:
                continue
            if _genre_families(g):
                tags.append(g); seen.add(lg)
                if len(tags) >= 2:
                    break
        # If we don't have 2 yet, supplement from the artist tags (still
        # CM-ordered: position 1 = artist primary, etc.).
        if len(tags) < 2:
            for g in artist_parts:
                lg = g.lower()
                if lg in seen:
                    continue
                if _genre_families(g):
                    tags.append(g); seen.add(lg)
                    if len(tags) >= 2:
                        break
        # Umbrella deepening: if our positions 1+2 resolve to an umbrella family
        # ONLY (currently just 'electronic' — the family that covers dance,
        # edm, house, techno, dubstep, trance, idm... too broad to define a
        # lane), AND the artist's tag list has specific subgenres available
        # (jungle, dnb, garage, breaks, breakcore, etc.), promote those. This
        # handles brand-new releases where CM has no track-level tags yet so
        # the gems pipeline fell back to the artist's umbrella primary (e.g.
        # Paradox: artist primary='dance' → gems-track='dance, electronic' →
        # lane would be {electronic} = flood. Deepen to 'drum & bass' +
        # 'jungle' from his position-3+ tags so we land on {dnb, jungle}).
        UMBRELLA_FAMILIES = {'electronic'}
        tag_fams = _genre_families(*tags) if tags else set()
        if tag_fams and tag_fams.issubset(UMBRELLA_FAMILIES):
            specific = []
            for g in artist_parts:
                lg = g.lower()
                fams = _genre_families(g)
                if not fams or fams.issubset(UMBRELLA_FAMILIES):
                    continue
                # de-dup against existing tags AND prior specific picks
                if lg in seen or any(t.lower() == lg for t in specific):
                    continue
                specific.append(g)
                if len(specific) >= 2:
                    break
            if specific:
                tags = specific
                print(f"  URL analysis: umbrella-only lane deepened to specific: {tags}")
        # Final fallback: first raw artist tag (very rare — no resolvable tags
        # anywhere). Avoids empty lane = filter-disabled flood.
        genre = ', '.join(tags) if tags else (artist_parts[0] if artist_parts else '')
    print(f"  URL analysis: match genre='{genre}' | track='{track_genre}' | artist='{artist_genre}' | dropdown='{dropdown_genre}'")

    # Always scan fresh via Mac worker — Spotify desktop playback through Loopback
    # No cached GEMS features, no preview URLs — full quality capture only
    features = None

    # Create job for Mac worker — insert directly as pending_features with spotify_url
    if not features:
        job_id = str(__import__('uuid').uuid4())
        now = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()

        # Check queue position before creating job
        queue_ahead = 0
        if job_mgr._supabase:
            try:
                q = job_mgr._supabase.table('analysis_jobs').select('id').in_(
                    'status', ['pending_features', 'capturing']
                ).execute()
                queue_ahead = len(q.data) if q.data else 0
            except Exception:
                pass
            if queue_ahead > 0:
                print(f"  URL analysis: {queue_ahead} job(s) ahead in queue")

            try:
                job_mgr._supabase.table('analysis_jobs').insert({
                    'id': job_id, 'token': token, 'status': 'pending_features',
                    'spotify_url': spotify_url,
                    'features': '{}', 'matches': '[]', 'playlists': '{}',
                    'related_artists': '[]', 'credits': '{}', 'curator_emails': '{}',
                    'confidence_map': '{}', 'progress': '{}',
                    'created_at': now, 'updated_at': now,
                }).execute()
                print(f"  URL analysis: created pending_features job {job_id[:8]}")
            except Exception as e:
                print(f"  URL analysis: job create failed: {e}")
        job_mgr._mem[job_id] = {'id': job_id, 'status': 'pending_features', 'spotify_url': spotify_url}
        # CRITICAL: Pause local scripts BEFORE Mac worker captures audio
        # GEMS uses Spotify playback — if it's running it will contaminate the capture
        global _last_api_activity
        _last_api_activity = time.time()
        _notify_local_pipeline('user_active')

        # Longer timeout if there's a queue (150s base + 90s per job ahead)
        timeout = 150 + (queue_ahead * 90)
        print(f"  URL analysis: waiting for Mac worker (up to {timeout}s, {queue_ahead} ahead)...")
        deadline = time.time() + timeout

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

        def _check_queue_position():
            """Check how many jobs are ahead of ours in the queue."""
            if not job_mgr._supabase:
                return 0
            try:
                q = job_mgr._supabase.table('analysis_jobs').select('id,created_at').in_(
                    'status', ['pending_features', 'capturing']
                ).lt('created_at', now).execute()
                return len(q.data) if q.data else 0
            except Exception:
                return 0

        loop = asyncio.get_event_loop()
        while time.time() < deadline:
            result = await loop.run_in_executor(None, _poll_supabase_for_features)
            if result:
                features = result
                print(f"  URL analysis: features received from Mac worker")
                break
            await asyncio.sleep(3)

    if not features:
        if job_id:
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

    all_found = matcher.find_matches(
        features, genre_hint=genre or '', artist_genre_hint=artist_genre or '',
        top_n=fetch_n, threshold=0.55,
    )

    # Track-to-track genre family filter. The user side is the TRACK's OWN genre
    # (the tightening) — never the artist's back-catalog union, which is what used
    # to let an alternative track inherit a country/reggae lane. The candidate side
    # still checks each candidate's track AND artist genres, so an off-lane act
    # can't slip through on sparse track tags.
    track_user_families = _genre_families(genre or '')
    artist_user_families = _genre_families(artist_genre or '')
    # Kept broad (track ∪ artist) for the looser flattery pass downstream.
    user_families = track_user_families | artist_user_families

    # Canonical gate — shared single source of truth in track_matcher.py
    # (match_in_lane / in_lane_families). Rules: sparse-data drop, cf-overlap
    # with lane, EXCLUSIVE_FAMILIES foreign drop (incl. reggae; 'electronic'
    # deliberately NOT exclusive — it's an umbrella over dnb/garage/etc.),
    # electronic-subgenre identity split (jungle ≠ dnb ≠ breakcore),
    # rock-cluster contamination drop for pure electronic-subgenre lanes,
    # primary-foreign drop with artist-soup bypass.
    in_lane = match_in_lane

    # Exclude self-matches (the artist being analyzed)
    exclude_artist_id = None
    if track_artist_cm_data:
        exclude_artist_id = str(track_artist_cm_data.get('cm_id', ''))
    if exclude_artist_id:
        all_found = [m for m in all_found if str(m.get('artist_id', '')) != exclude_artist_id]

    # Save unfiltered matches for flattery (uses looser filtering)
    all_matches_unfiltered = list(all_found)

    # Primary gate: keep candidates whose genre overlaps the user's TRACK lane.
    track_filtered = [m for m in all_found if in_lane(m, track_user_families)]
    # Safety relax — if the track lane starves the pool, widen to the artist
    # family union so the user never gets a near-empty result.
    MIN_AFTER_TRACK_FILTER = 25
    if (track_user_families and artist_user_families
            and len(track_filtered) < MIN_AFTER_TRACK_FILTER):
        relaxed = track_user_families | artist_user_families
        all_found = [m for m in all_found if in_lane(m, relaxed)]
        print(f"  Family filter (track-to-track): track lane kept {len(track_filtered)} "
              f"(<{MIN_AFTER_TRACK_FILTER}); relaxed to track∪artist → {len(all_found)}")
    else:
        all_found = track_filtered
        print(f"  Family filter (track-to-track overlap): {len(all_matches_unfiltered)} → {len(all_found)} matches")

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
    recs, high_converter_gems_url = _generate_recommendations(features, found_matches,
                                      gems_by_artist=gems_by_artist)
    # Stash for the originality pass below (user_profile is built later in this path)
    _url_sonic_originality = _compute_originality(features, high_converter_gems_url)

    # Flattery matches — higher-tier artists from unfiltered pool (before genre filter)
    flattery_matches = []
    if user_tier and all_matches_unfiltered:
        tier_order_map = {t: i for i, t in enumerate(TIER_RANGES.keys())}
        user_tier_num = tier_order_map.get(user_tier, 0)
        # Market banding: demote foreign-market targets to a second band, but
        # only when the user is an Anglophone-market artist themselves.
        user_non_native = _is_non_native_market(artist_genre or '', track_genre or '', genre or '')
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
            # Scope trajectory targets to the TRACK lane (same as Similar Artists),
            # not the broad artist union — the union carried the artist's
            # back-catalog reggae/country, which is exactly what was bleeding a
            # reggae act into the top trajectory slot for an alternative track.
            # Canonical gate (track_matcher.match_in_lane) — same rules and
            # same candidate extraction as the Similar Artists table, so the
            # trajectory pool can never drift from what the table shows.
            if track_user_families:
                if not match_in_lane(m, track_user_families):
                    continue
                shared = candidate_lane_families(m) & track_user_families
                total_boost += 0.05 * len(shared)
            cand_pronoun = m.get('pronoun_title', 'They')
            # Slight market penalty: nudge foreign-market targets down so they
            # interleave with same-market peers instead of stacking on top.
            nn_penalty = NON_NATIVE_TRAJECTORY_PENALTY if (not user_non_native and _cand_non_native(m)) else 0.0
            score = m.get('similarity', 0) + total_boost - nn_penalty
            flattery_candidates.append((cand_tier_num, score, m, cand_pronoun))

        # Sort by tier (highest first), then market-weighted sonic similarity.
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

    # Cap user conversion at 15% — same sanity bound applied to peer pool.
    # Prevents stale-follower outliers (e.g. 1.1K followers / 61 listeners)
    # from showing as "top 1%" when the underlying ratio is noise.
    if u_conversion is not None and u_conversion > 15.0:
        u_conversion = 15.0

    MAX_REASONABLE_CONVERSION = 15.0
    MAX_REASONABLE_RATIO = MAX_REASONABLE_CONVERSION / 2.326  # ≈ 6.45
    u_fol_listener_ratio = None
    if u_listeners > 0 and u_followers > 0:
        u_fol_listener_ratio = round(u_followers / u_listeners, 3)
    if u_fol_listener_ratio is not None and u_fol_listener_ratio > MAX_REASONABLE_RATIO:
        u_fol_listener_ratio = MAX_REASONABLE_RATIO

    if u_listeners > 0 and all_found:
        # Filter out anomalous conversion rates (retired artists, bad data)
        match_conversions = [
            m['conversion_rate'] for m in all_found
            if m.get('conversion_rate') is not None
            and 0 < m['conversion_rate'] <= MAX_REASONABLE_CONVERSION
        ]
        # Raw ratio peer pool — see memory/reference_conversion_metrics_research.md
        match_ratios = []
        for m in all_found:
            ml = m.get('listeners') or 0
            mf = m.get('followers') or 0
            if ml > 0 and mf > 0:
                r = mf / ml
                if 0 < r <= MAX_REASONABLE_RATIO:
                    match_ratios.append(r)
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
                'peer_pool_total': len(all_found),
            }
            if match_ratios:
                sorted_r = sorted(match_ratios)
                rn = len(sorted_r)
                conv_comparison['peer_ratio_median']    = round(sorted_r[rn // 2], 3)
                conv_comparison['peer_ratio_top_25']    = round(sorted_r[int(rn * 0.75)], 3)
                conv_comparison['peer_ratio_bottom_25'] = round(sorted_r[rn // 4], 3)
                conv_comparison['peer_ratio_p99']       = round(sorted_r[min(rn - 1, int(rn * 0.99))], 3)

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
            additional_revenue = additional_fans * REVENUE_PER_FAN_PER_YEAR

        # Estimated save rate range from follower conversion (proxy)
        conv = u_conversion or 0.5
        if conv < 0.5:
            save_low, save_high = 2.0, 4.0
        elif conv < 1.0:
            save_low, save_high = 3.0, 5.0
        elif conv < 2.0:
            save_low, save_high = 4.0, 7.0
        elif conv < 3.0:
            save_low, save_high = 6.0, 10.0
        elif conv < 5.0:
            save_low, save_high = 8.0, 15.0
        else:
            save_low, save_high = 12.0, 20.0

        # Retention bucket from documented benchmarks (Chartlex):
        #   > 1.0   = stale-or-superstar (more followers than monthly listeners)
        #   0.1-1.0 = healthy retention band
        #   0.067-0.1 = marginal
        #   < 0.067 = shallow (width without depth)
        retention_bucket = None
        if u_fol_listener_ratio is not None:
            if u_fol_listener_ratio > 1.0:
                retention_bucket = 'stale_or_superstar'
            elif u_fol_listener_ratio >= 0.1:
                retention_bucket = 'healthy'
            elif u_fol_listener_ratio >= 0.067:
                retention_bucket = 'marginal'
            else:
                retention_bucket = 'shallow'

        # Track-level momentum: look up the scanned track's own popularity / cm_score /
        # playlist counts. Cache-first; on miss, fetch live from Spotify + Chartmetric
        # so the panel works for tracks not yet in our universe.
        track_momentum = None
        scanned_track_row = matcher._tracks.get(track_isrc) if track_isrc else None
        if not scanned_track_row and track_isrc and track_id:
            # On-demand fetch — 1 Spotify call + ~3 CM calls, ~3-5s added latency.
            # Result is used for this scan only (not cached back to universe yet).
            try:
                cm_refresh = os.getenv('REFRESH_TOKEN')
                cm_tok = get_cm_token(cm_refresh) if cm_refresh else None
                fetched = fetch_track_momentum(cm_tok, track_id, track_isrc) if cm_tok else None
                if fetched:
                    scanned_track_row = fetched
                    print(f"  Track momentum: live-fetched for {track_isrc} (not in universe cache)")
            except Exception as e:
                print(f"  Track momentum: live fetch failed for {track_isrc}: {e}")
        if scanned_track_row and all_found:
            track_momentum = _build_track_momentum(scanned_track_row, all_found, u_listeners)

        # Sonic originality + quadrant — uses the cohort _generate_recommendations
        # already computed above (high_converter_gems_url stashed earlier).
        quadrant = None
        if _url_sonic_originality and track_momentum:
            quadrant = _classify_quadrant(
                _url_sonic_originality.get('composite_score'),
                track_momentum.get('composite_percentile'),
            )

        # Pitch comparables — A&R-ready list. Pool is found_matches (same tier,
        # already genre-family-filtered by the matcher). Pass user_families +
        # primary family so the alignment + primary-share filters can scope
        # to the user's dominant lane (catches hybrid-vs-hybrid false positives).
        _url_user_fams = _genre_families(genre or '')
        _url_user_primary = _primary_genre_family(genre or '')
        pitch_comparables = _compute_pitch_comparables(
            found_matches, high_converter_gems_url, matcher._gems_by_isrc,
            user_families=_url_user_fams,
            user_primary_family=_url_user_primary,
        )
        # Cohort scatter for the Sonic Quadrant background cloud
        cohort_scatter = _compute_cohort_scatter(
            found_matches, high_converter_gems_url, matcher._gems_by_isrc,
            user_families=_url_user_fams,
            user_primary_family=_url_user_primary,
        )

        user_profile = {
            'name': lead.get('name', 'Artist'),
            'listeners': u_listeners,
            'followers': u_followers,
            'conversion_rate': u_conversion,
            'fol_listener_ratio': u_fol_listener_ratio,
            'retention_bucket': retention_bucket,
            'estimated_save_rate_low': save_low,
            'estimated_save_rate_high': save_high,
            'conversion_comparison': conv_comparison,
            'additional_fans': additional_fans,
            'additional_revenue': additional_revenue,
            'track_momentum': track_momentum,
            'sonic_originality': _url_sonic_originality,
            'quadrant': quadrant,
            'pitch_comparables': pitch_comparables,
            'cohort_scatter': cohort_scatter,
        }
        if track_momentum:
            print(f"  Track momentum: pop={track_momentum['scanned_popularity']} (p{int((track_momentum['percentile_popularity'] or 0)*100)}), cm={track_momentum['scanned_cm_score']}, pl={track_momentum['scanned_playlists']}, composite=p{int(track_momentum['composite_percentile']*100)}, gap=${track_momentum['gap_additional_revenue']}/yr")
        if _url_sonic_originality:
            print(f"  Sonic originality: score={_url_sonic_originality['composite_score']}/100, distance={_url_sonic_originality['distance']}, top_deviations={[d['feature']+' '+('+' if d['z']>0 else '')+str(d['z']) for d in _url_sonic_originality['top_deviations'][:3]]}, quadrant={quadrant['quadrant'] if quadrant else 'unclassified'}")
        print(f"  User profile built: conversion={u_conversion}, ratio={u_fol_listener_ratio}, bucket={retention_bucket}, est_save={save_low}-{save_high}%, fans_gap={additional_fans}, peers={conv_comparison.get('peer_count', 0)}")

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
            # Whole-track Integrated ESTIMATE from the worker's 3 stereo probes
            # (gated power-mean; backtested MAE ~0.5 dB). Absent on old jobs.
            'lufs_integrated_est': features.get('lufs_integrated_est'),
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
        'signature_recommendations': _generate_signature_recommendations(features, high_converter_gems_url),
        'recommendation_ranges': _generate_recommendation_ranges(features, high_converter_gems_url),
        'full_target_ranges': _generate_full_target_ranges(features, high_converter_gems_url),
        'source': {
            'type': 'spotify_url',
            'track_name': track_name,
            'artist_name': artist_name,
            # Both genre lanes, exposed separately so the UI can color-code them
            # (track = green, artist = white) and the user can verify the split.
            # Display uses the richer CM-live track tags when available (gems
            # snapshot can be a stale subset). Lane resolution upstream uses
            # `track_genre` (gems-first) — display divergence is intentional.
            'track_genres': track_genre_display or track_genre or '',
            'artist_genres': artist_genre or '',
            'match_genre': genre or '',   # what actually drove the match (dropdown > track > artist)
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

    # Send email (non-blocking — don't fail the request if email fails)
    try:
        send_results_email(lead['name'], lead['email'], result)
    except Exception as e:
        print(f"Email send error (non-fatal): {e}")

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

    _use_scan(lead)
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
    _t0 = time.time()
    artist_data = lookup_artist_by_spotify(spotify_url)
    print(f"[TIMING] artist fetch (chartmetric): {time.time()-_t0:.1f}s")
    if not artist_data:
        raise HTTPException(404, "Artist not found on Spotify/Chartmetric")

    listeners = artist_data.get('listeners', 0) or 0
    followers = artist_data.get('followers', 0) or 0
    tier = artist_data.get('tier') or _listeners_to_tier(int(listeners))
    conversion_rate = artist_data.get('conversion_rate', 0) or 0
    # Cap at 15% — same sanity bound the analyzer applies — so stale-follower
    # outliers (more followers than listeners) don't show an inflated rate.
    if conversion_rate > 15.0:
        conversion_rate = 15.0

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

    _artist_cm_id = artist_data.get('cm_id')

    # 1) Newest track: try cached GEMS features by ISRC (proven path).
    if top_track and top_track.get('isrc'):
        isrc = top_track['isrc']
        features = _lookup_gems_features(isrc)
        if features:
            print(f"Deal lookup: using cached GEMS features for newest-track ISRC {isrc}")
        else:
            print(f"Deal lookup: newest-track ISRC {isrc} not cached")

    # 2) Cache-first fallback: if the newest track isn't cached, reuse ANY already-
    #    analyzed track by this artist instead of a slow live scan. A cached track is
    #    still the artist's real music — a valid basis for the production-gap recs.
    #    Find the most representative cached track in-memory (instant), then load its
    #    canonical features via the same proven path.
    if not features and _artist_cm_id and matcher._gems_list:
        aid = str(_artist_cm_id)
        candidates = sorted(
            [g for g in matcher._gems_list
             if str(g.get('artist_id', '')) == aid and g.get('isrc')],
            key=lambda g: g.get('sp_track_popularity') or 0, reverse=True,
        )
        for g in candidates:
            features = _lookup_gems_features(g['isrc'])
            if features:
                print(f"Deal lookup: newest track uncached — using cached track "
                      f"{g['isrc']} for artist {aid} ({len(candidates)} cached candidates)")
                break

    # 3) No cached tracks for this artist (truly new/unknown): skip the slow live
    #    scan and stay instant. Qualification + conversion gap still render; sonic
    #    production recs are simply omitted for these rare cases ("skip recs, stay
    #    instant" — the deal calc's job is qualification, not the scan).
    if not features:
        print(f"Deal lookup: no cached features for artist {_artist_cm_id} — "
              f"skipping live scan (instant), no sonic recs this run")

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
            _tfm = time.time()
            all_matches = matcher.find_matches(features, genre_hint=genres_str, top_n=5000, threshold=0.55)
            print(f"[TIMING] find_matches: {time.time()-_tfm:.1f}s ({len(all_matches)} matches)")

            # Canonical genre gate (track_matcher.match_in_lane) — same rules as
            # Similar Artists / trajectory / email refs. Without this, the sonic
            # gap consensus and conversion-opportunity peers were derived from
            # ANY high-converting sound-alike regardless of genre (a country
            # artist could get production advice computed from hip-hop peers).
            # Lane: track tags first (gems primary/secondary, present when
            # features came from cache), artist first-2 resolvable tags as
            # fallback, electronic umbrella deepening — mirrors the URL path.
            deal_lane = set()
            track_tags = [t for t in (features.get('primary_genre'),
                                      features.get('secondary_genre')) if t]
            if track_tags:
                deal_lane = _genre_families(*track_tags)
            if not deal_lane and genres_str:
                seen_tags, lane_tags = set(), []
                for g in (p.strip() for p in genres_str.split(',') if p.strip()):
                    lg = g.lower()
                    if lg in seen_tags:
                        continue
                    if _genre_families(g):
                        lane_tags.append(g); seen_tags.add(lg)
                        if len(lane_tags) >= 2:
                            break
                deal_lane = _genre_families(*lane_tags) if lane_tags else set()
            if deal_lane == {'electronic'} and genres_str:
                from track_matcher import ELECTRONIC_SUBGENRES
                deep = _genre_families(genres_str) & ELECTRONIC_SUBGENRES
                if deep:
                    deal_lane = deal_lane | deep

            if deal_lane:
                in_lane_matches = [m for m in all_matches if match_in_lane(m, deal_lane)]
                # Relax if the lane starves the pool — sonic gap needs peers
                if len(in_lane_matches) >= 25:
                    print(f"Deal lookup: lane {deal_lane} kept {len(in_lane_matches)}/{len(all_matches)}")
                    all_matches = in_lane_matches
                else:
                    print(f"Deal lookup: lane {deal_lane} kept only {len(in_lane_matches)} — keeping ungated pool")

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
                        additional_revenue = additional_fans * REVENUE_PER_FAN_PER_YEAR

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
            additional_revenue = additional_fans * REVENUE_PER_FAN_PER_YEAR
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

    print(f"[TIMING] through sonic + platform: {time.time()-_t0:.1f}s")

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

    print(f"[TIMING] through history + events (total): {time.time()-_t0:.1f}s")

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
# Chat widget — visitor <-> owner-via-Telegram bridge
# ---------------------------------------------------------------------------
def _telegram_api(method: str, payload: dict):
    """Call the Telegram Bot API. Returns parsed JSON or None."""
    if not TELEGRAM_BOT_TOKEN:
        return None
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}",
            json=payload,
            timeout=10,
        )
        return r.json()
    except Exception as e:
        print(f"Telegram {method} error: {e}")
        return None


@app.post("/api/chat/send")
async def chat_send(data: dict):
    """Visitor sends a message: store it and forward to the owner's Telegram."""
    conv = (data.get('conversation_id') or '').strip()
    text = (data.get('text') or '').strip()
    contact = (data.get('contact') or '').strip() or None
    if not conv or not text:
        raise HTTPException(400, "conversation_id and text required")

    row_id = None
    if supabase:
        try:
            res = supabase.table('chat_messages').insert({
                'conversation_id': conv,
                'sender': 'visitor',
                'text': text,
                'visitor_contact': contact,
                'created_at': datetime.utcnow().isoformat(),
            }).execute()
            if res.data:
                row_id = res.data[0].get('id')
        except Exception as e:
            print(f"chat_send insert error: {e}")

    # Forward to the owner's Telegram with a conversation tag. The owner replies
    # to that Telegram message; the webhook maps the reply back to this thread.
    if TELEGRAM_BOT_TOKEN and TELEGRAM_OWNER_CHAT_ID:
        header = f"💬 [#{conv[:8]}]"
        if contact:
            header += f"  ·  {contact}"
        body = f"{header}\n{text}\n\n↩️ Reply to this message to respond."
        sent = _telegram_api('sendMessage', {
            'chat_id': TELEGRAM_OWNER_CHAT_ID,
            'text': body,
        })
        try:
            tg_msg_id = sent['result']['message_id'] if sent and sent.get('ok') else None
        except Exception:
            tg_msg_id = None
        # Record which Telegram message corresponds to this conversation so a
        # reply-to can be routed back to the right visitor.
        if tg_msg_id and supabase and row_id is not None:
            try:
                supabase.table('chat_messages').update({
                    'telegram_message_id': tg_msg_id,
                }).eq('id', row_id).execute()
            except Exception as e:
                print(f"chat_send tg-map error: {e}")

    return {"ok": True}


@app.get("/api/chat/poll")
async def chat_poll(conversation_id: str, after_id: int = 0):
    """Widget polls for new owner replies in this conversation."""
    if not supabase:
        return {"messages": []}
    try:
        res = (
            supabase.table('chat_messages')
            .select('id,text,created_at')
            .eq('conversation_id', conversation_id)
            .eq('sender', 'owner')
            .gt('id', after_id)
            .order('id')
            .execute()
        )
        return {"messages": res.data or []}
    except Exception as e:
        print(f"chat_poll error: {e}")
        return {"messages": []}


@app.post("/api/chat/telegram-webhook")
async def chat_telegram_webhook(update: dict, secret: str = ''):
    """Telegram calls this when the owner replies. Route the reply to the
    correct visitor thread via the replied-to message id."""
    if not TELEGRAM_WEBHOOK_SECRET or secret != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(403, "forbidden")

    msg = update.get('message') or {}
    text = (msg.get('text') or '').strip()
    reply_to = msg.get('reply_to_message') or {}
    reply_to_id = reply_to.get('message_id')

    # Ignore empty messages and bot commands (e.g. /start).
    if not text or text.startswith('/') or not supabase:
        return {"ok": True}

    try:
        conv = None
        # Precise routing: owner replied to a specific forwarded message.
        if reply_to_id:
            found = (
                supabase.table('chat_messages')
                .select('conversation_id')
                .eq('telegram_message_id', reply_to_id)
                .limit(1)
                .execute()
            )
            if found.data:
                conv = found.data[0]['conversation_id']
        # Fallback: a plain message with no reply-to (watch / notification quick
        # replies drop the reply context) routes to the most recent visitor thread.
        if not conv:
            recent = (
                supabase.table('chat_messages')
                .select('conversation_id')
                .eq('sender', 'visitor')
                .order('id', desc=True)
                .limit(1)
                .execute()
            )
            if recent.data:
                conv = recent.data[0]['conversation_id']
        if conv:
            supabase.table('chat_messages').insert({
                'conversation_id': conv,
                'sender': 'owner',
                'text': text,
                'created_at': datetime.utcnow().isoformat(),
            }).execute()
    except Exception as e:
        print(f"chat_webhook error: {e}")

    return {"ok": True}


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
        # no-cache => browser must revalidate index.html on every load, so a new
        # deploy (with bumped ?v= on JS/CSS) is picked up immediately instead of
        # serving a stale cached page. The versioned static assets stay cacheable.
        return FileResponse(str(index), headers={"Cache-Control": "no-cache"})
    return {"message": "Sonic Analyzer API is running. Put index.html in static/"}


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
