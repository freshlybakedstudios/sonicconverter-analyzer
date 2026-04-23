"""
Mac Worker Daemon — Local audio capture for Spotify URL analysis.

Uses the same approach as the GEMS pipeline:
  1. Play track via Spotify Web API
  2. Seek to 3 positions (25%, 50%, 75%)
  3. Record 4s samples from Loopback audio device via sounddevice
  4. Pick highest-energy sample
  5. Extract 30+ features with librosa + pyloudnorm

Polls Supabase `analysis_jobs` for jobs with status='pending_features'.

Usage:
    python3 mac_worker.py

Requires:
    - Loopback audio app running (or BlackHole)
    - Spotify desktop app running and logged in
    - SUPABASE_URL, SUPABASE_SERVICE_KEY, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET env vars
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import requests
import sounddevice as sd

from dotenv import load_dotenv

load_dotenv()

POLL_INTERVAL = 5  # seconds
SAMPLE_RATE = 48000
SAMPLE_DURATION = 4  # seconds per sample point
RETRY_WINDOW = 120  # seconds — retry failed jobs created within this window

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
def _supabase_headers():
    project_ref = SUPABASE_URL.split('//', 1)[1].split('.', 1)[0]
    return {
        'apikey': SUPABASE_KEY,
        'Authorization': f"Bearer {SUPABASE_KEY}",
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Prefer': 'return=representation',
        'sb-project-ref': project_ref,
    }


_last_poll_log = 0

def poll_pending_jobs():
    """Check for jobs with status='pending_features'."""
    global _last_poll_log
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/analysis_jobs?status=eq.pending_features&order=created_at.asc&limit=1",
            headers=_supabase_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            jobs = resp.json()
            if jobs:
                print(f"Found pending job: {jobs[0]['id'][:8]}")
                return jobs[0]
        else:
            print(f"Poll response: {resp.status_code}")
    except Exception as e:
        print(f"Poll error: {e}")
    # Log heartbeat every 60s so we know it's alive
    now = time.time()
    if now - _last_poll_log > 60:
        _last_poll_log = now
        print(f"Polling... (no pending jobs)")
    return None


def poll_retryable_jobs():
    """Check for recently failed jobs that should be retried.

    Picks up jobs with status error/capture_failed/extraction_failed
    that were created within RETRY_WINDOW seconds.
    """
    try:
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=RETRY_WINDOW)).isoformat()
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/analysis_jobs"
            f"?status=in.(error,capture_failed,extraction_failed)"
            f"&created_at=gte.{cutoff}"
            f"&order=created_at.asc&limit=1",
            headers=_supabase_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            jobs = resp.json()
            if jobs:
                print(f"Found retryable job: {jobs[0]['id'][:8]} (status={jobs[0]['status']})")
                return jobs[0]
    except Exception as e:
        print(f"Retry poll error: {e}")
    return None


def update_job(job_id: str, status: str, features: dict = None):
    """Update job status and features in Supabase."""
    payload = {'status': status}
    if features:
        payload['features'] = json.dumps(features)
    try:
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/analysis_jobs?id=eq.{job_id}",
            json=payload,
            headers=_supabase_headers(),
            timeout=10,
        )
    except Exception as e:
        print(f"Update error: {e}")


# ---------------------------------------------------------------------------
# Spotify Web API helpers
# ---------------------------------------------------------------------------
_spotify_bearer = None
_spotify_bearer_ts = 0
_using_backup = False


def _get_spotify_token(force_backup=False):
    """Get Spotify Web API bearer token. Falls back to backup credentials on 429."""
    global _spotify_bearer, _spotify_bearer_ts, _using_backup
    if _spotify_bearer and (time.time() - _spotify_bearer_ts) < 3500 and not force_backup:
        return _spotify_bearer

    if force_backup or _using_backup:
        client_id = os.getenv('SPOTIFY_CLIENT_ID_BACKUP')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET_BACKUP')
        refresh_token = os.getenv('SPOTIFY_REFRESH_TOKEN_BACKUP')
        label = 'backup'
    else:
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        refresh_token = os.getenv('SPOTIFY_REFRESH_TOKEN')
        label = 'primary'

    if refresh_token and client_id and client_secret:
        resp = requests.post(
            'https://accounts.spotify.com/api/token',
            data={
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': client_id,
                'client_secret': client_secret,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            _spotify_bearer = resp.json()['access_token']
            _spotify_bearer_ts = time.time()
            if force_backup and not _using_backup:
                _using_backup = True
                print(f"Switched to backup Spotify credentials")
            return _spotify_bearer
        print(f"Spotify refresh failed ({label}): {resp.status_code} {resp.text[:100]}")

    print("No SPOTIFY_REFRESH_TOKEN set — cannot control playback")
    return None


def _switch_to_backup():
    """Force switch to backup Spotify credentials."""
    global _spotify_bearer, _spotify_bearer_ts
    _spotify_bearer = None
    _spotify_bearer_ts = 0
    return _get_spotify_token(force_backup=True)


def _extract_track_id(url: str) -> str | None:
    """Extract Spotify track ID from URL."""
    if 'track/' in url:
        return url.split('track/')[-1].split('?')[0].split('/')[0]
    if 'spotify:track:' in url:
        return url.split('spotify:track:')[1].split('?')[0]
    return None


def _get_track_info(track_id: str) -> dict | None:
    """Get track metadata (duration, playability). Retries on 429."""
    token = _get_spotify_token()
    if not token:
        return None
    for attempt in range(5):
        resp = requests.get(
            f'https://api.spotify.com/v1/tracks/{track_id}',
            headers={'Authorization': f'Bearer {token}'},
            params={'market': 'US'},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            retry_after = int(resp.headers.get('Retry-After', 5))
            if retry_after > 60:
                # Long ban — switch to backup credentials
                print(f"Track info 429 — Retry-After {retry_after}s, switching to backup")
                token = _switch_to_backup()
                if token:
                    continue
            else:
                print(f"Track info 429 — waiting {retry_after}s (attempt {attempt+1}/5)")
                time.sleep(retry_after)
                continue
        print(f"Track info failed: {resp.status_code}")
        return None
    print(f"Track info failed: 429 after 5 retries")
    return None


def _ensure_device_active() -> str | None:
    """Make sure Spotify has an active device. Returns device_id or None."""
    token = _get_spotify_token()
    if not token:
        return None

    headers = {'Authorization': f'Bearer {token}'}

    def _get_devices():
        r = requests.get('https://api.spotify.com/v1/me/player/devices', headers=headers, timeout=10)
        return r.json().get('devices', []) if r.status_code == 200 else []

    devices = _get_devices()
    if not devices:
        print("No Spotify devices found — opening Spotify...")
        subprocess.run(['open', '-a', 'Spotify'], check=False)
        time.sleep(8)
        devices = _get_devices()

    if not devices:
        return None

    # Match GEMS pipeline device selection — prefer MacBook desktop app
    # This ensures audio routes through Loopback correctly
    for dev in devices:
        name = dev.get('name', '')
        if dev.get('is_active') or 'MacBook' in name:
            dev_id = dev['id']
            print(f"  Using device: {name} (id: {dev_id})")
            requests.put(
                'https://api.spotify.com/v1/me/player',
                headers={**headers, 'Content-Type': 'application/json'},
                json={'device_ids': [dev_id], 'play': False},
                timeout=10,
            )
            time.sleep(1)
            return dev_id

    # Fallback: first non-web device
    for dev in devices:
        name_lower = dev.get('name', '').lower()
        if 'web player' not in name_lower and 'chrome' not in name_lower:
            dev_id = dev['id']
            print(f"  Using device (fallback): {dev['name']}")
    requests.put(
        'https://api.spotify.com/v1/me/player',
        headers={**headers, 'Content-Type': 'application/json'},
        json={'device_ids': [dev_id], 'play': False},
        timeout=10,
    )
    time.sleep(1)
    return dev_id


def _play_track(track_id: str, device_id: str = None) -> bool:
    """Play a track via Spotify Web API, targeting specific device."""
    token = _get_spotify_token()
    if not token:
        return False

    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    # Force a clean state: pause first, transfer to device, then play
    # This fixes the issue where GEMS left Spotify in a confused state
    if device_id:
        # Explicit transfer to ensure the device is truly active
        requests.put(
            'https://api.spotify.com/v1/me/player',
            headers=headers,
            json={'device_ids': [device_id], 'play': False},
            timeout=10,
        )
        time.sleep(1)

    # Pause any existing playback and let Spotify fully release the previous track
    requests.put(
        'https://api.spotify.com/v1/me/player/pause',
        headers=headers,
        timeout=5,
    )
    time.sleep(3)

    url = 'https://api.spotify.com/v1/me/player/play'
    if device_id:
        url += f'?device_id={device_id}'

    resp = requests.put(
        url,
        headers=headers,
        json={'uris': [f'spotify:track:{track_id}']},
        timeout=10,
    )
    if resp.status_code in (200, 204):
        return True
    print(f"Play failed: {resp.status_code} {resp.text[:100]}")
    return False


def _seek_to(position_ms: int):
    """Seek playback to position."""
    token = _get_spotify_token()
    if not token:
        return
    requests.put(
        f'https://api.spotify.com/v1/me/player/seek?position_ms={position_ms}',
        headers={'Authorization': f'Bearer {token}'},
        timeout=10,
    )


def _pause_playback():
    """Pause Spotify."""
    token = _get_spotify_token()
    if token:
        requests.put(
            'https://api.spotify.com/v1/me/player/pause',
            headers={'Authorization': f'Bearer {token}'},
            timeout=5,
        )


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
def _find_loopback_device(retry_coreaudio=True) -> int | None:
    """Find a Loopback or BlackHole audio input device.

    If not found and retry_coreaudio=True, restarts coreaudiod and tries again.
    This handles the common case where Loopback disappears after sleep/wake.
    """
    # Force sounddevice to re-query (it caches the device list)
    sd._terminate()
    sd._initialize()

    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        channels = dev['max_input_channels']
        if channels >= 2 and ('loopback' in name or 'blackhole' in name):
            print(f"Found audio device: {dev['name']} (index {i}, {channels}ch)")
            return i

    if retry_coreaudio:
        print("No Loopback device found — restarting coreaudiod...")
        try:
            subprocess.run(['sudo', 'killall', 'coreaudiod'], capture_output=True, timeout=10)
            time.sleep(5)  # coreaudiod takes a few seconds to come back
            return _find_loopback_device(retry_coreaudio=False)
        except Exception as e:
            print(f"coreaudiod restart failed: {e}")

    return None


def _record_sample(device_idx: int, duration: float = SAMPLE_DURATION) -> tuple:
    """Record audio from loopback device, return (mono float32 array, stereo buffer) or (None, None)."""
    try:
        buf = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=2,
            device=device_idx,
            blocking=True,
        )
        mono = buf.mean(axis=1).astype(np.float32)
        # Validate not silent
        if np.max(np.abs(mono)) < 0.001:
            return None, None
        return mono, buf
    except Exception as e:
        print(f"Record error: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Feature extraction (same as GEMS pipeline)
# ---------------------------------------------------------------------------
def extract_features_from_audio(audio: np.ndarray, audio_stereo: np.ndarray = None) -> dict:
    """Extract 30+ audio features — mirrors fixed_gems_pipeline_v2.py."""
    import librosa
    import pyloudnorm as pyln

    features = {}

    # === STEREO FEATURES ===
    if audio_stereo is not None and audio_stereo.ndim == 2 and audio_stereo.shape[1] == 2:
        left = audio_stereo[:, 0].astype(np.float64)
        right = audio_stereo[:, 1].astype(np.float64)
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        mid_energy = float(np.sqrt(np.mean(mid ** 2)))
        side_energy = float(np.sqrt(np.mean(side ** 2)))
        features['stereo_width'] = float(side_energy / max(mid_energy, 1e-10))
        features['mid_side_ratio'] = float(mid_energy / max(mid_energy + side_energy, 1e-10))
        if len(left) > 0:
            corr = np.corrcoef(left, right)[0, 1]
            features['stereo_correlation'] = float(corr) if np.isfinite(corr) else 1.0
        else:
            features['stereo_correlation'] = 1.0
    else:
        features['stereo_width'] = 0.0
        features['mid_side_ratio'] = 1.0
        features['stereo_correlation'] = 1.0

    # --- Loudness ---
    meter = pyln.Meter(SAMPLE_RATE)
    try:
        features['lufs_integrated'] = float(meter.integrated_loudness(audio))
    except Exception:
        features['lufs_integrated'] = -30.0

    # Loudness range
    block_size = int(0.4 * SAMPLE_RATE)
    loudness_values = []
    for i in range(0, len(audio) - block_size, block_size // 2):
        try:
            bl = meter.integrated_loudness(audio[i:i + block_size])
            if np.isfinite(bl):
                loudness_values.append(bl)
        except Exception:
            pass
    if len(loudness_values) >= 2:
        features['loudness_range'] = float(
            np.percentile(loudness_values, 95) - np.percentile(loudness_values, 10)
        )
    else:
        features['loudness_range'] = 0.0

    # --- Rhythm ---
    tempo, beats = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE)
    features['bpm'] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    onset_env = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE)
    features['beat_strength'] = float(np.mean(onset_env))
    onset_frames = librosa.onset.onset_detect(y=audio, sr=SAMPLE_RATE)
    duration_s = len(audio) / SAMPLE_RATE
    features['onset_rate'] = float(len(onset_frames) / duration_s) if duration_s > 0 else 0.0

    # --- Tonal ---
    chroma = librosa.feature.chroma_cqt(y=audio, sr=SAMPLE_RATE)
    chroma_mean = np.mean(chroma, axis=1)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = int(np.argmax(chroma_mean))
    features['key'] = pitch_classes[key_idx]

    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    major_corr = float(np.corrcoef(chroma_mean, np.roll(major_profile, key_idx))[0, 1])
    minor_corr = float(np.corrcoef(chroma_mean, np.roll(minor_profile, key_idx))[0, 1])
    features['scale'] = 'major' if major_corr > minor_corr else 'minor'
    features['key_strength'] = float(max(major_corr, minor_corr))

    # --- Spectral ---
    centroids = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
    features['brightness'] = float(np.mean(centroids))
    features['brightness_variance'] = float(np.var(centroids))
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)[0]
    features['spectral_rolloff'] = float(np.mean(rolloff))
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)[0]
    features['spectral_complexity'] = float(np.mean(bw) / SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr'] = float(np.mean(zcr))

    # --- 7-Band tonal balance ---
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)
    total_energy = np.sum(fft)
    if total_energy > 0:
        bands = [
            ('sub_ratio', 20, 60), ('bass_ratio', 60, 250),
            ('low_mid_ratio', 250, 500), ('mid_ratio', 500, 2000),
            ('high_mid_ratio', 2000, 4000), ('presence_ratio', 4000, 8000),
            ('air_ratio', 8000, 20000),
        ]
        for name, lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            features[name] = float(np.sum(fft[mask]) / total_energy)

    # Harmonic distortion estimate (THD approximation)
    if total_energy > 0:
        # Find fundamental frequency from spectral centroid
        fundamental_idx = np.argmax(fft[1:]) + 1  # skip DC
        fundamental_energy = float(fft[fundamental_idx])
        # Sum energy at harmonics (2x, 3x, 4x, 5x fundamental)
        harmonic_energy = 0.0
        for h in range(2, 6):
            h_idx = fundamental_idx * h
            if h_idx < len(fft):
                harmonic_energy += float(fft[h_idx])
        features['harmonic_distortion'] = float(harmonic_energy / max(fundamental_energy, 1e-10))
    else:
        features['harmonic_distortion'] = 0.0

    # --- Energy / Dynamics ---
    rms = float(np.sqrt(np.mean(audio ** 2)))
    features['energy'] = rms

    # True peak (inter-sample) via 4x oversampling
    from scipy import signal as scipy_signal
    try:
        upsampled = scipy_signal.resample(audio, len(audio) * 4)
        features['true_peak_dbfs'] = float(20 * np.log10(max(np.max(np.abs(upsampled)), 1e-10)))
    except Exception:
        features['true_peak_dbfs'] = float(20 * np.log10(max(np.max(np.abs(audio)), 1e-10)))

    p95 = np.percentile(np.abs(audio), 95)
    p10 = max(np.percentile(np.abs(audio), 10), 1e-10)
    features['dynamic_range'] = min(float(20 * np.log10(p95 / p10)), 60.0)
    features['crest_factor'] = float(20 * np.log10(np.max(np.abs(audio)) / max(rms, 1e-10)))
    features['compression_amount'] = max(0.0, 1.0 - (features['crest_factor'] / 26.0))

    # --- Advanced ---
    stft = librosa.stft(audio)
    spec_db = librosa.amplitude_to_db(np.abs(stft))
    spec_flux = np.sum(np.diff(spec_db, axis=1), axis=0)
    features['spectral_flux'] = float(np.mean(np.abs(spec_flux)))
    features['dissonance'] = float(np.mean(zcr) * features['spectral_flux'] / 1000)

    # Attack time
    attack_idx = int(np.argmax(onset_env > 0.9 * np.max(onset_env))) if np.max(onset_env) > 0 else 0
    features['attack_time'] = float(attack_idx / SAMPLE_RATE * len(audio) / max(len(onset_env), 1))
    features['attack_time'] = max(features['attack_time'], 0.1)

    # Danceability
    if len(beats) > 1:
        beat_diffs = np.diff(beats)
        tempo_stability = 1.0 - float(np.std(beat_diffs) / max(np.mean(beat_diffs), 1e-10))
        features['danceability'] = max(0.0, float(tempo_stability * features['beat_strength']))
    elif features['bpm'] > 0:
        # Fallback: estimate from BPM and beat_strength when beat frames are sparse
        # BPM was detected via autocorrelation even if onset frames weren't found
        bpm_factor = min(features['bpm'] / 120.0, 1.5)  # normalize around 120 BPM
        features['danceability'] = max(0.0, float(bpm_factor * features['beat_strength'] * 0.5))
    else:
        features['danceability'] = 0.0

    # Clean NaN
    for k, v in features.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    return features


# ---------------------------------------------------------------------------
# Job processor
# ---------------------------------------------------------------------------
def process_job(job: dict, loopback_device: int):
    """Process a single pending_features job using GEMS-style capture."""
    job_id = job['id']
    spotify_url = job.get('spotify_url', '')
    # Fallback: check progress field (older jobs stored URL there)
    if not spotify_url:
        progress = job.get('progress')
        if isinstance(progress, str):
            try:
                progress = json.loads(progress)
            except (json.JSONDecodeError, TypeError):
                progress = {}
        if isinstance(progress, dict):
            spotify_url = progress.get('spotify_url', '')

    if not spotify_url:
        print(f"[{job_id[:8]}] No spotify_url in job, skipping")
        update_job(job_id, 'error')
        return

    track_id = _extract_track_id(spotify_url)
    if not track_id:
        print(f"[{job_id[:8]}] Cannot parse track ID from: {spotify_url}")
        update_job(job_id, 'error')
        return

    print(f"[{job_id[:8]}] Processing track {track_id}...")
    update_job(job_id, 'capturing')

    # Get track info for duration
    info = _get_track_info(track_id)
    if not info:
        print(f"[{job_id[:8]}] Cannot get track info")
        update_job(job_id, 'error')
        return

    duration_ms = info.get('duration_ms', 0)
    track_name = info.get('name', '')
    artist_names = ', '.join(a['name'] for a in info.get('artists', []))
    print(f"[{job_id[:8]}] {track_name} — {artist_names} ({duration_ms / 1000:.0f}s)")

    if duration_ms < 30000:
        print(f"[{job_id[:8]}] Track too short ({duration_ms / 1000:.1f}s)")
        update_job(job_id, 'error')
        return

    # Stop GEMS/discovery BEFORE touching Spotify — they fight over playback
    _gems_paused = []
    try:
        import subprocess as _sp
        result = _sp.run(['pm2', 'jlist'], capture_output=True, text=True, timeout=10)
        for p in json.loads(result.stdout):
            if p.get('pm2_env', {}).get('status') == 'online' and p['name'] in ('gems', 'discovery'):
                _sp.run(['pm2', 'stop', p['name']], capture_output=True, timeout=30)
                _gems_paused.append(p['name'])
        if _gems_paused:
            print(f"[{job_id[:8]}] Paused {', '.join(_gems_paused)} for capture")
            time.sleep(5)  # Let Spotify fully release previous GEMS track
    except Exception as e:
        print(f"[{job_id[:8]}] Warning: could not pause scripts: {e}")

    # Ensure Spotify is active
    device_id = _ensure_device_active()
    if not device_id:
        print(f"[{job_id[:8]}] No Spotify device available")
        update_job(job_id, 'error')
        return

    # Play the track — retry up to 3 times
    for attempt in range(3):
        if _play_track(track_id, device_id):
            time.sleep(2)
            # Verify playback is actually happening
            token = _get_spotify_token()
            if token:
                try:
                    state = requests.get(
                        'https://api.spotify.com/v1/me/player',
                        headers={'Authorization': f'Bearer {token}'},
                        timeout=5,
                    )
                    if state.status_code == 200:
                        data = state.json()
                        if data.get('is_playing'):
                            break
                        print(f"[{job_id[:8]}] Playback not active (attempt {attempt+1}/3), retrying...")
                    else:
                        print(f"[{job_id[:8]}] Player state check failed: {state.status_code}")
                except Exception:
                    pass
            else:
                break  # No token, just hope it's playing
        else:
            print(f"[{job_id[:8]}] Play API failed (attempt {attempt+1}/3)")
        time.sleep(2)
    else:
        print(f"[{job_id[:8]}] Failed to start playback after 3 attempts")
        update_job(job_id, 'error')
        return

    # Quick audio routing check (same as GEMS pipeline)
    try:
        test_audio = sd.rec(int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                            channels=2, device=loopback_device, blocking=True)
        if np.max(np.abs(test_audio)) < 0.001:
            print(f"[{job_id[:8]}] Audio routing check: no signal, waiting 3s...")
            time.sleep(3)
    except Exception:
        pass

    # Sample at 3 positions: 25%, 50%, 75%
    sample_points = [
        int(duration_ms * 0.25),
        int(duration_ms * 0.50),
        int(duration_ms * 0.75),
    ]

    audio_samples = []
    stereo_samples = []
    energy_levels = []

    for i, pos_ms in enumerate(sample_points):
        # Ensure playback is active before seeking — Spotify can silently stop
        token = _get_spotify_token()
        if token:
            try:
                state = requests.get(
                    'https://api.spotify.com/v1/me/player',
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=5,
                )
                if state.status_code == 200 and not state.json().get('is_playing'):
                    print(f"[{job_id[:8]}] Spotify stopped before sample {i + 1}, restarting play...")
                    _play_track(track_id, device_id)
                    time.sleep(2)
            except Exception:
                pass

        _seek_to(pos_ms)
        time.sleep(1.5)  # Match GEMS pipeline settle time

        mono, stereo_buf = _record_sample(loopback_device)
        if mono is not None:
            energy = float(np.sqrt(np.mean(mono ** 2)))
            audio_samples.append(mono)
            stereo_samples.append(stereo_buf)
            energy_levels.append(energy)
            print(f"[{job_id[:8]}] Sample {i + 1}/3 at {pos_ms / 1000:.0f}s — energy={energy:.4f}")
        else:
            print(f"[{job_id[:8]}] Sample {i + 1}/3 at {pos_ms / 1000:.0f}s — silent/failed")

    _pause_playback()


    if not audio_samples:
        print(f"[{job_id[:8]}] All samples failed — no audio captured")
        update_job(job_id, 'capture_failed')
        return

    # Use highest-energy sample
    best_idx = int(np.argmax(energy_levels))
    audio = audio_samples[best_idx]
    audio_stereo = stereo_samples[best_idx]
    print(f"[{job_id[:8]}] Using sample {best_idx + 1} (energy={energy_levels[best_idx]:.4f})")

    # Extract features
    try:
        features = extract_features_from_audio(audio, audio_stereo=audio_stereo)
    except Exception as e:
        print(f"[{job_id[:8]}] Feature extraction failed: {e}")
        update_job(job_id, 'extraction_failed')
        return

    # Validate
    if features.get('lufs_integrated', -100) < -55:
        print(f"[{job_id[:8]}] LUFS too low ({features['lufs_integrated']:.1f}), capture likely failed")
        update_job(job_id, 'capture_failed')
        return
    if features.get('energy', 0) < 0.001:
        print(f"[{job_id[:8]}] Energy too low ({features['energy']:.4f}), capture likely failed")
        update_job(job_id, 'capture_failed')
        return

    # Success
    update_job(job_id, 'features_ready', features)
    print(f"[{job_id[:8]}] Features extracted — BPM={features.get('bpm', 0):.0f}, "
          f"key={features.get('key', '?')} {features.get('scale', '')}, "
          f"LUFS={features.get('lufs_integrated', 0):.1f}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        sys.exit(1)

    loopback_device = _find_loopback_device()
    if loopback_device is None:
        print("Error: No Loopback or BlackHole audio device found")
        print("Available devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                print(f"  {i}: {dev['name']} ({dev['max_input_channels']}ch input)")
        sys.exit(1)

    print(f"Mac Worker started")
    print(f"  Audio device: {sd.query_devices(loopback_device)['name']}")
    print(f"  Polling every {POLL_INTERVAL}s")
    print(f"  Retry window: {RETRY_WINDOW}s for failed jobs")

    import threading
    import subprocess as _sp

    def _pause_local_scripts():
        """Stop gems/discovery before audio capture."""
        paused = []
        try:
            result = _sp.run(['pm2', 'jlist'], capture_output=True, text=True, timeout=10)
            for p in json.loads(result.stdout):
                if p.get('pm2_env', {}).get('status') == 'online' and p['name'] in ('gems', 'discovery'):
                    _sp.run(['pm2', 'stop', p['name']], capture_output=True, timeout=30)
                    paused.append(p['name'])
            if paused:
                print(f"  Paused {', '.join(paused)} for capture")
                time.sleep(5)  # Let Spotify fully release previous GEMS track
        except Exception as e:
            print(f"  Warning: could not pause scripts: {e}")
        return paused

    def _resume_local_scripts(paused):
        """Resume gems/discovery after capture."""
        for name in paused:
            try:
                _sp.run(['pm2', 'restart', name], capture_output=True, timeout=30)
            except Exception:
                pass
        if paused:
            print(f"  Resumed {', '.join(paused)}")

    def _ensure_loopback(current_idx):
        """Re-validate loopback device before each job. Returns new index or None."""
        try:
            dev = sd.query_devices(current_idx)
            name = dev['name'].lower()
            if dev['max_input_channels'] >= 2 and ('loopback' in name or 'blackhole' in name):
                return current_idx
        except Exception:
            pass
        print("Loopback device lost — attempting recovery...")
        new_idx = _find_loopback_device()
        if new_idx is not None:
            print(f"Loopback recovered: device index {new_idx}")
        else:
            print("Loopback recovery FAILED — skipping job")
        return new_idx

    def _run_job(job, device_idx):
        """Run a job with timeout protection. Returns True if features were delivered."""
        paused = _pause_local_scripts()
        success = False
        JOB_TIMEOUT = 120

        def _run():
            nonlocal success
            try:
                process_job(job, device_idx)
                # Check if job actually succeeded
                if job_mgr_check(job['id']):
                    success = True
            except Exception as e:
                print(f"Job processing error: {e}")
                try:
                    update_job(job['id'], 'error')
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=JOB_TIMEOUT)
        if t.is_alive():
            print(f"[{job['id'][:8]}] Job timed out after {JOB_TIMEOUT}s — moving on")
            try:
                update_job(job['id'], 'error')
            except Exception:
                pass

        _resume_local_scripts(paused)
        return success

    def job_mgr_check(job_id):
        """Check if a job reached features_ready status."""
        try:
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/analysis_jobs?id=eq.{job_id}&select=status",
                headers=_supabase_headers(),
                timeout=10,
            )
            if resp.status_code == 200 and resp.json():
                return resp.json()[0].get('status') == 'features_ready'
        except Exception:
            pass
        return False

    while True:
        try:
            job = poll_pending_jobs()
        except Exception as e:
            print(f"Poll exception: {e}")
            job = None

        # If no pending job, check for recently failed jobs to retry
        if not job:
            try:
                job = poll_retryable_jobs()
                if job:
                    # Reset status to pending so we process it
                    update_job(job['id'], 'pending_features')
            except Exception as e:
                print(f"Retry poll exception: {e}")
                job = None

        if job:
            # Re-validate loopback device before every job
            loopback_device = _ensure_loopback(loopback_device)
            if loopback_device is None:
                print(f"[{job['id'][:8]}] No audio device — marking error")
                update_job(job['id'], 'error')
                # Keep trying to recover the device
                time.sleep(10)
                loopback_device = _find_loopback_device()
                if loopback_device is not None:
                    print(f"Loopback recovered after wait: device index {loopback_device}")
            else:
                _run_job(job, loopback_device)

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
