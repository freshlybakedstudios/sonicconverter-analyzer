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
SAMPLE_RATE = 44100
SAMPLE_DURATION = 4  # seconds per sample point

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


def poll_pending_jobs():
    """Check for jobs with status='pending_features'."""
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/analysis_jobs?status=eq.pending_features&order=created_at.asc&limit=1",
            headers=_supabase_headers(),
            timeout=10,
        )
        if resp.status_code == 200 and resp.json():
            return resp.json()[0]
    except Exception as e:
        print(f"Poll error: {e}")
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


def _get_spotify_token():
    """Get Spotify Web API bearer token (user auth via refresh token or client creds)."""
    global _spotify_bearer, _spotify_bearer_ts
    if _spotify_bearer and (time.time() - _spotify_bearer_ts) < 3500:
        return _spotify_bearer

    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    refresh_token = os.getenv('SPOTIFY_REFRESH_TOKEN')

    if refresh_token and client_id and client_secret:
        # User auth — needed for playback control
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
            return _spotify_bearer
        print(f"Spotify refresh failed: {resp.status_code} {resp.text[:100]}")

    print("No SPOTIFY_REFRESH_TOKEN set — cannot control playback")
    return None


def _extract_track_id(url: str) -> str | None:
    """Extract Spotify track ID from URL."""
    if 'track/' in url:
        return url.split('track/')[-1].split('?')[0].split('/')[0]
    if 'spotify:track:' in url:
        return url.split('spotify:track:')[1].split('?')[0]
    return None


def _get_track_info(track_id: str) -> dict | None:
    """Get track metadata (duration, playability)."""
    token = _get_spotify_token()
    if not token:
        return None
    resp = requests.get(
        f'https://api.spotify.com/v1/tracks/{track_id}',
        headers={'Authorization': f'Bearer {token}'},
        params={'market': 'US'},
        timeout=10,
    )
    if resp.status_code == 200:
        return resp.json()
    print(f"Track info failed: {resp.status_code}")
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

    # Prefer the computer/desktop device
    for dev in devices:
        if dev.get('type', '').lower() in ('computer', 'desktop'):
            dev_id = dev['id']
            print(f"  Using device: {dev['name']} ({dev['type']})")
            # Transfer playback to this device to make it active
            requests.put(
                'https://api.spotify.com/v1/me/player',
                headers={**headers, 'Content-Type': 'application/json'},
                json={'device_ids': [dev_id], 'play': False},
                timeout=10,
            )
            time.sleep(1)
            return dev_id

    # Fallback: use first device
    dev = devices[0]
    dev_id = dev['id']
    print(f"  Using device: {dev['name']} ({dev.get('type', '?')})")
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
def _find_loopback_device() -> int | None:
    """Find a Loopback or BlackHole audio input device."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        channels = dev['max_input_channels']
        if channels >= 2 and ('loopback' in name or 'blackhole' in name):
            print(f"Found audio device: {dev['name']} (index {i}, {channels}ch)")
            return i
    return None


def _record_sample(device_idx: int, duration: float = SAMPLE_DURATION) -> np.ndarray | None:
    """Record audio from loopback device, return mono float32 array."""
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
            return None
        return mono
    except Exception as e:
        print(f"Record error: {e}")
        return None


# ---------------------------------------------------------------------------
# Feature extraction (same as GEMS pipeline)
# ---------------------------------------------------------------------------
def extract_features_from_audio(audio: np.ndarray) -> dict:
    """Extract 30+ audio features — mirrors fixed_gems_pipeline_v2.py."""
    import librosa
    import pyloudnorm as pyln

    features = {}

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

    # --- Energy / Dynamics ---
    rms = float(np.sqrt(np.mean(audio ** 2)))
    features['energy'] = rms
    p95 = np.percentile(np.abs(audio), 95)
    p10 = max(np.percentile(np.abs(audio), 10), 1e-10)
    features['dynamic_range'] = float(p95 / p10)
    features['crest_factor'] = float(np.max(np.abs(audio)) / max(rms, 1e-10))
    features['compression_amount'] = 1.0 - (features['crest_factor'] / 20.0)

    # --- Advanced ---
    stft = librosa.stft(audio)
    spec_db = librosa.amplitude_to_db(np.abs(stft))
    spec_flux = np.sum(np.diff(spec_db, axis=1), axis=0)
    features['spectral_flux'] = float(np.mean(np.abs(spec_flux)))
    features['dissonance'] = float(np.mean(zcr) * features['spectral_flux'] / 1000)

    # Attack time
    attack_idx = int(np.argmax(onset_env > 0.9 * np.max(onset_env))) if np.max(onset_env) > 0 else 0
    features['attack_time'] = float(attack_idx / SAMPLE_RATE * len(audio) / max(len(onset_env), 1))

    # Danceability
    if len(beats) > 1:
        beat_diffs = np.diff(beats)
        tempo_stability = 1.0 - float(np.std(beat_diffs) / max(np.mean(beat_diffs), 1e-10))
        features['danceability'] = float(tempo_stability * features['beat_strength'])
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

    # Ensure Spotify is active
    device_id = _ensure_device_active()
    if not device_id:
        print(f"[{job_id[:8]}] No Spotify device available")
        update_job(job_id, 'error')
        return

    # Play the track
    if not _play_track(track_id, device_id):
        print(f"[{job_id[:8]}] Failed to start playback")
        update_job(job_id, 'error')
        return

    time.sleep(2)  # Let playback start

    # Sample at 3 positions: 25%, 50%, 75%
    sample_points = [
        int(duration_ms * 0.25),
        int(duration_ms * 0.50),
        int(duration_ms * 0.75),
    ]

    audio_samples = []
    energy_levels = []

    for i, pos_ms in enumerate(sample_points):
        _seek_to(pos_ms)
        time.sleep(1.5)  # Let playback settle

        mono = _record_sample(loopback_device)
        if mono is not None:
            energy = float(np.sqrt(np.mean(mono ** 2)))
            audio_samples.append(mono)
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
    print(f"[{job_id[:8]}] Using sample {best_idx + 1} (energy={energy_levels[best_idx]:.4f})")

    # Extract features
    try:
        features = extract_features_from_audio(audio)
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

    while True:
        job = poll_pending_jobs()
        if job:
            try:
                process_job(job, loopback_device)
            except Exception as e:
                print(f"Job processing error: {e}")
                try:
                    update_job(job['id'], 'error')
                except Exception:
                    pass
        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
