"""
Mac Worker Daemon — Local audio capture for Spotify URL analysis.

Polls Supabase `analysis_jobs` for jobs with status='pending_features',
captures audio from Spotify via BlackHole virtual audio device,
runs feature extraction, and updates the job with results.

Usage:
    python mac_worker.py

Requires:
    - BlackHole audio driver installed
    - Spotify desktop app running
    - SUPABASE_URL and SUPABASE_SERVICE_KEY env vars set
"""

import json
import os
import subprocess
import sys
import tempfile
import time

from dotenv import load_dotenv

load_dotenv()


POLL_INTERVAL = 5  # seconds
CAPTURE_DURATION = 35  # seconds of audio to capture
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')


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
    import requests
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
    import requests
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


def capture_spotify_audio(duration: int = CAPTURE_DURATION) -> str | None:
    """
    Capture system audio via BlackHole for `duration` seconds.
    Returns path to the captured WAV file, or None on failure.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.close()
    output_path = tmp.name

    try:
        # Use ffmpeg to capture from BlackHole virtual audio device
        cmd = [
            'ffmpeg', '-y',
            '-f', 'avfoundation',
            '-i', ':BlackHole 2ch',
            '-t', str(duration),
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 10)
        if result.returncode == 0 and os.path.getsize(output_path) > 1000:
            return output_path
        else:
            print(f"FFmpeg failed: {result.stderr[:200]}")
            os.unlink(output_path)
            return None
    except Exception as e:
        print(f"Capture error: {e}")
        try:
            os.unlink(output_path)
        except Exception:
            pass
        return None


def open_spotify_track(spotify_url: str):
    """Open a Spotify track URL in the desktop app."""
    # Convert web URL to Spotify URI
    if 'open.spotify.com/track/' in spotify_url:
        track_id = spotify_url.split('track/')[1].split('?')[0].split('/')[0]
        uri = f"spotify:track:{track_id}"
    else:
        uri = spotify_url

    subprocess.run(['open', uri], timeout=5)
    time.sleep(3)  # Wait for playback to start


def process_job(job: dict):
    """Process a single pending job."""
    job_id = job['id']
    print(f"Processing job {job_id[:8]}...")

    # Mark as processing
    update_job(job_id, 'capturing')

    # Get the Spotify URL from the job token or metadata
    # The URL should be stored when the job was created
    # For now, we skip if no URL is available
    features_str = job.get('features', '{}')
    if isinstance(features_str, str):
        try:
            features = json.loads(features_str)
        except (json.JSONDecodeError, TypeError):
            features = {}
    else:
        features = features_str or {}

    if features:
        # Features already provided, just mark ready
        update_job(job_id, 'features_ready', features)
        print(f"Job {job_id[:8]}: Features already present, marked ready")
        return

    # Try to capture audio
    audio_path = capture_spotify_audio()
    if not audio_path:
        print(f"Job {job_id[:8]}: Audio capture failed")
        update_job(job_id, 'capture_failed')
        return

    # Extract features
    try:
        from audio_analyzer import extract_features
        features = extract_features(audio_path)
        update_job(job_id, 'features_ready', features)
        print(f"Job {job_id[:8]}: Features extracted, marked ready")
    except Exception as e:
        print(f"Job {job_id[:8]}: Feature extraction failed: {e}")
        update_job(job_id, 'extraction_failed')
    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        sys.exit(1)

    print(f"Mac Worker started — polling every {POLL_INTERVAL}s")
    print(f"Supabase: {SUPABASE_URL[:50]}...")

    while True:
        job = poll_pending_jobs()
        if job:
            process_job(job)
        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
