#!/usr/bin/env python3
"""
Startup script for Railway deployment.
Downloads GEMS cache from GCS if missing OR if GCS has a newer version than
the local copy. (Railway uses a persistent volume, so without the freshness
check daily cache rebuilds never reach prod — the local file always "exists".)
"""
import os
import subprocess
import sys
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

CACHE_PATH = Path(__file__).parent / 'cache' / 'universe' / 'gems_universe.json'
CACHE_URL = 'https://storage.googleapis.com/fbs-static-assets/gems_universe.json'


def download_cache():
    """Download the GEMS universe cache from GCS."""
    print("📦 Downloading cache from GCS...")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(CACHE_URL, CACHE_PATH)
        print(f"✅ Cache downloaded to {CACHE_PATH}")
    except Exception as e:
        print(f"❌ Cache download failed: {e}")
        raise


def gcs_last_modified_epoch():
    """HEAD the GCS cache object for its Last-Modified header; return epoch
    seconds, or None on any error (callers fall back to skipping refresh)."""
    try:
        req = urllib.request.Request(CACHE_URL, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as resp:
            lm = resp.headers.get('Last-Modified')
            if lm:
                return parsedate_to_datetime(lm).timestamp()
    except Exception as e:
        print(f"⚠️  GCS freshness check failed (using local cache): {e}")
    return None


def main():
    if CACHE_PATH.exists():
        local_mtime = CACHE_PATH.stat().st_mtime
        remote_mtime = gcs_last_modified_epoch()
        if remote_mtime and remote_mtime > local_mtime + 1:
            print(f"🔄 Cache stale (local mtime {local_mtime:.0f} < GCS {remote_mtime:.0f}) — refreshing")
            download_cache()
        else:
            print(f"✅ Cache up to date at {CACHE_PATH}")
    else:
        download_cache()

    port = os.getenv('PORT', '8000')
    print(f"🚀 Starting server on port {port}...")

    subprocess.run([
        sys.executable, '-m', 'uvicorn',
        'app:app',
        '--host', '0.0.0.0',
        '--port', port
    ])


if __name__ == '__main__':
    main()
