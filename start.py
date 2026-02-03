#!/usr/bin/env python3
"""
Startup script for Railway deployment.
Downloads GEMS cache from GCS if missing, then starts server.
"""
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

CACHE_PATH = Path(__file__).parent / 'cache' / 'universe' / 'gems_universe.json'
CACHE_URL = 'https://storage.googleapis.com/fbs-static-assets/gems_universe.json'


def download_cache():
    """Download the GEMS universe cache from GCS."""
    print("📦 Downloading cache from GCS...")

    # Create directory if needed
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(CACHE_URL, CACHE_PATH)
        print(f"✅ Cache downloaded to {CACHE_PATH}")
    except Exception as e:
        print(f"❌ Cache download failed: {e}")
        raise


def main():
    if CACHE_PATH.exists():
        print(f"✅ Cache already exists at {CACHE_PATH}")
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
