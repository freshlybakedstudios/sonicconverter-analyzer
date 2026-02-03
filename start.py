#!/usr/bin/env python3
"""
Startup script for Railway deployment.
Starts server immediately, builds GEMS cache in background if missing.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path

CACHE_PATH = Path(__file__).parent / 'cache' / 'universe' / 'gems_universe.json'


def build_cache_background():
    """Build the GEMS universe cache in a background thread."""
    print("📦 Cache not found — building from Supabase in background...")

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY')

    if not url or not key:
        print("❌ SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        return

    try:
        from build_gems_universe_cache import UniverseCacheBuilder
        builder = UniverseCacheBuilder(url, key)
        builder.build()
        print("✅ Cache built successfully - restart the app or it will load on next request")
    except Exception as e:
        print(f"❌ Cache build failed: {e}")


def main():
    if CACHE_PATH.exists():
        print(f"✅ Cache already exists at {CACHE_PATH}")
    else:
        # Start cache build in background thread
        thread = threading.Thread(target=build_cache_background, daemon=True)
        thread.start()
        print("⏳ Cache building in background - app may not work until complete")

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
