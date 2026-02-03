#!/usr/bin/env python3
"""
Startup script for Railway deployment.
Builds GEMS cache if missing, then starts the server.
"""
import os
import subprocess
import sys
from pathlib import Path

CACHE_PATH = Path(__file__).parent / 'cache' / 'universe' / 'gems_universe.json'


def ensure_cache():
    """Build the GEMS universe cache if it doesn't exist."""
    if CACHE_PATH.exists():
        print(f"✅ Cache already exists at {CACHE_PATH}")
        return

    print("📦 Cache not found — building from Supabase...")

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY')

    if not url or not key:
        print("❌ SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        sys.exit(1)

    from build_gems_universe_cache import UniverseCacheBuilder
    builder = UniverseCacheBuilder(url, key)
    builder.build()
    print("✅ Cache built successfully")


def main():
    ensure_cache()

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
