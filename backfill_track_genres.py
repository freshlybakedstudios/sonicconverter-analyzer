#!/usr/bin/env python3
"""
backfill_track_genres.py — Refresh stale/broken tracks.track_genres
(and optionally tracks.recent_track_genres) from live CM data.

Why: the discovery import (discovery_events_work.py) had a priority-chain
bug that wrote single-tag values when CM actually had rich tags available.
Result in tracks: ~71K rows with NULL / 'Others' / 'Genre ID: N' /
single-genre values when the underlying CM track has 10+ tags.

This script uses the new _extract_track_genres helper (a union of all
CM genre-bearing fields, deduped) and PATCHes only rows where the
fresh value DIFFERS from the stored value, preserving updated_at as
a meaningful change-detection signal.

Coordination (per PM2 agent validation):
  pm2 stop discovery
  pm2 stop mac-worker          # also competes for 1 req/s CM budget
  python3 backfill_track_genres.py --fix-broken-only --include-recent
  pm2 start discovery
  pm2 start mac-worker

Usage:
    # Smallest validation run
    python3 backfill_track_genres.py --fix-broken-only --limit 50

    # Fix only NULL/'Others'/'Genre ID:' rows (~70K, ~20h)
    python3 backfill_track_genres.py --fix-broken-only

    # Also refresh recent_track_genres column (~127K subset, ~34h total)
    python3 backfill_track_genres.py --fix-broken-only --include-recent

    # Full universe refresh — every track, every recent (~78h)
    python3 backfill_track_genres.py --include-recent

    # Resume after Ctrl-C
    python3 backfill_track_genres.py --fix-broken-only      # auto-resumes
"""

import argparse
import json
import os
import signal
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from chartmetric_lookup import (
    _resolve_isrc_to_cm_track_id,
    _extract_track_genres,
    _rate_wait,
    get_cm_token,
    CM_TRACK_META_URL,
)

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_KEY')
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')

STATE_FILE = Path(__file__).parent / 'backfill_track_genres.state.json'
PAGE_SIZE = 1000
LOG_INTERVAL_S = 30

_BASE_HEADERS = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'}

stop_requested = False
def _sig_handler(signum, frame):
    global stop_requested
    stop_requested = True
    print('\n[backfill] signal received — flushing and exiting cleanly...')
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        'mode': None,
        'offset': 0,
        'processed': 0,
        'top_updated': 0,
        'recent_updated': 0,
        'unchanged': 0,
        'failed': 0,
        'skipped': 0,
        'started_at': time.time(),
    }


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fetch_tracks_page(offset, limit, fix_broken_only=False, include_recent=False):
    """Fetch a page of tracks. When --fix-broken-only, filter to rows where
    track_genres is NULL / 'Others' / starts with 'Genre ID' (or, if
    include_recent, where recent_track_genres has the same problem)."""
    cols = 'artist_id,isrc,top_track,track_genres,recent_track_isrc,recent_track,recent_track_genres'
    params = {'select': cols, 'order': 'artist_id.asc', 'offset': offset, 'limit': limit}
    if fix_broken_only:
        broken = (
            'track_genres.is.null,'
            'track_genres.eq.Others,'
            'track_genres.like.Genre ID*'
        )
        if include_recent:
            broken += (
                ',recent_track_genres.is.null,'
                'recent_track_genres.eq.Others,'
                'recent_track_genres.like.Genre ID*'
            )
        params['or'] = f'({broken})'
    r = requests.get(f'{SUPABASE_URL}/rest/v1/tracks', params=params, headers=_BASE_HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_track_meta(token, cm_track_id):
    _rate_wait()
    r = requests.get(
        CM_TRACK_META_URL.format(track_id=cm_track_id),
        headers={'Authorization': f'Bearer {token}', 'X-Accept-Partial-Data': 'true'},
        timeout=30,
    )
    if r.status_code == 200:
        return r.json().get('obj', {}) or {}
    return None


def patch_row(artist_id, isrc, updates):
    """PATCH a tracks row by composite (artist_id, isrc). Skip if updates empty."""
    if not updates:
        return
    headers = {**_BASE_HEADERS, 'Content-Type': 'application/json', 'Prefer': 'return=minimal'}
    params = {'artist_id': f'eq.{artist_id}', 'isrc': f'eq.{isrc}'}
    r = requests.patch(f'{SUPABASE_URL}/rest/v1/tracks', params=params, json=updates, headers=headers, timeout=30)
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fix-broken-only', action='store_true',
                    help='Only target NULL / Others / Genre ID rows (~70K)')
    ap.add_argument('--include-recent', action='store_true',
                    help='Also refresh recent_track_genres column')
    ap.add_argument('--limit', type=int, default=None, help='Cap total processed rows')
    ap.add_argument('--fresh', action='store_true', help='Ignore state file, start over')
    args = ap.parse_args()

    if args.fresh and STATE_FILE.exists():
        STATE_FILE.unlink()

    state = load_state()
    mode_key = f'broken={args.fix_broken_only},recent={args.include_recent}'
    if state.get('mode') and state['mode'] != mode_key:
        print(f'[backfill] State is for mode={state["mode"]!r}, you asked for mode={mode_key!r}.')
        print('[backfill] Use --fresh to start over, or run the matching mode to resume.')
        return
    state['mode'] = mode_key

    print(f'[backfill] mode={mode_key} resume_offset={state["offset"]} '
          f'processed={state["processed"]} top_updated={state["top_updated"]} '
          f'recent_updated={state["recent_updated"]}')

    token = get_cm_token(REFRESH_TOKEN)
    last_log = time.time()
    page_offset = state['offset']

    while not stop_requested:
        if args.limit is not None and state['processed'] >= args.limit:
            print(f'[backfill] hit --limit={args.limit}, stopping.')
            break

        page = fetch_tracks_page(page_offset, PAGE_SIZE,
                                 fix_broken_only=args.fix_broken_only,
                                 include_recent=args.include_recent)
        if not page:
            print('[backfill] page returned 0 — done.')
            break

        rows_processed = 0
        limit_hit = False
        for row in page:
            if stop_requested:
                break
            if args.limit is not None and state['processed'] >= args.limit:
                limit_hit = True
                break

            artist_id = row.get('artist_id')
            isrc = row.get('isrc')
            existing_top = row.get('track_genres') or ''
            recent_isrc = row.get('recent_track_isrc')
            existing_recent = row.get('recent_track_genres') or ''

            updates = {}

            # Top track
            if isrc:
                try:
                    cm_tid = _resolve_isrc_to_cm_track_id(token, isrc)
                    if cm_tid:
                        meta = fetch_track_meta(token, cm_tid)
                        if meta is not None:
                            fresh = _extract_track_genres(meta)
                            if fresh and fresh != existing_top:
                                updates['track_genres'] = fresh
                                state['top_updated'] += 1
                except Exception as e:
                    state['failed'] += 1
                    if '401' in str(e):
                        token = get_cm_token(REFRESH_TOKEN)
                    else:
                        print(f'[backfill] top {artist_id}/{isrc}: {type(e).__name__}: {str(e)[:80]}')

            # Recent track (optional)
            if args.include_recent and recent_isrc:
                try:
                    cm_tid_r = _resolve_isrc_to_cm_track_id(token, recent_isrc)
                    if cm_tid_r:
                        meta_r = fetch_track_meta(token, cm_tid_r)
                        if meta_r is not None:
                            fresh_r = _extract_track_genres(meta_r)
                            if fresh_r and fresh_r != existing_recent:
                                updates['recent_track_genres'] = fresh_r
                                state['recent_updated'] += 1
                except Exception as e:
                    state['failed'] += 1
                    if '401' in str(e):
                        token = get_cm_token(REFRESH_TOKEN)
                    else:
                        print(f'[backfill] recent {artist_id}/{recent_isrc}: {type(e).__name__}: {str(e)[:80]}')

            # Apply (skip when nothing to write — keeps updated_at meaningful)
            if updates:
                try:
                    patch_row(artist_id, isrc, updates)
                except Exception as e:
                    state['failed'] += 1
                    print(f'[backfill] PATCH {artist_id}/{isrc}: {type(e).__name__}: {str(e)[:80]}')
            else:
                state['unchanged'] += 1

            state['processed'] += 1
            rows_processed += 1

            if rows_processed % 25 == 0:
                state['offset'] = page_offset + rows_processed
                save_state(state)

            now = time.time()
            if now - last_log > LOG_INTERVAL_S:
                elapsed = now - state['started_at']
                rate = state['processed'] / elapsed if elapsed > 0 else 0
                print(f'[backfill] processed={state["processed"]} '
                      f'top_updated={state["top_updated"]} '
                      f'recent_updated={state["recent_updated"]} '
                      f'unchanged={state["unchanged"]} failed={state["failed"]} | '
                      f'{rate:.2f}/s ({rate*3600:.0f}/hr)')
                last_log = now

        page_offset += rows_processed
        state['offset'] = page_offset
        save_state(state)
        if limit_hit or stop_requested:
            break
        if len(page) < PAGE_SIZE:
            print('[backfill] reached last page, done.')
            break

    save_state(state)
    elapsed = time.time() - state['started_at']
    print(f'\n[backfill] DONE. processed={state["processed"]} '
          f'top_updated={state["top_updated"]} '
          f'recent_updated={state["recent_updated"]} '
          f'unchanged={state["unchanged"]} failed={state["failed"]} | '
          f'elapsed {elapsed/3600:.2f}h')


if __name__ == '__main__':
    main()
