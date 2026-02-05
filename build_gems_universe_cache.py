#!/usr/bin/env python3
"""
Build or incrementally update the GEMS universe cache.
This pre-joins all GEMS, tracks, artists, and tier data for instant access.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from supabase import create_client


GEMS_COLUMNS = [
    'isrc',
    'sub_ratio', 'bass_ratio', 'low_mid_ratio', 'mid_ratio', 'high_mid_ratio', 'presence_ratio', 'air_ratio',
    'energy', 'dynamic_range', 'loudness_range', 'attack_time', 'compression_amount', 'crest_factor',
    'beat_strength', 'onset_rate', 'danceability',
    'lufs_integrated', 'dissonance', 'key_strength', 'zcr',
    'emotion_1', 'emotion_1_score', 'emotion_2', 'emotion_2_score', 'emotion_3', 'emotion_3_score', 'emotion_4', 'emotion_4_score',
    'primary_genre', 'secondary_genre', 'overall_description', 'sonic_signature_text', 'bpm'
]


class UniverseCacheBuilder:
    def __init__(self, supabase_url, service_key, max_pages=None):
        self.supabase = create_client(supabase_url, service_key)
        root = Path(__file__).resolve().parent
        self.cache_dir = root / 'cache' / 'universe'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / 'gems_universe.json'
        # Optional page cap to prevent infinite loops; None = unlimited
        self.max_pages = max_pages

    def load_existing_cache(self):
        """Load existing cache if it exists."""
        if not self.cache_path.exists():
            return None

        print(f"📂 Loading existing cache from {self.cache_path}")
        try:
            with self.cache_path.open('r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"   ✅ Loaded cache from {cache.get('build_date', 'unknown date')}")
            print(f"   📊 Stats: {len(cache.get('gems', []))} GEMS, {len(cache.get('tracks', {}))} tracks, {len(cache.get('artists', {}))} artists")
            return cache
        except (OSError, json.JSONDecodeError) as e:
            print(f"   ⚠️  Failed to load cache: {e}")
            return None

    def fetch_all_gems(self, existing_isrcs=None):
        """Fetch all GEMS records, optionally only new ones."""
        print("\n🎵 Fetching GEMS records...")
        start = time.time()

        all_gems = []
        page = 0
        page_size = 1000
        total_fetched = 0

        while True:
            # Use limit + offset for proper pagination (range() is buggy in Python client)
            offset = page * page_size
            query = self.supabase.table('gems_complete_analysis')\
                .select(','.join(GEMS_COLUMNS))\
                .limit(page_size)\
                .offset(offset)

            batch = query.execute()

            if not batch.data:
                break

            total_fetched += len(batch.data)
            new_count_before = len(all_gems)

            for row in batch.data:
                isrc = row.get('isrc')
                if not isrc:
                    continue

                # Skip if we already have this ISRC (for incremental updates)
                if existing_isrcs and isrc in existing_isrcs:
                    continue

                all_gems.append(row)

            new_in_batch = len(all_gems) - new_count_before
            print(f"   📦 Page {page + 1}: {len(batch.data)} records ({new_in_batch} new, {len(all_gems)} total new)...", flush=True)

            # Break if we got fewer records than page size (last page)
            if len(batch.data) < page_size:
                print(f"   🛑 Last page (got {len(batch.data)} < {page_size})")
                break

            page += 1

            # Optional safety check to prevent runaway loops; None = no cap
            if self.max_pages is not None and page >= self.max_pages:
                print(f"   ⚠️  Safety limit reached at page {page} (max_pages={self.max_pages})")
                break

        elapsed = time.time() - start
        print(f"\n   ✅ Fetched {total_fetched:,} total records, {len(all_gems):,} new GEMS in {elapsed:.1f}s")
        return all_gems

    def fetch_tracks_batch(self, isrc_list):
        """Fetch track metadata for ISRCs in batches.

        Handles both top_track (isrc column) and recent_track (recent_track_isrc column).
        First pass matches on tracks.isrc, second pass tries unmatched ISRCs
        against tracks.recent_track_isrc so the full GEMS pool is used.
        """
        print(f"\n🎸 Fetching track metadata for {len(isrc_list)} ISRCs...")
        start = time.time()

        tracks = {}
        batch_size = 100
        empty_batches = 0
        low_match_batches = 0
        top_track_matches = 0
        recent_track_matches = 0

        # --- Pass 1: match on tracks.isrc (top track) ---
        for i in range(0, len(isrc_list), batch_size):
            batch = isrc_list[i:i+batch_size]
            rows = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    rows = self.supabase.table('tracks')\
                        .select('isrc, artist_id, track_genres, top_track, spotify_url')\
                        .in_('isrc', batch)\
                        .execute()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"\n   ⚠️  Timeout on batch {i//batch_size}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"\n   ❌ Batch {i//batch_size} failed after {max_retries} retries, skipping {len(batch)} ISRCs")
                        rows = None
                        break

            if rows is None or not rows.data:
                empty_batches += 1
                continue

            if len(rows.data) < len(batch) * 0.5:
                low_match_batches += 1

            for r in rows.data:
                isrc = r['isrc']
                tracks[isrc] = {
                    'artist_id': r.get('artist_id'),
                    'track_genres': r.get('track_genres', ''),
                    'top_track': r.get('top_track', ''),
                    'spotify_url': r.get('spotify_url', '')
                }
                top_track_matches += 1

            print(f"   📦 Pass 1 (top track): {min(i + batch_size, len(isrc_list))}/{len(isrc_list)}...", end='\r')

        print(f"\n   ✅ Pass 1: {top_track_matches} matched via top track ISRC")

        # --- Pass 2: match remaining ISRCs on tracks.recent_track_isrc ---
        missing_isrcs = [isrc for isrc in isrc_list if isrc not in tracks]
        if missing_isrcs:
            print(f"   🔄 Pass 2: trying {len(missing_isrcs)} unmatched ISRCs against recent_track_isrc...")
            for i in range(0, len(missing_isrcs), batch_size):
                batch = missing_isrcs[i:i+batch_size]
                rows = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        rows = self.supabase.table('tracks')\
                            .select('recent_track_isrc, artist_id, recent_track, recent_track_spotify_url, recent_track_genres')\
                            .in_('recent_track_isrc', batch)\
                            .execute()
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"\n   ⚠️  Timeout on recent batch, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            print(f"\n   ❌ Recent batch failed after {max_retries} retries, skipping")
                            rows = None
                            break

                if rows and rows.data:
                    for r in rows.data:
                        isrc = r['recent_track_isrc']
                        if isrc and isrc not in tracks:
                            tracks[isrc] = {
                                'artist_id': r.get('artist_id'),
                                'track_genres': r.get('recent_track_genres', ''),
                                'top_track': r.get('recent_track', ''),
                                'spotify_url': r.get('recent_track_spotify_url', '')
                            }
                            recent_track_matches += 1

                print(f"   📦 Pass 2 (recent track): {min(i + batch_size, len(missing_isrcs))}/{len(missing_isrcs)}...", end='\r')

            print(f"\n   🆕 Pass 2: {recent_track_matches} matched via recent_track_isrc")

        elapsed = time.time() - start
        print(f"\n   ✅ Fetched {len(tracks)} total track records in {elapsed:.1f}s")
        print(f"      Top track: {top_track_matches} | Recent track: {recent_track_matches} | Unmatched: {len(isrc_list) - len(tracks)}")
        if empty_batches > 0:
            print(f"   ⚠️  {empty_batches} batches returned empty!")
        if low_match_batches > 0:
            print(f"   ℹ️  {low_match_batches} batches had <50% match rate")
        return tracks

    def fetch_artists_batch(self, artist_ids):
        """Fetch artist info for artist IDs in batches."""
        print(f"\n🎤 Fetching artist metadata for {len(artist_ids)} artists...")
        start = time.time()

        artists = {}
        batch_size = 100
        artist_list = list(artist_ids)

        for i in range(0, len(artist_list), batch_size):
            batch = artist_list[i:i+batch_size]
            # Retry logic for timeout errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    rows = self.supabase.table('artists')\
                        .select('*')\
                        .in_('id', batch)\
                        .execute()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"\n   ⚠️  Timeout on batch, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise

            for r in rows.data:
                artists[r['id']] = {
                    'name': r.get('name'),
                    'spotify_url': r.get('spotify_url'),
                    'pronoun_title': r.get('pronoun_title', ''),
                    'genres': r.get('genres', ''),
                    'is_band': r.get('is_band'),
                    'ensemble_type': r.get('ensemble_type'),
                    'project_type_title': r.get('project_type_title'),
                    'project_type': r.get('project_type'),
                    'profile_type': r.get('profile_type'),
                    'formation_title': r.get('formation_title'),
                    'formation_type': r.get('formation_type'),
                    'formation_label': r.get('formation_label'),
                }

            print(f"   📦 Processed {min(i + batch_size, len(artist_list))}/{len(artist_list)} artists...", end='\r')

        elapsed = time.time() - start
        print(f"\n   ✅ Fetched {len(artists)} artist records in {elapsed:.1f}s")
        return artists

    def fetch_brand_affinity_batch(self, artist_ids):
        """Fetch brand affinity rows for artists."""
        if not artist_ids:
            return {}

        print(f"\n👜 Fetching brand affinity for {len(artist_ids)} artists...")
        start = time.time()

        affinity = {}
        batch_size = 500
        artist_list = list(artist_ids)

        for i in range(0, len(artist_list), batch_size):
            batch = artist_list[i:i+batch_size]
            # Retry logic for timeout errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    rows = self.supabase.table('artist_brand_affinity')\
                        .select('artist_id,brand_id,brand_name,type')\
                        .in_('artist_id', batch)\
                        .execute()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"\n   ⚠️  Timeout on batch, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise

            for r in rows.data or []:
                aid = r.get('artist_id')
                if aid is None:
                    continue
                entry = {
                    'brand_id': r.get('brand_id'),
                    'brand_name': r.get('brand_name'),
                    'type': r.get('type') or 'unknown'
                }
                affinity.setdefault(aid, []).append(entry)

            print(f"   📦 Processed {min(i + batch_size, len(artist_list))}/{len(artist_list)} artists for brand affinity...", end='\r')

        elapsed = time.time() - start
        total_affinity = sum(len(v) for v in affinity.values())
        print(f"\n   ✅ Fetched {total_affinity} brand affinity rows in {elapsed:.1f}s")
        return affinity

    def fetch_events_batch(self, artist_ids):
        """Fetch upcoming events for artists."""
        if not artist_ids:
            return {}

        print(f"\n🎟️  Fetching events for {len(artist_ids)} artists...")
        start = time.time()

        events = {}
        batch_size = 500
        artist_list = list(artist_ids)

        for i in range(0, len(artist_list), batch_size):
            batch = artist_list[i:i+batch_size]
            # Retry logic for timeout errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    rows = self.supabase.table('artist_events')\
                        .select('*')\
                        .in_('artist_id', batch)\
                        .execute()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"\n   ⚠️  Timeout on batch, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise

            for r in rows.data or []:
                aid = r.get('artist_id')
                if aid is None:
                    continue
                events.setdefault(aid, []).append(r)

            print(f"   📦 Processed {min(i + batch_size, len(artist_list))}/{len(artist_list)} artists for events...", end='\r')

        # Sort events per artist by start date for consistency
        for aid, ev_list in events.items():
            ev_list.sort(key=lambda e: e.get('start_date') or '')

        elapsed = time.time() - start
        total_events = sum(len(v) for v in events.values())
        print(f"\n   ✅ Fetched {total_events} events in {elapsed:.1f}s")
        return events

    def fetch_tiers_batch(self, artist_ids):
        """Fetch ALL 58 columns from last 5 snapshots for ML analysis."""
        print(f"\n🎯 Fetching FULL snapshot data (ALL 58 columns, last 5 snapshots) for {len(artist_ids)} artists...")
        start = time.time()

        tiers = {}
        artist_list = list(artist_ids)
        max_retries = 3

        # Fetch last 5 COMPLETE snapshots per artist (all 58 columns)
        for i, aid in enumerate(artist_list):
            retry_count = 0
            success = False
            result = None

            while retry_count < max_retries and not success:
                try:
                    # Get last 5 snapshots with ALL columns
                    result = self.supabase.table('artists_history')\
                        .select('*')\
                        .eq('artist_id', aid)\
                        .order('snapshot_date', desc=True)\
                        .limit(5)\
                        .execute()
                    success = True
                except Exception as e:
                    retry_count += 1
                    if 'timeout' in str(e).lower() or 'timed out' in str(e).lower() or 'disconnected' in str(e).lower():
                        wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                        print(f"\n   ⚠️  Timeout on artist {aid} (#{i+1}/{len(artist_list)}), retry {retry_count}/{max_retries} after {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n   ❌ Error on artist {aid}: {e}")
                        break

            if not success or not result:
                print(f"\n   ⚠️  Skipping artist {aid} after {max_retries} retries")
                continue

            if result.data:
                # Latest snapshot at top level (backward compatible + tier calculation)
                latest = result.data[0]
                listeners = latest.get('sp_monthly_listeners', 0)
                followers = latest.get('sp_followers', 0)

                # Build history array with ALL columns from previous snapshots
                history = []
                for row in result.data[1:]:  # Skip first (latest)
                    # Store the ENTIRE row (all 58 columns)
                    history.append(dict(row))

                tiers[aid] = {
                    # Top-level keys for backward compatibility (sonic matcher)
                    'listeners': listeners,
                    'followers': followers,
                    'snapshot_date': latest.get('snapshot_date'),
                    'tier': self._tier_from_listeners(listeners),
                    # Latest snapshot: ALL columns for ML
                    'latest_full': dict(latest),
                    # Historical snapshots: ALL columns for ML time-series
                    'history': history
                }

            # Progress every 50 artists
            if (i + 1) % 50 == 0 or (i + 1) == len(artist_list):
                print(f"   📦 Processed {i + 1}/{len(artist_list)} artists...", end='\r')

            # Small delay every 100 artists to avoid overwhelming Supabase
            if (i + 1) % 100 == 0:
                time.sleep(0.5)

        elapsed = time.time() - start
        snapshots_fetched = sum(1 + len(t.get('history', [])) for t in tiers.values())
        print(f"\n   ✅ Fetched FULL snapshot data for {len(tiers)} artists ({snapshots_fetched:,} total snapshots, ALL 58 columns) in {elapsed:.1f}s")
        return tiers

    def _tier_from_listeners(self, listeners):
        """Calculate tier from listener count."""
        try:
            value = float(listeners)
        except (TypeError, ValueError):
            return 'unknown'

        if value >= 1_000_000:
            return 'superstar'
        if value >= 100_000:
            return 'established'
        if value >= 50_000:
            return 'rising'
        if value >= 10_000:
            return 'mid'
        if value >= 5_000:
            return 'emerging'
        return 'micro'

    def build_emotion_index(self, gems_records):
        """Build reverse index: emotion -> [list of ISRCs]."""
        print("\n🎭 Building emotion index...")
        start = time.time()

        emotion_index = {}

        for record in gems_records:
            isrc = record.get('isrc')
            if not isrc:
                continue

            for i in range(1, 5):
                emotion = record.get(f'emotion_{i}')
                if emotion:
                    if emotion not in emotion_index:
                        emotion_index[emotion] = []
                    emotion_index[emotion].append(isrc)

        elapsed = time.time() - start
        print(f"   ✅ Built emotion index with {len(emotion_index)} emotions in {elapsed:.1f}s")
        return emotion_index

    def build_full_cache(self):
        """Build complete cache from scratch."""
        print("\n" + "="*70)
        print("🏗️  BUILDING COMPLETE GEMS UNIVERSE CACHE")
        print("="*70)

        total_start = time.time()

        # Step 1: Fetch all GEMS
        gems_records = self.fetch_all_gems()

        if not gems_records:
            print("❌ No GEMS records found!")
            return None

        # Step 2: Fetch all tracks
        isrc_list = [r['isrc'] for r in gems_records if r.get('isrc')]
        tracks = self.fetch_tracks_batch(isrc_list)

        # Step 3: Fetch all artists
        artist_ids = set()
        for track in tracks.values():
            if track.get('artist_id'):
                artist_ids.add(track['artist_id'])

        artists = self.fetch_artists_batch(artist_ids)

        # Step 4a: Fetch brand affinity
        brand_affinity = self.fetch_brand_affinity_batch(artist_ids)

        # Step 4: Fetch all tiers
        tiers = self.fetch_tiers_batch(artist_ids)

        # Step 4b: Fetch events
        events = self.fetch_events_batch(artist_ids)

        # Step 5: Build emotion index
        emotion_index = self.build_emotion_index(gems_records)

        # Build final cache structure
        cache = {
            'build_date': datetime.now().isoformat(),
            'version': '1.0',
            'gems': gems_records,
            'tracks': tracks,
            'artists': artists,
            'tiers': tiers,
            'brand_affinity': brand_affinity,
            'events': events,
            'emotion_index': emotion_index,
            'stats': {
                'total_gems': len(gems_records),
                'total_tracks': len(tracks),
                'total_artists': len(artists),
                'total_tiers': len(tiers),
                'total_emotions': len(emotion_index),
                'total_brand_affinity': sum(len(v) for v in brand_affinity.values()),
                'total_events': sum(len(v) for v in events.values())
            }
        }

        total_elapsed = time.time() - total_start

        print("\n" + "="*70)
        print("📊 CACHE BUILD COMPLETE")
        print("="*70)
        print(f"   GEMS records:    {cache['stats']['total_gems']:,}")
        print(f"   Tracks:          {cache['stats']['total_tracks']:,}")
        print(f"   Artists:         {cache['stats']['total_artists']:,}")
        print(f"   Tiers:           {cache['stats']['total_tiers']:,}")
        print(f"   Emotions:        {cache['stats']['total_emotions']:,}")
        print(f"   Total time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print("="*70)

        return cache

    def incremental_update(self):
        """Incrementally update existing cache with new records."""
        print("\n" + "="*70)
        print("🔄 INCREMENTAL CACHE UPDATE")
        print("="*70)

        existing_cache = self.load_existing_cache()

        if not existing_cache:
            print("   ⚠️  No existing cache found. Building from scratch...")
            return self.build_full_cache()

        total_start = time.time()

        # Get existing ISRCs
        existing_isrcs = {g['isrc'] for g in existing_cache.get('gems', []) if g.get('isrc')}
        print(f"   📊 Existing cache has {len(existing_isrcs)} GEMS records")

        # Fetch only new GEMS
        new_gems = self.fetch_all_gems(existing_isrcs)

        if not new_gems:
            print("   ✅ No new GEMS records found. Cache is up to date!")
            return existing_cache

        print(f"\n   🆕 Found {len(new_gems)} new GEMS records")

        # Fetch data for new ISRCs
        new_isrcs = [g['isrc'] for g in new_gems if g.get('isrc')]
        new_tracks = self.fetch_tracks_batch(new_isrcs)

        # Fetch artists that we don't already have
        new_artist_ids = set()
        for track in new_tracks.values():
            aid = track.get('artist_id')
            if aid and str(aid) not in existing_cache.get('artists', {}):
                new_artist_ids.add(aid)

        if new_artist_ids:
            new_artists = self.fetch_artists_batch(new_artist_ids)
            new_tiers = self.fetch_tiers_batch(new_artist_ids)
        else:
            new_artists = {}
            new_tiers = {}
            print("\n   ℹ️  No new artists to fetch")

        # Merge with existing cache
        cache = existing_cache.copy()
        cache['gems'].extend(new_gems)
        cache['tracks'].update(new_tracks)
        cache['artists'].update(new_artists)
        cache['tiers'].update(new_tiers)

        # Refresh brand affinity and events for all artists in cache (keeps data current even if no new GEMS)
        all_artist_ids = set()
        for aid in cache['artists'].keys():
            try:
                all_artist_ids.add(int(aid))
            except Exception:
                all_artist_ids.add(aid)

        brand_affinity = self.fetch_brand_affinity_batch(all_artist_ids)
        events = self.fetch_events_batch(all_artist_ids)

        cache['brand_affinity'] = brand_affinity
        cache['events'] = events

        # Rebuild emotion index
        cache['emotion_index'] = self.build_emotion_index(cache['gems'])

        # Update metadata
        cache['build_date'] = datetime.now().isoformat()
        cache['last_update'] = datetime.now().isoformat()
        cache['stats'] = {
            'total_gems': len(cache['gems']),
            'total_tracks': len(cache['tracks']),
            'total_artists': len(cache['artists']),
            'total_tiers': len(cache['tiers']),
            'total_emotions': len(cache['emotion_index']),
            'new_gems_added': len(new_gems),
            'new_artists_added': len(new_artists),
            'total_brand_affinity': sum(len(v) for v in brand_affinity.values()),
            'total_events': sum(len(v) for v in events.values())
        }

        total_elapsed = time.time() - total_start

        print("\n" + "="*70)
        print("📊 INCREMENTAL UPDATE COMPLETE")
        print("="*70)
        print(f"   New GEMS added:   {len(new_gems):,}")
        print(f"   New tracks:       {len(new_tracks):,}")
        print(f"   New artists:      {len(new_artists):,}")
        print(f"   Total GEMS:       {cache['stats']['total_gems']:,}")
        print(f"   Total time:       {total_elapsed:.1f}s")
        print("="*70)

        return cache

    def save_cache(self, cache):
        """Save cache to disk."""
        print(f"\n💾 Saving cache to {self.cache_path}...")
        start = time.time()

        try:
            with self.cache_path.open('w', encoding='utf-8') as f:
                json.dump(cache, f, separators=(',', ':'))

            elapsed = time.time() - start
            file_size = self.cache_path.stat().st_size / (1024 * 1024)  # MB

            print(f"   ✅ Cache saved successfully in {elapsed:.1f}s")
            print(f"   📁 File size: {file_size:.1f} MB")
            print(f"   📍 Location: {self.cache_path}")

            return True
        except OSError as e:
            print(f"   ❌ Failed to save cache: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Build or update GEMS universe cache")
    parser.add_argument('--rebuild', action='store_true', help='Force complete rebuild instead of incremental update')
    parser.add_argument('--supabase-url', default=os.getenv('SUPABASE_URL', 'https://irjslmczmrwpmzdhfneq.supabase.co'))
    parser.add_argument('--supabase-key', default=os.getenv('SUPABASE_SERVICE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlyanNsbWN6bXJ3cG16ZGhmbmVxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NjcxNTg4NywiZXhwIjoyMDYyMjkxODg3fQ.d-oH6Wh4MYAL3c1qakQv-ntyaI2S0L2zMZaiQ8q8kIY'))
    parser.add_argument('--max-pages', type=int, default=None, help='Optional max pages to fetch (page size 1000). None = unlimited.')
    args = parser.parse_args()

    builder = UniverseCacheBuilder(args.supabase_url, args.supabase_key, max_pages=args.max_pages)

    if args.rebuild:
        cache = builder.build_full_cache()
    else:
        cache = builder.incremental_update()

    if cache:
        builder.save_cache(cache)
        print("\n✅ Done! Use --use-cache flag with final_sonic_matcher.py for instant matching.")
    else:
        print("\n❌ Cache build failed!")


if __name__ == '__main__':
    main()
