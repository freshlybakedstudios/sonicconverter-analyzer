"""
Real-time Chartmetric lookup for unknown artists.

When a user provides a Spotify URL that isn't in the GEMS cache, this module
queries Chartmetric to fetch the full artist profile — identical to what the
main discovery pipeline (discovery_events_work.py) collects — and upserts to
Supabase so the artist is fully populated for future cache rebuilds.

Returns a lightweight dict for the web analyzer to use immediately (genres,
tier, conversion rate, top track).
"""

import html
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from requests.exceptions import HTTPError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chartmetric API endpoints (same as discovery_events_work.py)
# ---------------------------------------------------------------------------
CM_TOKEN_URL = "https://api.chartmetric.com/api/token"
CM_SEARCH_URL = "https://api.chartmetric.com/api/search"
CM_ARTIST_URL = "https://api.chartmetric.com/api/artist/{artist_id}"
CM_URLS_URL = "https://api.chartmetric.com/api/artist/{artist_id}/urls"
CM_TRACKS_URL = "https://api.chartmetric.com/api/artist/{artist_id}/tracks"
CM_TRACK_META_URL = "https://api.chartmetric.com/api/track/{track_id}"
CM_TRACK_PLAYLISTS_URL = "https://api.chartmetric.com/api/track/{track_id}/spotify/{status}/playlists"

# Rate limiting — thread-safe, 1 req/s CM plan
_last_call = 0.0
_RATE_INTERVAL = 1.1  # ~0.9 req/s, safely under 1 req/s CM limit
_rate_lock = threading.Lock()

# Retry settings — match discovery_events_work.py exactly
MAX_RETRIES = 5
BACKOFF_FACTOR = 3

# Track pagination — match discovery_events_work.py
TRACK_FETCH_LIMIT = 100
MAX_TRACK_PAGES = None  # fetch all pages

# Track name exclusion patterns — match discovery_events_work.py
TRACK_NAME_EXCLUDE_PATTERNS = [
    re.compile(r'\blive\b', re.IGNORECASE),
    re.compile(r'\bremix\b', re.IGNORECASE),
]

# Solo track detection roles — match discovery_events_work.py
MAIN_ARTIST_ROLES = {
    'main', 'primary', 'lead', 'main artist', 'lead artist'
}
NON_SOLO_ARTIST_ROLES = {
    'featured', 'feature', 'guest', 'featuring', 'support',
    'secondary', 'remixer', 'composer', 'producer', 'writer'
}


def _rate_wait():
    global _last_call
    with _rate_lock:
        now = time.time()
        gap = now - _last_call
        if gap < _RATE_INTERVAL:
            time.sleep(_RATE_INTERVAL - gap)
        _last_call = time.time()


def _retry(fn, *args, **kwargs):
    """Retry with backoff on 429/5xx — same pattern as discovery_events_work.py."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except HTTPError as e:
            status = getattr(e.response, 'status_code', None)
            if status == 429:
                retry_after = e.response.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait = max(1.0, float(retry_after) / 1000.0 + 0.5)
                    except (ValueError, TypeError):
                        wait = delay
                else:
                    wait = delay
                logger.warning(f"Rate limit 429 — retry {attempt+1}/{MAX_RETRIES} after {wait:.1f}s")
                time.sleep(wait)
                delay = min(delay * 1.5, 15.0)  # Cap at 15s
            elif status in (500, 502, 503, 504):
                logger.warning(f"Server {status} — retry {attempt+1}/{MAX_RETRIES} after {delay}s")
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                raise
        except Exception:
            raise
    logger.error(f"Exceeded {MAX_RETRIES} retries for {fn.__name__}")
    return None


# ---------------------------------------------------------------------------
# Chartmetric auth
# ---------------------------------------------------------------------------
_cached_token = None
_token_ts = 0.0


def get_cm_token(refresh_token: str) -> str | None:
    """Get a bearer token, caching for 55 minutes."""
    global _cached_token, _token_ts
    if _cached_token and (time.time() - _token_ts) < 55 * 60:
        return _cached_token

    _rate_wait()
    resp = requests.post(CM_TOKEN_URL, json={"refreshtoken": refresh_token}, timeout=30)
    if resp.status_code != 200:
        logger.error(f"Token request failed: {resp.status_code} {resp.text}")
        return None
    data = resp.json()
    token = data.get('token') or (data.get('data') or {}).get('token')
    if token:
        _cached_token = token
        _token_ts = time.time()
        logger.info("Chartmetric token obtained")
    return token


# ---------------------------------------------------------------------------
# Search: Spotify URL → CM artist ID
# ---------------------------------------------------------------------------
def _search_artist_by_spotify(token: str, spotify_url: str) -> int | None:
    """Search Chartmetric for an artist by their Spotify URL, return CM artist ID."""
    def _call():
        _rate_wait()
        resp = requests.get(
            CM_SEARCH_URL,
            params={"q": spotify_url},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        artists = resp.json().get('obj', {}).get('artists', [])
        if artists:
            return artists[0].get('id')
        return None
    return _retry(_call)


# ---------------------------------------------------------------------------
# Fetch artist metadata (same as discovery_events_work.py fetch_artist_metadata)
# ---------------------------------------------------------------------------
def _fetch_metadata(token: str, cm_id: int) -> dict | None:
    def _call():
        _rate_wait()
        resp = requests.get(
            CM_ARTIST_URL.format(artist_id=cm_id),
            headers={
                "Authorization": f"Bearer {token}",
                "X-Accept-Partial-Data": "true",
            },
            timeout=30,
        )
        if resp.status_code == 406:
            logger.warning(f"Partial data not available for {cm_id}")
            return None
        resp.raise_for_status()
        return resp.json().get('obj', {})
    return _retry(_call)


# ---------------------------------------------------------------------------
# Fetch artist URLs (same as discovery_events_work.py fetch_artist_urls)
# ---------------------------------------------------------------------------
def _fetch_urls(token: str, cm_id: int) -> dict:
    def _call():
        _rate_wait()
        resp = requests.get(
            CM_URLS_URL.format(artist_id=cm_id),
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        url_dict = {}
        for item in resp.json().get('obj', []):
            domain = item.get('domain')
            urls = item.get('url', [])
            if domain and urls:
                url_dict[domain] = urls[0]
        return url_dict
    return _retry(_call) or {}


# ---------------------------------------------------------------------------
# Solo track detection — same as discovery_events_work.py
# ---------------------------------------------------------------------------
def _normalize_artist_name(name):
    if not isinstance(name, str):
        return ""
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def _extract_track_artist_ids(track):
    ids = set()
    cm_values = track.get('cm_artist')
    if isinstance(cm_values, list):
        for value in cm_values:
            try:
                ids.add(int(value))
            except (TypeError, ValueError):
                continue
    artists_meta = track.get('artists')
    if isinstance(artists_meta, list):
        for entry in artists_meta:
            if not isinstance(entry, dict):
                continue
            try:
                artist_id = entry.get('id')
                if artist_id is not None:
                    ids.add(int(artist_id))
            except (TypeError, ValueError):
                continue
    return ids


def _extract_track_artist_names(track):
    names = set()
    raw_names = track.get('artist_names')
    if isinstance(raw_names, list):
        for name in raw_names:
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    artists_meta = track.get('artists')
    if isinstance(artists_meta, list):
        for entry in artists_meta:
            if isinstance(entry, dict):
                name = entry.get('name')
                if isinstance(name, str) and name.strip():
                    names.add(name.strip())
    return names


def _is_solo_track(track, artist_id, artist_name=None):
    ids = _extract_track_artist_ids(track)
    try:
        target_id = int(artist_id)
    except (TypeError, ValueError):
        target_id = None

    role = track.get('artist_type')
    if isinstance(role, str):
        normalized_role = ' '.join(role.replace('_', ' ').replace('-', ' ').lower().split())
        if normalized_role in NON_SOLO_ARTIST_ROLES:
            return False
        if normalized_role in MAIN_ARTIST_ROLES:
            if not ids or target_id is None or target_id in ids:
                return True

    if ids:
        if target_id is not None:
            return ids == {target_id}
        return len(ids) == 1

    names = _extract_track_artist_names(track)
    if names:
        normalized_names = {_normalize_artist_name(name) for name in names if _normalize_artist_name(name)}
        if artist_name:
            target_name = _normalize_artist_name(artist_name)
            if target_name:
                return normalized_names == {target_name}
        return len(normalized_names) == 1

    return True


def _track_name_filtered(name: Optional[str]) -> bool:
    if not name:
        return False
    for pattern in TRACK_NAME_EXCLUDE_PATTERNS:
        if pattern.search(name):
            return True
    return False


# ---------------------------------------------------------------------------
# Fetch top track — same logic as discovery_events_work.py fetch_artist_top_track
# Selects newest solo track (not score-based), with name/collab filtering,
# paginated track fetching, and post-metadata re-validation.
# ---------------------------------------------------------------------------
def _fetch_top_track(token: str, cm_id: int, artist_name: str = None) -> dict | None:
    def _call():
        headers = {"Authorization": f"Bearer {token}"}

        # --- Paginated track fetch (same as discovery_events_work.py) ---
        url = CM_TRACKS_URL.format(artist_id=cm_id)
        offset = 0
        all_tracks = []
        total_available = None
        page = 0

        while True:
            if MAX_TRACK_PAGES is not None and page >= MAX_TRACK_PAGES:
                break
            _rate_wait()
            params = {'limit': TRACK_FETCH_LIMIT, 'offset': offset}
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json() or {}
            batch = data.get('obj') or []
            total_available = data.get('total', total_available)

            if not batch:
                break

            all_tracks.extend(batch)

            if total_available is not None and len(all_tracks) >= total_available:
                break
            if len(batch) < TRACK_FETCH_LIMIT:
                break

            offset += TRACK_FETCH_LIMIT
            page += 1

        if not all_tracks:
            return None

        # --- Filter + date-sort candidates (same as discovery_events_work.py) ---
        dated_candidates = []
        undated_candidates = []

        for track in all_tracks:
            track_name = track.get('name', 'Unknown')

            # Name filter (skip live/remix)
            if _track_name_filtered(track_name):
                continue

            # Solo filter (skip collaborations)
            if not _is_solo_track(track, cm_id, artist_name):
                continue

            release_dates = track.get('release_dates') or []
            valid_dates = []
            for date_str in release_dates:
                try:
                    d = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if d.year >= 2006:
                        valid_dates.append(d)
                except Exception:
                    continue

            if valid_dates:
                most_recent_date = max(valid_dates)
                track['_most_recent_date'] = most_recent_date
                dated_candidates.append(track)
            else:
                undated_candidates.append(track)

        # Sort dated by newest first, then append undated
        ordered_candidates = sorted(
            dated_candidates,
            key=lambda x: x['_most_recent_date'],
            reverse=True,
        )
        ordered_candidates.extend(undated_candidates)

        if not ordered_candidates:
            return None

        # --- Iterate through candidates, fetch metadata, re-validate ---
        for track in ordered_candidates:
            track_id = track.get('id') or track.get('cm_track')
            selected_release_date = None
            if '_most_recent_date' in track:
                selected_release_date = track['_most_recent_date'].strftime('%Y-%m-%d')

            # Fetch full track metadata
            if track_id:
                _rate_wait()
                meta_resp = requests.get(
                    CM_TRACK_META_URL.format(track_id=track_id),
                    headers=headers, timeout=30,
                )
                if meta_resp.status_code == 200:
                    metadata = meta_resp.json().get('obj', {})
                    track.update(metadata)

            # Re-check name filter after metadata update
            track_name = track.get('name', 'Unknown')
            if _track_name_filtered(track_name):
                continue

            # Re-check solo filter after metadata update
            if not _is_solo_track(track, cm_id, artist_name):
                continue

            if selected_release_date:
                track['selected_release_date'] = selected_release_date

            # Stash catalog size so caller can surface it
            track['_catalog_size'] = total_available or len(all_tracks)

            # Fetch playlist details (always, same as discovery_events_work.py)
            if track_id:
                try:
                    editorial_info, independent_info = _fetch_track_playlists(token, track_id)
                    track['editorial_playlists_info'] = editorial_info
                    track['independent_playlists_info'] = independent_info
                except Exception as e:
                    logger.error(f"Playlist fetch failed for track {track_id}: {e}")
                    track['editorial_playlists_info'] = 'N/A'
                    track['independent_playlists_info'] = 'N/A'
            else:
                track['editorial_playlists_info'] = 'N/A'
                track['independent_playlists_info'] = 'N/A'

            return track

        # All candidates were filtered out
        return None

    return _retry(_call)


# ---------------------------------------------------------------------------
# Fetch track playlists — same as discovery_events_work.py fetch_track_playlists
# ---------------------------------------------------------------------------
def _fetch_track_playlists(token: str, track_id: int) -> tuple[str, str]:
    editorial_list = []
    indie_list = []

    for status in ('current', 'past'):
        _rate_wait()
        # Params match discovery_events_work.py exactly
        params = {
            'editorial': True,
            'indie': True,
            'personalized': True,
            'chart': True,
            'newMusicFriday': True,
            'radio': True,
            'brand': True,
            'majorCurator': True,
            'popularIndie': True,
            'thisIs': True,
            'fullyPersonalized': True,
            'audiobook': False,
            'limit': 50,
            'offset': 0,
            'sortColumn': 'followers',
        }
        resp = requests.get(
            CM_TRACK_PLAYLISTS_URL.format(track_id=track_id, status=status),
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=30,
        )
        if resp.status_code == 404:
            continue
        resp.raise_for_status()

        for item in resp.json().get('obj', []):
            pl = item.get('playlist', {})
            if not pl:
                continue
            # Skip playlists with missing critical fields (same validation as discovery)
            if not pl.get('playlist_id') or not pl.get('name'):
                continue
            info = {
                'name': pl.get('name'),
                'tags': ", ".join(t.get('name', '') for t in pl.get('tags', [])) or 'N/A',
                'link': f"https://open.spotify.com/playlist/{pl.get('playlist_id')}",
                'followers': pl.get('followers') or 0,
            }
            if pl.get('editorial'):
                editorial_list.append(info)
            else:
                indie_list.append(info)

        if status == 'current':
            time.sleep(1.0)

    def _fmt(lst, limit=5):
        if not lst:
            return "N/A"
        top = sorted(lst, key=lambda x: x['followers'], reverse=True)[:limit]
        return "; ".join(
            f"Name: {p['name']} | Genres: {p['tags']} | Link: {p['link']}"
            for p in top
        )

    return _fmt(editorial_list), _fmt(indie_list)


# ---------------------------------------------------------------------------
# Supabase upsert — identical payloads to discovery_events_work.py
# ---------------------------------------------------------------------------
def _supabase_headers(api_key: str, project_ref: str) -> dict:
    return {
        'apikey': api_key,
        'Authorization': f"Bearer {api_key}",
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Prefer': 'return=representation',
        'sb-project-ref': project_ref,
    }


def _upsert_artist(supa_url: str, api_key: str, project_ref: str,
                    meta: dict, artist_urls: dict):
    """Upsert artist row — same payload as discovery_events_work.py upsert_artist."""
    career_status = meta.get('career_status', {})

    # Extract genres — primary + secondary (matches discovery_events_work.py lines 1608-1620)
    genres_list = []
    genres_obj = meta.get('genres', {})
    if isinstance(genres_obj, dict):
        # Primary genre first
        primary = genres_obj.get('primary', {})
        if primary and primary.get('name'):
            genres_list.append(primary['name'])
        # Then secondary genres
        for g in genres_obj.get('secondary', []):
            if isinstance(g, dict) and g.get('name'):
                genres_list.append(g['name'])

    moods_list = [m['name'] for m in meta.get('moods', []) if isinstance(m, dict) and m.get('name')]
    activities_list = [a['name'] for a in meta.get('activities', []) if isinstance(a, dict) and a.get('name')]

    # Clean HTML from description (same as discovery_events_work.py)
    description = meta.get('description', '')
    if description:
        description = re.sub(r'<[^>]+>', '', description)
        description = description.replace('&amp;', '&')
        description = description.replace('&lt;', '<')
        description = description.replace('&gt;', '>')
        description = description.replace('&quot;', '"')
        description = description.replace('&#39;', "'")
        description = description.replace('&#34;', '"')
        description = description.replace('&#x1f4a6;', '\U0001f4a6')
        description = html.unescape(description)

    payload = {
        'id': str(meta.get('id')),
        'name': meta.get('name'),
        'code2': meta.get('code2'),
        'pronoun_title': meta.get('pronoun_title'),
        'band': 'Yes' if str(meta.get('band', '')).lower() == 'true' else 'No',
        'record_label': meta.get('record_label'),
        'genres': ', '.join(genres_list) if genres_list else None,
        'moods': ', '.join(moods_list) if moods_list else None,
        'activities': ', '.join(activities_list) if activities_list else None,
        'career_status_stage': career_status.get('stage'),
        'career_status_trend': career_status.get('trend'),
        'hometown_city': meta.get('hometown_city'),
        'booking_agent': meta.get('booking_agent'),
        'press_contact': meta.get('press_contact'),
        'band_members': meta.get('band_members'),
        'general_manager': meta.get('general_manager'),
    }

    urls = artist_urls or {}
    if urls.get('spotify'):
        payload['spotify_url'] = urls['spotify']
    if urls.get('instagram'):
        payload['instagram_url'] = urls['instagram']
    if urls.get('website'):
        payload['website_url'] = urls['website']
    if urls.get('facebook'):
        payload['facebook_url'] = urls['facebook']
    if description:
        payload['description'] = description
    image_url = meta.get('image_url') or meta.get('cover_url')
    if image_url:
        payload['image_url'] = image_url

    headers = _supabase_headers(api_key, project_ref)
    url = f"{supa_url}/rest/v1/artists"
    artist_id = meta.get('id')

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        logger.info(f"CM lookup: created artist {meta.get('name')} ({artist_id})")
    except HTTPError as e:
        if e.response.status_code == 409:
            resp = requests.patch(f"{url}?id=eq.'{artist_id}'", json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            logger.info(f"CM lookup: updated artist {meta.get('name')} ({artist_id})")
        else:
            raise


def _upsert_track(supa_url: str, api_key: str, project_ref: str,
                   artist_id: int, track_data: dict):
    """Upsert track row — same payload as discovery_events_work.py upsert_track."""
    if not track_data:
        return

    headers = _supabase_headers(api_key, project_ref)

    # Check if track already exists
    check_resp = requests.get(
        f"{supa_url}/rest/v1/tracks?artist_id=eq.{artist_id}",
        headers=headers, timeout=30,
    )
    check_resp.raise_for_status()
    if check_resp.json():
        logger.info(f"CM lookup: track already exists for artist {artist_id}")
        return

    spotify_url = None
    if track_data.get('spotify_track_ids'):
        spotify_url = f"https://open.spotify.com/track/{track_data['spotify_track_ids'][0]}"

    stats = track_data.get('cm_statistics', {}) or {}

    # Extract moods and activities
    moods = track_data.get('moods', [])
    activities = track_data.get('activities', [])
    mood_str = None
    if moods:
        mood_names = [m.get('name', '') for m in moods if isinstance(m, dict) and m.get('name')]
        mood_str = ', '.join(mood_names) if mood_names else None
    activity_str = None
    if activities:
        activity_names = [a.get('name', '') for a in activities if isinstance(a, dict) and a.get('name')]
        activity_str = ', '.join(activity_names) if activity_names else None

    # Track genre extraction hierarchy (same as discovery_events_work.py lines 1293-1309)
    track_genres_value = None
    if isinstance(track_data.get('genre'), str) and track_data.get('genre').strip():
        track_genres_value = track_data.get('genre').strip()
    elif isinstance(track_data.get('genres'), list):
        names = [g.strip() for g in track_data['genres'] if isinstance(g, str) and g.strip()]
        if names:
            track_genres_value = ', '.join(names)
    elif isinstance(track_data.get('tags'), list):
        tags = [t.strip() for t in track_data['tags'] if isinstance(t, str) and t.strip()]
        if tags:
            track_genres_value = ', '.join(tags)
    elif isinstance(track_data.get('tags'), str) and track_data.get('tags').strip():
        track_genres_value = track_data.get('tags').strip()
    if not track_genres_value:
        track_genres_value = 'Others'

    payload = {
        'artist_id': str(artist_id),
        'top_track': track_data.get('name'),
        'track_genres': track_genres_value,
        'spotify_url': spotify_url,
        'editorial_playlists': stats.get('num_sp_editorial_playlists') or 0,
        'user_playlists': stats.get('num_sp_playlists') or 0,
        'mood': mood_str,
        'activity': activity_str,
        'spotify_plays': stats.get('sp_streams') or 0,
        'editorial_playlists_info': track_data.get('editorial_playlists_info', 'N/A'),
        'independent_playlists_info': track_data.get('independent_playlists_info', 'N/A'),
        'isrc': track_data.get('isrc'),
        'cm_track': track_data.get('id') or track_data.get('cm_track'),
        'release_date': track_data.get('selected_release_date'),
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }

    resp = requests.post(f"{supa_url}/rest/v1/tracks", json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    logger.info(f"CM lookup: saved track '{track_data.get('name')}' for artist {artist_id}")


def _insert_history(supa_url: str, api_key: str, project_ref: str, meta: dict):
    """Insert history snapshot — same payload as discovery_events_work.py insert_history."""
    stats = meta.get('cm_statistics', {})

    # cm_artist_score fallback (same as discovery_events_work.py lines 1783-1790)
    cm_score = (
        meta.get('cm_artist_score')
        or meta.get('cm_score')
        or (stats.get('cm_artist_score') if isinstance(stats, dict) else None)
    )
    if cm_score is None:
        cm_score = meta.get('cm_artist_rank')

    snapshot = {
        'artist_id': str(meta.get('id')),
        'snapshot_date': datetime.now(timezone.utc).isoformat(),
        'cm_artist_score': cm_score,
        'cm_artist_rank': meta.get('cm_artist_rank'),
        'rank_eg': meta.get('rank_eg'),
        'rank_fb': meta.get('rank_fb'),
        # Spotify
        'sp_followers': stats.get('sp_followers'),
        'sp_monthly_listeners': stats.get('sp_monthly_listeners'),
        'sp_followers_to_listeners_ratio': stats.get('sp_followers_to_listeners_ratio'),
        'sp_listeners_to_followers_ratio': stats.get('sp_listeners_to_followers_ratio'),
        'sp_popularity': stats.get('sp_popularity'),
        'sp_playlist_total_reach': stats.get('sp_playlist_total_reach'),
        'spotify_playlist_count': stats.get('num_sp_playlists'),
        # TikTok
        'tiktok_followers': stats.get('tiktok_followers'),
        'tiktok_likes': stats.get('tiktok_likes'),
        'tiktok_track_posts': stats.get('tiktok_track_posts'),
        'tiktok_top_video_views': stats.get('tiktok_top_video_views'),
        # YouTube
        'ycs_subscribers': stats.get('ycs_subscribers'),
        'ycs_views': stats.get('ycs_views'),
        'youtube_daily_video_views': stats.get('youtube_daily_video_views'),
        'youtube_monthly_video_views': stats.get('youtube_monthly_video_views'),
        'youtube_playlist_count': stats.get('num_yt_playlists'),
        # Instagram
        'ins_followers': stats.get('ins_followers'),
        # Other streaming
        'pandora_listeners_28_day': stats.get('pandora_listeners_28_day'),
        'pandora_lifetime_streams': stats.get('pandora_lifetime_streams'),
        'pandora_lifetime_stations_added': stats.get('pandora_lifetime_stations_added'),
        'shazam_count': stats.get('shazam_count'),
        'deezer_fans': stats.get('deezer_fans'),
        'deezer_playlist_count': stats.get('num_de_playlists'),
        'deezer_playlist_total_reach': stats.get('de_playlist_total_reach'),
        'soundcloud_followers': stats.get('soundcloud_followers'),
        'soundcloud_plays': stats.get('soundcloud_plays'),
        'itunes_playlist_count': stats.get('num_am_playlists'),
        'amazon_playlist_count': stats.get('num_az_playlists'),
        # Social
        'facebook_followers': stats.get('facebook_followers'),
        'ts_followers': stats.get('twitter_followers'),
        'songkick_fans': stats.get('songkick_fans'),
        'genius_views': stats.get('genius_pageviews'),
        'twitch_followers': stats.get('twitch_followers'),
        # Asian platforms
        'line_music_artist_likes': stats.get('line_music_artist_likes'),
        'line_music_likes': stats.get('line_music_likes'),
        'line_music_mv_plays': stats.get('line_music_mv_plays'),
        'line_music_plays': stats.get('line_music_plays'),
        'melon_artist_fans': stats.get('melon_artist_fans'),
        'melon_likes': stats.get('melon_likes'),
        'melon_video_likes': stats.get('melon_video_likes'),
        'melon_video_views': stats.get('melon_video_views'),
        # Boomplay
        'boomplay_streams': stats.get('boomplay_plays'),
        'boomplay_favorites': stats.get('boomplay_favorites'),
    }

    headers = _supabase_headers(api_key, project_ref)
    try:
        resp = requests.post(f"{supa_url}/rest/v1/artists_history", json=snapshot, headers=headers, timeout=30)
        resp.raise_for_status()
        logger.info(f"CM lookup: snapshot created for artist {meta.get('id')}")
    except HTTPError as e:
        if e.response.status_code == 409:
            logger.debug(f"CM lookup: snapshot already exists today for {meta.get('id')}")
        else:
            raise


def _do_upsert(supa_url: str, api_key: str, project_ref: str,
               meta: dict, artist_urls: dict, track_data: dict):
    """Run all 3 Supabase upserts. Called from a background thread."""
    try:
        _upsert_artist(supa_url, api_key, project_ref, meta, artist_urls)
        _upsert_track(supa_url, api_key, project_ref, meta.get('id'), track_data)
        _insert_history(supa_url, api_key, project_ref, meta)
    except Exception as e:
        logger.error(f"CM lookup: Supabase upsert failed: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
TIER_RANGES = {
    'micro': (0, 5_000),
    'emerging': (5_000, 10_000),
    'mid': (10_000, 50_000),
    'rising': (50_000, 100_000),
    'established': (100_000, 1_000_000),
    'superstar': (1_000_000, float('inf')),
}


def _listeners_to_tier(listeners: float) -> str:
    for tier, (lo, hi) in TIER_RANGES.items():
        if lo <= listeners < hi:
            return tier
    return 'unknown'


def lookup_artist_by_spotify(spotify_url: str) -> dict | None:
    """
    Look up a Spotify artist in Chartmetric in real time.

    Mirrors discovery_events_work.py exactly:
    - Same API endpoints, same data extraction, same Supabase payloads
    - Same track selection (newest solo track, not score-based)
    - Same filters (live/remix name filter, collaboration filter)
    - Same playlist params, same genre hierarchy, same cm_score fallback

    Returns a dict for the web analyzer, plus fires a background Supabase upsert.
    Returns None on failure.
    """
    refresh_token = os.getenv('REFRESH_TOKEN')
    if not refresh_token:
        logger.warning("CM lookup: REFRESH_TOKEN not set, skipping")
        return None

    try:
        token = get_cm_token(refresh_token)
        if not token:
            return None

        # 1) Search by Spotify URL
        cm_id = _search_artist_by_spotify(token, spotify_url)
        if not cm_id:
            logger.info(f"CM lookup: no artist found for {spotify_url}")
            return None

        logger.info(f"CM lookup: found CM artist {cm_id} for {spotify_url}")

        # 2) Fetch full metadata
        meta = _fetch_metadata(token, cm_id)
        if not meta:
            return None

        artist_name = meta.get('name', '')

        # 3) Fetch URLs
        artist_urls = _fetch_urls(token, cm_id)

        # 4) Fetch top track (with playlists) — passes artist_name for solo detection
        track_data = _fetch_top_track(token, cm_id, artist_name=artist_name)

        # Extract fields for the analyzer
        stats = meta.get('cm_statistics', {})
        listeners = float(stats.get('sp_monthly_listeners') or 0)
        followers = float(stats.get('sp_followers') or 0)

        # Genres — primary + secondary (same as discovery_events_work.py)
        genres_list = []
        genres_obj = meta.get('genres', {})
        if isinstance(genres_obj, dict):
            primary = genres_obj.get('primary', {})
            if primary and primary.get('name'):
                genres_list.append(primary['name'])
            for g in genres_obj.get('secondary', []):
                if isinstance(g, dict) and g.get('name'):
                    genres_list.append(g['name'])
        genres_str = ', '.join(genres_list) if genres_list else ''

        career = meta.get('career_status', {})

        # Conversion rate — same formula as track_matcher.py
        conversion_rate = None
        if listeners > 0 and followers > 0:
            conversion_rate = round((followers * 0.1) / (listeners * 4.3) * 100, 2)

        # Top track summary
        top_track_info = None
        if track_data:
            track_spotify = None
            if track_data.get('spotify_track_ids'):
                track_spotify = f"https://open.spotify.com/track/{track_data['spotify_track_ids'][0]}"
            top_track_info = {
                'name': track_data.get('name', ''),
                'isrc': track_data.get('isrc', ''),
                'spotify_url': track_spotify or '',
            }

        # Catalog size from track pagination
        catalog_size = (track_data or {}).get('_catalog_size', 20)

        result = {
            'cm_id': cm_id,
            'name': artist_name,
            'genres': genres_str,
            'career_stage': career.get('stage', ''),
            'listeners': listeners,
            'followers': followers,
            'tier': _listeners_to_tier(listeners),
            'conversion_rate': conversion_rate,
            'catalog_size': catalog_size,
            'top_track': top_track_info,
            '_meta': meta,
            '_urls': artist_urls,
            '_track_data': track_data,
        }

        # 5) Fire-and-forget Supabase upsert in background thread
        supa_url = os.getenv('SUPABASE_URL')
        supa_key = os.getenv('SUPABASE_SERVICE_KEY')
        if supa_url and supa_key:
            project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
            t = threading.Thread(
                target=_do_upsert,
                args=(supa_url, supa_key, project_ref, meta, artist_urls, track_data),
                daemon=True,
            )
            t.start()
            logger.info("CM lookup: Supabase upsert started in background")

        return result

    except Exception as e:
        logger.error(f"CM lookup failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Historical listener data from Chartmetric
# ---------------------------------------------------------------------------
CM_STAT_SPOTIFY_URL = "https://api.chartmetric.com/api/artist/{artist_id}/stat/spotify"


def fetch_listener_history(token: str, cm_id: int, days: int = 365) -> list[dict]:
    """
    Fetch monthly listener history from Chartmetric stat/spotify endpoint.
    Returns list of {date: "YYYY-MM-DD", listeners: number} sorted ascending.
    Fetches in 90-day chunks (CM API limit), samples one point per month.
    """
    until_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')

    all_points = []
    seen_dates = set()

    # Fetch in 90-day chunks
    chunk_start = datetime.strptime(since_date, '%Y-%m-%d')
    end_date = datetime.strptime(until_date, '%Y-%m-%d')

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=90), end_date)

        def _call(s=chunk_start.strftime('%Y-%m-%d'), e=chunk_end.strftime('%Y-%m-%d')):
            _rate_wait()
            resp = requests.get(
                CM_STAT_SPOTIFY_URL.format(artist_id=cm_id),
                params={'since': s, 'until': e},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get('obj', {}).get('listeners', [])

        try:
            chunk_data = _retry(_call) or []
            for point in chunk_data:
                ts = point.get('timestp', '')
                val = point.get('value', 0)
                date_key = ts[:10] if ts else ''
                if date_key and date_key not in seen_dates and val and val > 0:
                    seen_dates.add(date_key)
                    all_points.append({'date': date_key, 'listeners': int(val)})
        except Exception as e:
            logger.warning(f"CM history chunk failed for {cm_id}: {e}")

        chunk_start = chunk_end
        time.sleep(1.1)  # rate limit between chunks

    if not all_points:
        return []

    # Sort by date ascending
    all_points.sort(key=lambda p: p['date'])

    # Sample one point per month (first of each month wins)
    monthly = []
    last_month = ''
    for p in all_points:
        month_key = p['date'][:7]  # YYYY-MM
        if month_key != last_month:
            monthly.append(p)
            last_month = month_key

    # Always include the last point
    if all_points and (not monthly or monthly[-1] != all_points[-1]):
        monthly.append(all_points[-1])

    return monthly


def fetch_artist_events(token: str, cm_id: int, lookback_days: int = 730, lookahead_days: int = 365) -> list[dict]:
    """
    Fetch past + future events from Chartmetric for an artist.
    Returns raw event dicts with 'start_date', 'venue_capacity', etc.
    Past events use status='past', future use status='current'.
    """
    CM_EVENTS_URL = "https://api.chartmetric.com/api/artist/{artist_id}/{status}/events"
    PAGE_LIMIT = 50
    MAX_PAGES = 10
    all_events = []

    for status, from_days, to_days in [
        ("past", lookback_days, 0),
        ("current", 0, -abs(lookahead_days)),
    ]:
        offset = 0
        pages = 0
        while pages < MAX_PAGES:
            def _call(o=offset):
                _rate_wait()
                resp = requests.get(
                    CM_EVENTS_URL.format(artist_id=cm_id, status=status),
                    params={
                        'fromDaysAgo': from_days,
                        'toDaysAgo': to_days,
                        'limit': PAGE_LIMIT,
                        'offset': o,
                    },
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                )
                if resp.status_code != 200:
                    return {}
                return resp.json() or {}

            try:
                page = _retry(_call) or {}
            except Exception:
                break
            page_items = page.get('obj') or []
            all_events.extend(page_items)
            if len(page_items) < PAGE_LIMIT:
                break
            offset += PAGE_LIMIT
            pages += 1

    return all_events


# ---------------------------------------------------------------------------
# ISRC → CM Track ID resolution with Supabase caching
# ---------------------------------------------------------------------------
CM_SEARCH_TRACKS_URL = "https://api.chartmetric.com/api/search"
CM_RELATED_ARTISTS_URL = "https://api.chartmetric.com/api/artist/{artist_id}/relatedartists"


def _resolve_isrc_to_cm_track_id(token: str, isrc: str) -> int | None:
    """
    Resolve an ISRC to a Chartmetric track ID.
    Checks Supabase cache first, then falls back to CM search API.
    """
    if not isrc:
        return None

    # Check Supabase cache
    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')
    if supa_url and supa_key:
        try:
            headers = _supabase_headers(supa_key, supa_url.split('//', 1)[1].split('.', 1)[0])
            resp = requests.get(
                f"{supa_url}/rest/v1/isrc_cm_track_map?isrc=eq.{isrc}&select=cm_track_id",
                headers=headers, timeout=10,
            )
            if resp.status_code == 200 and resp.json():
                cached = resp.json()[0]
                cm_id = cached.get('cm_track_id')
                if cm_id:
                    return int(cm_id)
        except Exception as e:
            logger.debug(f"ISRC cache check failed for {isrc}: {e}")

    # Search CM by ISRC
    def _call():
        _rate_wait()
        resp = requests.get(
            CM_SEARCH_TRACKS_URL,
            params={"q": isrc, "type": "tracks"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        tracks = resp.json().get('obj', {}).get('tracks', [])
        if tracks:
            return tracks[0].get('id')
        return None

    cm_track_id = _retry(_call)

    # Cache the result
    if cm_track_id and supa_url and supa_key:
        try:
            project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
            headers = _supabase_headers(supa_key, project_ref)
            payload = {
                'isrc': isrc,
                'cm_track_id': cm_track_id,
                'resolved_at': datetime.now(timezone.utc).isoformat(),
            }
            requests.post(
                f"{supa_url}/rest/v1/isrc_cm_track_map",
                json=payload, headers=headers, timeout=10,
            )
        except Exception as e:
            logger.debug(f"ISRC cache write failed for {isrc}: {e}")

    return cm_track_id


# ---------------------------------------------------------------------------
# Structured playlist fetching (returns list of dicts instead of strings)
# ---------------------------------------------------------------------------
def _upsert_playlist_tracks(isrc: str, artist_name: str, track_name: str,
                            playlists: list[dict]):
    """Upsert playlist-track associations into playlist_tracks table."""
    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')
    if not supa_url or not supa_key or not isrc or not playlists:
        return
    try:
        headers = {
            'apikey': supa_key,
            'Authorization': f'Bearer {supa_key}',
            'Content-Type': 'application/json',
            'Prefer': 'resolution=ignore-duplicates',
        }
        rows = []
        for pl in playlists:
            pid = pl.get('playlist_id', '')
            if not pid:
                continue
            rows.append({
                'playlist_id': pid,
                'spotify_track_id': isrc,  # best we have without Spotify track ID
                'isrc': isrc,
                'track_name': track_name or '',
                'artist_names': artist_name or '',
                'playlist_name': pl.get('name', ''),
                'added_at': (pl.get('added_at') or '')[:10] or None,
                'position': 0,
            })
        if rows:
            resp = requests.post(
                f"{supa_url}/rest/v1/playlist_tracks",
                json=rows, headers=headers, timeout=15,
            )
            if resp.status_code < 300:
                logger.debug(f"Upserted {len(rows)} playlist_tracks for ISRC {isrc}")
            else:
                logger.debug(f"playlist_tracks upsert status {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.debug(f"playlist_tracks upsert failed for {isrc}: {e}")


def _fetch_track_playlists_structured(token: str, track_id: int,
                                      max_stale_days: int = 180,
                                      cache_days: int = 7,
                                      isrc: str = '',
                                      artist_name: str = '',
                                      track_name: str = '') -> list[dict]:
    """
    Fetch playlists for a track, returning structured data for scoring.
    Checks Supabase cache first (valid for cache_days).
    Filters out stale playlists not updated within max_stale_days.
    """
    import json as _json
    from datetime import datetime, timezone

    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')

    # --- Check cache ---
    if supa_url and supa_key:
        try:
            project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
            headers = _supabase_headers(supa_key, project_ref)
            resp = requests.get(
                f"{supa_url}/rest/v1/track_playlists_cache"
                f"?cm_track_id=eq.{track_id}&select=playlists,fetched_at",
                headers=headers, timeout=10,
            )
            if resp.status_code == 200 and resp.json():
                row = resp.json()[0]
                fetched_at = row.get('fetched_at', '')
                if fetched_at:
                    dt = datetime.fromisoformat(fetched_at.replace('Z', '+00:00'))
                    age_days = (datetime.now(timezone.utc) - dt).days
                    if age_days <= cache_days:
                        cached = row.get('playlists', [])
                        if isinstance(cached, str):
                            cached = _json.loads(cached)
                        logger.debug(f"Playlist cache hit for track {track_id} ({len(cached)} playlists, {age_days}d old)")
                        return cached
        except Exception as e:
            logger.debug(f"Playlist cache check failed for track {track_id}: {e}")

    # --- Fetch from CM API ---
    results = []

    for status in ('current', 'past'):
        def _call(s=status):
            _rate_wait()
            params = {
                'editorial': True,
                'indie': True,
                'personalized': True,
                'chart': True,
                'newMusicFriday': True,
                'radio': True,
                'brand': True,
                'majorCurator': True,
                'popularIndie': True,
                'thisIs': True,
                'fullyPersonalized': True,
                'audiobook': False,
                'limit': 100,
                'offset': 0,
                'sortColumn': 'followers',
            }
            resp = requests.get(
                CM_TRACK_PLAYLISTS_URL.format(track_id=track_id, status=s),
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=30,
            )
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return resp.json().get('obj', [])

        items = _retry(_call) or []

        # Debug: log first playlist's raw fields to see what CM gives us
        if items and status == 'current':
            sample = items[0].get('playlist', {})
            logger.warning(f"CM playlist raw keys: {list(sample.keys())}")
            logger.warning(f"CM playlist sample: curator_id={sample.get('curator_id')}, "
                          f"owner_name={sample.get('owner_name')}, "
                          f"curator_name={sample.get('curator_name')}, "
                          f"owner_id={sample.get('owner_id')}, "
                          f"spotify_owner_id={sample.get('spotify_owner_id')}, "
                          f"user_id={sample.get('user_id')}")

        for item in items:
            pl = item.get('playlist', {})
            if not pl or not pl.get('playlist_id') or not pl.get('name'):
                continue
            followers = pl.get('followers') or 0
            if followers < 1:
                continue
            last_updated = pl.get('last_updated') or pl.get('updated_at') or ''
            added_at = item.get('added_at') or ''

            # Filter stale playlists — use best available date
            best_date = last_updated or added_at
            if best_date and max_stale_days > 0:
                try:
                    dt = datetime.fromisoformat(best_date.replace('Z', '+00:00'))
                    days_old = (datetime.now(timezone.utc) - dt).days
                    if days_old > max_stale_days:
                        continue
                except (ValueError, TypeError):
                    pass

            tags = [t.get('name', '') for t in pl.get('tags', []) if t.get('name')]
            results.append({
                'name': pl.get('name', ''),
                'playlist_id': pl.get('playlist_id', ''),
                'cm_playlist_id': pl.get('id', ''),
                'link': f"https://open.spotify.com/playlist/{pl.get('playlist_id')}",
                'followers': followers,
                'tags': tags,
                'editorial': bool(pl.get('editorial')),
                'curator_name': pl.get('curator_name') or pl.get('owner_name') or '',
                'cm_curator_id': pl.get('curator_id') or pl.get('owner_id') or '',
                'spotify_user_id': pl.get('user_id') or '',
                'status': status,
                'last_updated': last_updated,
                'added_at': added_at,
            })

        if status == 'current':
            time.sleep(1.0)

    # --- Write to cache ---
    if supa_url and supa_key and results:
        try:
            project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
            headers = _supabase_headers(supa_key, project_ref)
            # Upsert — update if exists, insert if not
            headers['Prefer'] = 'resolution=merge-duplicates,return=representation'
            payload = {
                'cm_track_id': track_id,
                'playlists': _json.dumps(results),
                'fetched_at': datetime.now(timezone.utc).isoformat(),
            }
            requests.post(
                f"{supa_url}/rest/v1/track_playlists_cache",
                json=payload, headers=headers, timeout=10,
            )
        except Exception as e:
            logger.debug(f"Playlist cache write failed for track {track_id}: {e}")

    # --- Upsert into playlist_tracks for long-term storage ---
    if results and isrc:
        _upsert_playlist_tracks(isrc, artist_name, track_name, results)

    return results


# ---------------------------------------------------------------------------
# Curator contact resolution
# ---------------------------------------------------------------------------
def _fetch_curator_contact(token: str, cm_curator_id: int,
                           cache_days: int = 30) -> dict:
    """
    Fetch curator social URLs and submission email from Chartmetric.
    Checks Supabase cache first (valid for cache_days).
    Returns {email, instagram_url, facebook_url, website_url, submission_url, ...}
    """
    import json as _json
    from datetime import datetime, timezone

    result = {}
    if not cm_curator_id:
        return result

    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')

    # --- Check cache ---
    if supa_url and supa_key:
        try:
            project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
            headers = _supabase_headers(supa_key, project_ref)
            resp = requests.get(
                f"{supa_url}/rest/v1/curator_contacts_cache"
                f"?curator_key=eq.{cm_curator_id}&select=contact_data,fetched_at",
                headers=headers, timeout=10,
            )
            if resp.status_code == 200 and resp.json():
                row = resp.json()[0]
                fetched_at = row.get('fetched_at', '')
                if fetched_at:
                    dt = datetime.fromisoformat(fetched_at.replace('Z', '+00:00'))
                    age_days = (datetime.now(timezone.utc) - dt).days
                    if age_days <= cache_days:
                        cached = row.get('contact_data', {})
                        if isinstance(cached, str):
                            cached = _json.loads(cached)
                        logger.debug(f"Curator cache hit for {cm_curator_id} ({age_days}d old)")
                        return cached
        except Exception as e:
            logger.debug(f"Curator cache check failed for {cm_curator_id}: {e}")

    # --- Fetch from CM API ---

    # 1) Get curator profile (may have submission email)
    def _call_profile():
        _rate_wait()
        resp = requests.get(
            f"https://api.chartmetric.com/api/curator/spotify/{cm_curator_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json().get('obj', {})

    try:
        profile = _retry(_call_profile) or {}
        raw = profile.get('profile_raw') or {}

        # Check submission fields
        sub = raw.get('submission') or {}
        if not sub and isinstance(raw.get('submissions'), list) and raw['submissions']:
            sub = raw['submissions'][0]
        if sub.get('email'):
            result['email'] = sub['email']
        if sub.get('url'):
            result['submission_url'] = sub['url']
        if sub.get('accepting'):
            result['accepts_submissions'] = True

        result['curator_name'] = profile.get('name') or ''
    except Exception as e:
        logger.debug(f"Curator profile fetch failed for {cm_curator_id}: {e}")

    # 2) Get curator social URLs
    def _call_urls():
        _rate_wait()
        resp = requests.get(
            f"https://api.chartmetric.com/api/curator/spotify/{cm_curator_id}/urls",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return resp.json().get('obj', [])

    try:
        urls = _retry(_call_urls) or []
        for item in urls:
            domain = (item.get('domain') or '').lower()
            url_list = item.get('url') or []
            if isinstance(url_list, str):
                url_list = [url_list]
            url = url_list[0] if url_list else ''
            if not url:
                continue
            if domain == 'instagram':
                result['instagram_url'] = url
            elif domain == 'facebook':
                result['facebook_url'] = url
            elif domain == 'website':
                result['website_url'] = url
            elif domain == 'twitter':
                result['twitter_url'] = url
            elif domain == 'groover':
                result['groover_url'] = url
            elif domain == 'submithub':
                result['submithub_url'] = url
    except Exception as e:
        logger.debug(f"Curator URLs fetch failed for {cm_curator_id}: {e}")

    # --- Write to cache (even empty results, to avoid re-querying) ---
    if supa_url and supa_key:
        try:
            project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
            headers = _supabase_headers(supa_key, project_ref)
            headers['Prefer'] = 'resolution=merge-duplicates,return=representation'
            payload = {
                'curator_key': str(cm_curator_id),
                'contact_data': _json.dumps(result),
                'fetched_at': datetime.now(timezone.utc).isoformat(),
            }
            requests.post(
                f"{supa_url}/rest/v1/curator_contacts_cache",
                json=payload, headers=headers, timeout=10,
            )
        except Exception as e:
            logger.debug(f"Curator cache write failed for {cm_curator_id}: {e}")

    return result


# ---------------------------------------------------------------------------
# Related Artists
# ---------------------------------------------------------------------------
def _fetch_related_artists(token: str, cm_id: int, limit: int = 50) -> list[dict]:
    """
    Fetch CM related artists for audience overlap cross-referencing.
    Returns list of {name, cm_id, spotify_url} dicts.
    """
    def _call():
        _rate_wait()
        resp = requests.get(
            CM_RELATED_ARTISTS_URL.format(artist_id=cm_id),
            params={"limit": limit},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        items = resp.json().get('obj', [])
        results = []
        for item in items:
            results.append({
                'name': item.get('name', ''),
                'cm_id': item.get('id'),
                'spotify_url': item.get('spotify_url') or '',
            })
        return results

    return _retry(_call) or []


# ---------------------------------------------------------------------------
# Upsert extracted audio features into gems_complete_analysis
# ---------------------------------------------------------------------------
def _upsert_gems_features(isrc: str, features: dict, genre: str = '',
                           secondary_genre: str = ''):
    """
    Insert extracted audio features into gems_complete_analysis.
    Uses INSERT with ON CONFLICT DO NOTHING so we never overwrite
    richer data from the full GEMS pipeline.
    """
    if not isrc or not features:
        return

    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')
    if not supa_url or not supa_key:
        return

    # Map extract_features() keys → gems_complete_analysis columns
    # Only include columns that exist in the table
    record = {'isrc': isrc}

    # Core audio features
    audio_keys = [
        'sub_ratio', 'bass_ratio', 'low_mid_ratio', 'mid_ratio',
        'high_mid_ratio', 'presence_ratio', 'air_ratio',
        'lufs_integrated', 'loudness_range', 'energy', 'dynamic_range',
        'crest_factor', 'compression_amount', 'attack_time',
        'brightness', 'brightness_variance', 'spectral_rolloff',
        'spectral_complexity', 'spectral_flux',
        'key', 'scale', 'key_strength', 'zcr', 'dissonance',
        'bpm', 'beat_strength', 'onset_rate', 'danceability',
    ]
    for k in audio_keys:
        v = features.get(k)
        if v is not None:
            record[k] = v

    # Emotions
    for i in range(1, 5):
        ek = f'emotion_{i}'
        esk = f'emotion_{i}_score'
        if features.get(ek):
            record[ek] = features[ek]
        if features.get(esk) is not None:
            record[esk] = features[esk]

    # Genre
    if genre:
        # Split "Pop, Rock" into primary + secondary
        parts = [g.strip() for g in genre.split(',') if g.strip()]
        if parts:
            record['primary_genre'] = parts[0]
        if len(parts) > 1:
            record['secondary_genre'] = ', '.join(parts[1:])
    if secondary_genre and 'secondary_genre' not in record:
        record['secondary_genre'] = secondary_genre

    # Timestamp
    record['analyzed_at'] = datetime.now(timezone.utc).isoformat()

    try:
        project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
        headers = _supabase_headers(supa_key, project_ref)
        # ON CONFLICT DO NOTHING — don't overwrite existing (richer) data
        headers['Prefer'] = 'resolution=ignore-duplicates,return=representation'

        resp = requests.post(
            f"{supa_url}/rest/v1/gems_complete_analysis",
            json=record,
            headers=headers,
            timeout=30,
        )
        if resp.status_code in (200, 201):
            if resp.json():
                logger.info(f"GEMS upsert: inserted new record for ISRC {isrc}")
            else:
                logger.info(f"GEMS upsert: ISRC {isrc} already exists (skipped)")
        elif resp.status_code == 409:
            logger.info(f"GEMS upsert: ISRC {isrc} already exists (conflict)")
        else:
            resp.raise_for_status()
    except Exception as e:
        logger.error(f"GEMS upsert failed for ISRC {isrc}: {e}")


def _lookup_gems_features(isrc: str) -> dict | None:
    """Look up cached audio features from gems_complete_analysis by ISRC.
    Returns a features dict compatible with extract_features() output, or None."""
    if not isrc:
        return None
    supa_url = os.getenv('SUPABASE_URL')
    supa_key = os.getenv('SUPABASE_SERVICE_KEY')
    if not supa_url or not supa_key:
        return None

    audio_keys = [
        'sub_ratio', 'bass_ratio', 'low_mid_ratio', 'mid_ratio',
        'high_mid_ratio', 'presence_ratio', 'air_ratio',
        'lufs_integrated', 'loudness_range', 'energy', 'dynamic_range',
        'crest_factor', 'compression_amount', 'attack_time',
        'brightness', 'brightness_variance', 'spectral_rolloff',
        'spectral_complexity', 'spectral_flux',
        'key', 'scale', 'key_strength', 'zcr', 'dissonance',
        'bpm', 'beat_strength', 'onset_rate', 'danceability',
        'emotion_1', 'emotion_1_score', 'emotion_2', 'emotion_2_score',
        'emotion_3', 'emotion_3_score', 'emotion_4', 'emotion_4_score',
    ]
    select_cols = ','.join(audio_keys)

    try:
        project_ref = supa_url.split('//', 1)[1].split('.', 1)[0]
        headers = _supabase_headers(supa_key, project_ref)
        resp = requests.get(
            f"{supa_url}/rest/v1/gems_complete_analysis"
            f"?isrc=eq.{isrc}&select={select_cols}&limit=1",
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            rows = resp.json()
            if rows:
                row = rows[0]
                # Only return if we have core features (not just an empty row)
                if row.get('bpm') is not None and row.get('energy') is not None:
                    features = {k: v for k, v in row.items() if v is not None}
                    logger.info(f"GEMS cache hit for ISRC {isrc} ({len(features)} features)")
                    return features
    except Exception as e:
        logger.debug(f"GEMS feature lookup failed for {isrc}: {e}")
    return None


# ---------------------------------------------------------------------------
# Track Credits (producers/writers)
# ---------------------------------------------------------------------------
def _extract_track_credits(token: str, cm_track_id: int) -> dict:
    """
    Fetch track metadata and extract artist roles (producer, writer, etc.).
    Returns {producers: [{name, cm_id, role}], writers: [{name, cm_id, role}]}
    """
    def _call():
        _rate_wait()
        resp = requests.get(
            CM_TRACK_META_URL.format(track_id=cm_track_id),
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json().get('obj', {})

    metadata = _retry(_call)
    if not metadata:
        return {'producers': [], 'writers': []}

    producers = []
    writers = []

    artists = metadata.get('artists', [])
    if isinstance(artists, list):
        for artist in artists:
            if not isinstance(artist, dict):
                continue
            role = (artist.get('role') or artist.get('artist_type') or '').lower()
            name = artist.get('name', '')
            cm_id = artist.get('id')

            entry = {'name': name, 'cm_id': cm_id, 'role': role}

            if any(r in role for r in ('producer', 'production')):
                producers.append(entry)
            elif any(r in role for r in ('writer', 'songwriter', 'composer', 'lyricist')):
                writers.append(entry)

    return {'producers': producers, 'writers': writers}
