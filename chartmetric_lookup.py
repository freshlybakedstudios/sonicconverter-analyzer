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

# Rate limiting — same 1.5 s cadence as discovery_events_work.py
_last_call = 0.0
_RATE_INTERVAL = 1.5

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
                wait = (float(retry_after) / 1000.0 + 0.5) if retry_after else delay
                logger.warning(f"Rate limit 429 — retry {attempt+1}/{MAX_RETRIES} after {wait:.1f}s")
                time.sleep(wait)
                delay *= 1.5
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

        result = {
            'cm_id': cm_id,
            'name': artist_name,
            'genres': genres_str,
            'career_stage': career.get('stage', ''),
            'listeners': listeners,
            'followers': followers,
            'tier': _listeners_to_tier(listeners),
            'conversion_rate': conversion_rate,
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
