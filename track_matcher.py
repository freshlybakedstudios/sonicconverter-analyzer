"""
Sonic matching engine for uploaded tracks against the GEMS universe cache.
Adapted from final_sonic_matcher.py - runs entirely from the pre-built cache.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize(value, min_val, max_val, default=0.5):
    if value is None:
        return default
    val = float(value)
    return max(0, min(1, (val - min_val) / (max_val - min_val)))


def _genre_words(genre_str: Optional[str]) -> Set[str]:
    if not genre_str or genre_str == 'unknown':
        return set()
    return {w.strip() for w in genre_str.lower().replace(',', ' ').split() if w.strip()}


def _parse_track_genres(genre_str: Optional[str]) -> Set[str]:
    if not genre_str:
        return set()
    lower = genre_str.lower()
    if 'genre id:' in lower or lower in {'others', 'other', 'various'}:
        return set()
    return {t.strip() for t in lower.replace(',', ' ').split() if t.strip()}


INCOMPATIBLE_GENRE_PAIRS = {
    frozenset(['death', 'folk']),
    frozenset(['death', 'acoustic']),
    frozenset(['death', 'singer-songwriter']),
    frozenset(['death', 'classical']),
    frozenset(['death', 'ambient']),
    frozenset(['death', 'meditation']),
    frozenset(['deathcore', 'folk']),
    frozenset(['deathcore', 'acoustic']),
    frozenset(['deathcore', 'classical']),
    frozenset(['deathcore', 'ambient']),
    frozenset(['deathcore', 'singer-songwriter']),
    frozenset(['death', 'gospel']),
    frozenset(['death', 'worship']),
    frozenset(['death', 'christian']),
    frozenset(['deathcore', 'gospel']),
    frozenset(['deathcore', 'worship']),
    frozenset(['deathcore', 'christian']),
    frozenset(['classical', 'dubstep']),
    frozenset(['classical', 'trap']),
    frozenset(['classical', 'drill']),
    frozenset(['classical', 'hardcore']),
    frozenset(['classical', 'deathcore']),
    frozenset(['classical', 'death']),
    frozenset(['opera', 'trap']),
    frozenset(['opera', 'drill']),
    frozenset(['opera', 'metal']),
    frozenset(['opera', 'death']),
    frozenset(['meditation', 'metal']),
    frozenset(['meditation', 'hardcore']),
    frozenset(['meditation', 'deathcore']),
    frozenset(['meditation', 'death']),
    frozenset(['meditation', 'punk']),
    frozenset(['ambient', 'death']),
    frozenset(['ambient', 'deathcore']),
    frozenset(['children', 'metal']),
    frozenset(['children', 'hardcore']),
    frozenset(['children', 'death']),
    frozenset(['children', 'deathcore']),
    frozenset(['nursery', 'metal']),
    frozenset(['nursery', 'hardcore']),
    frozenset(['nursery', 'death']),
    frozenset(['bluegrass', 'death']),
    frozenset(['bluegrass', 'deathcore']),
    frozenset(['bluegrass', 'dubstep']),
    frozenset(['country', 'death']),
    frozenset(['country', 'deathcore']),
    frozenset(['metal', 'flamenco']),
    frozenset(['metalcore', 'flamenco']),
    frozenset(['hardcore', 'flamenco']),
}

# ---------------------------------------------------------------------------
# Genre family clustering — used to penalise cross-family matches
# Maps individual genre words → a family identifier.
# Only unambiguous words are mapped; unknown words are simply ignored.
# ---------------------------------------------------------------------------
GENRE_FAMILIES = {
    # Rock
    'rock': 'rock', 'indie': 'rock', 'alternative': 'rock', 'alt': 'rock',
    'shoegaze': 'rock', 'grunge': 'rock', 'britpop': 'rock',
    'post-rock': 'rock', 'noise-rock': 'rock', 'surf': 'rock',
    'glam': 'rock',
    # Metal
    'metal': 'metal', 'death': 'metal', 'black': 'metal', 'doom': 'metal',
    'thrash': 'metal', 'speed': 'metal', 'metalcore': 'metal',
    'deathcore': 'metal', 'sludge': 'metal', 'djent': 'metal',
    'nu-metal': 'metal',
    # Punk
    'punk': 'punk', 'post-hardcore': 'punk', 'emo': 'punk',
    'post-punk': 'punk', 'screamo': 'punk', 'crust': 'punk',
    'pop-punk': 'punk',
    # Pop
    'pop': 'pop', 'synthpop': 'pop', 'electropop': 'pop',
    'bubblegum': 'pop', 'k-pop': 'pop', 'j-pop': 'pop',
    'dance-pop': 'pop',
    # Hip-hop
    'hip-hop': 'hiphop', 'rap': 'hiphop', 'trap': 'hiphop',
    'drill': 'hiphop', 'grime': 'hiphop',
    # Electronic
    'electronic': 'electronic', 'edm': 'electronic', 'house': 'electronic',
    'techno': 'electronic', 'trance': 'electronic', 'dubstep': 'electronic',
    'dnb': 'electronic', 'ambient': 'electronic', 'idm': 'electronic',
    'electronica': 'electronic',
    # R&B / Soul
    'r&b': 'rnb', 'rnb': 'rnb', 'soul': 'rnb', 'funk': 'rnb',
    'neo-soul': 'rnb', 'motown': 'rnb',
    # Country / Folk
    'country': 'folk', 'folk': 'folk', 'bluegrass': 'folk',
    'americana': 'folk', 'singer-songwriter': 'folk', 'acoustic': 'folk',
    # Jazz
    'jazz': 'jazz', 'swing': 'jazz', 'bebop': 'jazz', 'bossa': 'jazz',
    # Classical
    'classical': 'classical', 'orchestra': 'classical',
    'chamber': 'classical', 'baroque': 'classical', 'opera': 'classical',
    # Latin
    'latin': 'latin', 'reggaeton': 'latin', 'salsa': 'latin',
    'bachata': 'latin', 'cumbia': 'latin',
    # Reggae
    'reggae': 'reggae', 'ska': 'reggae', 'dancehall': 'reggae',
}


def _genre_families(genre_words: Set[str]) -> Set[str]:
    """Return the set of genre families for a set of genre words."""
    families = set()
    for w in genre_words:
        fam = GENRE_FAMILIES.get(w)
        if fam:
            families.add(fam)
    return families


class TrackMatcher:
    """Loads the GEMS universe cache and matches uploaded track features against it."""

    def __init__(self, cache_path: Optional[str] = None):
        if cache_path is None:
            cache_path = str(Path(__file__).resolve().parent / 'cache' / 'universe' / 'gems_universe.json')
        self.cache_path = cache_path
        self.cache = None
        self._gems_list = []
        self._tracks = {}
        self._artists = {}
        self._tiers = {}
        self._emotion_index = {}

    def load_cache(self):
        """Load the 1.2 GB universe cache into memory."""
        print(f"Loading universe cache from {self.cache_path}...")
        start = time.time()
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            self.cache = json.load(f)
        elapsed = time.time() - start

        self._gems_list = self.cache.get('gems', [])
        self._gems_by_isrc = {g['isrc']: g for g in self._gems_list if g.get('isrc')}
        self._tracks = self.cache.get('tracks', {})
        self._artists = self.cache.get('artists', {})
        self._tiers = self.cache.get('tiers', {})
        self._emotion_index = self.cache.get('emotion_index', {})

        stats = self.cache.get('stats', {})
        print(f"  Loaded in {elapsed:.1f}s — {stats.get('total_gems', 0):,} GEMS records")

    def _build_profile(self, row: Dict) -> Dict:
        """Convert a GEMS cache row into a normalized profile dict."""
        return {
            'sub_ratio': row.get('sub_ratio', 0.02),
            'bass_ratio': row.get('bass_ratio', 0.2),
            'low_mid_ratio': row.get('low_mid_ratio', 0.1),
            'mid_ratio': row.get('mid_ratio', 0.25),
            'high_mid_ratio': row.get('high_mid_ratio', 0.2),
            'presence_ratio': row.get('presence_ratio', 0.15),
            'air_ratio': row.get('air_ratio', 0.2),
            'energy': row.get('energy', 0.5),
            'dynamic_range': _normalize(row.get('dynamic_range'), 5, 30, 15),
            'loudness_range': _normalize(row.get('loudness_range'), 1, 15, 5),
            'attack_time': _normalize(row.get('attack_time'), 0.01, 10, 2),
            'compression_amount': _normalize(row.get('compression_amount'), 0, 2, 1),
            'crest_factor': _normalize(row.get('crest_factor'), 1, 10, 4),
            'beat_strength': _normalize(row.get('beat_strength'), 0.5, 2.0, 1.0),
            'onset_rate': _normalize(row.get('onset_rate'), 0, 10, 5),
            'danceability': _normalize(row.get('danceability'), 0.8, 1.5, 1.1),
            'lufs_integrated': _normalize(row.get('lufs_integrated'), -30, -6, -12),
            'dissonance': row.get('dissonance', 0.1),
            'key_strength': row.get('key_strength', 0.5),
            'zcr': _normalize(row.get('zcr'), 0, 0.15, 0.05),
            'emotion_1': row.get('emotion_1', 'neutral'),
            'emotion_1_score': row.get('emotion_1_score', 0.5),
            'emotion_2': row.get('emotion_2', 'neutral'),
            'emotion_2_score': row.get('emotion_2_score', 0),
            'emotion_3': row.get('emotion_3', 'neutral'),
            'emotion_3_score': row.get('emotion_3_score', 0),
            'emotion_4': row.get('emotion_4', 'neutral'),
            'emotion_4_score': row.get('emotion_4_score', 0),
            'primary_genre': row.get('primary_genre', 'unknown'),
            'secondary_genre': row.get('secondary_genre', 'unknown'),
            'bpm': row.get('bpm', 120),
        }

    def _build_uploaded_profile(self, features: Dict) -> Dict:
        """Normalize uploaded track features into the same profile format."""
        return {
            'sub_ratio': features.get('sub_ratio', 0.02),
            'bass_ratio': features.get('bass_ratio', 0.2),
            'low_mid_ratio': features.get('low_mid_ratio', 0.1),
            'mid_ratio': features.get('mid_ratio', 0.25),
            'high_mid_ratio': features.get('high_mid_ratio', 0.2),
            'presence_ratio': features.get('presence_ratio', 0.15),
            'air_ratio': features.get('air_ratio', 0.2),
            'energy': features.get('energy', 0.5),
            'dynamic_range': _normalize(features.get('dynamic_range'), 5, 30, 15),
            'loudness_range': _normalize(features.get('loudness_range'), 1, 15, 5),
            'attack_time': _normalize(features.get('attack_time'), 0.01, 10, 2),
            'compression_amount': _normalize(features.get('compression_amount'), 0, 2, 1),
            'crest_factor': _normalize(features.get('crest_factor'), 1, 10, 4),
            'beat_strength': _normalize(features.get('beat_strength'), 0.5, 2.0, 1.0),
            'onset_rate': _normalize(features.get('onset_rate'), 0, 10, 5),
            'danceability': _normalize(features.get('danceability'), 0.8, 1.5, 1.1),
            'lufs_integrated': _normalize(features.get('lufs_integrated'), -30, -6, -12),
            'dissonance': features.get('dissonance', 0.1),
            'key_strength': features.get('key_strength', 0.5),
            'zcr': _normalize(features.get('zcr'), 0, 0.15, 0.05),
            'emotion_1': features.get('emotion_1', 'neutral'),
            'emotion_1_score': features.get('emotion_1_score', 0.5),
            'emotion_2': features.get('emotion_2', 'neutral'),
            'emotion_2_score': features.get('emotion_2_score', 0),
            'emotion_3': features.get('emotion_3', 'neutral'),
            'emotion_3_score': features.get('emotion_3_score', 0),
            'emotion_4': features.get('emotion_4', 'neutral'),
            'emotion_4_score': features.get('emotion_4_score', 0),
            'primary_genre': features.get('primary_genre', 'unknown'),
            'secondary_genre': features.get('secondary_genre', 'unknown'),
            'bpm': features.get('bpm', 120),
        }

    @staticmethod
    def _similarity(profile_a: Dict, profile_b: Dict,
                    track_genres_a: Set[str], track_genres_b: Set[str]):
        """33-feature weighted similarity. Returns (score, breakdown)."""
        breakdown = {}

        def safe_diff(key):
            va = profile_a.get(key)
            vb = profile_b.get(key)
            if va is None or vb is None:
                return 0.5
            return 1 - abs(va - vb)

        freq_scores = {
            b: safe_diff(b)
            for b in ['sub_ratio', 'bass_ratio', 'low_mid_ratio', 'mid_ratio',
                       'high_mid_ratio', 'presence_ratio', 'air_ratio']
        }
        breakdown['frequency_spectrum'] = {'score': float(np.mean(list(freq_scores.values()))), 'weight': 0.35}

        for key, w in [
            ('energy', 0.06), ('dynamic_range', 0.03), ('loudness_range', 0.02),
            ('attack_time', 0.02), ('compression_amount', 0.01), ('crest_factor', 0.01),
            ('beat_strength', 0.05), ('onset_rate', 0.03), ('danceability', 0.02),
            ('dissonance', 0.05), ('key_strength', 0.03), ('zcr', 0.02),
        ]:
            breakdown[key] = {'score': safe_diff(key), 'weight': w}

        # Emotion similarity
        emo_a, emo_b = [], []
        for idx in range(1, 5):
            ea = profile_a.get(f'emotion_{idx}', 'neutral')
            sa = profile_a.get(f'emotion_{idx}_score', 0)
            eb = profile_b.get(f'emotion_{idx}', 'neutral')
            sb = profile_b.get(f'emotion_{idx}_score', 0)
            if ea == eb:
                emo_a.append(sa); emo_b.append(sb)
            else:
                emo_a.append(0); emo_b.append(0)
        if sum(emo_a) > 0 and sum(emo_b) > 0:
            emo_sim = float(np.dot(emo_a, emo_b) / (np.linalg.norm(emo_a) * np.linalg.norm(emo_b)))
        else:
            emo_sim = 0
        breakdown['emotion'] = {'score': emo_sim, 'weight': 0.05}

        # Genre similarity (Jaccard)
        ga = (_genre_words(profile_a.get('primary_genre'))
              | _genre_words(profile_a.get('secondary_genre'))
              | track_genres_a)
        gb = (_genre_words(profile_b.get('primary_genre'))
              | _genre_words(profile_b.get('secondary_genre'))
              | track_genres_b)
        if ga and gb:
            genre_score = len(ga & gb) / len(ga | gb)
        else:
            genre_score = 0
        breakdown['genre'] = {'score': genre_score, 'weight': 0.08}  # Genre nudge — keeps results in the right lane

        # LUFS
        lufs_score = 1 - abs((profile_a.get('lufs_integrated') or 0) - (profile_b.get('lufs_integrated') or 0))
        breakdown['lufs_integrated'] = {'score': lufs_score, 'weight': 0.05}

        # Boost sonic features
        breakdown['frequency_spectrum']['weight'] = 0.38
        breakdown['emotion']['weight'] = 0.07

        total = sum(item['score'] * item['weight'] for item in breakdown.values())
        return total, breakdown

    def find_matches(self, features: Dict, genre_hint: str = '',
                     top_n: int = 20, threshold: float = 0.60) -> List[Dict]:
        """
        Find the top-N most sonically similar artists from the cache.

        Parameters
        ----------
        features : dict
            Raw features from audio_analyzer.extract_features().
        genre_hint : str
            Optional genre to use as the uploaded track's genre tag.
        top_n : int
            Number of matches to return.
        threshold : float
            Minimum similarity to include.
        """
        if not self._gems_list:
            raise RuntimeError("Cache not loaded — call load_cache() first")

        target_profile = self._build_uploaded_profile(features)
        # Set genre on the profile if provided
        if genre_hint:
            target_profile['primary_genre'] = genre_hint

        target_emotions = {
            target_profile.get('emotion_1'),
            target_profile.get('emotion_2'),
            target_profile.get('emotion_3'),
            target_profile.get('emotion_4'),
        } - {None, ''}

        target_genre_words = _genre_words(genre_hint)

        # Pre-filter by emotion overlap (candidates must share at least 1 emotion)
        candidate_isrcs = set()
        for emo in target_emotions:
            if emo in self._emotion_index:
                candidate_isrcs.update(self._emotion_index[emo])

        # If no emotion hits, scan all
        if not candidate_isrcs:
            candidate_isrcs = set(self._gems_by_isrc.keys())

        all_scored = []  # Score ALL tracks first, dedupe later

        for isrc in candidate_isrcs:
            row = self._gems_by_isrc.get(isrc)
            if not row:
                continue

            profile = self._build_profile(row)

            # Quick emotion pre-check (need at least 1 overlap)
            cand_emotions = {
                profile.get('emotion_1'),
                profile.get('emotion_2'),
                profile.get('emotion_3'),
                profile.get('emotion_4'),
            } - {None, ''}

            track_data = self._tracks.get(isrc, {})
            track_genres = _parse_track_genres(track_data.get('track_genres', ''))
            artist_id = track_data.get('artist_id')
            if not artist_id:
                continue

            similarity, breakdown = self._similarity(
                target_profile, profile, target_genre_words, track_genres
            )

            # Sonic penalties
            energy_d = abs(_float(profile.get('energy')) - _float(target_profile.get('energy')))
            beat_d = abs(_float(profile.get('beat_strength')) - _float(target_profile.get('beat_strength')))
            dance_d = abs(_float(profile.get('danceability')) - _float(target_profile.get('danceability')))
            onset_d = abs(_float(profile.get('onset_rate')) - _float(target_profile.get('onset_rate')))
            bpm_d = abs((_float(profile.get('bpm')) or 0) - (_float(target_profile.get('bpm')) or 0))

            penalty = 0.0
            if energy_d > 0.18:
                penalty += (energy_d - 0.18) * 0.6
            if beat_d > 0.18:
                penalty += (beat_d - 0.18) * 0.5
            if dance_d > 0.22:
                penalty += (dance_d - 0.22) * 0.4
            if onset_d > 0.22:
                penalty += (onset_d - 0.22) * 0.4
            if bpm_d > 28:
                penalty += (bpm_d - 28) * 0.0015
            penalty = min(penalty, 0.10)

            similarity = max(0.0, similarity - penalty)
            if similarity < threshold:
                continue

            # Emotion bonus
            emotion_overlap = len(cand_emotions & target_emotions)
            emotion_bonus = min(0.08, 0.05 * (emotion_overlap / 4.0))
            if emotion_bonus > 0:
                emotion_bonus = max(0.0, emotion_bonus - penalty * 0.5)
            similarity = min(1.0, similarity + emotion_bonus)

            if similarity < threshold:
                continue

            # Incompatible genre filter
            artist_data = self._artists.get(str(artist_id), {})
            artist_genres = _parse_track_genres(artist_data.get('genres', ''))
            cand_all = track_genres | artist_genres
            skip = False
            for pair in INCOMPATIBLE_GENRE_PAIRS:
                if target_genre_words & pair and cand_all & pair:
                    if (target_genre_words & pair) != (cand_all & pair):
                        skip = True
                        break
            if skip:
                continue

            # Genre family penalty — penalise candidates from a different genre family
            # so that e.g. "art rock" prioritises rock tracks over metal tracks.
            if target_genre_words:
                target_fams = _genre_families(target_genre_words)
                cand_fams = _genre_families(cand_all)
                if target_fams and cand_fams:
                    overlap = target_fams & cand_fams
                    if not overlap:
                        # Completely different family (rock vs metal) → 10 % penalty
                        similarity *= 0.90
                    elif cand_fams - target_fams:
                        # Crossover — shares a family but also has a foreign one
                        # (e.g. "rock metal" vs "rock") → 4 % penalty
                        similarity *= 0.96
                    if similarity < threshold:
                        continue

            # Tier + conversion data
            tier_data = self._tiers.get(str(artist_id), {})
            tier = tier_data.get('tier', 'unknown')
            listeners = _float(tier_data.get('listeners'), 0)
            followers = _float(tier_data.get('followers'), 0)

            conversion_rate = None
            if listeners > 0 and followers > 0:
                conversion_rate = round((followers * 0.1) / (listeners * 4.3) * 100, 2)

            all_scored.append({
                'isrc': isrc,
                'artist_id': artist_id,
                'name': artist_data.get('name', 'Unknown'),
                'spotify_url': artist_data.get('spotify_url', ''),
                'tier': tier,
                'listeners': listeners,
                'followers': followers,
                'conversion_rate': conversion_rate,
                'similarity': round(similarity, 4),
                'emotion_overlap': emotion_overlap,
                'track_name': track_data.get('top_track', ''),
                'track_url': track_data.get('spotify_url', ''),
                'track_genres': sorted(track_genres) if track_genres else [],
                'artist_genres': sorted(artist_genres) if artist_genres else [],
                'primary_genre': profile.get('primary_genre', ''),
                'secondary_genre': profile.get('secondary_genre', ''),
                'emotions': [
                    profile.get('emotion_1', ''),
                    profile.get('emotion_2', ''),
                    profile.get('emotion_3', ''),
                    profile.get('emotion_4', ''),
                ],
                'breakdown': {k: {'score': round(v['score'], 4), 'weight': v['weight']}
                              for k, v in breakdown.items()},
            })

        # Sort by similarity descending, then dedupe by artist (keep best match per artist)
        all_scored.sort(key=lambda m: m['similarity'], reverse=True)

        matches = []
        seen_artists = set()
        for m in all_scored:
            if m['artist_id'] not in seen_artists:
                seen_artists.add(m['artist_id'])
                matches.append(m)
                if len(matches) >= top_n:
                    break

        return matches

    def get_gems_for_artists(self, artist_ids: list) -> Dict[str, Dict]:
        """Return {artist_id: gems_row} for a list of artist IDs (first track per artist)."""
        aid_set = set(str(a) for a in artist_ids)
        aid_to_isrc = {}
        for isrc, tdata in self._tracks.items():
            aid = str(tdata.get('artist_id', ''))
            if aid in aid_set and aid not in aid_to_isrc:
                aid_to_isrc[aid] = isrc

        gems_by_isrc = {g['isrc']: g for g in self._gems_list if g.get('isrc')}
        results = {}
        for aid in artist_ids:
            isrc = aid_to_isrc.get(str(aid))
            if isrc and isrc in gems_by_isrc:
                results[str(aid)] = gems_by_isrc[isrc]
        return results
