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
    words = {w.strip() for w in genre_str.lower().replace(',', ' ').split() if w.strip()}
    # When specific sub-genre qualifiers are present, strip broad parent words
    # that would cause false matches (e.g. "black metal" shouldn't match "rock")
    _EXTREME_QUALIFIERS = {'black', 'death', 'brutal', 'doom', 'stoner', 'sludge',
                           'thrash', 'grind', 'goregrind', 'deathcore'}
    if words & _EXTREME_QUALIFIERS:
        words.discard('rock')  # "european rock" etc. is noise for extreme metal
    return words


def _parse_track_genres(genre_str: Optional[str]) -> Set[str]:
    """Split a comma-joined CM genre string into a set of TAG PHRASES.

    Multi-word tags like 'southeast asian pop' or 'us devotional & spiritual'
    are preserved as single entries — previously this function also split on
    spaces and slashes, which fragmented those phrases into bare tokens like
    'pop' or 'spiritual'. The lone 'pop' token then resolved (incorrectly)
    to the pop family, leaking 'southeast asian pop' artists into a pop lane
    and 'us devotional & spiritual' artists into a gospel lane via 'spiritual'.

    _genre_families (the downstream resolver) handles each phrase as: full-
    phrase match first, then regional-prefix stripping, then a word-level
    fallback for genuinely-unrecognised multi-word tags. So the families
    that legitimately come from word-level matches still surface; we just
    don't fragment proactively.
    """
    if not genre_str:
        return set()
    lower = genre_str.lower()
    if 'genre id:' in lower or lower in {'others', 'other', 'various'}:
        return set()
    return {t.strip() for t in lower.split(',') if t.strip()}


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
    # Metal sub-genre clashes — brutal/death vs softer metal-adjacent
    frozenset(['death', 'lofi']),
    frozenset(['death', 'indie']),
    frozenset(['death', 'pop']),
    frozenset(['death', 'emo']),
    frozenset(['death', 'skramz']),
    frozenset(['brutal', 'lofi']),
    frozenset(['brutal', 'indie']),
    frozenset(['brutal', 'pop']),
    frozenset(['brutal', 'emo']),
    frozenset(['brutal', 'skramz']),
    frozenset(['deathcore', 'lofi']),
    frozenset(['deathcore', 'indie']),
    frozenset(['deathcore', 'pop']),
    frozenset(['deathcore', 'emo']),
    frozenset(['deathcore', 'skramz']),
    frozenset(['goregrind', 'lofi']),
    frozenset(['goregrind', 'indie']),
    frozenset(['goregrind', 'pop']),
    frozenset(['goregrind', 'emo']),
    # Metal vs indie/pop
    frozenset(['metal', 'indie']),
    frozenset(['metal', 'lofi']),
    frozenset(['metalcore', 'lofi']),
    frozenset(['metalcore', 'indie']),
    # Black metal vs rock/pop/soft
    frozenset(['black', 'rock']),
    frozenset(['black', 'pop']),
    frozenset(['black', 'indie']),
    frozenset(['black', 'folk']),
    frozenset(['black', 'acoustic']),
    frozenset(['black', 'singer-songwriter']),
    frozenset(['black', 'country']),
    frozenset(['black', 'lofi']),
    # Metal sub-genre clashes — black metal vs slow/groovy metal
    frozenset(['black', 'stoner']),
    frozenset(['black', 'doom']),
    frozenset(['black', 'sludge']),
    frozenset(['black', 'groove']),
    frozenset(['black', 'nu']),
    frozenset(['black', 'funk']),
    frozenset(['black', 'hair']),
    frozenset(['black', 'southern']),
    frozenset(['black', 'glam']),
    # Death/brutal vs slow/groovy metal
    frozenset(['death', 'stoner']),
    frozenset(['death', 'doom']),
    frozenset(['death', 'sludge']),
    frozenset(['death', 'groove']),
    frozenset(['death', 'nu']),
    frozenset(['death', 'funk']),
    frozenset(['death', 'hair']),
    frozenset(['death', 'glam']),
    frozenset(['brutal', 'stoner']),
    frozenset(['brutal', 'doom']),
    frozenset(['brutal', 'sludge']),
    frozenset(['brutal', 'groove']),
    frozenset(['brutal', 'nu']),
    frozenset(['brutal', 'hair']),
    frozenset(['brutal', 'glam']),
    frozenset(['deathcore', 'stoner']),
    frozenset(['deathcore', 'doom']),
    frozenset(['deathcore', 'sludge']),
    frozenset(['deathcore', 'groove']),
    frozenset(['deathcore', 'nu']),
    frozenset(['deathcore', 'hair']),
    frozenset(['deathcore', 'glam']),
    frozenset(['goregrind', 'stoner']),
    frozenset(['goregrind', 'doom']),
    frozenset(['goregrind', 'sludge']),
    frozenset(['goregrind', 'nu']),
    frozenset(['goregrind', 'hair']),
    frozenset(['goregrind', 'glam']),
    # Stoner/doom vs thrash/speed
    frozenset(['stoner', 'thrash']),
    frozenset(['stoner', 'speed']),
    frozenset(['doom', 'thrash']),
    frozenset(['doom', 'speed']),
    frozenset(['sludge', 'thrash']),
    frozenset(['sludge', 'speed']),
    # Extreme metal vs hip-hop/rap/r&b — broad "metal" excluded to allow nu-metal/rap-metal crossovers
    frozenset(['death', 'hip-hop']),
    frozenset(['death', 'rap']),
    frozenset(['death', 'r&b']),
    frozenset(['death', 'soul']),
    frozenset(['death', 'k-pop']),
    frozenset(['death', 'korean']),
    frozenset(['death', 'reggaeton']),
    frozenset(['death', 'latin']),
    frozenset(['deathcore', 'hip-hop']),
    frozenset(['deathcore', 'rap']),
    frozenset(['deathcore', 'r&b']),
    frozenset(['deathcore', 'reggaeton']),
    frozenset(['brutal', 'hip-hop']),
    frozenset(['brutal', 'rap']),
    frozenset(['brutal', 'r&b']),
    frozenset(['black', 'hip-hop']),
    frozenset(['black', 'rap']),
    frozenset(['black', 'r&b']),
    frozenset(['goregrind', 'hip-hop']),
    frozenset(['goregrind', 'rap']),
    frozenset(['goregrind', 'r&b']),
    frozenset(['thrash', 'hip-hop']),
    frozenset(['thrash', 'rap']),
    # Extreme metal vs electronic/dance
    frozenset(['death', 'house']),
    frozenset(['death', 'techno']),
    frozenset(['death', 'trance']),
    frozenset(['death', 'dance']),
    frozenset(['brutal', 'house']),
    frozenset(['brutal', 'techno']),
    frozenset(['brutal', 'trance']),
    frozenset(['brutal', 'dance']),
    frozenset(['black', 'house']),
    frozenset(['black', 'techno']),
    frozenset(['black', 'trance']),
    frozenset(['black', 'dance']),
}

# ---------------------------------------------------------------------------
# Genre family clustering — loaded from genre_mapping.json
# Maps full genre strings (e.g. "melodic death metal") → family ("metal").
# Used to penalise cross-family matches in the similarity scoring.
# ---------------------------------------------------------------------------
_GENRE_FAMILY_MAP: Dict[str, str] = {}

def _load_genre_mapping():
    """Flatten genre_mapping.json into a single {genre_string: family} dict."""
    mapping_path = Path(__file__).parent / 'genre_mapping.json'
    if not mapping_path.exists():
        print(f"⚠️  genre_mapping.json not found at {mapping_path}")
        return
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for category in data.get('genre_mapping', {}).values():
        if isinstance(category, dict):
            for genre_str, family in category.items():
                _GENRE_FAMILY_MAP[genre_str.lower().strip()] = family
    print(f"  Loaded {len(_GENRE_FAMILY_MAP)} genre→family mappings")

_load_genre_mapping()


_REGION_PREFIXES = (
    'us ', 'uk ', 'australian ', 'brazilian ', 'canadian ', 'french ',
    'german ', 'japanese ', 'mexican ', 'indian ', 'korean ', 'latin ',
    'east asia ', 'southeast asian ', 'african ', 'european ', 'mena ',
    'oceania ', 'scandinavian ', 'spanish ', 'italian ', 'dutch ',
    'russian ', 'chinese ', 'turkish ', 'arabic ', 'nordic ',
    'colombian ', 'argentine ', 'chilean ', 'peruvian ', 'swedish ',
    'norwegian ', 'danish ', 'finnish ', 'polish ', 'czech ',
)


def _genre_families(*genre_strings: str) -> Set[str]:
    """Resolve genre families from raw genre strings using the comprehensive mapping.

    Resolution order per input string:
      1. Try the FULL phrase as-is — preserves slash-containing tags like
         'singer/songwriter' and 'hip-hop/rap' that would otherwise be
         fragmented into bare 'singer'/'songwriter' tokens that don't map.
      2. Comma/slash split — for legacy callers that hand in a multi-tag
         string like 'jungle, breakcore, atmospheric dnb'.
      3. Per-term: full match → regional-prefix strip → word-level fallback.
    """
    families: Set[str] = set()
    for gs in genre_strings:
        if not gs or gs == 'unknown':
            continue
        # Step 1: try the full input phrase first
        full = gs.strip().lower()
        if full in _GENRE_FAMILY_MAP:
            families.add(_GENRE_FAMILY_MAP[full])
            continue
        # Step 2: split on comma OR slash (handles multi-tag strings)
        terms = [t.strip().lower() for t in gs.replace('/', ',').split(',') if t.strip()]
        for term in terms:
            if term in _GENRE_FAMILY_MAP:
                families.add(_GENRE_FAMILY_MAP[term])
                continue
            # Strip regional prefix and retry (e.g. "us hip-hop" → "hip-hop")
            stripped = term
            for prefix in _REGION_PREFIXES:
                if term.startswith(prefix):
                    stripped = term[len(prefix):]
                    break
            if stripped != term and stripped in _GENRE_FAMILY_MAP:
                families.add(_GENRE_FAMILY_MAP[stripped])
                continue
            # Word-level fallback for unrecognised multi-word genres
            for word in term.split():
                if word in _GENRE_FAMILY_MAP:
                    families.add(_GENRE_FAMILY_MAP[word])
    return families


# =============================================================================
# Canonical genre gate — SINGLE SOURCE OF TRUTH for lane filtering.
#
# Consumers: app.py URL path (Similar Artists + flattery/trajectory),
# app.py upload path (same), scripts/backfill_reference_artists.py (email
# reference artists). Any new surface that filters candidates by genre lane
# must call these instead of copying the rules — three hand-copies of this
# logic drifted apart once already (the email refs missed the reggae fix and
# the electronic-subgenre split for a month).
# =============================================================================

# Strong, identity-defining families that must not cross-contaminate a lane
# even when a candidate also touches the lane via a shared family tag.
# 'reggae' included so reggae acts stop bleeding into alt-rock lanes.
# 'electronic' is NOT exclusive: it's an umbrella that legitimately covers
# dnb/garage/house/techno — excluding it would drop the very peers we want.
EXCLUSIVE_FAMILIES = frozenset({'metal', 'hip-hop', 'country',
                                'classical', 'jazz', 'latin', 'reggae'})

# Each identity-defining within itself: jungle ≠ dnb ≠ garage ≠ breakcore
# (per taxonomy decision — "drum and bass is not jungle").
ELECTRONIC_SUBGENRES = frozenset({'jungle', 'dnb', 'garage', 'breaks', 'breakcore'})

_ROCK_CLUSTER = frozenset({'rock', 'indie', 'pop', 'punk', 'r&b'})


def lane_strong_foreign(allowed: Set[str]) -> Set[str]:
    """Lane-aware strong-foreign family set (applied at cf + primary levels)."""
    strong = set(EXCLUSIVE_FAMILIES)
    if not (allowed & {'rock', 'indie', 'pop', 'punk'}):
        strong |= {'pop', 'rock'}
    if allowed & ELECTRONIC_SUBGENRES:
        strong |= (ELECTRONIC_SUBGENRES - allowed)
        if not (allowed & _ROCK_CLUSTER):
            strong |= {'r&b', 'indie', 'punk'}
    return strong


def in_lane_families(cf: Set[str], primary_fams: Set[str],
                     artist_soup_fams: Set[str], allowed: Set[str]) -> bool:
    """Pure-set canonical gate. True = candidate belongs in the lane.

    cf            candidate families across ALL fields (primary + secondary +
                  artist genres + track genres)
    primary_fams  families of the candidate's primary genre alone
    artist_soup_fams  families across the candidate's artist_genres soup
    allowed       the lane (empty = no lane → pass everything)

    Rules (in order):
      1. Sparse-data drop: cf empty → drop (no positive identity proof)
      2. cf must overlap the lane
      3. Foreign-exclusive drop: cf carries an EXCLUSIVE family outside lane
      4. Electronic-subgenre lanes (non-rock-cluster): rock-cluster
         contamination anywhere in cf drops the candidate (hyperpop-tagged
         breakcore — Planet 1999, Lealani). Electronic-cluster cross-
         pollination (dnb in cf for a jungle lane) is deliberately allowed.
      5. Primary-foreign drop with artist-soup bypass: candidate whose
         PRIMARY family is strong-foreign and out-of-lane drops UNLESS the
         artist soup carries a lane tag (jungle producer with one DnB-tagged
         release keeps; pure DnB act still drops).
    """
    if not allowed:
        return True
    if not cf:
        return False
    if not (cf & allowed):
        return False
    if (cf & EXCLUSIVE_FAMILIES) - allowed:
        return False
    strong_foreign = lane_strong_foreign(allowed)
    if (allowed & ELECTRONIC_SUBGENRES) and not (allowed & _ROCK_CLUSTER):
        if cf & _ROCK_CLUSTER:
            return False
    if primary_fams and (primary_fams & strong_foreign) and not (primary_fams & allowed):
        if not (artist_soup_fams & allowed):
            return False
    return True


def candidate_lane_families(m: Dict) -> Set[str]:
    """Canonical candidate-family extraction for matcher match dicts:
    primary + secondary + artist genres + track genres."""
    cand = []
    for field in ('primary_genre', 'secondary_genre'):
        g = (m.get(field) or '').strip()
        if g:
            cand.append(g)
    cand.extend(m.get('artist_genres') or [])
    cand.extend(m.get('track_genres') or [])
    return _genre_families(*cand)


# Categorically blocked genres — never surfaced as comps/matches regardless of
# sonic similarity. Seasonal + kids/novelty productions are bright, simple, and
# vocal-forward, so they score spuriously close to airy pop captures, but they
# are never a legitimate reference artist. Matched as raw substrings against ALL
# of a candidate's genre tags (these map to the 'pop' family, so the lane gate
# alone won't catch them). Applies everywhere match_in_lane is used.
BLOCKED_GENRE_TOKENS = (
    'christmas', 'holiday', 'seasonal',
    "children", "child's", 'kids', 'nursery', 'lullab',  # lullaby/lullabies
)


def is_blocked_genre(m: Dict) -> bool:
    """True if any of the candidate's raw genre tags is a categorically
    blocked genre (christmas / holiday / children's / kids)."""
    parts = []
    for field in ('primary_genre', 'secondary_genre'):
        g = m.get(field)
        if g:
            parts.append(g)
    parts.extend(m.get('artist_genres') or [])
    parts.extend(m.get('track_genres') or [])
    blob = ' , '.join(parts).lower()
    return any(tok in blob for tok in BLOCKED_GENRE_TOKENS)


def match_in_lane(m: Dict, allowed: Set[str]) -> bool:
    """Canonical gate applied to a matcher match dict.

    Categorically blocked genres (christmas/holiday/children's) are dropped
    first, before any lane logic — they're never valid comps even with no lane.
    """
    if is_blocked_genre(m):
        return False
    if not allowed:
        return True
    primary = (m.get('primary_genre') or '').strip()
    primary_fams = _genre_families(primary) if primary else set()
    artist_soup_fams: Set[str] = set()
    for g in (m.get('artist_genres') or []):
        artist_soup_fams |= _genre_families(g)
    return in_lane_families(candidate_lane_families(m), primary_fams,
                            artist_soup_fams, allowed)


# --- Sub-scene refinement gates (origin: email reference-artist picks; ---
# --- shared here so web surfaces can adopt them without re-copying)    ---

# Inside the metal family the sub-scenes don't cross-pitch: a metalcore band
# should not get a power-metal reference. Keyword → coarse sub-scene group.
METAL_SUBSCENES = {
    'core': ('metalcore', 'deathcore', 'post-hardcore', 'post hardcore',
             'mathcore', 'melodic death', 'melodeath', 'nu-metalcore',
             'hardcore', 'screamo', 'emocore'),
    'power': ('power metal', 'symphonic metal', 'epic metal'),
    'trad': ('heavy metal', 'thrash metal', 'thrash', 'speed metal',
             'nwobhm', 'traditional metal', 'classic metal'),
    'extreme': ('death metal', 'black metal', 'grindcore', 'deathgrind',
                'blackened'),
    'doom': ('doom metal', 'sludge metal', 'stoner metal', 'stoner doom',
             'drone metal', 'funeral doom', 'post-metal'),
    'alt': ('nu metal', 'nu-metal', 'alternative metal', 'rap metal',
            'industrial metal', 'rapcore', 'groove metal'),
    'prog': ('progressive metal', 'djent', 'tech metal', 'technical death',
             'math metal'),
}

# Aggressive-music markers across any family — used as a coarse signal check:
# an aggressive user should get aggressive candidates.
AGGRESSIVE_TOKENS = (
    'metal', 'metalcore', 'deathcore', 'hardcore', 'thrash', 'grindcore',
    'grind', 'doom', 'punk', 'post-hardcore', '-core', 'djent', 'sludge',
    'black metal', 'death metal', 'screamo',
)


def metal_subscene_groups(genre_str: Optional[str]) -> Set[str]:
    """Return the METAL_SUBSCENES group keys present in a genre string."""
    if not genre_str:
        return set()
    gl = genre_str.lower()
    return {group for group, keywords in METAL_SUBSCENES.items()
            if any(k in gl for k in keywords)}


def has_aggressive_signal(genre_str: Optional[str]) -> bool:
    """True if the genre string contains any aggressive-music marker."""
    if not genre_str:
        return False
    gl = genre_str.lower()
    return any(tok in gl for tok in AGGRESSIVE_TOKENS)


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
        self._cand_cache = {}

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

        # Enrich each GEMS row with conversion_rate from tier data
        # so sonic_gap filtering (row_cr > threshold) actually works
        enriched = 0
        for gem in self._gems_list:
            isrc = gem.get('isrc')
            if not isrc:
                continue
            track_data = self._tracks.get(isrc, {})
            artist_id = str(track_data.get('artist_id', ''))
            if not artist_id:
                continue
            tier_data = self._tiers.get(artist_id, {})
            a_listeners = _float(tier_data.get('listeners'), 0)
            a_followers = _float(tier_data.get('followers'), 0)
            if a_listeners > 0 and a_followers > 0:
                cr = round((a_followers * 0.1) / (a_listeners * 4.3) * 100, 2)
                gem['conversion_rate'] = cr
                enriched += 1

        # Precompute static per-candidate data used by find_matches: the normalized
        # profile, parsed track/artist genres, genre families, and emotion set. These
        # derive ONLY from static cache rows and were previously rebuilt for EVERY
        # track on EVERY query — the bulk of find_matches' cost. Building them once
        # here changes no math: _similarity and all target-dependent logic are
        # untouched; candidates without an artist_id are skipped exactly as before.
        self._cand_cache = {}
        for isrc, row in self._gems_by_isrc.items():
            track_data = self._tracks.get(isrc, {})
            artist_id = track_data.get('artist_id')
            if not artist_id:
                continue
            artist_data = self._artists.get(str(artist_id), {})
            profile = self._build_profile(row)
            cand_emotions = {
                profile.get('emotion_1'), profile.get('emotion_2'),
                profile.get('emotion_3'), profile.get('emotion_4'),
            } - {None, ''}
            self._cand_cache[isrc] = {
                'profile': profile,
                'cand_emotions': cand_emotions,
                'track_genres': _parse_track_genres(track_data.get('track_genres', '')),
                'artist_genres': _parse_track_genres(artist_data.get('genres', '')),
                'cand_fams': _genre_families(
                    track_data.get('track_genres', ''),
                    artist_data.get('genres', ''),
                    profile.get('primary_genre', ''),
                    profile.get('secondary_genre', ''),
                ),
            }

        stats = self.cache.get('stats', {})
        print(f"  Loaded in {elapsed:.1f}s — {stats.get('total_gems', 0):,} GEMS records, {enriched:,} enriched with conversion_rate, {len(self._cand_cache):,} candidate profiles precomputed")

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
                    track_genres_a: Set[str], track_genres_b: Set[str],
                    artist_genres_a: Set[str] = None, artist_genres_b: Set[str] = None):
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

        # Genre similarity — two separate channels so we never compare the
        # user's artist genres against a candidate's track genres (a category
        # mismatch). Track channel is the primary lane signal; artist channel is
        # a light "same kind of act" verification. Weights sum to the old 0.08
        # so the sonic-vs-genre balance is unchanged.
        #
        # Track channel (Jaccard): user track genre vs candidate track genre.
        ga = (_genre_words(profile_a.get('primary_genre'))
              | _genre_words(profile_a.get('secondary_genre'))
              | (track_genres_a or set()))
        gb = (_genre_words(profile_b.get('primary_genre'))
              | _genre_words(profile_b.get('secondary_genre'))
              | (track_genres_b or set()))
        track_genre_score = len(ga & gb) / len(ga | gb) if (ga and gb) else 0
        breakdown['genre'] = {'score': track_genre_score, 'weight': 0.06}

        # Artist channel (Jaccard): user artist genre vs candidate artist genre.
        # Light secondary verification — never the main driver. After the
        # _parse_track_genres change to preserve multi-word phrases (no more
        # space-splitting), enrich each side with word-level tokens too so the
        # Jaccard still picks up shared genre WORDS even when the phrases
        # don't align exactly (e.g. 'indie pop' vs 'pop' would otherwise
        # fail to overlap; with word enrichment, both share 'pop').
        def _enrich(tags):
            out = set(tags or [])
            for t in (tags or []):
                for w in t.replace('/', ' ').split():
                    w = w.strip()
                    if w:
                        out.add(w)
            return out
        aa = _enrich(artist_genres_a)
        ab = _enrich(artist_genres_b)
        artist_genre_score = len(aa & ab) / len(aa | ab) if (aa and ab) else 0
        breakdown['artist_genre'] = {'score': artist_genre_score, 'weight': 0.02}

        # LUFS
        lufs_score = 1 - abs((profile_a.get('lufs_integrated') or 0) - (profile_b.get('lufs_integrated') or 0))
        breakdown['lufs_integrated'] = {'score': lufs_score, 'weight': 0.05}

        # Boost sonic features
        breakdown['frequency_spectrum']['weight'] = 0.38
        breakdown['emotion']['weight'] = 0.07

        total = sum(item['score'] * item['weight'] for item in breakdown.values())
        return total, breakdown

    def find_matches(self, features: Dict, genre_hint: str = '',
                     top_n: int = 20, threshold: float = 0.60,
                     artist_genre_hint: str = '') -> List[Dict]:
        """
        Find the top-N most sonically similar artists from the cache.

        Parameters
        ----------
        features : dict
            Raw features from audio_analyzer.extract_features().
        genre_hint : str
            The analyzed track's OWN genre — the primary track-to-track lane
            signal (and the hard family gate upstream).
        top_n : int
            Number of matches to return.
        threshold : float
            Minimum similarity to include.
        artist_genre_hint : str
            The analyzed artist's genres — used only for the light artist-channel
            verification in _similarity, never as the primary lane signal.
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
        target_artist_genre_words = _parse_track_genres(artist_genre_hint)
        _EXTREME = {'black', 'death', 'brutal', 'doom', 'thrash',
                    'deathcore', 'goregrind', 'grind', 'sludge'}
        is_extreme = bool(target_genre_words & _EXTREME)
        # Build target families from primary + first few genres (skip noisy CM tail)
        # CM primary genre is reliable, first 3-4 secondaries usually good, tail is noise
        # e.g. The Prodigy: "electronic, dance, rock, breakbeat" = good, "uk hip-hop/rap" = noise
        if genre_hint and ',' in genre_hint:
            genre_parts = [g.strip() for g in genre_hint.split(',')]
            core_genres = ', '.join(genre_parts[:6])  # primary + first 5 secondaries
            target_fams = _genre_families(core_genres)
        else:
            target_fams = _genre_families(genre_hint)
        # Primary genre families — used to boost candidates that share the primary
        primary_genre = genre_hint.split(',')[0].strip() if genre_hint else ''
        primary_fams = _genre_families(primary_genre) if primary_genre else target_fams
        if is_extreme:
            target_fams.discard('rock')  # CM tags extreme metal with "rock" — strip it

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
            # Static per-candidate data precomputed at load (no artist_id -> absent,
            # same candidates skipped as before). _similarity below is unchanged.
            cand = self._cand_cache.get(isrc)
            if not cand:
                continue

            profile = cand['profile']
            cand_emotions = cand['cand_emotions']
            track_genres = cand['track_genres']
            cand_artist_genres = cand['artist_genres']

            track_data = self._tracks.get(isrc, {})
            artist_id = track_data.get('artist_id')

            similarity, breakdown = self._similarity(
                target_profile, profile, target_genre_words, track_genres,
                target_artist_genre_words, cand_artist_genres,
            )

            # Sonic penalties
            energy_d = abs(_float(profile.get('energy')) - _float(target_profile.get('energy')))
            beat_d = abs(_float(profile.get('beat_strength')) - _float(target_profile.get('beat_strength')))
            dance_d = abs(_float(profile.get('danceability')) - _float(target_profile.get('danceability')))
            onset_d = abs(_float(profile.get('onset_rate')) - _float(target_profile.get('onset_rate')))
            # BPM distance is tempo-octave AND triplet-ratio invariant: BPM
            # detectors regularly alias fast-percussive tracks (jungle/dnb at
            # ~170, metal blast beats at ~220-250) to half-tempo (1/2) or to
            # 2/3 of true tempo (triplet/off-beat lock-on). Without this, real
            # 170-BPM jungle peers get penalized vs a track detected at 117.
            ta = _float(target_profile.get('bpm')) or 0
            tb = _float(profile.get('bpm')) or 0
            if ta > 0 and tb > 0:
                bpm_d = min(
                    abs(ta - tb),
                    abs(2*ta - tb), abs(ta - 2*tb),       # 1/2 (octave) aliasing
                    abs(1.5*ta - tb), abs(ta - 1.5*tb),   # 2/3 (triplet) aliasing
                )
            else:
                bpm_d = abs(ta - tb)

            penalty = 0.0
            if energy_d > 0.15:
                penalty += (energy_d - 0.15) * 0.8
            if beat_d > 0.15:
                penalty += (beat_d - 0.15) * 0.7
            if dance_d > 0.18:
                penalty += (dance_d - 0.18) * 0.5
            if onset_d > 0.18:
                penalty += (onset_d - 0.18) * 0.5
            if bpm_d > 20:
                penalty += (bpm_d - 20) * 0.003
            penalty = min(penalty, 0.20)

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
            artist_genres = cand['artist_genres']
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
                cand_fams = cand['cand_fams']
                if target_fams:
                    if not cand_fams:
                        # Candidate has no recognisable genre → heavy penalty
                        if is_extreme:
                            similarity *= 0.60  # 40% penalty for no-genre vs extreme
                        else:
                            similarity *= 0.75  # 25% penalty for no-genre vs normal
                    else:
                        overlap = target_fams & cand_fams
                        if not overlap:
                            # Completely different family → hard filter
                            # e.g. pure hip-hop matching electronic, or pure pop matching metal
                            continue
                        # Penalise weak overlap — candidate shares few of the user's families
                        # e.g. pure electronic DJ matching a hip-hop/r&b/electronic/rock artist
                        overlap_ratio = len(overlap) / len(target_fams)
                        if overlap_ratio < 0.5:
                            # Shares <50% of families → significant penalty
                            similarity *= (0.82 + 0.18 * overlap_ratio)  # 10-18% penalty
                        if cand_fams - target_fams:
                            # Crossover — shares a family but also has a foreign one
                            similarity *= 0.96
                    # Boost candidates that share the PRIMARY genre family
                    # e.g. R&B artist matching R&B candidate gets +5% over hip-hop candidate
                    if primary_fams and cand_fams:
                        if primary_fams & cand_fams:
                            similarity = min(1.0, similarity * 1.05)  # 5% boost for primary match
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
                'code2': artist_data.get('code2', ''),
                'pronoun_title': artist_data.get('pronoun_title', ''),
                'tier': tier,
                'listeners': listeners,
                'followers': followers,
                'conversion_rate': conversion_rate,
                # Track-level momentum signals from the universe cache.
                # Populated from tracks.sp_track_popularity / cm_track_score (or the
                # recent_track_* equivalents when recent track filled this slot — the
                # cache builder unifies key names). Either may be None for older rows
                # that haven't been backfilled yet; downstream callers use .get(...).
                'sp_track_popularity': track_data.get('sp_track_popularity'),
                'cm_track_score': track_data.get('cm_track_score'),
                # Track-level playlist counts (Spotify editorial + user) — same
                # unified shape: source is top-track or recent-track depending on
                # which pass populated this row in the cache.
                'editorial_playlists': track_data.get('editorial_playlists', 0),
                'user_playlists': track_data.get('user_playlists', 0),
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
