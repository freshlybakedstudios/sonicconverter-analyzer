"""
Audio feature extraction for uploaded tracks.
Ported from GEMS/fixed_gems_pipeline_v2.py - adapted for file upload (no Loopback needed).
"""

import math

import numpy as np
import librosa
import pyloudnorm as pyln
from typing import Dict


class GenreAwareEmotionDetector:
    """Genre-aware GEMS 9 emotion detection. Ported from gems_emotion_detector.py."""

    GENRE_GROUPS = {
        'bass_heavy': ['hip-hop', 'electronic', 'reggae', 'latin'],
        'extreme_metal': ['black metal', 'death metal', 'brutal death metal',
                          'melodic death metal', 'technical death metal',
                          'deathcore', 'goregrind', 'grindcore', 'doom metal',
                          'sludge metal', 'thrash metal', 'black', 'death',
                          'brutal', 'doom', 'sludge', 'thrash', 'grind'],
        'energy_focused': ['rock', 'punk', 'metal', 'metalcore', 'hardcore',
                           'power metal', 'progressive metal', 'nu metal'],
        'balanced': ['pop', 'r&b', 'country', 'indie'],
        'soft': ['folk', 'jazz', 'classical', 'ambient', 'singer-songwriter', 'gospel', 'world'],
    }

    BASS_HEAVY_EMOTIONS = {
        'power': {
            'lufs_integrated': ('gt', -12, 0.3),
            'bass_ratio': ('gt', 0.15, 0.3),
            'energy': ('gt', 0.3, 0.2),
            'compression_amount': ('gt', 0.8, 0.2),
            'beat_strength': ('gt', 1.0, 0.1),
        },
        'nostalgia': {
            'brightness': ('range', (1200, 3500), 0.25),
            'energy': ('range', (0.2, 0.7), 0.2),
            'mid_ratio': ('gt', 0.25, 0.3),
            'air_ratio': ('lt', 0.18, 0.15),
            'bpm': ('range', (60, 120), 0.1),
        },
        'tension': {
            'dissonance': ('gt', 0.06, 0.25),
            'spectral_complexity': ('gt', 0.07, 0.2),
            'compression_amount': ('gt', 0.75, 0.15),
            'onset_rate': ('gt', 3.5, 0.15),
            'brightness': ('gt', 2500, 0.1),
        },
    }

    # Extreme metal emotions — tuned to match what metal tracks in the
    # GEMS universe cache actually have: power (23%), intense (18%),
    # aggressive (13%), tension (7%). Dark/brooding are too rare (<2%)
    # to be useful for matching. Thresholds are very low so these always
    # outscore nostalgia/tenderness from universal emotions.
    EXTREME_METAL_EMOTIONS = {
        'power': {
            'lufs_integrated': ('gt', -18, 0.25),
            'bass_ratio': ('gt', 0.05, 0.2),
            'energy': ('gt', 0.1, 0.3),
            'compression_amount': ('gt', 0.4, 0.15),
            'brightness': ('gt', 1500, 0.1),
        },
        'aggressive': {
            'energy': ('gt', 0.1, 0.3),
            'lufs_integrated': ('gt', -18, 0.2),
            'dissonance': ('gt', 0.02, 0.25),
            'brightness': ('gt', 1200, 0.15),
            'onset_rate': ('gt', 1.5, 0.1),
        },
        'intense': {
            'energy': ('gt', 0.1, 0.3),
            'spectral_complexity': ('gt', 0.03, 0.25),
            'lufs_integrated': ('gt', -18, 0.2),
            'compression_amount': ('gt', 0.4, 0.15),
            'onset_rate': ('gt', 2, 0.1),
        },
        'tension': {
            'dissonance': ('gt', 0.03, 0.3),
            'spectral_complexity': ('gt', 0.03, 0.25),
            'compression_amount': ('gt', 0.3, 0.15),
            'onset_rate': ('gt', 1.5, 0.15),
            'brightness': ('gt', 1200, 0.1),
        },
    }

    ENERGY_FOCUSED_EMOTIONS = {
        'power': {
            'lufs_integrated': ('gt', -13, 0.3),
            'bass_ratio': ('gt', 0.08, 0.2),
            'energy': ('gt', 0.25, 0.3),
            'compression_amount': ('gt', 0.75, 0.15),
            'brightness': ('gt', 2500, 0.2),
        },
        'aggressive': {
            'energy': ('gt', 0.2, 0.3),
            'lufs_integrated': ('gt', -14, 0.25),
            'dissonance': ('gt', 0.05, 0.2),
            'brightness': ('gt', 2200, 0.15),
            'onset_rate': ('gt', 3, 0.1),
        },
        'intense': {
            'energy': ('gt', 0.25, 0.3),
            'spectral_complexity': ('gt', 0.06, 0.25),
            'lufs_integrated': ('gt', -13, 0.2),
            'compression_amount': ('gt', 0.7, 0.15),
            'brightness': ('gt', 2000, 0.1),
        },
        'dark': {
            'brightness': ('lt', 2200, 0.3),
            'bass_ratio': ('gt', 0.10, 0.25),
            'energy': ('range', (0.2, 0.8), 0.2),
            'lufs_integrated': ('gt', -16, 0.15),
            'air_ratio': ('lt', 0.12, 0.1),
        },
        'brooding': {
            'energy': ('range', (0.15, 0.6), 0.25),
            'brightness': ('lt', 2000, 0.2),
            'bass_ratio': ('gt', 0.12, 0.2),
            'dissonance': ('gt', 0.04, 0.15),
            'spectral_complexity': ('gt', 0.04, 0.1),
        },
        'tension': {
            'dissonance': ('gt', 0.06, 0.3),
            'spectral_complexity': ('gt', 0.06, 0.2),
            'compression_amount': ('gt', 0.6, 0.15),
            'onset_rate': ('gt', 3, 0.15),
            'brightness': ('gt', 2500, 0.1),
        },
        'nostalgia': {
            'brightness': ('range', (1000, 2800), 0.25),
            'energy': ('range', (0.15, 0.6), 0.2),
            'mid_ratio': ('gt', 0.22, 0.2),
            'dynamic_range': ('gt', 12, 0.15),
            'bpm': ('range', (60, 120), 0.1),
        },
    }

    BALANCED_EMOTIONS = {
        'power': {
            'lufs_integrated': ('gt', -11, 0.3),
            'bass_ratio': ('gt', 0.15, 0.2),
            'energy': ('gt', 0.35, 0.3),
            'compression_amount': ('gt', 0.8, 0.1),
            'beat_strength': ('gt', 1.2, 0.1),
        },
        'nostalgia': {
            'brightness': ('range', (1000, 2200), 0.25),
            'energy': ('range', (0.2, 0.6), 0.2),
            'mid_ratio': ('gt', 0.24, 0.2),
            'air_ratio': ('lt', 0.12, 0.15),
            'bpm': ('range', (60, 110), 0.1),
        },
        'tension': {
            'dissonance': ('gt', 0.08, 0.3),
            'spectral_complexity': ('gt', 0.08, 0.2),
            'compression_amount': ('gt', 0.65, 0.15),
            'onset_rate': ('gt', 4, 0.15),
            'brightness': ('gt', 2800, 0.1),
        },
    }

    SOFT_EMOTIONS = {
        'power': {
            'lufs_integrated': ('gt', -12, 0.4),
            'energy': ('gt', 0.35, 0.4),
            'compression_amount': ('gt', 0.8, 0.1),
            'beat_strength': ('gt', 1.2, 0.1),
        },
        'nostalgia': {
            'brightness': ('range', (800, 2000), 0.3),
            'energy': ('range', (0.1, 0.5), 0.25),
            'mid_ratio': ('gt', 0.22, 0.2),
            'air_ratio': ('lt', 0.10, 0.15),
            'dynamic_range': ('gt', 15, 0.1),
        },
        'tension': {
            'dissonance': ('gt', 0.05, 0.3),
            'spectral_complexity': ('gt', 0.06, 0.2),
            'compression_amount': ('gt', 0.6, 0.15),
            'onset_rate': ('gt', 3, 0.15),
            'brightness': ('gt', 2200, 0.1),
        },
        'wonder': {
            'spectral_complexity': ('gt', 0.05, 0.3),
            'brightness': ('range', (1000, 3000), 0.25),
            'air_ratio': ('gt', 0.08, 0.2),
            'energy': ('range', (0.1, 0.6), 0.2),
            'loudness_range': ('gt', 5, 0.15),
            'mid_ratio': ('gt', 0.25, 0.1),
        },
    }

    UNIVERSAL_EMOTIONS = {
        'tenderness': {
            'energy': ('lt', 0.4, 0.3),
            'lufs_integrated': ('lt', -14, 0.25),
            'dissonance': ('lt', 0.04, 0.2),
            'brightness': ('range', (1200, 2200), 0.15),
            'dynamic_range': ('gt', 6, 0.1),
        },
        'joyfulness': {
            'brightness': ('gt', 2200, 0.25),
            'bpm': ('range', (110, 140), 0.2),
            'energy': ('gt', 0.55, 0.2),
            'dissonance': ('lt', 0.05, 0.15),
            'high_mid_ratio': ('gt', 0.14, 0.15),
            'danceability': ('gt', 0.6, 0.05),
        },
        'sadness': {
            'energy': ('lt', 0.35, 0.3),
            'brightness': ('lt', 1600, 0.25),
            'bpm': ('lt', 85, 0.2),
            'lufs_integrated': ('lt', -15, 0.15),
            'dissonance': ('range', (0.03, 0.07), 0.1),
        },
        'peacefulness': {
            'energy': ('lt', 0.3, 0.35),
            'dissonance': ('lt', 0.03, 0.25),
            'onset_rate': ('lt', 2, 0.2),
            'spectral_complexity': ('lt', 0.05, 0.15),
            'dynamic_range': ('gt', 8, 0.05),
        },
        'wonder': {
            'spectral_complexity': ('gt', 0.07, 0.25),
            'brightness': ('range', (1800, 3200), 0.2),
            'air_ratio': ('gt', 0.12, 0.2),
            'energy': ('range', (0.3, 0.7), 0.15),
            'loudness_range': ('gt', 6, 0.15),
        },
        'transcendence': {
            'air_ratio': ('gt', 0.15, 0.3),
            'spectral_complexity': ('gt', 0.09, 0.25),
            'brightness': ('gt', 2500, 0.2),
            'energy': ('range', (0.4, 0.8), 0.15),
            'loudness_range': ('gt', 8, 0.1),
        },
    }

    # ---- V8 scoring (2026-06-10) — TWIN of GEMS/gems_emotion_detector.py ----
    # KEEP IN SYNC with the GEMS detector: the matcher compares the live
    # user's emotion labels against universe labels slot-by-slot (5% weight),
    # so the two implementations must produce the same labels for the same
    # features. Changes: anchored sigmoid scoring (0.8 at threshold, spreads
    # frozen from the 235K universe), dead-threshold repairs (compression
    # rules were calibrated pre crest-dB-migration), major/minor mode wired
    # into valence feelings, and gate x evidence structure (definitional
    # features multiply). See GEMS/emotion_rederive_prototype.py for the
    # validation history.

    V8_OFFSET = math.log(0.8 / 0.2)  # sigmoid(V8_OFFSET) = 0.8 at threshold
    GATE_FLOOR = 0.2

    THRESHOLD_REPAIRS = {
        ('bass_heavy', 'power', 'compression_amount'): 0.660,
        ('bass_heavy', 'tension', 'compression_amount'): 0.660,
        ('bass_heavy', 'joyfulness', 'energy'): 0.373,
        ('bass_heavy', 'peacefulness', 'onset_rate'): 4.250,
        ('bass_heavy', 'peacefulness', 'spectral_complexity'): 0.072,
        ('bass_heavy', 'transcendence', 'loudness_range'): 4.320,
        ('energy_focused', 'power', 'compression_amount'): 0.633,
        ('energy_focused', 'intense', 'compression_amount'): 0.633,
        ('energy_focused', 'joyfulness', 'energy'): 0.340,
        ('energy_focused', 'peacefulness', 'spectral_complexity'): 0.070,
        ('energy_focused', 'transcendence', 'loudness_range'): 3.158,
        # extreme_metal uses the energy_focused repairs for universal emotions
        ('extreme_metal', 'joyfulness', 'energy'): 0.340,
        ('extreme_metal', 'peacefulness', 'spectral_complexity'): 0.070,
        ('extreme_metal', 'transcendence', 'loudness_range'): 3.158,
        ('balanced', 'power', 'compression_amount'): 0.629,
        ('balanced', 'joyfulness', 'energy'): 0.335,
        ('balanced', 'transcendence', 'loudness_range'): 4.145,
        ('soft', 'power', 'compression_amount'): 0.609,
        ('soft', 'joyfulness', 'energy'): 0.314,
    }

    GROUP_SPREADS = {
        'balanced': {
            'air_ratio': 0.041562, 'bass_ratio': 0.031633, 'beat_strength': 0.112053,
            'bpm': 19.171055, 'brightness': 556.339426, 'compression_amount': 0.050304,
            'danceability': 0.126757, 'dissonance': 0.020517, 'dynamic_range': 3.892159,
            'energy': 0.050931, 'high_mid_ratio': 0.028616, 'loudness_range': 1.131408,
            'lufs_integrated': 1.518536, 'mid_ratio': 0.0464, 'onset_rate': 1.625,
            'spectral_complexity': 0.010362,
        },
        'soft': {
            'air_ratio': 0.050487, 'bass_ratio': 0.040219, 'beat_strength': 0.1453,
            'bpm': 20.145408, 'brightness': 682.943725, 'compression_amount': 0.058987,
            'danceability': 0.167432, 'dissonance': 0.024171, 'dynamic_range': 4.059437,
            'energy': 0.064316, 'high_mid_ratio': 0.030534, 'loudness_range': 1.429242,
            'lufs_integrated': 2.175985, 'mid_ratio': 0.056244, 'onset_rate': 1.75,
            'spectral_complexity': 0.01364,
        },
        'bass_heavy': {
            'air_ratio': 0.04502, 'bass_ratio': 0.031337, 'beat_strength': 0.156749,
            'bpm': 18.157728, 'brightness': 572.479191, 'compression_amount': 0.051409,
            'danceability': 0.150893, 'dissonance': 0.027934, 'dynamic_range': 4.747274,
            'energy': 0.055297, 'high_mid_ratio': 0.022525, 'loudness_range': 1.186059,
            'lufs_integrated': 1.401391, 'mid_ratio': 0.040999, 'onset_rate': 1.1875,
            'spectral_complexity': 0.009933,
        },
        'energy_focused': {
            'air_ratio': 0.035334, 'bass_ratio': 0.025942, 'beat_strength': 0.079953,
            'bpm': 21.110983, 'brightness': 469.557702, 'compression_amount': 0.046074,
            'danceability': 0.115272, 'dissonance': 0.01637, 'dynamic_range': 2.409165,
            'energy': 0.04697, 'high_mid_ratio': 0.031489, 'loudness_range': 0.895832,
            'lufs_integrated': 1.461087, 'mid_ratio': 0.038491, 'onset_rate': 2.0,
            'spectral_complexity': 0.008886,
        },
    }
    # extreme_metal is a web-only refinement of energy_focused — same universe
    # cohort, so it shares those spreads.
    GROUP_SPREADS['extreme_metal'] = GROUP_SPREADS['energy_focused']

    GATE_FEATURES = {
        'power':         ['lufs_integrated'],
        'aggressive':    ['energy', 'dissonance'],
        'intense':       ['energy'],
        'dark':          ['brightness'],
        'brooding':      ['brightness'],
        'tension':       ['dissonance'],
        'nostalgia':     ['air_ratio', 'brightness'],
        'tenderness':    ['energy'],
        'joyfulness':    ['brightness'],
        'sadness':       ['energy'],
        'peacefulness':  ['energy', 'dissonance'],
        'wonder':        ['spectral_complexity'],
        'transcendence': ['air_ratio'],
    }

    MODE_GATES = {'sadness': 'minor', 'brooding': 'minor', 'joyfulness': 'major'}
    MODE_RULES = {'dark': ('minor', 0.10), 'tension': ('minor', 0.10),
                  'peacefulness': ('major', 0.05)}

    def get_genre_group(self, genre: str) -> str:
        if not genre:
            return 'balanced'
        genre_lower = genre.lower()
        for group, genres in self.GENRE_GROUPS.items():
            # Exact match first
            if genre_lower in genres:
                return group
            # Substring match: "black metal" contains "metal" → energy_focused
            for g in genres:
                if g in genre_lower:
                    return group
        return 'balanced'

    @staticmethod
    def _sigmoid(x):
        if x > 30:
            return 1.0
        if x < -30:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _score_rule(self, group, value, comparison, threshold, feature):
        """Anchored continuous score for one rule (0.8 at threshold)."""
        spread = self.GROUP_SPREADS.get(group, {}).get(feature)
        if spread is None or spread <= 0:
            if comparison == 'gt':
                return 1.0 if value > threshold else 0.0
            if comparison == 'lt':
                return 1.0 if value < threshold else 0.0
            lo, hi = threshold
            return 1.0 if lo <= value <= hi else 0.0
        if comparison == 'gt':
            return self._sigmoid(self.V8_OFFSET + (value - threshold) / spread)
        if comparison == 'lt':
            return self._sigmoid(self.V8_OFFSET + (threshold - value) / spread)
        if comparison == 'range':
            lo, hi = threshold
            return (self._sigmoid(self.V8_OFFSET + (value - lo) / spread)
                    * self._sigmoid(self.V8_OFFSET + (hi - value) / spread))
        return 0.0

    def detect(self, features: Dict, genre_hint: str = '') -> Dict:
        group = self.get_genre_group(genre_hint)
        if group == 'bass_heavy':
            genre_emotions = self.BASS_HEAVY_EMOTIONS
        elif group == 'extreme_metal':
            genre_emotions = self.EXTREME_METAL_EMOTIONS
        elif group == 'energy_focused':
            genre_emotions = self.ENERGY_FOCUSED_EMOTIONS
        elif group == 'soft':
            genre_emotions = self.SOFT_EMOTIONS
        else:
            genre_emotions = self.BALANCED_EMOTIONS

        all_emotions = {**genre_emotions, **self.UNIVERSAL_EMOTIONS}

        # For extreme metal, suppress soft emotions so they can't outscore
        # aggression/tension — keeps user-side labels aligned with what the
        # universe carries for extreme genres (aggressive/intense/power).
        _SOFT_EMOTIONS = {'tenderness', 'joyfulness', 'peacefulness', 'nostalgia', 'wonder'}

        track_scale = features.get('scale')
        key_strength = features.get('key_strength') or 0.0
        try:
            key_strength = max(0.0, min(1.0, float(key_strength)))
        except (TypeError, ValueError):
            key_strength = 0.0

        scores = {}
        for emotion, rules in all_emotions.items():
            gate_feats = set(self.GATE_FEATURES.get(emotion, []))
            gate = 1.0
            total, wsum = 0.0, 0.0
            for feat_name, (comp, thresh, weight) in rules.items():
                val = features.get(feat_name)
                if not isinstance(val, (int, float)):
                    continue
                thresh = self.THRESHOLD_REPAIRS.get((group, emotion, feat_name), thresh)
                s = self._score_rule(group, val, comp, thresh, feat_name)
                if feat_name in gate_feats:
                    gate *= self.GATE_FLOOR + (1.0 - self.GATE_FLOOR) * s
                else:
                    total += s * weight
                    wsum += weight

            if emotion in self.MODE_GATES:
                if track_scale in ('major', 'minor'):
                    match = 1.0 if track_scale == self.MODE_GATES[emotion] else 0.0
                    ms = 1.0 - key_strength * (1.0 - match)
                    gate *= self.GATE_FLOOR + (1.0 - self.GATE_FLOOR) * ms
            elif emotion in self.MODE_RULES and track_scale in ('major', 'minor'):
                want, mw = self.MODE_RULES[emotion]
                total += (1.0 if track_scale == want else 0.0) * key_strength * mw
                wsum += mw

            evidence = (total / wsum) if wsum > 0 else 1.0
            score = gate * evidence
            if group == 'extreme_metal' and emotion in _SOFT_EMOTIONS:
                score *= 0.4  # Cap soft emotions at ~40% of raw score
            scores[emotion] = score

        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_4 = sorted_emotions[:4]

        return {
            'emotions': [(e, round(s, 4)) for e, s in top_4],
            'all_scores': {k: round(v, 4) for k, v in scores.items()},
            'genre_group': group,
        }


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

_emotion_detector = GenreAwareEmotionDetector()


def extract_features(file_path: str, genre_hint: str = '') -> Dict:
    """
    Load an audio file and extract the full 33-feature profile
    matching the GEMS cache format.

    Parameters
    ----------
    file_path : str
        Path to mp3/wav file on disk.
    genre_hint : str, optional
        Primary genre string for emotion detection thresholds.

    Returns
    -------
    dict  with all feature keys used by the matcher + emotion data.
    """
    # Load audio at 44.1 kHz — mono for features, stereo for spatial analysis
    y, sr = librosa.load(file_path, sr=44100, mono=True)
    # Also load stereo if available
    y_stereo_raw, _ = librosa.load(file_path, sr=44100, mono=False)
    if y_stereo_raw.ndim == 2 and y_stereo_raw.shape[0] == 2:
        # librosa returns (channels, samples), transpose to (samples, channels)
        y_stereo_full = y_stereo_raw.T
    else:
        y_stereo_full = None

    duration_samples = len(y)
    duration_sec = duration_samples / sr

    if duration_sec < 5:
        raise ValueError("Audio file too short for analysis (< 5 seconds)")

    # 3-point sampling (25%, 50%, 75%) - take 8-second segments
    segment_len = int(min(8, duration_sec * 0.2) * sr)
    sample_points = [0.25, 0.50, 0.75]
    segments = []
    stereo_segments = []
    energies = []

    for pct in sample_points:
        start = int(duration_samples * pct) - segment_len // 2
        start = max(0, min(start, duration_samples - segment_len))
        seg = y[start : start + segment_len]
        if len(seg) < sr:  # skip if less than 1 second
            continue
        segments.append(seg)
        if y_stereo_full is not None:
            stereo_segments.append(y_stereo_full[start : start + segment_len])
        else:
            stereo_segments.append(None)
        energies.append(float(np.sqrt(np.mean(seg ** 2))))

    if not segments:
        raise ValueError("Could not extract valid audio segments")

    # Use highest-energy segment for feature extraction
    best_idx = int(np.argmax(energies))
    audio = segments[best_idx]
    audio_stereo = stereo_segments[best_idx]

    features = _extract_core(audio, sr, audio_stereo=audio_stereo)

    # --- Whole-track loudness (display-only, NOT used for peer comparison) ---
    # The universe (gems_complete_analysis) was measured on short highest-energy
    # captures, so `lufs_integrated` must stay chunk-based to remain
    # apples-to-apples with peers. These two fields are true BS.1770 over the
    # FULL file (stereo when available, matching Pro Tools / Insight) so the
    # artist sees the same number their mastering meter shows.
    try:
        _wt_meter = pyln.Meter(sr)
        # Mono files measure as dual-mono: a DAW meters a mono file on a stereo
        # bus with both channels fed (+3.01 dB vs single-channel), and the
        # Loopback capture path sees the same track as 2 channels too.
        _wt_audio = y_stereo_full if y_stereo_full is not None else np.column_stack((y, y))
        _wt = _wt_meter.integrated_loudness(_wt_audio)
        if np.isfinite(_wt):  # pyloudnorm returns -inf (not NaN) on all-gated audio
            features['lufs_whole_track'] = float(_wt)
        # EBU-style LRA estimate: 3 s short-term windows, 1 s hop, p95 - p10
        _win, _hop = int(3 * sr), sr
        _st = []
        for _i in range(0, len(y) - _win, _hop):
            _bl = _wt_meter.integrated_loudness(y[_i:_i + _win])
            if not np.isnan(_bl) and _bl > -70:  # gate silence
                _st.append(_bl)
        if len(_st) > 3:
            features['lra_whole_track'] = float(np.percentile(_st, 95) - np.percentile(_st, 10))
    except Exception:
        pass  # display-only — never fail the scan over it

    # Emotion detection
    emo = _emotion_detector.detect(features, genre_hint)
    top = emo['emotions']
    for i in range(4):
        if i < len(top):
            features[f'emotion_{i+1}'] = top[i][0]
            features[f'emotion_{i+1}_score'] = top[i][1]
        else:
            features[f'emotion_{i+1}'] = 'neutral'
            features[f'emotion_{i+1}_score'] = 0.0

    features['emotion_summary'] = emo

    # Auto-detect genre from audio features if no hint provided
    if not genre_hint:
        features['detected_genre'] = _estimate_genre(features)
    else:
        features['detected_genre'] = genre_hint

    return features


def _estimate_genre(f: Dict) -> str:
    """
    Estimate a rough genre category from audio features.
    Uses BPM, energy, beat_strength, compression, dynamic_range,
    spectral characteristics, and frequency ratios.
    Returns a genre string matching common GEMS primary_genre values.
    """
    bpm = f.get('bpm', 120)
    energy = f.get('energy', 0.2)
    beat_str = f.get('beat_strength', 1.0)
    compression = f.get('compression_amount', 0.5)
    dynamic_range = f.get('dynamic_range', 15)
    bass_ratio = f.get('bass_ratio', 0.15)
    sub_ratio = f.get('sub_ratio', 0.05)
    mid_ratio = f.get('mid_ratio', 0.25)
    presence_ratio = f.get('presence_ratio', 0.12)
    air_ratio = f.get('air_ratio', 0.1)
    brightness = f.get('brightness', 2000)
    spectral_complexity = f.get('spectral_complexity', 0.1)
    dissonance = f.get('dissonance', 0.1)
    danceability = f.get('danceability', 1.1)
    key_strength = f.get('key_strength', 0.3)
    lufs = f.get('lufs_integrated', -14)
    onset_rate = f.get('onset_rate', 5)

    scores = {}

    # Electronic / EDM: loud, compressed, strong beats, high danceability, 120-140 BPM
    scores['electronic'] = (
        (0.2 if 118 <= bpm <= 140 else 0.05) +
        (0.2 if compression > 0.7 else 0.05) +
        (0.2 if beat_str > 1.3 else 0.05) +
        (0.2 if danceability > 1.3 else 0.05) +
        (0.1 if sub_ratio > 0.08 else 0.02) +
        (0.1 if lufs > -10 else 0.03)
    )

    # Hip-hop / Rap: heavy bass/sub, moderate BPM, strong beat
    scores['hip-hop'] = (
        (0.2 if 70 <= bpm <= 110 else 0.05) +
        (0.25 if bass_ratio > 0.18 else 0.05) +
        (0.15 if sub_ratio > 0.06 else 0.03) +
        (0.15 if beat_str > 1.0 else 0.05) +
        (0.15 if compression > 0.6 else 0.05) +
        (0.1 if onset_rate < 6 else 0.03)
    )

    # Pop: moderate everything, clear mids/presence, 100-130 BPM
    scores['pop'] = (
        (0.2 if 95 <= bpm <= 135 else 0.05) +
        (0.2 if 0.2 <= mid_ratio <= 0.35 else 0.05) +
        (0.15 if presence_ratio > 0.1 else 0.05) +
        (0.15 if 0.4 <= compression <= 0.8 else 0.05) +
        (0.15 if key_strength > 0.2 else 0.05) +
        (0.15 if 10 <= dynamic_range <= 25 else 0.05)
    )

    # Rock: high energy, wide dynamic range, bright, moderate-fast BPM
    scores['rock'] = (
        (0.2 if energy > 0.25 else 0.05) +
        (0.2 if 110 <= bpm <= 165 else 0.05) +
        (0.15 if dynamic_range > 15 else 0.05) +
        (0.15 if brightness > 2500 else 0.05) +
        (0.15 if onset_rate > 6 else 0.05) +
        (0.15 if dissonance > 0.08 else 0.05)
    )

    # R&B / Soul: moderate tempo, warm (less brightness), strong mids
    scores['r&b'] = (
        (0.2 if 75 <= bpm <= 115 else 0.05) +
        (0.2 if mid_ratio > 0.22 else 0.05) +
        (0.15 if brightness < 2500 else 0.05) +
        (0.15 if bass_ratio > 0.15 else 0.05) +
        (0.15 if key_strength > 0.15 else 0.05) +
        (0.15 if 0.3 <= compression <= 0.7 else 0.05)
    )

    # Indie / Alternative: high spectral complexity, varied dynamics, moderate energy
    scores['indie'] = (
        (0.2 if spectral_complexity > 0.08 else 0.05) +
        (0.2 if dynamic_range > 12 else 0.05) +
        (0.15 if 0.08 <= energy <= 0.3 else 0.05) +
        (0.15 if dissonance > 0.05 else 0.05) +
        (0.15 if air_ratio > 0.1 else 0.05) +
        (0.15 if compression < 0.7 else 0.05)
    )

    # Folk / Acoustic: low energy, high dynamics, strong key, airy
    scores['folk'] = (
        (0.2 if energy < 0.15 else 0.05) +
        (0.2 if dynamic_range > 18 else 0.05) +
        (0.2 if key_strength > 0.25 else 0.05) +
        (0.15 if air_ratio > 0.12 else 0.05) +
        (0.15 if compression < 0.5 else 0.05) +
        (0.1 if bpm < 130 else 0.03)
    )

    # Classical / Ambient: very low energy, huge dynamics, low beat strength
    scores['classical'] = (
        (0.25 if energy < 0.08 else 0.03) +
        (0.25 if dynamic_range > 22 else 0.05) +
        (0.2 if beat_str < 0.8 else 0.05) +
        (0.15 if compression < 0.4 else 0.05) +
        (0.15 if key_strength > 0.3 else 0.05)
    )

    best = max(scores, key=scores.get)
    return best


def _extract_core(audio: np.ndarray, sr: int, audio_stereo: np.ndarray = None) -> Dict:
    """Extract all 33 audio features from a mono float32 array."""
    features: Dict = {}

    # === STEREO FEATURES ===
    if audio_stereo is not None and audio_stereo.ndim == 2 and audio_stereo.shape[1] == 2:
        left = audio_stereo[:, 0].astype(np.float64)
        right = audio_stereo[:, 1].astype(np.float64)
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        mid_energy = float(np.sqrt(np.mean(mid ** 2)))
        side_energy = float(np.sqrt(np.mean(side ** 2)))
        features['stereo_width'] = float(side_energy / max(mid_energy, 1e-10))
        features['mid_side_ratio'] = float(mid_energy / max(mid_energy + side_energy, 1e-10))
        if len(left) > 0:
            corr = np.corrcoef(left, right)[0, 1]
            features['stereo_correlation'] = float(corr) if np.isfinite(corr) else 1.0
        else:
            features['stereo_correlation'] = 1.0
    else:
        features['stereo_width'] = 0.0
        features['mid_side_ratio'] = 1.0
        features['stereo_correlation'] = 1.0

    # --- LOUDNESS (LUFS) ---
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    features['lufs_integrated'] = float(loudness) if not np.isnan(loudness) else -30.0

    block_size = int(0.4 * sr)
    lv = []
    for i in range(0, len(audio) - block_size, block_size // 2):
        blk = audio[i : i + block_size]
        if len(blk) == block_size:
            bl = meter.integrated_loudness(blk)
            if not np.isnan(bl):
                lv.append(bl)
    features['loudness_range'] = float(np.percentile(lv, 95) - np.percentile(lv, 10)) if len(lv) > 1 else 0.0

    # --- RHYTHM ---
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    features['bpm'] = float(tempo) if np.isscalar(tempo) else float(tempo[0]) if len(tempo) > 0 else 120.0
    features['beat_strength'] = float(np.mean(librosa.onset.onset_strength(y=audio, sr=sr)))

    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    features['onset_rate'] = float(len(onset_frames) / (len(audio) / sr))

    # --- TONAL ---
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_index = int(np.argmax(chroma_mean))
    features['key'] = pitch_classes[key_index]

    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, key_index))[0, 1]
    minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, key_index))[0, 1]
    features['scale'] = 'major' if major_corr > minor_corr else 'minor'
    features['key_strength'] = float(max(major_corr, minor_corr)) if not np.isnan(major_corr) else 0.5

    # --- SPECTRAL ---
    centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features['brightness'] = float(np.mean(centroids))
    features['brightness_variance'] = float(np.var(centroids))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features['spectral_rolloff'] = float(np.mean(rolloff))

    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features['spectral_complexity'] = float(np.mean(bw) / sr)

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr'] = float(np.mean(zcr))

    # --- 7-BAND FREQUENCY ANALYSIS ---
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    bands = {
        'sub_ratio': (20, 60),
        'bass_ratio': (60, 250),
        'low_mid_ratio': (250, 500),
        'mid_ratio': (500, 2000),
        'high_mid_ratio': (2000, 4000),
        'presence_ratio': (4000, 8000),
        'air_ratio': (8000, 20000),
    }
    total_energy = np.sum(fft)
    if total_energy > 0:
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            features[name] = float(np.sum(fft[mask]) / total_energy)
    else:
        defaults = [0.10, 0.20, 0.15, 0.25, 0.15, 0.10, 0.05]
        for name, val in zip(bands.keys(), defaults):
            features[name] = val

    # Harmonic distortion estimate (THD approximation)
    if total_energy > 0:
        # Find fundamental frequency from spectral centroid
        fundamental_idx = np.argmax(fft[1:]) + 1  # skip DC
        fundamental_energy = float(fft[fundamental_idx])
        # Sum energy at harmonics (2x, 3x, 4x, 5x fundamental)
        harmonic_energy = 0.0
        for h in range(2, 6):
            h_idx = fundamental_idx * h
            if h_idx < len(fft):
                harmonic_energy += float(fft[h_idx])
        features['harmonic_distortion'] = float(harmonic_energy / max(fundamental_energy, 1e-10))
    else:
        features['harmonic_distortion'] = 0.0

    # --- ENERGY & DYNAMICS ---
    features['energy'] = float(np.sqrt(np.mean(audio ** 2)))

    # True peak (inter-sample) via 4x oversampling
    from scipy import signal as scipy_signal
    try:
        upsampled = scipy_signal.resample(audio, len(audio) * 4)
        features['true_peak_dbfs'] = float(20 * np.log10(max(np.max(np.abs(upsampled)), 1e-10)))
    except Exception:
        features['true_peak_dbfs'] = float(20 * np.log10(max(np.max(np.abs(audio)), 1e-10)))

    p95 = np.percentile(np.abs(audio), 95)
    p10 = np.percentile(np.abs(audio), 10)
    features['dynamic_range'] = min(float(20 * np.log10(p95 / p10)), 60.0) if p10 > 1e-10 else 10.0

    features['crest_factor'] = float(20 * np.log10(np.max(np.abs(audio)) / max(features['energy'], 1e-10))) if features['energy'] > 0 else 1.0

    # Attack time
    envelope = np.abs(librosa.onset.onset_strength(y=audio, sr=sr))
    if len(envelope) > 0:
        attack_samples = np.argmax(envelope > 0.9 * np.max(envelope))
        features['attack_time'] = float(attack_samples / sr * len(audio) / len(envelope))
    else:
        features['attack_time'] = 0.0
    features['attack_time'] = max(features['attack_time'], 0.1)

    features['compression_amount'] = max(0.0, 1.0 - (features['crest_factor'] / 26.0))

    # Spectral flux
    stft = librosa.stft(audio)
    spectral_flux = np.sum(np.diff(librosa.amplitude_to_db(np.abs(stft))), axis=0)
    features['spectral_flux'] = float(np.mean(np.abs(spectral_flux)))

    features['dissonance'] = float(np.mean(zcr) * features['spectral_flux'] / 1000)

    # Danceability
    if len(beats) > 1:
        diffs = np.diff(beats)
        tempo_stability = 1.0 - np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else 0.5
    elif features.get('bpm', 0) > 0:
        # Fallback: estimate from BPM when beat frames are sparse
        bpm_factor = min(features['bpm'] / 120.0, 1.5)
        tempo_stability = bpm_factor * 0.5
    else:
        tempo_stability = 0.5
    features['danceability'] = max(0.0, float(tempo_stability * features['beat_strength']))

    # Clean NaN
    for k, v in features.items():
        if isinstance(v, float) and np.isnan(v):
            features[k] = 0.0

    return features
