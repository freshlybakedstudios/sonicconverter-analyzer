"""
Audio feature extraction for uploaded tracks.
Ported from GEMS/fixed_gems_pipeline_v2.py - adapted for file upload (no Loopback needed).
"""

import numpy as np
import librosa
import pyloudnorm as pyln
from typing import Dict


class GenreAwareEmotionDetector:
    """Genre-aware GEMS 9 emotion detection. Ported from gems_emotion_detector.py."""

    GENRE_GROUPS = {
        'bass_heavy': ['hip-hop', 'electronic', 'reggae', 'latin'],
        'energy_focused': ['rock', 'punk', 'metal'],
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

    def get_genre_group(self, genre: str) -> str:
        if not genre:
            return 'balanced'
        genre_lower = genre.lower()
        for group, genres in self.GENRE_GROUPS.items():
            if genre_lower in genres:
                return group
        return 'balanced'

    def _calc_feature_score(self, value, comparison, threshold):
        if comparison == 'gt':
            return min(1.0, value / threshold) if value >= threshold else (value / threshold) * 0.8
        elif comparison == 'lt':
            return 1.0 if value <= threshold else min(1.0, (threshold / (value + 0.001)) * 0.8)
        elif comparison == 'range':
            min_val, max_val = threshold
            if min_val <= value <= max_val:
                return 1.0
            elif value < min_val:
                return max(0.0, 1.0 - (min_val - value) / min_val * 0.6)
            else:
                return max(0.0, 1.0 - (value - max_val) / max_val * 0.6)
        return 0.0

    def detect(self, features: Dict, genre_hint: str = '') -> Dict:
        group = self.get_genre_group(genre_hint)
        if group == 'bass_heavy':
            genre_emotions = self.BASS_HEAVY_EMOTIONS
        elif group == 'energy_focused':
            genre_emotions = self.ENERGY_FOCUSED_EMOTIONS
        elif group == 'soft':
            genre_emotions = self.SOFT_EMOTIONS
        else:
            genre_emotions = self.BALANCED_EMOTIONS

        all_emotions = {**genre_emotions, **self.UNIVERSAL_EMOTIONS}

        scores = {}
        for emotion, rules in all_emotions.items():
            total = 0.0
            for feat_name, (comp, thresh, weight) in rules.items():
                val = features.get(feat_name, 0)
                total += self._calc_feature_score(val, comp, thresh) * weight
            scores[emotion] = min(1.0, total)

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
    # Load audio at 44.1 kHz mono
    y, sr = librosa.load(file_path, sr=44100, mono=True)
    duration_samples = len(y)
    duration_sec = duration_samples / sr

    if duration_sec < 5:
        raise ValueError("Audio file too short for analysis (< 5 seconds)")

    # 3-point sampling (25%, 50%, 75%) - take 8-second segments
    segment_len = int(min(8, duration_sec * 0.2) * sr)
    sample_points = [0.25, 0.50, 0.75]
    segments = []
    energies = []

    for pct in sample_points:
        start = int(duration_samples * pct) - segment_len // 2
        start = max(0, min(start, duration_samples - segment_len))
        seg = y[start : start + segment_len]
        if len(seg) < sr:  # skip if less than 1 second
            continue
        segments.append(seg)
        energies.append(float(np.sqrt(np.mean(seg ** 2))))

    if not segments:
        raise ValueError("Could not extract valid audio segments")

    # Use highest-energy segment for feature extraction
    best_idx = int(np.argmax(energies))
    audio = segments[best_idx]

    features = _extract_core(audio, sr)

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


def _extract_core(audio: np.ndarray, sr: int) -> Dict:
    """Extract all 33 audio features from a mono float32 array."""
    features: Dict = {}

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

    # --- ENERGY & DYNAMICS ---
    features['energy'] = float(np.sqrt(np.mean(audio ** 2)))

    p95 = np.percentile(np.abs(audio), 95)
    p10 = np.percentile(np.abs(audio), 10)
    features['dynamic_range'] = float(p95 / p10) if p10 > 0 else 10.0

    features['crest_factor'] = float(np.max(np.abs(audio)) / features['energy']) if features['energy'] > 0 else 1.0

    # Attack time
    envelope = np.abs(librosa.onset.onset_strength(y=audio, sr=sr))
    if len(envelope) > 0:
        attack_samples = np.argmax(envelope > 0.9 * np.max(envelope))
        features['attack_time'] = float(attack_samples / sr * len(audio) / len(envelope))
    else:
        features['attack_time'] = 0.0

    features['compression_amount'] = 1.0 - (features['crest_factor'] / 20.0)

    # Spectral flux
    stft = librosa.stft(audio)
    spectral_flux = np.sum(np.diff(librosa.amplitude_to_db(np.abs(stft))), axis=0)
    features['spectral_flux'] = float(np.mean(np.abs(spectral_flux)))

    features['dissonance'] = float(np.mean(zcr) * features['spectral_flux'] / 1000)

    # Danceability
    if len(beats) > 1:
        diffs = np.diff(beats)
        tempo_stability = 1.0 - np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else 0.5
    else:
        tempo_stability = 0.5
    features['danceability'] = float(tempo_stability * features['beat_strength'])

    # Clean NaN
    for k, v in features.items():
        if isinstance(v, float) and np.isnan(v):
            features[k] = 0.0

    return features
