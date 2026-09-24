"""Musical key detection (Krumhansl-Schmuckler).

2026-09-23. Damion Yang reported "Cerulean (The Sea)" (B major) coming back as
F# major. Both feature extractors were calling the LOUDEST pitch class the
tonic (`argmax(chroma)`), which lands on the fifth whenever the bass or a pad
sits on it — F# is the fifth of B. It also decided major/minor by rotating a
scale mask to that already-wrong tonic, so a wrong tonic could drag the mode
wrong too, and `scale` feeds the emotion detector's major/minor gates.

This correlates the chroma against all 24 rotated Krumhansl-Kessler profiles
and takes the best, which is the standard method and does not assume the
tonic is loud. It also returns a confidence (the margin over the best
candidate rooted on a different tonic) so the UI can say "low confidence"
instead of stating a key it cannot support — Damion's own suggestion, and the
right call: a wrong key is quickly falsifiable and casts doubt on everything
else on the page.

NOTE: `key_strength` is deliberately NOT computed here. Its old formula stays
exactly as it was in both extractors, because that value is stored for all
320k tracks in the universe cache and has calibrated percentile bounds; a new
formula would silently break those comparisons.
"""
import numpy as np

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Kessler probe-tone profiles
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# A margin of this much correlation over the best rival tonic reads as a
# confident call; below ~0.25 of it we tell the user it is a guess.
_FULL_MARGIN = 0.25
LOW_CONFIDENCE = 0.25


def detect_key(chroma_mean):
    """(key, scale, confidence) from a 12-bin mean chroma vector.

    Returns (None, None, 0.0) when the input cannot support a call, so the
    caller can fall back or report no key rather than invent one.
    """
    v = np.asarray(chroma_mean, dtype=float).ravel()
    if v.size != 12 or not np.all(np.isfinite(v)) or float(v.sum()) <= 0:
        return None, None, 0.0
    vc = v - v.mean()
    if not np.any(vc):
        return None, None, 0.0

    scored = []
    for i in range(12):
        for name, prof in (('major', _KS_MAJOR), ('minor', _KS_MINOR)):
            p = np.roll(prof, i)
            p = p - p.mean()
            den = float(np.sqrt(float((vc ** 2).sum()) * float((p ** 2).sum())))
            scored.append((float((vc * p).sum() / den) if den else 0.0, i, name))
    scored.sort(key=lambda s: -s[0])

    best = scored[0]
    rival = next((s for s in scored[1:] if s[1] != best[1]), scored[1])
    confidence = float(np.clip((best[0] - rival[0]) / _FULL_MARGIN, 0.0, 1.0))
    return PITCH_CLASSES[best[1]], best[2], confidence


def key_label(key, scale, confidence):
    """What to print. Never states a key the numbers cannot support."""
    if not key:
        return 'N/A'
    base = f"{key} {scale}".strip()
    return base if confidence >= LOW_CONFIDENCE else f"{base} (low confidence)"
