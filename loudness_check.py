"""Free loudness checker (2026-09-08). Behind LOUDNESS_CHECKER_ENABLED=1.

POST /api/loudness-check  (multipart file, no token)  ->  JSON measurements
GET  /loudness-checker                                  ->  static page

Measures the WHOLE file, stereo, BS.1770 via pyloudnorm (standard LUFS, unlike
the rig's mono-mean window numbers). Nothing is stored; the temp file is
deleted. Rate-limited per IP (in-memory) so it can't be used as a batch API.
"""
import os, time, tempfile, threading
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()
_HITS: dict = {}
_LOCK = threading.Lock()
MAX_MB = 80
PER_HOUR = 12
SR = 48000
PLATFORMS = [("Spotify", -14.0), ("Apple Music", -16.0), ("YouTube", -14.0),
             ("Amazon Music", -14.0), ("Tidal", -14.0), ("Deezer", -15.0)]


def _rate_ok(ip: str) -> bool:
    now = time.time()
    with _LOCK:
        hits = [t for t in _HITS.get(ip, []) if now - t < 3600]
        if len(hits) >= PER_HOUR:
            _HITS[ip] = hits
            return False
        hits.append(now); _HITS[ip] = hits
        return True


def measure(path: str) -> dict:
    import librosa, pyloudnorm as pyln
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    if sr != SR:
        y = np.stack([librosa.resample(y[c], orig_sr=sr, target_sr=SR) for c in range(y.shape[0])])
        sr = SR
    stereo = y[:2].T.astype(np.float64)  # (n, 2)
    n = len(stereo)
    if n < sr * 3:
        raise ValueError("File is shorter than 3 seconds")
    meter = pyln.Meter(sr)
    integrated = float(meter.integrated_loudness(stereo))
    # short-term (3 s) loudness, hop 1 s -> max and loudness range (p95 - p10 of gated blocks)
    win, hop = sr * 3, sr
    st = []
    for s in range(0, n - win + 1, hop):
        try:
            v = meter.integrated_loudness(stereo[s:s + win])
            if np.isfinite(v) and v > -70:
                st.append(v)
        except Exception:
            pass
    st = np.array(st) if st else np.array([integrated])
    gated = st[st > (integrated - 20)]
    lra = float(np.percentile(gated, 95) - np.percentile(gated, 10)) if len(gated) > 2 else 0.0
    sample_peak = float(20 * np.log10(max(np.max(np.abs(stereo)), 1e-10)))
    try:
        from scipy.signal import resample_poly
        up = resample_poly(stereo, 4, 1, axis=0)
        true_peak = float(20 * np.log10(max(np.max(np.abs(up)), 1e-10)))
    except Exception:
        true_peak = sample_peak
    rms = float(np.sqrt(np.mean(stereo ** 2)))
    crest = float(20 * np.log10(max(np.max(np.abs(stereo)), 1e-10) / max(rms, 1e-10)))
    clipped = int(np.sum(np.abs(stereo) >= 0.9999))
    return {
        "duration_s": round(n / sr, 1),
        "lufs_integrated": round(integrated, 1),
        "lufs_short_term_max": round(float(np.max(st)), 1),
        "loudness_range_lu": round(lra, 1),
        "true_peak_dbtp": round(true_peak, 2),
        "sample_peak_dbfs": round(sample_peak, 2),
        "crest_factor_db": round(crest, 1),
        "clipped_samples": clipped,
        "platform_gain_db": {name: round(target - integrated, 1) for name, target in PLATFORMS},
    }


@router.post("/api/loudness-check")
async def loudness_check(request: Request, file: UploadFile = File(...)):
    ip = (request.headers.get("x-forwarded-for") or request.client.host or "?").split(",")[0].strip()
    if not _rate_ok(ip):
        raise HTTPException(429, "Too many checks from this connection. Try again in an hour.")
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ("mp3", "wav", "flac", "ogg", "m4a", "aac", "aiff", "aif"):
        raise HTTPException(400, f"Unsupported file type: .{ext}")
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_MB} MB)")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    try:
        tmp.write(data); tmp.close()
        t0 = time.time()
        out = measure(tmp.name)
        out["measured_in_s"] = round(time.time() - t0, 1)
        out["filename"] = file.filename
        return JSONResponse(out)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        print(f"loudness-check failed: {e}")
        raise HTTPException(500, "Could not decode that file")
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass


@router.get("/loudness-checker")
async def loudness_checker_page():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "loudness-checker.html"))
