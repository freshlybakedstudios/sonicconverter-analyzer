#!/usr/bin/env python3
"""FAST, NETWORK-FREE cache patch: apply the 1,032 pop-flip corrections
directly into the on-disk universe cache by RECOMPUTING emotions locally from
the features already in the cache (no Supabase — the earlier version hung on a
network fetch). Rebuild emotion index, upload to GCS, restart Railway.
"""
import os, sys, json, time, subprocess
from pathlib import Path

t0 = time.time()
sys.path.insert(0, '/Users/alexanderalmgren/artist-pipeline/GEMS')
from gems_emotion_detector import GenreAwareGEMS9EmotionDetector
det = GenreAwareGEMS9EmotionDetector()

# 1. isrcs to fix (local file — no network)
isrcs = {r['isrc'] for r in json.load(open('/tmp/pop_flip_FULL_backup.json'))}
print(f"flip set: {len(isrcs)} isrcs", flush=True)

# 2. load cache
cache_path = Path('/Users/alexanderalmgren/artist-pipeline/sonicconverter-web/cache/universe/gems_universe.json')
print("loading cache...", flush=True)
cache = json.load(cache_path.open())
print(f"  loaded {len(cache.get('gems',[]))} gems in {time.time()-t0:.0f}s", flush=True)

# 3. patch + recompute emotions locally from each row's own features
EMO = ['emotion_1','emotion_2','emotion_3','emotion_4']
patched = 0
for g in cache['gems']:
    if g.get('isrc') not in isrcs:
        continue
    g['tonal_balance'] = 'pop'
    g['primary_genre'] = 'Pop'
    res = det.analyze_track(g, {'primary_genre': 'pop'})
    pairs = res['emotion_pairs']
    g['emotional_signature'] = res['primary_contradiction']
    for i in range(4):
        g[f'emotion_{i+1}'] = pairs[i][0] if len(pairs) > i else None
        g[f'emotion_{i+1}_score'] = round(pairs[i][1], 4) if len(pairs) > i else None
    patched += 1
print(f"  patched {patched} rows in {time.time()-t0:.0f}s", flush=True)

# verify Elle in-cache
elle = next((g for g in cache['gems'] if g.get('isrc') == 'US23A1564256'), None)
if elle:
    print(f"  ELLE in cache -> tonal={elle.get('tonal_balance')} emo1={elle.get('emotion_1')}", flush=True)

# 4. rebuild emotion index
idx = {}
for g in cache['gems']:
    iso = g.get('isrc')
    if not iso: continue
    for n in range(1,5):
        e = g.get(f'emotion_{n}')
        if e: idx.setdefault(e, []).append(iso)
cache['emotion_index'] = idx
print(f"  rebuilt emotion_index ({len(idx)} emotions) in {time.time()-t0:.0f}s", flush=True)

# 5. atomic save
tmp = cache_path.with_suffix('.json.tmp')
print("saving...", flush=True)
with tmp.open('w', encoding='utf-8') as f:
    json.dump(cache, f, separators=(',', ':'))
tmp.replace(cache_path)
print(f"  saved {cache_path.stat().st_size/1024/1024:.0f}MB in {time.time()-t0:.0f}s", flush=True)

# 6. GCS upload
print("uploading to GCS...", flush=True)
up = subprocess.run(['gsutil','-o','GSUtil:parallel_composite_upload_threshold=150M',
                     'cp', str(cache_path), 'gs://fbs-static-assets/gems_universe.json'],
                    capture_output=True, text=True)
print(f"  gcs rc={up.returncode} {up.stderr[-160:] if up.returncode else 'OK'}", flush=True)

# 7. Railway restart
if up.returncode == 0:
    print("restarting Railway...", flush=True)
    rs = subprocess.run(['railway','service','restart','--yes'], capture_output=True, text=True)
    print(f"  railway rc={rs.returncode} {(rs.stderr or rs.stdout)[-160:]}", flush=True)
print(f"DONE in {time.time()-t0:.0f}s", flush=True)
