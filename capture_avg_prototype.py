#!/usr/bin/env python3
"""PROTOTYPE: loudest-window vs averaged-3-window capture.

Re-captures a track through the live Mac-worker chain (Spotify -> Loopback),
but instead of analyzing ONLY the highest-energy window (current behavior),
it extracts features from all 3 windows (25/50/75%) and shows:
  - per-window brightness / bass / air
  - LOUDEST pick (what the analyzer does today)
  - AVERAGED across 3 windows (proposed)
  - the delta

Read-only: does NOT write to the DB or change any production code.
Usage: python3 capture_avg_prototype.py [spotify_track_url]
Default track = Elle Coves "Before I Fall Apart".
"""
import sys, time, json
import numpy as np
import sounddevice as sd

import mac_worker as W

TRACK_URL = sys.argv[1] if len(sys.argv) > 1 else \
    "https://open.spotify.com/track/1DTNVJ6MC8gmJTJ0WHJbAC"

SHOW = ['brightness', 'bass_ratio', 'sub_ratio', 'air_ratio', 'presence_ratio',
        'high_mid_ratio', 'mid_ratio', 'lufs_integrated', 'energy', 'dynamic_range']


def main():
    track_id = W._extract_track_id(TRACK_URL)
    print(f"track {track_id}")

    # pause gems/discovery so they don't fight over Spotify
    paused = []
    try:
        import subprocess as sp
        for p in json.loads(sp.run(['pm2', 'jlist'], capture_output=True, text=True, timeout=10).stdout):
            if p.get('pm2_env', {}).get('status') == 'online' and p['name'] in ('gems', 'discovery'):
                sp.run(['pm2', 'stop', p['name']], capture_output=True, timeout=30)
                paused.append(p['name'])
        if paused:
            print(f"paused {paused} for capture"); time.sleep(5)
    except Exception as e:
        print(f"(couldn't pause scripts: {e})")

    try:
        info = W._get_track_info(track_id)
        dur = info.get('duration_ms', 0)
        print(f"{info.get('name')} — {', '.join(a['name'] for a in info.get('artists', []))} ({dur/1000:.0f}s)")

        loop = W._find_loopback_device()
        if loop is None:
            print("NO LOOPBACK DEVICE — is the Loopback app running with the spotify device enabled?")
            return

        device_id = W._ensure_device_active()
        for _ in range(3):
            if W._play_track(track_id, device_id):
                time.sleep(2); break
            time.sleep(2)

        pts = [int(dur*0.25), int(dur*0.50), int(dur*0.75)]
        per_window = []
        for i, pos in enumerate(pts):
            W._seek_to(pos); time.sleep(1.5)
            mono, stereo = W._record_sample(loop)
            if mono is None:
                print(f"  window {i+1} @ {pos/1000:.0f}s: SILENT"); continue
            energy = float(np.sqrt(np.mean(mono**2)))
            feats = W.extract_features_from_audio(mono, audio_stereo=stereo)
            feats['_energy_sel'] = energy
            feats['_pos'] = pos/1000
            per_window.append(feats)
            print(f"  window {i+1} @ {pos/1000:.0f}s: energy={energy:.4f} "
                  f"brightness={feats.get('brightness',0):.0f} bass={feats.get('bass_ratio',0):.3f} air={feats.get('air_ratio',0):.3f}")
        W._pause_playback()

        if not per_window:
            print("no windows captured"); return

        loudest = max(per_window, key=lambda f: f['_energy_sel'])
        def avg(k): return float(np.mean([f.get(k, 0) for f in per_window]))

        print("\n" + "="*64)
        print(f"{'feature':18}{'LOUDEST (today)':>16}{'AVG of 3 (proposed)':>22}")
        print("="*64)
        for k in SHOW:
            lv, av = loudest.get(k, 0), avg(k)
            print(f"{k:18}{lv:>16.3f}{av:>22.3f}")
        print(f"\nloudest window was @ {loudest['_pos']:.0f}s")
        print(f"brightness: loudest {loudest.get('brightness',0):.0f}Hz vs avg {avg('brightness'):.0f}Hz "
              f"(Δ {loudest.get('brightness',0)-avg('brightness'):+.0f}Hz)")
        print(f"bass_ratio: loudest {loudest.get('bass_ratio',0):.3f} vs avg {avg('bass_ratio'):.3f}")
    finally:
        for n in paused:
            try:
                import subprocess as sp
                sp.run(['pm2', 'start', n], capture_output=True, timeout=30)
            except Exception:
                pass
        if paused:
            print(f"\nresumed {paused}")


if __name__ == '__main__':
    main()
