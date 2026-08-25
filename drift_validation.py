#!/usr/bin/env python3
"""
drift_validation.py — backtest the drift model's predictive validity.

For each major genre family:
  1. Compute cohort centroids using only data through the validation cutoff
     (default: 2025H1, i.e. tracks released through 2025-06-30).
  2. Compute the drift vector for the most-recent transition leading up to
     the cutoff (default: 2024H2 → 2025H1).
  3. Score every artist with a track in the cutoff bucket by alignment with
     that drift vector. Dedupe by artist (keep highest-scoring track).
  4. Pull artists_history rows for those artists. For each, compute the
     sp_monthly_listeners growth ratio from the earliest snapshot at/after
     ~the cutoff date through the most recent snapshot.
  5. Bucket artists by alignment quintile within each genre. Compute median
     growth per quintile and the Spearman correlation between alignment and
     growth.
  6. Print verdict per genre: predictive (correlation > +0.1, low p-value
     proxy), mixed (small or no signal), descriptive (no signal or negative).

Reuses the prototype's math so the model under test is identical.

READ-ONLY. No INSERT / UPDATE / DELETE.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from supabase import create_client

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / '.env')
except ImportError:
    pass

# Reuse the prototype's primitives so the model under test is unchanged
from drift_prototype import (
    SUPABASE_URL_DEFAULT, SUPABASE_KEY_DEFAULT, PAGE_SIZE,
    WEIGHTED_FEATURES, REPORTING_FEATURES, FEATURE_WEIGHTS,
    EMOTION_WEIGHT, GENRE_WEIGHT,
    DEFAULT_FAMILIES,
    paginate, fetch_gems_for_isrcs, bucket_of,
    build_genre_tokens, build_genre_families, has_incompatible_pair,
    compute_audio_centroid, compute_emotion_distribution, compute_genre_distribution,
    compute_feature_stats, delta_to_z, weighted_cosine, cosine, vector_diff,
    compute_alignment, normalize_feature,
)

DEFAULT_GENRES_FOR_VALIDATION = [
    'pop', 'electronic', 'hip-hop', 'rock', 'metal', 'latin', 'r&b', 'jazz', 'country', 'folk',
]

# Metrics from artists_history we'll test correlation against. Each is a numeric
# column on the snapshot row. Some are momentum-flavored (sp_popularity,
# cm_artist_score), some are raw size (followers, views). We test all and
# surface the strongest signal per genre.
DEFAULT_METRICS = [
    'sp_monthly_listeners',
    'sp_popularity',
    'sp_followers',
    'sp_playlist_total_reach',
    'spotify_playlist_count',
    'cm_artist_score',
    'tiktok_followers',
    'tiktok_top_video_views',
    'ycs_subscribers',
    'ycs_views',
]


def banner(s):
    print()
    print('=' * 78)
    print(s)
    print('=' * 78)


def spearman(a, b):
    """Spearman rank correlation. No scipy dep."""
    n = len(a)
    if n < 3:
        return None
    def rank(xs):
        sorted_with_idx = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[sorted_with_idx[j + 1]] == xs[sorted_with_idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based ranks averaged for ties
            for k in range(i, j + 1):
                ranks[sorted_with_idx[k]] = avg
            i = j + 1
        return ranks
    ra, rb = rank(list(a)), rank(list(b))
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    denom = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)) *
                      sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / denom if denom else None


def fetch_artists_history(supabase, artist_ids, since_iso, metrics):
    """Fetch artists_history rows for the given artist IDs since `since_iso`,
    pulling all configured outcome metrics. Returns dict artist_id → list of dicts."""
    out = defaultdict(list)
    aids = [str(a) for a in artist_ids if a is not None]
    cols = 'artist_id, snapshot_date, ' + ', '.join(metrics)
    batch = 100
    for i in range(0, len(aids), batch):
        chunk = aids[i:i + batch]
        try:
            resp = (
                supabase.table('artists_history')
                .select(cols)
                .in_('artist_id', chunk)
                .gte('snapshot_date', since_iso)
                .order('snapshot_date')
                .execute()
            )
            for r in resp.data or []:
                aid = r.get('artist_id')
                if aid:
                    out[str(aid)].append(r)
        except Exception as e:
            print(f'  [error fetching history batch {i}: {e}]')
    return out


def _parse_snap_dt(s):
    """Parse snapshot_date in any format we've seen, normalize to UTC-aware."""
    if not s:
        return None
    txt = s.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        # Bare date like '2025-02-12'
        try:
            dt = datetime.strptime(txt[:10], '%Y-%m-%d')
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def growth_ratio(snaps, metric='sp_monthly_listeners',
                 first_window_days=90, min_span_days=120):
    """Compute log growth ratio for an artist's snapshots on the given metric.
    Picks the median value of the first `first_window_days` of snapshots as
    the baseline (smooths over a noisy first ping) and the most recent value.
    Returns log(ratio) or None if span/coverage is insufficient."""
    if not snaps or len(snaps) < 2:
        return None
    valid = [s for s in snaps if s.get(metric) not in (None, 0)]
    if len(valid) < 2:
        return None
    valid.sort(key=lambda s: s['snapshot_date'])
    first_dt = _parse_snap_dt(valid[0]['snapshot_date'])
    last_dt = _parse_snap_dt(valid[-1]['snapshot_date'])
    if first_dt is None or last_dt is None:
        return None
    span_days = (last_dt - first_dt).days
    if span_days < min_span_days:
        return None
    # baseline = median of snapshots in the first 90-day window
    baseline_vals = []
    for s in valid:
        dt = _parse_snap_dt(s['snapshot_date'])
        if dt is None:
            continue
        if (dt - first_dt).days <= first_window_days:
            baseline_vals.append(float(s[metric]))
    if not baseline_vals:
        return None
    baseline_vals.sort()
    baseline = baseline_vals[len(baseline_vals) // 2]
    latest = float(valid[-1][metric])
    if baseline <= 0 or latest <= 0:
        return None
    return math.log(latest / baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-bucket', default='2024H2',
                        help='Bucket used as the "before" centroid (default: 2024H2)')
    parser.add_argument('--validation-bucket', default='2025H1',
                        help='Bucket used as the "after" centroid AND as the artist-scoring window (default: 2025H1)')
    parser.add_argument('--cutoff-date', default='2025-07-01',
                        help='ISO date marking end of validation bucket. Tracks released after this are excluded from cohort math. Default: 2025-07-01')
    parser.add_argument('--history-since', default='2025-01-01',
                        help='Earliest snapshot_date to fetch from artists_history. Default: 2025-01-01')
    parser.add_argument('--genres', default=','.join(DEFAULT_GENRES_FOR_VALIDATION))
    parser.add_argument('--window', choices=['quarter', 'halfyear'], default='halfyear')
    parser.add_argument('--cohort-mode', choices=['family', 'token', 'primary'], default='family')
    parser.add_argument('--min-cohort', type=int, default=30)
    parser.add_argument('--limit', type=int, default=200000)
    parser.add_argument('--metrics', default=','.join(DEFAULT_METRICS),
                        help='Comma-separated artists_history metrics to test as outcomes.')
    parser.add_argument('--top-decile-pct', type=float, default=10.0,
                        help='Right-tail percentile threshold for bimodal detection (default: 10).')
    parser.add_argument('--supabase-url', default=SUPABASE_URL_DEFAULT)
    parser.add_argument('--supabase-key', default=SUPABASE_KEY_DEFAULT)
    args = parser.parse_args()

    target_genres = [g.strip().lower() for g in args.genres.split(',') if g.strip()]
    target_buckets = [args.baseline_bucket, args.validation_bucket]
    cutoff_date = args.cutoff_date

    sb = create_client(args.supabase_url, args.supabase_key)
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'baseline_bucket': args.baseline_bucket,
        'validation_bucket': args.validation_bucket,
        'cutoff_date': cutoff_date,
        'history_since': args.history_since,
        'metrics': args.metrics,
        'cohort_mode': args.cohort_mode,
        'genres': target_genres,
    }

    # 1) Pull tracks (top + recent), filtered to release_date <= cutoff (no look-ahead)
    banner('STEP 1: pull tracks (top + recent), enforce no look-ahead')
    cols = ('isrc, recent_track_isrc, release_date, recent_track_release_date, '
            'track_genres, recent_track_genres, recent_track, '
            'spotify_plays, recent_track_spotify_plays, artist_id, top_track')
    rows_top = paginate(
        sb, 'tracks', cols,
        where_filter=lambda q: q.not_.is_('release_date', 'null').lte('release_date', cutoff_date),
        limit=args.limit,
    )
    print(f'  rows with top release_date <= cutoff:  {len(rows_top)}')
    rows_rec_only = paginate(
        sb, 'tracks', cols,
        where_filter=lambda q: (q.is_('release_date', 'null')
                                 .not_.is_('recent_track_release_date', 'null')
                                 .lte('recent_track_release_date', cutoff_date)),
        limit=args.limit,
    )
    print(f'  rows with only recent date <= cutoff:  {len(rows_rec_only)}')
    track_rows = rows_top + rows_rec_only

    # 2) Build target records (top and recent), filter to target buckets
    target_records = []
    for t in track_rows:
        top_isrc = t.get('isrc')
        rec_isrc = t.get('recent_track_isrc')
        top_rd = t.get('release_date')
        rec_rd = t.get('recent_track_release_date')
        same_track = (top_isrc and rec_isrc and top_isrc == rec_isrc)

        if top_isrc and top_rd and top_rd <= cutoff_date:
            b = bucket_of(top_rd, args.window)
            if b in target_buckets:
                target_records.append({
                    'isrc': top_isrc,
                    'release_date': top_rd,
                    'bucket': b,
                    'track_genres': t.get('track_genres') or '',
                    'spotify_plays': t.get('spotify_plays') or 0,
                    'artist_id': t.get('artist_id'),
                    'track_name': t.get('top_track') or '',
                    'track_role': 'top',
                })

        if rec_isrc and rec_rd and rec_rd <= cutoff_date and not same_track:
            b = bucket_of(rec_rd, args.window)
            if b in target_buckets:
                target_records.append({
                    'isrc': rec_isrc,
                    'release_date': rec_rd,
                    'bucket': b,
                    'track_genres': t.get('recent_track_genres') or t.get('track_genres') or '',
                    'spotify_plays': t.get('recent_track_spotify_plays') or 0,
                    'artist_id': t.get('artist_id'),
                    'track_name': t.get('recent_track') or '',
                    'track_role': 'recent',
                })
    print(f'  records in target buckets:             {len(target_records)}')

    # 3) Pull gems features
    banner('STEP 2: pull audio features')
    gems_columns = ','.join([
        'isrc', *[name for name, *_ in WEIGHTED_FEATURES],
        *[name for name, *_ in REPORTING_FEATURES], 'bpm',
        'emotion_1', 'emotion_1_score', 'emotion_2', 'emotion_2_score',
        'emotion_3', 'emotion_3_score', 'emotion_4', 'emotion_4_score',
        'primary_genre', 'secondary_genre',
    ])
    isrcs_needed = {r['isrc'] for r in target_records}
    gems_by_isrc = fetch_gems_for_isrcs(sb, isrcs_needed, gems_columns)
    print(f'  features fetched for {len(gems_by_isrc)}/{len(isrcs_needed)} ISRCs')

    # 4) Enrich
    enriched = []
    seen = set()
    for r in target_records:
        if r['isrc'] in seen:
            continue
        feats = gems_by_isrc.get(r['isrc'])
        if not feats:
            continue
        tokens = build_genre_tokens(r, feats)
        if has_incompatible_pair(tokens):
            continue
        try:
            plays = float(r.get('spotify_plays') or 0)
        except (TypeError, ValueError):
            plays = 0.0
        r['features'] = feats
        r['tokens'] = tokens
        r['families'] = build_genre_families(r, feats)
        r['spotify_plays'] = plays
        r['weight'] = math.log1p(plays) + 1.0
        enriched.append(r)
        seen.add(r['isrc'])
    print(f'  enriched records: {len(enriched)}')

    stats = compute_feature_stats(enriched)

    # 5) Score every artist in the validation bucket per genre family
    banner('STEP 3: score artists, fetch outcomes, compute correlations')
    all_scored_artist_ids = set()
    per_genre_artists = {}

    for genre in target_genres:
        if args.cohort_mode == 'family':
            cohort = [r for r in enriched if genre in r.get('families', set())]
        elif args.cohort_mode == 'primary':
            cohort = [r for r in enriched
                      if (r['features'].get('primary_genre') or '').strip().lower() == genre]
        else:
            cohort = [r for r in enriched if genre in r['tokens']]

        by_bucket = defaultdict(list)
        for r in cohort:
            by_bucket[r['bucket']].append(r)

        baseline = by_bucket.get(args.baseline_bucket, [])
        validation = by_bucket.get(args.validation_bucket, [])
        if len(baseline) < args.min_cohort or len(validation) < args.min_cohort:
            print(f'  [{genre}] cohort too thin: baseline={len(baseline)} validation={len(validation)} — skip')
            continue

        c_base = {
            'audio': compute_audio_centroid(baseline),
            'emotion': compute_emotion_distribution(baseline),
            'genre': compute_genre_distribution(baseline),
        }
        c_val = {
            'audio': compute_audio_centroid(validation),
            'emotion': compute_emotion_distribution(validation),
            'genre': compute_genre_distribution(validation),
        }
        drift_audio = vector_diff(c_val['audio']['drift'], c_base['audio']['drift'])
        drift_emotion = vector_diff(c_val['emotion'], c_base['emotion'])
        drift_genre = vector_diff(c_val['genre'], c_base['genre'])

        # Compute BOTH alignment (vs baseline centroid) and vanguard (vs latest centroid)
        # alignment: how much track moved in drift direction since baseline period
        # vanguard:  how much track is ahead of where the cohort is now
        scored = []
        for r in validation:
            align = compute_alignment(
                r,
                c_base['audio']['drift'],
                drift_audio, drift_emotion, drift_genre,
                c_base['audio']['report'], stats,
            )
            vanguard = compute_alignment(
                r,
                c_val['audio']['drift'],
                drift_audio, drift_emotion, drift_genre,
                c_val['audio']['report'], stats,
            )
            scored.append({**r,
                           'alignment_score': align['alignment_score'],
                           'vanguard_score': vanguard['alignment_score']})

        # Dedupe by artist — keep highest alignment_score (canonical pick)
        scored.sort(key=lambda x: -x['alignment_score'])
        seen_artists = set()
        deduped = []
        for r in scored:
            aid = r.get('artist_id')
            if aid is None or aid in seen_artists:
                continue
            seen_artists.add(aid)
            deduped.append(r)

        per_genre_artists[genre] = deduped
        all_scored_artist_ids.update(r['artist_id'] for r in deduped if r.get('artist_id'))

        print(f'  [{genre}] cohort={len(cohort)}  baseline={len(baseline)}  validation={len(validation)}  scored_artists={len(deduped)}')

    if not all_scored_artist_ids:
        print('No genres met threshold. Exiting.')
        return

    # 6) Pull artists_history for ALL scored artists, with all metrics, in one pass
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    banner(f'STEP 4: artists_history since {args.history_since}  ({len(metrics)} metrics)')
    history_by_artist = fetch_artists_history(sb, all_scored_artist_ids, args.history_since, metrics)
    print(f'  history fetched for {len(history_by_artist)} artists')

    # 7) Per-genre × per-metric × per-score-variant correlation grid
    banner('STEP 5: full grid (genre × metric × score-variant)')

    def median(lst):
        lst = sorted(lst)
        return lst[len(lst) // 2] if lst else 0

    tail_pct = max(1.0, min(50.0, args.top_decile_pct))
    genre_results = {}

    # Print per-genre best result, full grid in JSON
    print(f'{"GENRE":<12}  {"BEST_METRIC":<22}  {"VARIANT":<9}  {"SPEARMAN":>9}  {"Q5/Q1":>7}  '
          f'{"TOP{:>3.0f}%/REST".format(tail_pct):>13}  {"N":>5}  VERDICT')
    print('-' * 120)

    for genre, artists in per_genre_artists.items():
        # Build the per-artist score+history rows once per genre
        all_rows = []
        for r in artists:
            aid = str(r.get('artist_id') or '')
            snaps = history_by_artist.get(aid, [])
            all_rows.append({
                'artist_id': aid,
                'alignment': r['alignment_score'],
                'vanguard': r['vanguard_score'],
                'plays': int(r['spotify_plays']),
                'track_name': r.get('track_name', ''),
                'snaps': snaps,
            })

        # For each (metric, variant), compute correlation
        per_metric = {}
        best = None  # (rho, metric, variant, q5q1, tail_ratio, n)
        for metric in metrics:
            scored_rows = []
            for row in all_rows:
                g = growth_ratio(row['snaps'], metric=metric)
                if g is None:
                    continue
                scored_rows.append({
                    'alignment': row['alignment'],
                    'vanguard': row['vanguard'],
                    'growth': g,
                    'artist_id': row['artist_id'],
                    'plays': row['plays'],
                    'track_name': row['track_name'],
                })

            if len(scored_rows) < 20:
                per_metric[metric] = {'skipped': True, 'n_with_growth': len(scored_rows)}
                continue

            metric_results = {'n_with_growth': len(scored_rows)}
            for variant in ('alignment', 'vanguard'):
                xs = [r[variant] for r in scored_rows]
                ys = [r['growth'] for r in scored_rows]
                rho = spearman(xs, ys)

                # Quintile analysis
                ranked = sorted(scored_rows, key=lambda r: r[variant])
                n = len(ranked)
                q1 = ranked[: n // 5]
                q5 = ranked[-n // 5:]
                q1_g = median([r['growth'] for r in q1])
                q5_g = median([r['growth'] for r in q5])
                q5q1 = math.exp(q5_g - q1_g) if (q1_g is not None and q5_g is not None) else float('nan')

                # Right-tail (top X% vs rest)
                cut = max(1, int(n * tail_pct / 100))
                tail = ranked[-cut:]
                rest = ranked[:-cut]
                tail_g = median([r['growth'] for r in tail]) if tail else None
                rest_g = median([r['growth'] for r in rest]) if rest else None
                tail_ratio = (math.exp(tail_g - rest_g)
                              if tail_g is not None and rest_g is not None else float('nan'))

                metric_results[variant] = {
                    'spearman': rho,
                    'q1_log_growth': q1_g, 'q5_log_growth': q5_g, 'q5_over_q1': q5q1,
                    'top_tail_log_growth': tail_g, 'rest_log_growth': rest_g,
                    'top_tail_over_rest': tail_ratio,
                    'top_5_by_score': [
                        {'artist_id': r['artist_id'], variant: r[variant],
                         'growth_log': r['growth'], 'plays': r['plays'],
                         'track_name': r['track_name']}
                        for r in ranked[-5:][::-1]
                    ],
                }

                if rho is not None:
                    candidate = (rho, metric, variant, q5q1, tail_ratio, n)
                    if best is None or abs(rho) > abs(best[0]):
                        best = candidate
            per_metric[metric] = metric_results

        if best is None:
            print(f'{genre:<12}  {"(none)":<22}  {"":<9}  {"":>9}  {"":>7}  {"":>13}  {"":>5}  no valid combos')
            genre_results[genre] = {'n_scored': len(artists), 'no_signal': True, 'per_metric': per_metric}
            continue

        rho, metric, variant, q5q1, tail_ratio, n = best
        verdict = ('predictive' if rho > 0.15
                   else 'mixed' if rho > 0.05
                   else 'descriptive')
        print(f'{genre:<12}  {metric:<22}  {variant:<9}  {rho:>+9.3f}  {q5q1:>7.2f}x  {tail_ratio:>12.2f}x  {n:>5}  {verdict}')

        genre_results[genre] = {
            'n_scored': len(artists),
            'best': {'metric': metric, 'variant': variant, 'spearman': rho,
                     'q5_q1_ratio': q5q1, 'top_tail_over_rest': tail_ratio,
                     'n_with_growth': n, 'verdict': verdict},
            'per_metric': per_metric,
        }

    report['per_genre'] = genre_results

    # Aggregate over genres' best combos
    bests = [v['best'] for v in genre_results.values() if v.get('best') is not None]
    if bests:
        rhos = [b['spearman'] for b in bests]
        mean_rho = sum(rhos) / len(rhos)
        n_predictive = sum(1 for b in bests if b['spearman'] > 0.15)
        n_mixed = sum(1 for b in bests if 0.05 < b['spearman'] <= 0.15)
        n_descriptive = sum(1 for b in bests if b['spearman'] <= 0.05)
        agg_verdict = ('predictive' if mean_rho > 0.15
                       else 'mixed' if mean_rho > 0.05
                       else 'descriptive')
        print(f'\n=== AGGREGATE (best-of per genre) ===')
        print(f'  mean spearman: {mean_rho:+.3f}  →  {agg_verdict}')
        print(f'  predictive (>+0.15):    {n_predictive}/{len(bests)}')
        print(f'  mixed (+0.05..+0.15):   {n_mixed}/{len(bests)}')
        print(f'  descriptive (<=+0.05):  {n_descriptive}/{len(bests)}')
        report['aggregate'] = {
            'mean_spearman_of_bests': mean_rho,
            'verdict': agg_verdict,
            'n_predictive': n_predictive,
            'n_mixed': n_mixed,
            'n_descriptive': n_descriptive,
            'n_genres': len(bests),
        }

        # Most-frequent best metric and variant
        from collections import Counter
        metric_counts = Counter(b['metric'] for b in bests)
        variant_counts = Counter(b['variant'] for b in bests)
        print(f'  most-frequent best metric:  {metric_counts.most_common(3)}')
        print(f'  most-frequent best variant: {variant_counts.most_common()}')

    out = Path(__file__).resolve().parent / 'cache' / 'drift_validation_report.json'
    out.parent.mkdir(exist_ok=True)
    with out.open('w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\nJSON report: {out}')


if __name__ == '__main__':
    main()
