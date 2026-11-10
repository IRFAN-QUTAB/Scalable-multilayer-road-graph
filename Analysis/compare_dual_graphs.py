
import argparse
import itertools
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dual_closures import norm_id

METRICS = ['degree', 'betweenness_norm', 'pagerank', 'harmonic_closeness']


def normalise_name(name):
    s = unicodedata.normalize('NFKD', str(name))
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^\w\s]", ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def main():
    p = argparse.ArgumentParser(description='Comparison of dual graphs built with different road definitions.')
    p.add_argument('--rels-csv', required=True)
    p.add_argument('--edge-keys', required=True)
    p.add_argument('--metrics', nargs='+', required=True,
                   help='key=folder pairs; key is a column of edge_keys.csv (name, name_split, stroke), '
                        'folder contains node_metrics.csv of dual_metrics.py')
    p.add_argument('--output', required=True)
    p.add_argument('--top', type=float, default=10.0, help='top share (%%) used for the overlap')
    o = p.parse_args()
    os.makedirs(o.output, exist_ok=True)

    rels = pd.read_csv(o.rels_csv, dtype=str, usecols=lambda c: c in ('source_id', 'target_id', 'osmway_id', 'length', 'status'),
                       low_memory=False)
    keys = pd.read_csv(o.edge_keys, dtype=str, keep_default_na=False, na_values=[''])
    for df in (rels, keys):
        for c in ('source_id', 'target_id', 'osmway_id'):
            df[c] = norm_id(df[c].fillna(''))
    variants = dict(m.split('=', 1) for m in o.metrics)
    join = ['source_id', 'target_id', 'osmway_id']
    cols = list(dict.fromkeys(['name'] + list(variants)))
    seg = rels.merge(keys[join + cols].drop_duplicates(join), on=join, how='left')
    if 'status' in seg.columns:
        seg = seg[seg['status'].fillna('active').str.strip().str.lower() == 'active']
    seg['length'] = pd.to_numeric(seg['length'], errors='coerce').fillna(0.0)
    a = seg['source_id'].astype(str)
    b = seg['target_id'].astype(str)
    seg['seg'] = np.where(a < b, a + '|' + b, b + '|' + a) + '|' + seg['osmway_id'].astype(str)
    seg = seg.drop_duplicates('seg').reset_index(drop=True)
    total = seg['length'].sum()

    size_rows = []
    for key, folder in variants.items():
        nm = pd.read_csv(os.path.join(folder, 'node_metrics.csv'), dtype={'road': str})
        L = seg.groupby(key)['length'].sum().sort_values(ascending=False)
        top1 = L.head(max(1, int(round(len(L) * 0.01)))).sum()
        span = pd.to_numeric(nm['span_m'], errors='coerce') if 'span_m' in nm.columns else pd.Series(dtype=float)
        size_rows.append({'dual_graph': key, 'roads': len(nm), 'covered_length_km': L.sum() / 1000,
                          'covered_length_pct': 100 * L.sum() / total,
                          'length_mean_km': L.mean() / 1000, 'length_median_km': L.median() / 1000,
                          'length_p90_km': L.quantile(0.9) / 1000, 'length_max_km': L.max() / 1000,
                          'longest_road': L.index[0], 'top1pct_roads_length_share': 100 * top1 / L.sum(),
                          'span_median_km': span.median() / 1000 if len(span) else None,
                          'span_max_km': span.max() / 1000 if len(span) else None})
        for m in METRICS:
            seg[f'{key}:{m}'] = seg[key].map(dict(zip(nm['road'], nm[m])))
    pd.DataFrame(size_rows).to_csv(os.path.join(o.output, 'road_size.csv'), index=False)

    corr_rows = []
    for k1, k2 in itertools.combinations(variants, 2):
        for m in METRICS:
            s = seg[[f'{k1}:{m}', f'{k2}:{m}', 'length']].dropna()
            if len(s) < 3:
                continue
            rho = spearmanr(s.iloc[:, 0], s.iloc[:, 1]).statistic
            def top_set(col):
                order = s.sort_values(col, ascending=False, kind='mergesort')
                cum = order['length'].cumsum()
                return set(order.index[cum <= cum.iloc[-1] * o.top / 100])
            t1, t2 = top_set(f'{k1}:{m}'), top_set(f'{k2}:{m}')
            L = s['length']
            overlap = L[list(t1 & t2)].sum() / L[list(t1)].sum() if t1 else None
            corr_rows.append({'dual_graph_1': k1, 'dual_graph_2': k2, 'metric': m, 'segments': len(s),
                              'length_km': L.sum() / 1000, 'spearman': rho, f'top{o.top:g}pct_overlap': overlap})
    pd.DataFrame(corr_rows).to_csv(os.path.join(o.output, 'centrality_agreement.csv'), index=False)

    keys = ['nodes', 'directed_edges', 'undirected_edges', 'reciprocity', 'wcc_count', 'wcc_largest', 'scc_count',
            'scc_largest', 'degree_mean', 'degree_max', 'degree_max_road', 'P_k1', 'P_k2_4', 'betweenness_norm_max',
            'betweenness_norm_max_road', 'pagerank_max', 'pagerank_max_road', 'harmonic_max', 'harmonic_max_road',
            'harmonic_mean', 'clustering_mean', 'adapted_clustering_mean', 'avg_path_length', 'diameter',
            'reachable_share', 'assortativity_r', 'modularity_best', 'communities_best', 'conductance_mean',
            'powerlaw_alpha', 'powerlaw_alpha_sigma', 'powerlaw_xmin', 'powerlaw_n_tail', 'powerlaw_tail_share',
            'powerlaw_KS_D', 'powerlaw_gof_p_value', 'powerlaw_LR_vs_lognormal', 'powerlaw_p_vs_lognormal',
            'powerlaw_LR_vs_exponential', 'powerlaw_p_vs_exponential', 'powerlaw_LR_vs_truncated_power_law',
            'powerlaw_p_vs_truncated_power_law', 'powerlaw_LR_vs_stretched_exponential',
            'powerlaw_p_vs_stretched_exponential']
    table = {}
    for key, folder in variants.items():
        path = os.path.join(folder, 'summary.json')
        if os.path.exists(path):
            with open(path) as f:
                sm = json.load(f)
            table[key] = {k: sm.get(k) for k in keys}
    if table:
        pd.DataFrame(table).rename_axis('metric').to_csv(os.path.join(o.output, 'metrics_summary.csv'))

    if 'name' in seg.columns:
        nl = seg.dropna(subset=['name']).groupby('name')['length'].sum().reset_index()
        nl['normalised'] = nl['name'].map(normalise_name)
        groups = nl.groupby('normalised').filter(lambda g: len(g) > 1).sort_values(['normalised', 'length'], ascending=[True, False])
        groups.to_csv(os.path.join(o.output, 'name_variants.csv'), index=False)
        named = nl['length'].sum()
        var = {'names': len(nl), 'normalised_names': nl['normalised'].nunique(),
               'names_in_variant_groups': len(groups), 'variant_groups': groups['normalised'].nunique(),
               'length_in_variant_groups_pct': 100 * groups['length'].sum() / named if named else None,
               'unnamed_length_pct': 100 * seg.loc[seg['name'].isna(), 'length'].sum() / total}
        pd.Series(var).to_csv(os.path.join(o.output, 'name_variants_summary.csv'), header=False)
        print("Name variants:", var)
    print(pd.DataFrame(size_rows).to_string(index=False))
    print(pd.DataFrame(corr_rows).to_string(index=False))


if __name__ == '__main__':
    main()
