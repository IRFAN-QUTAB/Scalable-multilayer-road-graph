"""
Structural analysis of a dual graph (Section "Metrics" of the paper), computed on the CSV files written by
dual_graph.py (dual_nodes.csv, dual_edges.csv). The same script is used for every city and for
every dual graph (name-based, name_split, strokes), so that all the results are computed in the same way.

Metrics (formulas of the paper):
  degree            number of distinct neighbouring roads (directions ignored), in- and out-degree
  betweenness       Brandes, directed shortest paths (hops); raw value and value / ((N-1)(N-2))
  pagerank          PR(v) = (1-d) + d * sum_{u->v} PR(u)/L(u), d = 0.85, iterated until convergence
  harmonic          C_H(v) = 1/(N-1) * sum_{u != v} 1/d(v,u), directed distances (0 if no path)
  clustering        C(i) = 2 E_i / (k_i (k_i - 1)) on the undirected graph (0 if k_i < 2)
  adapted_cc        C_a(n) = (directed links between the neighbours of n) / (k_n (k_n - 1))
  components        weakly and strongly connected components
  path length       exact mean of the directed shortest-path lengths (hops) over all reachable ordered pairs
  diameter          exact largest directed shortest-path length
  assortativity     Newman r on the undirected graph (each edge once), and k_nn(k)
  communities       Louvain on the undirected graph, several seeds (modularity Q of the paper)
  conductance       per community: external / total directed edges (communities with more than 3 roads)
  power law         Clauset, Shalizi and Newman (2009): alpha, xmin, tail size, KS distance, bootstrap
                    goodness-of-fit p-value, likelihood-ratio tests against lognormal, exponential and
                    truncated power law

Usage:
  python3 dual_metrics.py --dual-dir rome_dual/name_split --label "Rome name_split" --output analysis/rome_name_split
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

import igraph as ig
import numpy as np
import pandas as pd


def clean_columns(df):
    """'id:ID(RoadOsm)' -> 'id', ':START_ID(RoadOsm)' -> 'source', ':END_ID(RoadOsm)' -> 'target', 'x:DOUBLE' -> 'x'"""
    cols = {}
    for c in df.columns:
        if c.startswith(':START_ID'):
            cols[c] = 'source'
        elif c.startswith(':END_ID'):
            cols[c] = 'target'
        elif c == ':TYPE':
            cols[c] = 'type'
        else:
            cols[c] = c.split(':')[0]
    return df.rename(columns=cols)


def load_dual(dual_dir):
    nodes = clean_columns(pd.read_csv(os.path.join(dual_dir, 'dual_nodes.csv'), dtype={0: str}, keep_default_na=False,
                                      na_values=['']))
    edges = clean_columns(pd.read_csv(os.path.join(dual_dir, 'dual_edges.csv'), dtype={0: str, 1: str},
                                      keep_default_na=False, na_values=['']))
    nodes['id'] = nodes['id'].astype(str)
    edges['source'] = edges['source'].astype(str)
    edges['target'] = edges['target'].astype(str)
    return nodes, edges


def build_graphs(nodes, edges):
    names = nodes['id'].tolist()
    index = {n: i for i, n in enumerate(names)}
    pairs = []
    unknown = self_loops = 0
    for s, t in zip(edges['source'], edges['target']):
        if s not in index or t not in index:
            unknown += 1
        elif s == t:
            self_loops += 1
        else:
            pairs.append((index[s], index[t]))
    n_raw = len(pairs)
    pairs = sorted(set(pairs))
    G = ig.Graph(n=len(names), edges=pairs, directed=True)
    G.vs['name'] = names
    U = G.as_undirected(mode='collapse')
    info = {'edges_in_file': int(len(edges)), 'edges_unknown_node': unknown, 'self_loops_removed': self_loops,
            'duplicate_edges_removed': n_raw - len(pairs)}
    return G, U, info


def pagerank_paper(G, d=0.85, tol=1e-12, max_iter=10000):
    """PR(v) = (1-d) + d * sum_{u in B(v)} PR(u)/L(u)  (formula of the paper, scores not normalised to 1)"""
    n = G.vcount()
    src = np.array([e.source for e in G.es], dtype=np.int64)
    dst = np.array([e.target for e in G.es], dtype=np.int64)
    out_deg = np.bincount(src, minlength=n).astype(float)
    pr = np.ones(n)
    for it in range(max_iter):
        contrib = np.zeros(n)
        np.add.at(contrib, dst, pr[src] / out_deg[src])
        new = (1 - d) + d * contrib
        delta = np.abs(new - pr).max()
        pr = new
        if delta < tol:
            return pr, it + 1
    return pr, max_iter


def adapted_clustering(G, U):
    """C_a(n) = number of directed links among the (undirected) neighbours of n / (k_n (k_n - 1))"""
    out_nb = [set(G.successors(v)) for v in range(G.vcount())]
    res = np.zeros(G.vcount())
    for v in range(G.vcount()):
        nb = U.neighbors(v)
        k = len(nb)
        if k < 2:
            continue
        nbs = set(nb)
        links = sum(len(out_nb[b] & nbs) - (1 if b in out_nb[b] else 0) for b in nb)
        res[v] = links / (k * (k - 1))
    return res


def path_length_stats(G):
    """exact mean and maximum of the directed shortest-path lengths over all reachable ordered pairs"""
    h = G.path_length_hist(directed=True)
    total = count = 0
    diameter = 0
    for start, end, c in h.bins():
        total += start * c
        count += c
        if c > 0:
            diameter = max(diameter, int(start))
    n = G.vcount()
    return {'avg_path_length': total / count if count else None, 'diameter': diameter,
            'reachable_pairs': int(count), 'unreachable_pairs': int(h.unconnected),
            'reachable_share': count / (n * (n - 1)) if n > 1 else None}


def newman_r(U, degree):
    """Newman (2002) assortativity, each undirected edge counted once (formula of the paper)"""
    j = np.array([degree[e.source] for e in U.es], dtype=float)
    k = np.array([degree[e.target] for e in U.es], dtype=float)
    a = np.mean(j * k)
    b = np.mean(0.5 * (j + k)) ** 2
    c = np.mean(0.5 * (j ** 2 + k ** 2))
    return (a - b) / (c - b) if c != b else None


def louvain_runs(U, seeds):
    runs = []
    best = None
    for seed in seeds:
        random.seed(seed)
        ig.set_random_number_generator(random)
        part = U.community_multilevel()
        q = U.modularity(part.membership)
        runs.append({'seed': seed, 'modularity': q, 'communities': len(set(part.membership))})
        if best is None or q > best[0]:
            best = (q, part.membership, seed)
    return pd.DataFrame(runs), best


def conductance(G, membership, min_size=4):
    comm = np.array(membership)
    sizes = pd.Series(comm).value_counts()
    total, internal = defaultdict(int), defaultdict(int)
    for e in G.es:
        cs, ct = comm[e.source], comm[e.target]
        total[cs] += 1
        if cs == ct:
            internal[cs] += 1
    rows = []
    for c, tot in total.items():
        if sizes.get(c, 0) < min_size:
            continue
        rows.append({'community': int(c), 'size': int(sizes[c]), 'internal_edges': internal[c],
                     'external_edges': tot - internal[c], 'total_edges': tot,
                     'conductance': (tot - internal[c]) / tot if tot else 0.0})
    return pd.DataFrame(rows).sort_values('size', ascending=False) if rows else pd.DataFrame()


def power_law_analysis(degrees, n_boot, seed=42):
    """Clauset et al. (2009): MLE alpha and xmin (KS minimisation), tail size, KS distance, bootstrap p-value,
    likelihood-ratio tests against alternative heavy-tailed distributions."""
    import powerlaw
    data = np.asarray([d for d in degrees if d > 0], dtype=float)
    fit = powerlaw.Fit(data, discrete=True, verbose=False)
    alpha, xmin, D = float(fit.power_law.alpha), float(fit.power_law.xmin), float(fit.power_law.D)
    n_tail = int(np.sum(data >= xmin))
    res = {'n': int(len(data)), 'alpha': alpha, 'alpha_sigma': float(fit.power_law.sigma), 'xmin': xmin,
           'n_tail': n_tail, 'tail_share': n_tail / len(data), 'KS_D': D}
    for alt in ('lognormal', 'exponential', 'truncated_power_law', 'stretched_exponential'):
        try:
            R, p = fit.distribution_compare('power_law', alt, normalized_ratio=True,
                                            nested=(alt == 'truncated_power_law'))
            res[f'LR_vs_{alt}'] = float(R)
            res[f'p_vs_{alt}'] = float(p)
        except Exception as e:
            res[f'LR_vs_{alt}'] = None
            res[f'p_vs_{alt}'] = None
            print(f"   likelihood ratio vs {alt} failed: {e}")
    try:
        res['lognormal_mu'] = float(fit.lognormal.mu)
        res['lognormal_sigma'] = float(fit.lognormal.sigma)
        res['truncated_alpha'] = float(fit.truncated_power_law.alpha)
        res['truncated_lambda'] = float(fit.truncated_power_law.Lambda)
        res['exponential_lambda'] = float(fit.exponential.Lambda)
    except Exception:
        pass
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        body = data[data < xmin]
        n = len(data)
        ge = 0
        for b in range(n_boot):
            n_t = rng.binomial(n, n_tail / n)
            tail = fit.power_law.generate_random(n_t) if n_t > 0 else np.array([])
            synth = np.concatenate([rng.choice(body, n - n_t, replace=True) if len(body) and n - n_t > 0 else np.array([]),
                                    np.asarray(tail, dtype=float)])
            f2 = powerlaw.Fit(synth, discrete=True, verbose=False)
            if f2.power_law.D >= D:
                ge += 1
        res['bootstrap_runs'] = n_boot
        res['gof_p_value'] = ge / n_boot
    return res, fit


def versions():
    import sys
    import scipy
    v = {'python': sys.version.split()[0], 'igraph': ig.__version__, 'numpy': np.__version__,
         'pandas': pd.__version__, 'scipy': scipy.__version__}
    try:
        import powerlaw
        v['powerlaw'] = getattr(powerlaw, '__version__', 'unknown')
    except ImportError:
        pass
    return v


def main():
    p = argparse.ArgumentParser(description='Structural metrics of a dual graph (CSV files of dual_graph.py).')
    p.add_argument('--dual-dir', required=True, help='folder with dual_nodes.csv and dual_edges.csv')
    p.add_argument('--output', required=True)
    p.add_argument('--label', default='')
    p.add_argument('--louvain-seeds', type=int, default=10)
    p.add_argument('--bootstrap', type=int, default=1000, help='bootstrap runs for the power-law p-value (0 = none)')
    o = p.parse_args()
    os.makedirs(o.output, exist_ok=True)
    t0 = time.time()
    times = {}

    def lap(name, t):
        times[name] = round(time.time() - t, 2)
        print(f"   {name}: {times[name]} s")

    print(f"=== {o.label or o.dual_dir} ===")
    t = time.time()
    nodes, edges = load_dual(o.dual_dir)
    G, U, info = build_graphs(nodes, edges)
    N, M = G.vcount(), G.ecount()
    lap('loading', t)
    print(f"   nodes {N}, directed edges {M}, undirected edges {U.ecount()}")

    t = time.time()
    degree = np.array(U.degree())
    df = pd.DataFrame({'road': G.vs['name'], 'degree': degree,
                       'in_degree': G.indegree(), 'out_degree': G.outdegree()})
    lap('degree', t)

    t = time.time()
    bc = np.array(G.betweenness(directed=True))
    df['betweenness'] = bc
    df['betweenness_norm'] = bc / ((N - 1) * (N - 2)) if N > 2 else 0.0
    lap('betweenness', t)

    t = time.time()
    pr, iters = pagerank_paper(G)
    df['pagerank'] = pr
    lap('pagerank', t)

    t = time.time()
    df['harmonic_closeness'] = G.harmonic_centrality(mode='out', normalized=True)
    lap('harmonic closeness', t)

    t = time.time()
    df['clustering'] = U.transitivity_local_undirected(mode='zero')
    df['adapted_clustering'] = adapted_clustering(G, U)
    lap('clustering', t)

    t = time.time()
    wcc = G.connected_components(mode='weak')
    scc = G.connected_components(mode='strong')
    lap('components', t)

    t = time.time()
    paths = path_length_stats(G)
    lap('path length and diameter', t)

    t = time.time()
    r = newman_r(U, degree)
    r_igraph = U.assortativity_degree(directed=False)
    knn_node = np.array([np.mean(degree[U.neighbors(v)]) if degree[v] > 0 else np.nan for v in range(N)])
    df['knn'] = knn_node
    lap('assortativity', t)

    t = time.time()
    runs, best = louvain_runs(U, list(range(1, o.louvain_seeds + 1)))
    df['community'] = best[1]
    cond = conductance(G, best[1])
    lap('louvain', t)

    t = time.time()
    try:
        pl, _ = power_law_analysis(degree, o.bootstrap)
    except ImportError:
        pl = {'error': 'pip install powerlaw'}
    except Exception as e:
        pl = {'error': f'power-law fit not possible: {e}'}
    lap('power law', t)

    extra = [c for c in ('name', 'distance', 'segment_count', 'junction_count', 'lane_km', 'lanes_known', 'span_m',
                         'pieces', 'max_gap_m') if c in nodes.columns]
    df = df.merge(nodes[['id'] + extra].rename(columns={'id': 'road'}), on='road', how='left')
    for c in ('degree', 'betweenness', 'pagerank', 'harmonic_closeness'):
        df[f'{c}_rank'] = df[c].rank(ascending=False, method='min').astype(int)

    pk = df['degree'].value_counts().sort_index().rename_axis('degree').reset_index(name='count')
    pk['probability'] = pk['count'] / N
    cc_spec = df.groupby('degree').agg(avg_clustering=('clustering', 'mean'),
                                       avg_adapted_clustering=('adapted_clustering', 'mean'),
                                       node_count=('road', 'count')).reset_index()
    knn_spec = df[df['degree'] > 0].groupby('degree').agg(knn=('knn', 'mean'), node_count=('road', 'count')).reset_index()

    summary = {
        'label': o.label, 'dual_dir': o.dual_dir,
        'nodes': N, 'directed_edges': M, 'undirected_edges': U.ecount(),
        'reciprocity': G.reciprocity(ignore_loops=True, mode='default'),
        'density_undirected': 2 * U.ecount() / (N * (N - 1)) if N > 1 else None,
        'wcc_count': len(wcc), 'wcc_largest': max(wcc.sizes()),
        'scc_count': len(scc), 'scc_largest': max(scc.sizes()),
        'degree_mean': float(degree.mean()), 'degree_median': float(np.median(degree)),
        'degree_max': int(degree.max()), 'degree_max_road': df.loc[df['degree'].idxmax(), 'road'],
        'degree_std': float(degree.std(ddof=1)),
        'P_k1': float(np.mean(degree == 1)), 'P_k2_4': float(np.mean((degree >= 2) & (degree <= 4))),
        'betweenness_norm_max': float(df['betweenness_norm'].max()),
        'betweenness_norm_max_road': df.loc[df['betweenness_norm'].idxmax(), 'road'],
        'betweenness_norm_mean': float(df['betweenness_norm'].mean()),
        'pagerank_max': float(pr.max()), 'pagerank_max_road': df.loc[df['pagerank'].idxmax(), 'road'],
        'pagerank_mean': float(pr.mean()), 'pagerank_iterations': iters,
        'harmonic_max': float(df['harmonic_closeness'].max()),
        'harmonic_max_road': df.loc[df['harmonic_closeness'].idxmax(), 'road'],
        'harmonic_mean': float(df['harmonic_closeness'].mean()),
        'clustering_mean': float(df['clustering'].mean()),
        'adapted_clustering_mean': float(df['adapted_clustering'].mean()),
        'global_transitivity': U.transitivity_undirected(),
        **paths,
        'assortativity_r': r, 'assortativity_r_igraph_check': r_igraph,
        'louvain_seeds': len(runs), 'modularity_best': best[0], 'modularity_best_seed': best[2],
        'modularity_mean': float(runs['modularity'].mean()), 'modularity_std': float(runs['modularity'].std(ddof=1)) if len(runs) > 1 else 0.0,
        'communities_best': len(set(best[1])),
        'communities_mean': float(runs['communities'].mean()),
        'communities_min': int(runs['communities'].min()), 'communities_max': int(runs['communities'].max()),
        'conductance_mean': float(cond['conductance'].mean()) if len(cond) else None,
        'conductance_min': float(cond['conductance'].min()) if len(cond) else None,
        'conductance_max': float(cond['conductance'].max()) if len(cond) else None,
        **{f'powerlaw_{k}': v for k, v in pl.items()},
        **{f'input_{k}': v for k, v in info.items()},
        'times_s': times, 'total_time_s': round(time.time() - t0, 2),
        'versions': versions(),
    }

    df.to_csv(os.path.join(o.output, 'node_metrics.csv'), index=False)
    pk.to_csv(os.path.join(o.output, 'degree_distribution.csv'), index=False)
    cc_spec.to_csv(os.path.join(o.output, 'clustering_spectrum.csv'), index=False)
    knn_spec.to_csv(os.path.join(o.output, 'assortativity_spectrum.csv'), index=False)
    runs.to_csv(os.path.join(o.output, 'louvain_runs.csv'), index=False)
    cond.to_csv(os.path.join(o.output, 'conductance.csv'), index=False)
    with open(os.path.join(o.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    top = []
    for c in ('degree', 'betweenness_norm', 'pagerank', 'harmonic_closeness'):
        for rank, row in enumerate(df.nlargest(10, c).itertuples(), 1):
            top.append({'metric': c, 'rank': rank, 'road': row.road, 'value': getattr(row, c)})
    pd.DataFrame(top).to_csv(os.path.join(o.output, 'top10.csv'), index=False)

    with open(os.path.join(o.output, 'summary.txt'), 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print("\n".join(f"   {k}: {v}" for k, v in summary.items() if k != 'times_s'))
    print(f"Results in {o.output} ({summary['total_time_s']} s)")


if __name__ == '__main__':
    main()
