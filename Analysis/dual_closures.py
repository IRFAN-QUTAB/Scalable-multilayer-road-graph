"""
Road-closure scenarios on a dual graph and their effect on the dual graph and on routing in the primal graph.

Scenarios
  RCS1 / RCS2 (paper)  close the top / bottom p% of the roads ranked by PageRank of the full dual graph
                       (p = 2, 5, 10, 20, 30, 50; number of roads = int(N p / 100)).
  length budgets       close roads until a fixed share of the total length of the primal network is removed
                       (1, 2, 5, 10, 20%), choosing the roads by: top PageRank, top betweenness, top degree,
                       bottom PageRank, and in random order (several repetitions). The removed length is then the
                       same for every criterion, so the effect of the selection can be separated from the amount
                       of road removed. The same budgets are used for every dual graph
                       (name-based, name_split, strokes), so the granularities can be compared.
  section closures     for the top --section-roads roads by PageRank: close only a short stretch of the road
                       (--section-length, default 1 km: the segments of the road whose middle point is within half
                       this length from the centre of the stretch, so both carriageways are closed) and, for
                       comparison, the whole road. The centre of the stretch is the middle point of the segment of
                       the road nearest to the length-weighted centre of the road. The effect is measured on the
                       origin-destination pairs of the whole area and on --local-pairs pairs of junctions within
                       --local-radius of the stretch (routes nearby): share of routes that used the closed
                       segments, share without a path, detour in length and in time, extra km and minutes of the
                       affected routes.

For every scenario
  size of the closure  roads closed, length removed (km and % of the network), lane-km removed (segments with a
                       lanes tag), road segments, junctions touched, extent (diagonal of the bounding box of the
                       closed segments) and mean extent of a closed road
  dual graph           metrics of the paper on the remaining roads: average path length and diameter (directed,
                       exact), weakly connected components and share of the largest, share of the largest strongly
                       connected component, average clustering and adapted clustering, Louvain modularity and
                       communities, average harmonic closeness, average betweenness / ((N-1)(N-2))
                       (betweenness only for RCS1/RCS2 unless --budget-betweenness)
  primal graph         driveable edges (status active, driveable True) without the closed roads:
                       share of the driveable junctions in the largest strongly connected component,
                       origin-destination pairs: share without a path, share whose original route used a closed
                       road, detour ratio (new length / original length) with shortest paths by length, and
                       detour ratio in time (new travel time / original travel time) with fastest paths
  travel time          length / speed; speed from the OSM tag maxspeed (numbers, "30 mph", "IT:urban" 50,
                       "IT:rural" 90, "IT:trunk" 110, "IT:motorway" 130; a list gives the mean); where maxspeed is
                       missing, the mean of the known speeds of the same highway type in the graph, and if the type
                       has no known speed, the mean of all known speeds (the method of OSMnx add_edge_speeds)

Also written (for the figures of the paper)
  closure_clustering_spectra.csv  average clustering and adapted clustering per degree, without closures and after
                       every RCS1 / RCS2 closure
  closed_roads.csv     the roads closed in every RCS1 / RCS2 scenario
  example_routes.csv   the junctions of the shortest route (by length) of --example-pairs origin-destination pairs,
                       one pair from each distance decile, without closures and after the RCS1 closures of
                       --example-levels (for the map of the routes)

Inputs: primal CSV files of the cell-based pipeline, edge_keys.csv and one dual graph folder of dual_graph.py.

Usage:
  python3 dual_closures.py --nodes-csv .../fullgraph_nodes.csv --rels-csv .../fullgraph_rels.csv \
      --edge-keys .../rome_dual/edge_keys.csv --dual-dir .../rome_dual/name_split --key name_split \
      --label "Rome name_split" --output .../analysis/rome_name_split_closures
"""
import argparse
import json
import math
import os
import time

import igraph as ig
import numpy as np
import pandas as pd

from dual_metrics import (load_dual, build_graphs, pagerank_paper, adapted_clustering, path_length_stats,
                          louvain_runs)


def norm_id(series):
    """ids as text without a trailing '.0' (the same id can be read as 1 or 1.0)"""
    s = series.astype(str).str.strip()
    return s.where(~s.str.endswith('.0'), s.str[:-2])


def is_true(series):
    return series.astype(str).str.strip().str.lower().isin(('true', '1', 'yes'))


IMPLICIT_SPEEDS = {'it:urban': 50.0, 'it:rural': 90.0, 'it:trunk': 110.0, 'it:motorway': 130.0}


def parse_maxspeed(value):
    """km/h from an OSM maxspeed value, or NaN (none, signals, variable, walk, missing)"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in ('', 'nan', 'none', 'signals', 'variable', 'walk'):
        return np.nan
    parts = [x.strip(" '\"") for x in text.replace('[', '').replace(']', '').replace('|', ';').replace(',', ';').split(';')]
    speeds = []
    for part in parts:
        if part in IMPLICIT_SPEEDS:
            speeds.append(IMPLICIT_SPEEDS[part])
            continue
        mph = 'mph' in part
        number = part.replace('mph', '').replace('km/h', '').replace('kmh', '').strip()
        try:
            v = float(number)
        except ValueError:
            continue
        if v > 0:
            speeds.append(v * 1.609344 if mph else v)
    return float(np.mean(speeds)) if speeds else np.nan


def first_highway(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'unknown'
    text = str(value).replace('[', '').replace(']', '').replace("'", '').replace('"', '')
    return text.replace(';', ',').split(',')[0].strip() or 'unknown'


def first_number(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    for token in str(value).replace('[', ' ').replace(']', ' ').replace(';', ' ').replace(',', ' ').split():
        try:
            return float(token)
        except ValueError:
            continue
    return np.nan


class Primal:
    def __init__(self, nodes_csv, rels_csv, edge_keys_csv, key):
        t = time.time()
        nodes = pd.read_csv(nodes_csv, usecols=['osmnode_id', 'x', 'y'], dtype={'osmnode_id': str}, low_memory=False)
        rels = pd.read_csv(rels_csv, dtype=str, low_memory=False)
        for c in ('length', 'driveable', 'status', 'lanes', 'osmway_id', 'maxspeed', 'highway'):
            if c not in rels.columns:
                rels[c] = np.nan
        rels = rels[['source_id', 'target_id', 'osmway_id', 'length', 'driveable', 'status', 'lanes',
                     'maxspeed', 'highway']].copy()
        keys = pd.read_csv(edge_keys_csv, dtype=str, keep_default_na=False, na_values=[''])
        for df in (rels, keys):
            df['source_id'] = norm_id(df['source_id'])
            df['target_id'] = norm_id(df['target_id'])
            df['osmway_id'] = norm_id(df['osmway_id'].fillna(''))
        nodes['osmnode_id'] = norm_id(nodes['osmnode_id'])

        join = ['source_id', 'target_id', 'osmway_id']
        k = keys[join + [key]].drop_duplicates(join)
        rels = rels.merge(k, on=join, how='left', validate='many_to_one')
        rels['key'] = rels[key]
        rels['length'] = pd.to_numeric(rels['length'], errors='coerce').fillna(0.0)
        rels['active'] = rels['status'].fillna('active').astype(str).str.strip().str.lower().eq('active')
        rels['drive'] = is_true(rels['driveable']) & rels['active']
        rels['lanes_n'] = rels['lanes'].map(first_number)

        index = {n: i for i, n in enumerate(nodes['osmnode_id'])}
        rels['u'] = rels['source_id'].map(index)
        rels['v'] = rels['target_id'].map(index)
        missing = rels['u'].isna() | rels['v'].isna()
        if missing.any():
            print(f"   {int(missing.sum())} primal edges with a node not in the nodes file are ignored")
            rels = rels[~missing].copy()
        rels['u'] = rels['u'].astype(np.int64)
        rels['v'] = rels['v'].astype(np.int64)

        rels['seg'] = list(zip(np.minimum(rels['u'], rels['v']), np.maximum(rels['u'], rels['v']), rels['osmway_id']))
        self.seg = rels[rels['active']].drop_duplicates('seg')[['seg', 'u', 'v', 'key', 'length', 'lanes_n']].copy()
        self.total_length_m = float(self.seg['length'].sum())
        lat0 = float(nodes['y'].mean())
        self.mx = nodes['x'].to_numpy(float) * 111320.0 * math.cos(math.radians(lat0))
        self.my = nodes['y'].to_numpy(float) * 111320.0

        d = rels[rels['drive']].reset_index(drop=True)
        d['speed_tag'] = d['maxspeed'].map(parse_maxspeed)
        d['hw'] = d['highway'].map(first_highway)
        type_mean = d.groupby('hw')['speed_tag'].mean()
        all_mean = float(d['speed_tag'].mean()) if d['speed_tag'].notna().any() else 50.0
        d['speed'] = d['speed_tag'].fillna(d['hw'].map(type_mean)).fillna(all_mean)
        d['tt'] = d['length'] / (d['speed'] / 3.6)
        dseg = d.drop_duplicates('seg')
        tagged = float(dseg.loc[dseg['speed_tag'].notna(), 'length'].sum()) / max(float(dseg['length'].sum()), 1e-9)
        self.speed_info = {'driveable_length_with_maxspeed_pct': round(100 * tagged, 1),
                           'mean_known_speed_kmh': round(all_mean, 1),
                           'speed_by_highway_kmh': {k: round(float(v), 1) for k, v in type_mean.dropna().items()},
                           'highway_types_without_known_speed': sorted(set(d['hw']) - set(type_mean.dropna().index))}
        print(f"   speeds: {self.speed_info['driveable_length_with_maxspeed_pct']}% of the driveable length has "
              f"maxspeed; mean known speed {all_mean:.1f} km/h")
        self.drive_key = d['key'].to_numpy(object)
        self.drive_seg = list(d['seg'])
        self.node_ids = nodes['osmnode_id'].to_numpy(object)
        self.G = ig.Graph(n=len(nodes), edges=list(zip(d['u'], d['v'])), directed=True)
        self.G.es['w'] = d['length'].to_numpy(float)
        self.G.es['tt'] = d['tt'].to_numpy(float)
        scc = self.G.connected_components(mode='strong')
        self.giant = np.array(scc[int(np.argmax(scc.sizes()))])
        self.n_drive_nodes = int(np.sum(np.array(self.G.degree()) > 0))
        self.keys_missing = int(rels['key'].isna().sum())
        print(f"   primal: {len(nodes)} junctions, {len(rels)} edges, {len(self.seg)} segments, "
              f"{self.total_length_m / 1000:.1f} km; driveable edges {len(d)}, largest SCC {len(self.giant)} junctions "
              f"({time.time() - t:.1f} s)")

    def od_pairs(self, n_origins, per_origin, seed):
        rng = np.random.default_rng(seed)
        origins = rng.choice(self.giant, size=min(n_origins, len(self.giant)), replace=False)
        pairs = []
        for o in origins:
            dests = rng.choice(self.giant[self.giant != o], size=per_origin, replace=False)
            pairs.extend((int(o), int(t)) for t in dests)
        return pairs

    def routes(self, pairs, closed_keys=None, weight='w', closed_segs=None):
        """shortest path of every pair (inf = no path) on the driveable graph without the closed roads (closed_keys)
        or without the closed road segments (closed_segs); weight 'w' = length (m), 'tt' = travel time (s)"""
        if closed_keys:
            keep = [i for i, k in enumerate(self.drive_key) if not (isinstance(k, str) and k in closed_keys)]
            H = self.G.subgraph_edges(keep, delete_vertices=False)
        elif closed_segs:
            keep = [i for i, sg in enumerate(self.drive_seg) if sg not in closed_segs]
            H = self.G.subgraph_edges(keep, delete_vertices=False)
        else:
            H = self.G
        by_origin = {}
        for o, t in pairs:
            by_origin.setdefault(o, []).append(t)
        out = {}
        for o, ts in by_origin.items():
            dist = H.distances(source=[o], target=ts, weights=weight, mode='out')[0]
            for t, dd in zip(ts, dist):
                out[(o, t)] = dd
        scc = H.connected_components(mode='strong')
        return out, max(scc.sizes())

    def base_route_keys(self, pairs, weight='w'):
        """road keys used by the original shortest (w) or fastest (tt) route of every pair"""
        by_origin = {}
        for o, t in pairs:
            by_origin.setdefault(o, []).append(t)
        used = {}
        for o, ts in by_origin.items():
            paths = self.G.get_shortest_paths(o, to=ts, weights=weight, mode='out', output='epath')
            for t, p in zip(ts, paths):
                used[(o, t)] = {self.drive_key[e] for e in p if isinstance(self.drive_key[e], str)}
        return used

    def route_junctions(self, pairs, closed_keys=None, weight='w'):
        """osm ids of the junctions of the shortest route of every pair (empty list = no path)"""
        if closed_keys:
            keep = [i for i, k in enumerate(self.drive_key) if not (isinstance(k, str) and k in closed_keys)]
            H = self.G.subgraph_edges(keep, delete_vertices=False)
        else:
            H = self.G
        out = {}
        for o, t in pairs:
            path = H.get_shortest_paths(o, to=[t], weights=weight, mode='out', output='vpath')[0]
            out[(o, t)] = [str(self.node_ids[v]) for v in path] if (path and path[-1] == t) else []
        return out

    def base_route_segs(self, pairs, weight='w'):
        """road segments used by the original shortest (w) or fastest (tt) route of every pair"""
        by_origin = {}
        for o, t in pairs:
            by_origin.setdefault(o, []).append(t)
        used = {}
        for o, ts in by_origin.items():
            paths = self.G.get_shortest_paths(o, to=ts, weights=weight, mode='out', output='epath')
            for t, p in zip(ts, paths):
                used[(o, t)] = {self.drive_seg[e] for e in p}
        return used

    def road_section(self, key, length_m):
        """short stretch of a road: the segments of the road whose middle point is within length_m / 2 of the
        centre of the stretch (the middle point of the segment nearest to the length-weighted centre of the road).
        Returns (segments of the stretch as a DataFrame, centre x, centre y) or None"""
        s = self.seg[self.seg['key'] == key]
        if s.empty:
            return None
        mx = (self.mx[s['u'].to_numpy()] + self.mx[s['v'].to_numpy()]) / 2
        my = (self.my[s['u'].to_numpy()] + self.my[s['v'].to_numpy()]) / 2
        w = s['length'].to_numpy(float)
        if w.sum() > 0:
            gx, gy = float(np.average(mx, weights=w)), float(np.average(my, weights=w))
        else:
            gx, gy = float(mx.mean()), float(my.mean())
        k = int(np.argmin(np.hypot(mx - gx, my - gy)))
        cx, cy = float(mx[k]), float(my[k])
        sel = np.hypot(mx - cx, my - cy) <= length_m / 2
        return s[sel], cx, cy

    def local_pairs(self, cx, cy, radius_m, n_pairs, rng):
        """random pairs of different junctions of the largest strongly connected component within radius_m"""
        near = self.giant[np.hypot(self.mx[self.giant] - cx, self.my[self.giant] - cy) <= radius_m]
        if len(near) < 2:
            return []
        pairs = set()
        for _ in range(20 * n_pairs):
            if len(pairs) >= n_pairs:
                break
            a, b = rng.choice(near, size=2, replace=False)
            pairs.add((int(a), int(b)))
        return sorted(pairs)

    def closure_size(self, closed_keys):
        s = self.seg[self.seg['key'].isin(closed_keys)]
        if s.empty:
            return {'removed_length_km': 0.0, 'removed_length_pct': 0.0, 'removed_lane_km': 0.0,
                    'lanes_tagged_share': None, 'removed_segments': 0, 'junctions_touched': 0, 'extent_km': 0.0}
        nodes = np.unique(np.concatenate([s['u'].to_numpy(), s['v'].to_numpy()]))
        x, y = self.mx[nodes], self.my[nodes]
        known = s['lanes_n'].notna()
        length = float(s['length'].sum())
        return {'removed_length_km': length / 1000, 'removed_length_pct': 100 * length / self.total_length_m,
                'removed_lane_km': float((s.loc[known, 'length'] * s.loc[known, 'lanes_n']).sum()) / 1000,
                'lanes_tagged_share': float(s.loc[known, 'length'].sum()) / length if length else None,
                'removed_segments': int(len(s)), 'junctions_touched': int(len(nodes)),
                'extent_km': float(math.hypot(x.max() - x.min(), y.max() - y.min())) / 1000}


def dual_after_closure(G, closed_idx, seeds, with_bc, spectrum=False):
    keep = sorted(set(range(G.vcount())) - set(closed_idx))
    H = G.induced_subgraph(keep)
    U = H.as_undirected(mode='collapse')
    n = H.vcount()
    m = {'remaining_roads': n, 'remaining_dual_edges': H.ecount()}
    if n < 3:
        return (m, None) if spectrum else m
    wcc = H.connected_components(mode='weak')
    scc = H.connected_components(mode='strong')
    m.update({'wcc_count': len(wcc), 'wcc_largest_share': max(wcc.sizes()) / n, 'scc_largest_share': max(scc.sizes()) / n})
    p = path_length_stats(H)
    m.update({'avg_path_length': p['avg_path_length'], 'diameter': p['diameter'], 'reachable_share': p['reachable_share']})
    m['avg_harmonic_closeness'] = float(np.mean(H.harmonic_centrality(mode='out', normalized=True)))
    cc = np.array(U.transitivity_local_undirected(mode='zero'))
    acc = np.array(adapted_clustering(H, U))
    m['avg_clustering'] = float(np.mean(cc))
    m['avg_adapted_clustering'] = float(np.mean(acc))
    spec = None
    if spectrum:
        spec = pd.DataFrame({'degree': U.degree(), 'cc': cc, 'acc': acc}).groupby('degree').agg(
            avg_clustering=('cc', 'mean'), avg_adapted_clustering=('acc', 'mean'), node_count=('cc', 'size')).reset_index()
    runs, best = louvain_runs(U, seeds)
    m.update({'modularity': best[0], 'communities': len(set(best[1])), 'modularity_mean': float(runs['modularity'].mean())})
    if with_bc:
        m['avg_betweenness_norm'] = float(np.mean(H.betweenness(directed=True))) / ((n - 1) * (n - 2))
    return (m, spec) if spectrum else m


def take_until(order, lengths, budget_m):
    """roads in the given order until their total length reaches the budget"""
    chosen, total = [], 0.0
    for i in order:
        if total >= budget_m:
            break
        chosen.append(i)
        total += lengths[i]
    return chosen


def main():
    p = argparse.ArgumentParser(description='Road-closure scenarios on a dual graph (dual metrics and primal routing).')
    p.add_argument('--nodes-csv', required=True, help='primal nodes CSV (osmnode_id, x, y)')
    p.add_argument('--rels-csv', required=True, help='primal relationships CSV')
    p.add_argument('--edge-keys', required=True, help='edge_keys.csv written by dual_graph.py')
    p.add_argument('--dual-dir', required=True)
    p.add_argument('--key', required=True, choices=['name', 'name_split', 'stroke'],
                   help='column of edge_keys.csv that identifies the roads of this dual graph')
    p.add_argument('--output', required=True)
    p.add_argument('--label', default='')
    p.add_argument('--percentages', type=float, nargs='+', default=[2, 5, 10, 20, 30, 50])
    p.add_argument('--budgets', type=float, nargs='+', default=[1, 2, 5, 10, 20],
                   help='length budgets in %% of the total length of the primal network')
    p.add_argument('--random-repeats', type=int, default=10)
    p.add_argument('--origins', type=int, default=100)
    p.add_argument('--per-origin', type=int, default=10, help='destinations per origin (pairs = origins x per-origin)')
    p.add_argument('--louvain-seeds', type=int, default=5)
    p.add_argument('--budget-betweenness', action='store_true', help='also compute betweenness for the budget scenarios')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--section-roads', type=int, default=5,
                   help='section closures for the top N roads by PageRank (0 = none)')
    p.add_argument('--section-length', type=float, default=1000.0, help='length of the closed stretch (m)')
    p.add_argument('--local-radius', type=float, default=3000.0,
                   help='radius (m) around the stretch for the origin-destination pairs nearby')
    p.add_argument('--local-pairs', type=int, default=200, help='origin-destination pairs nearby, per road')
    p.add_argument('--example-pairs', type=int, default=10,
                   help='origin-destination pairs whose routes are written for the map (one per distance range)')
    p.add_argument('--example-levels', type=float, nargs='+', default=[2, 5],
                   help='RCS1 percentages whose routes are written for the map')
    o = p.parse_args()
    os.makedirs(o.output, exist_ok=True)
    t0 = time.time()
    seeds = list(range(1, o.louvain_seeds + 1))

    print(f"=== {o.label or o.dual_dir} ===")
    nodes, edges = load_dual(o.dual_dir)
    G, _, _ = build_graphs(nodes, edges)
    names = np.array(G.vs['name'], dtype=object)
    N = G.vcount()
    length_by_road = dict(zip(nodes['id'].astype(str), pd.to_numeric(nodes['distance'], errors='coerce').fillna(0.0)))
    road_len = np.array([length_by_road.get(r, 0.0) for r in names])
    print(f"   dual: {N} roads, {G.ecount()} edges")

    primal = Primal(o.nodes_csv, o.rels_csv, o.edge_keys, o.key)
    dual_keys = set(names)
    primal_keys = set(primal.seg['key'].dropna())
    print(f"   roads of the dual graph not found in edge_keys.csv: {len(dual_keys - primal_keys)}; "
          f"keys in edge_keys.csv not in the dual graph: {len(primal_keys - dual_keys)}")

    pr, _ = pagerank_paper(G)
    bc = np.array(G.betweenness(directed=True))
    deg = np.array(G.as_undirected(mode='collapse').degree(), dtype=float)
    def order(values, descending=True):
        df = pd.DataFrame({'v': values, 'id': names, 'i': range(N)})
        return df.sort_values(['v', 'id'], ascending=[not descending, True], kind='mergesort')['i'].to_numpy()
    ranks = {'pagerank_top': order(pr), 'pagerank_bottom': order(pr, False), 'betweenness_top': order(bc),
             'degree_top': order(deg)}
    pd.DataFrame({'road': names, 'pagerank': pr, 'betweenness': bc, 'degree': deg, 'length_m': road_len}).to_csv(
        os.path.join(o.output, 'road_rankings.csv'), index=False)

    pairs = primal.od_pairs(o.origins, o.per_origin, o.seed)
    base, base_giant = primal.routes(pairs)
    used = primal.base_route_keys(pairs)
    base_t, _ = primal.routes(pairs, weight='tt')
    used_t = primal.base_route_keys(pairs, weight='tt')
    bx = np.hypot(primal.mx[[a for a, _ in pairs]] - primal.mx[[b for _, b in pairs]],
                  primal.my[[a for a, _ in pairs]] - primal.my[[b for _, b in pairs]])
    pd.DataFrame({'origin': [a for a, _ in pairs], 'destination': [b for _, b in pairs], 'straight_km': bx / 1000,
                  'route_km': [base[q] / 1000 for q in pairs],
                  'route_min': [base_t[q] / 60 for q in pairs]}).to_csv(os.path.join(o.output, 'od_pairs.csv'), index=False)
    print(f"   {len(pairs)} origin-destination pairs; median route {np.median([base[q] for q in pairs]) / 1000:.2f} km, "
          f"median travel time {np.median([base_t[q] for q in pairs]) / 60:.1f} min")

    rows, od_rows, spec_rows, closed_rows = [], [], [], []

    def run(scenario, criterion, level, repeat, chosen, with_bc):
        t = time.time()
        closed = set(names[chosen]) if len(chosen) else set()
        row = {'variant': o.label, 'scenario': scenario, 'criterion': criterion, 'level': level, 'repeat': repeat,
               'roads_closed': len(chosen), 'roads_closed_pct': 100 * len(chosen) / N,
               'mean_road_extent_km': None}
        row.update(primal.closure_size(closed))
        if len(chosen) and 'span_m' in nodes.columns:
            span = dict(zip(nodes['id'].astype(str), pd.to_numeric(nodes['span_m'], errors='coerce')))
            row['mean_road_extent_km'] = float(np.nanmean([span.get(r, np.nan) for r in sorted(closed)])) / 1000
        if scenario in ('none', 'RCS1', 'RCS2'):
            m, spec = dual_after_closure(G, chosen, seeds, with_bc, spectrum=True)
            row.update(m)
            if spec is not None:
                spec.insert(0, 'level', level)
                spec.insert(0, 'scenario', scenario)
                spec_rows.append(spec)
            closed_rows.extend({'scenario': scenario, 'level': level, 'road': rd} for rd in names[chosen])
        else:
            row.update(dual_after_closure(G, chosen, seeds, with_bc))
        dist, giant = primal.routes(pairs, closed)
        row['primal_largest_scc_share'] = giant / len(primal.giant)
        disc, ratios, affected, aff_ratios = 0, [], 0, []
        for q in pairs:
            hit = bool(used[q] & closed)
            affected += hit
            if math.isinf(dist[q]):
                disc += 1
            else:
                r = dist[q] / base[q] if base[q] > 0 else 1.0
                ratios.append(r)
                if hit:
                    aff_ratios.append(r)
        dist_t, _ = primal.routes(pairs, closed, weight='tt')
        ratios_t, aff_t, affected_t = [], [], 0
        for q in pairs:
            hit = bool(used_t[q] & closed)
            affected_t += hit
            if not math.isinf(dist_t[q]):
                r = dist_t[q] / base_t[q] if base_t[q] > 0 else 1.0
                ratios_t.append(r)
                if hit:
                    aff_t.append(r)
            od_rows.append((scenario, criterion, level, repeat, q[0], q[1], dist[q], dist_t[q]))
        ratios = np.array(ratios)
        ratios_t = np.array(ratios_t)
        row.update({'od_pairs': len(pairs), 'od_route_affected_pct': 100 * affected / len(pairs),
                    'od_disconnected_pct': 100 * disc / len(pairs),
                    'detour_mean': float(ratios.mean()) if len(ratios) else None,
                    'detour_median': float(np.median(ratios)) if len(ratios) else None,
                    'detour_p90': float(np.percentile(ratios, 90)) if len(ratios) else None,
                    'detour_max': float(ratios.max()) if len(ratios) else None,
                    'detour_affected_mean': float(np.mean(aff_ratios)) if aff_ratios else None,
                    'od_fastest_route_affected_pct': 100 * affected_t / len(pairs),
                    'time_detour_mean': float(ratios_t.mean()) if len(ratios_t) else None,
                    'time_detour_median': float(np.median(ratios_t)) if len(ratios_t) else None,
                    'time_detour_p90': float(np.percentile(ratios_t, 90)) if len(ratios_t) else None,
                    'time_detour_max': float(ratios_t.max()) if len(ratios_t) else None,
                    'time_detour_affected_mean': float(np.mean(aff_t)) if aff_t else None,
                    'time_s': round(time.time() - t, 2)})
        rows.append(row)
        print(f"   {scenario:9s} {criterion:16s} {level:>5} r{repeat}: {len(chosen)} roads, "
              f"{row['removed_length_pct']:.1f}% length, WCC {row.get('wcc_count')}, "
              f"no path {row['od_disconnected_pct']:.1f}%, detour length {row['detour_mean']}, "
              f"time {row['time_detour_mean']} ({row['time_s']} s)")

    run('none', 'none', 0, 0, [], True)
    for pct in o.percentages:
        cnt = int(N * pct / 100)
        run('RCS1', 'pagerank_top', pct, 0, list(ranks['pagerank_top'][:cnt]), True)
        run('RCS2', 'pagerank_bottom', pct, 0, list(ranks['pagerank_bottom'][:cnt]), True)
    rng = np.random.default_rng(o.seed)
    for b in o.budgets:
        budget_m = primal.total_length_m * b / 100
        for crit in ('pagerank_top', 'betweenness_top', 'degree_top', 'pagerank_bottom'):
            run('budget', crit, b, 0, take_until(ranks[crit], road_len, budget_m), o.budget_betweenness)
        for r in range(1, o.random_repeats + 1):
            run('budget', 'random', b, r, take_until(rng.permutation(N), road_len, budget_m), o.budget_betweenness)

    if spec_rows:
        pd.concat(spec_rows, ignore_index=True).to_csv(os.path.join(o.output, 'closure_clustering_spectra.csv'), index=False)
    pd.DataFrame(closed_rows, columns=['scenario', 'level', 'road']).to_csv(
        os.path.join(o.output, 'closed_roads.csv'), index=False)
    if o.example_pairs > 0:
        order = np.argsort(bx, kind='mergesort')
        picks = sorted(set(int(order[min(len(order) - 1, int((i + 0.5) * len(order) / o.example_pairs))])
                           for i in range(o.example_pairs)), key=lambda i: bx[i])
        ex_pairs = [pairs[i] for i in picks]
        ex_rows = []
        for scenario, level, closed in [('none', 0, None)] + [
                ('RCS1', pct, set(names[list(ranks['pagerank_top'][:int(N * pct / 100)])])) for pct in o.example_levels]:
            paths = primal.route_junctions(ex_pairs, closed)
            dist, _ = primal.routes(ex_pairs, closed)
            for k, q in enumerate(ex_pairs, 1):
                ex_rows.append({'scenario': scenario, 'level': level, 'pair': k,
                                'origin': primal.node_ids[q[0]], 'destination': primal.node_ids[q[1]],
                                'straight_km': float(bx[picks[k - 1]]) / 1000,
                                'route_km': dist[q] / 1000 if not math.isinf(dist[q]) else None,
                                'junctions': ' '.join(paths[q])})
        pd.DataFrame(ex_rows).to_csv(os.path.join(o.output, 'example_routes.csv'), index=False)

    sec_rows, sec_segments = [], []
    if o.section_roads > 0:
        rng_local = np.random.default_rng(o.seed + 1)
        used_seg = primal.base_route_segs(pairs)
        used_seg_t = primal.base_route_segs(pairs, weight='tt')

        def effect(prefix, prs, base_w, base_tt, used_w, used_tt, hit_w, hit_tt, closed_keys, closed_segs):
            dist, giant = primal.routes(prs, closed_keys, closed_segs=closed_segs)
            dist_t, _ = primal.routes(prs, closed_keys, weight='tt', closed_segs=closed_segs)
            aff = [q for q in prs if hit_w(used_w[q])]
            aff_t = [q for q in prs if hit_tt(used_tt[q])]
            disc = sum(1 for q in prs if math.isinf(dist[q]))
            ok = [q for q in aff if not math.isinf(dist[q]) and base_w[q] > 0]
            ok_t = [q for q in aff_t if not math.isinf(dist_t[q]) and base_tt[q] > 0]
            n = max(len(prs), 1)
            out = {f'{prefix}pairs': len(prs),
                   f'{prefix}route_affected_pct': 100 * len(aff) / n,
                   f'{prefix}disconnected_pct': 100 * disc / n,
                   f'{prefix}detour_affected_mean': float(np.mean([dist[q] / base_w[q] for q in ok])) if ok else None,
                   f'{prefix}extra_km_affected_mean': float(np.mean([(dist[q] - base_w[q]) / 1000 for q in ok])) if ok else None,
                   f'{prefix}fastest_route_affected_pct': 100 * len(aff_t) / n,
                   f'{prefix}time_detour_affected_mean': float(np.mean([dist_t[q] / base_tt[q] for q in ok_t])) if ok_t else None,
                   f'{prefix}extra_min_affected_mean': float(np.mean([(dist_t[q] - base_tt[q]) / 60 for q in ok_t])) if ok_t else None}
            return out, giant

        for rank, i in enumerate(ranks['pagerank_top'][:o.section_roads], 1):
            road = names[i]
            found = primal.road_section(road, o.section_length)
            if found is None or found[0].empty:
                print(f"   section: road {road} has no primal segments, skipped")
                continue
            segs, cx, cy = found
            closed_segs = set(segs['seg'])
            local = primal.local_pairs(cx, cy, o.local_radius, o.local_pairs, rng_local)
            l_base, _ = primal.routes(local)
            l_base_t, _ = primal.routes(local, weight='tt')
            l_used = primal.base_route_keys(local)
            l_used_t = primal.base_route_keys(local, weight='tt')
            l_used_seg = primal.base_route_segs(local)
            l_used_seg_t = primal.base_route_segs(local, weight='tt')
            whole = primal.closure_size({road})
            for kind in ('section', 'whole_road'):
                t = time.time()
                if kind == 'section':
                    ck, cs = None, closed_segs
                    hit = lambda used_set: bool(used_set & closed_segs)
                    g_used, g_used_t, lu, lu_t = used_seg, used_seg_t, l_used_seg, l_used_seg_t
                    closed_km = float(segs['length'].sum()) / 1000
                    n_segments = int(len(segs))
                else:
                    ck, cs = {road}, None
                    hit = lambda used_set: road in used_set
                    g_used, g_used_t, lu, lu_t = used, used_t, l_used, l_used_t
                    closed_km = whole['removed_length_km']
                    n_segments = whole['removed_segments']
                row = {'variant': o.label, 'road': road, 'pagerank_rank': rank, 'closure': kind,
                       'road_length_km': whole['removed_length_km'], 'road_segments': whole['removed_segments'],
                       'road_extent_km': whole['extent_km'], 'closed_length_km': closed_km,
                       'closed_segments': n_segments, 'section_length_m': o.section_length,
                       'local_radius_m': o.local_radius}
                g, giant = effect('', pairs, base, base_t, g_used, g_used_t, hit, hit, ck, cs)
                row.update(g)
                row['primal_largest_scc_share'] = giant / len(primal.giant)
                l, _ = effect('local_', local, l_base, l_base_t, lu, lu_t, hit, hit, ck, cs)
                row.update(l)
                row['time_s'] = round(time.time() - t, 2)
                sec_rows.append(row)
                print(f"   section   {kind:10s} #{rank} {road}: closed {closed_km:.2f} km, "
                      f"local pairs {row['local_pairs']}, local routes affected {row['local_route_affected_pct']:.1f}%, "
                      f"local no path {row['local_disconnected_pct']:.1f}%, extra km {row['local_extra_km_affected_mean']}, "
                      f"extra min {row['local_extra_min_affected_mean']} ({row['time_s']} s)")
            for sg, ln in zip(segs['seg'], segs['length']):
                sec_segments.append({'road': road, 'pagerank_rank': rank, 'junction_a': primal.node_ids[sg[0]],
                                     'junction_b': primal.node_ids[sg[1]], 'osmway_id': sg[2], 'length_m': ln})
        pd.DataFrame(sec_rows).to_csv(os.path.join(o.output, 'section_closures.csv'), index=False)
        pd.DataFrame(sec_segments).to_csv(os.path.join(o.output, 'section_segments.csv'), index=False)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(o.output, 'closures.csv'), index=False)
    pd.DataFrame(od_rows, columns=['scenario', 'criterion', 'level', 'repeat', 'origin', 'destination', 'route_m',
                                   'route_s']).to_csv(
        os.path.join(o.output, 'od_routes.csv'), index=False)
    num = res.select_dtypes('number').columns.drop(['level', 'repeat'])
    rnd = res[res['criterion'] == 'random'].groupby('level')[list(num)].agg(['mean', 'std'])
    rnd.columns = [f"{a}_{b}" for a, b in rnd.columns]
    rnd.reset_index().to_csv(os.path.join(o.output, 'closures_random_summary.csv'), index=False)
    info = {'label': o.label, 'key': o.key, 'dual_roads': N, 'primal_length_km': primal.total_length_m / 1000,
            'primal_edges_without_key': primal.keys_missing, 'od_pairs': len(pairs),
            'driveable_largest_scc': int(len(primal.giant)), 'speeds': primal.speed_info,
            'section_closures': {'roads': o.section_roads, 'section_length_m': o.section_length,
                                 'local_radius_m': o.local_radius, 'local_pairs': o.local_pairs},
            'total_time_s': round(time.time() - t0, 1)}
    with open(os.path.join(o.output, 'closures_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Results in {o.output} ({info['total_time_s']} s)")


if __name__ == '__main__':
    main()
