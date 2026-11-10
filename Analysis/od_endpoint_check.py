"""
Origin-destination pairs without a path after the closures of dual_closures.py: how many of them have an origin or a
destination that is left without any open driveable road, and how many are separated by the closures although both
end points still have an open road.

A closure removes every segment of the closed roads. A junction that lies only on closed roads (for example a junction
in the middle of a closed road) has no open road left, so every pair that starts or ends there has no path. The other
pairs without a path are separated by the closures.

The closed roads of every scenario are selected again in the same way as dual_closures.py (same rankings from
road_rankings.csv, same length budgets, same random generator), and the result is checked against closures.csv
(number of closed roads, removed length, pairs without a path). No route is computed again: the distances are read
from od_routes.csv.

Output (in the closures folder): od_endpoint_check.csv, one row per scenario:
  pairs_no_path                    pairs without a path (as in closures.csv)
  no_path_endpoint_without_road    of these, pairs whose origin has no open outgoing road or whose destination has
                                   no open incoming road
  no_path_separated                the other pairs without a path
  pairs_endpoints_open             pairs whose origin and destination are not junctions of a closed road
  no_path_endpoints_open           pairs without a path among them
  (and the same values in % of all pairs, and in % of pairs_endpoints_open for the last one)

Section closures (section_closures.csv): the nearby pairs of each of the top roads are drawn again in the same way as
dual_closures.py (same random generator), their routes are computed again without the 1 km stretch and without the
whole road, and the same split is written to od_endpoint_check_sections.csv (checked against the share of nearby pairs
without a path in section_closures.csv).

Usage:
  python3 od_endpoint_check.py --nodes-csv .../fullgraph_nodes.csv --rels-csv .../fullgraph_rels.csv \
      --edge-keys .../edge_keys.csv --key name --closures .../analysis/rome_name_closures
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

from dual_closures import Primal, take_until


def main():
    p = argparse.ArgumentParser(description='End points of the origin-destination pairs without a path.')
    p.add_argument('--nodes-csv', required=True)
    p.add_argument('--rels-csv', required=True)
    p.add_argument('--edge-keys', required=True)
    p.add_argument('--key', required=True, choices=['name', 'name_split', 'stroke'])
    p.add_argument('--closures', required=True, help='output folder of dual_closures.py')
    p.add_argument('--percentages', type=float, nargs='+', default=[2, 5, 10, 20, 30, 50])
    p.add_argument('--budgets', type=float, nargs='+', default=[1, 2, 5, 10, 20])
    p.add_argument('--random-repeats', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--section-roads', type=int, default=5)
    p.add_argument('--section-length', type=float, default=1000.0)
    p.add_argument('--local-radius', type=float, default=3000.0)
    p.add_argument('--local-pairs', type=int, default=200)
    o = p.parse_args()

    rk = pd.read_csv(os.path.join(o.closures, 'road_rankings.csv'), dtype={'road': str}, keep_default_na=False,
                     na_values=[])
    names = rk['road'].to_numpy(object)
    N = len(names)
    road_len = pd.to_numeric(rk['length_m'], errors='coerce').fillna(0.0).to_numpy(float)

    def order(values, descending=True):
        df = pd.DataFrame({'v': values, 'id': names, 'i': range(N)})
        return df.sort_values(['v', 'id'], ascending=[not descending, True], kind='mergesort')['i'].to_numpy()

    pr = pd.to_numeric(rk['pagerank']).to_numpy(float)
    ranks = {'pagerank_top': order(pr), 'pagerank_bottom': order(pr, False),
             'betweenness_top': order(pd.to_numeric(rk['betweenness']).to_numpy(float)),
             'degree_top': order(pd.to_numeric(rk['degree']).to_numpy(float))}

    primal = Primal(o.nodes_csv, o.rels_csv, o.edge_keys, o.key)
    edges = np.array(primal.G.get_edgelist(), dtype=np.int64)
    eu, ev = edges[:, 0], edges[:, 1]
    n_nodes = primal.G.vcount()
    dkey = primal.drive_key

    scen = [('none', 'none', 0, 0, [])]
    for pct in o.percentages:
        cnt = int(N * pct / 100)
        scen.append(('RCS1', 'pagerank_top', pct, 0, list(ranks['pagerank_top'][:cnt])))
        scen.append(('RCS2', 'pagerank_bottom', pct, 0, list(ranks['pagerank_bottom'][:cnt])))
    rng = np.random.default_rng(o.seed)
    for b in o.budgets:
        budget_m = primal.total_length_m * b / 100
        for crit in ('pagerank_top', 'betweenness_top', 'degree_top', 'pagerank_bottom'):
            scen.append(('budget', crit, b, 0, take_until(ranks[crit], road_len, budget_m)))
        for r in range(1, o.random_repeats + 1):
            scen.append(('budget', 'random', b, r, take_until(rng.permutation(N), road_len, budget_m)))

    cl = pd.read_csv(os.path.join(o.closures, 'closures.csv'))
    od = pd.read_csv(os.path.join(o.closures, 'od_routes.csv'))
    od['route_m'] = pd.to_numeric(od['route_m'], errors='coerce')
    groups = dict(list(od.groupby(['scenario', 'criterion', 'level', 'repeat'])))

    rows, problems = [], []
    for scenario, criterion, level, repeat, chosen in scen:
        closed = set(names[chosen]) if len(chosen) else set()
        is_closed = np.array([isinstance(k, str) and k in closed for k in dkey], dtype=bool)
        op = ~is_closed
        out_open = np.bincount(eu[op], minlength=n_nodes)
        in_open = np.bincount(ev[op], minlength=n_nodes)
        on_closed = np.zeros(n_nodes, dtype=bool)
        on_closed[eu[is_closed]] = True
        on_closed[ev[is_closed]] = True

        g = groups.get((scenario, criterion, float(level), int(repeat)))
        if g is None:
            g = groups.get((scenario, criterion, level, repeat))
        if g is None:
            problems.append(f'{scenario} {criterion} {level} r{repeat}: not in od_routes.csv')
            continue
        o_ = g['origin'].to_numpy(np.int64)
        d_ = g['destination'].to_numpy(np.int64)
        nop = ~np.isfinite(g['route_m'].to_numpy(float))
        no_road = (out_open[o_] == 0) | (in_open[d_] == 0)
        ends_open = ~(on_closed[o_] | on_closed[d_])
        n = len(g)
        row = {'scenario': scenario, 'criterion': criterion, 'level': level, 'repeat': repeat,
               'roads_closed': len(chosen), 'pairs': n,
               'pairs_no_path': int(nop.sum()),
               'no_path_endpoint_without_road': int((nop & no_road).sum()),
               'no_path_separated': int((nop & ~no_road).sum()),
               'pairs_endpoints_open': int(ends_open.sum()),
               'no_path_endpoints_open': int((nop & ends_open).sum())}
        row['no_path_pct'] = 100 * row['pairs_no_path'] / n
        row['no_path_endpoint_without_road_pct'] = 100 * row['no_path_endpoint_without_road'] / n
        row['no_path_separated_pct'] = 100 * row['no_path_separated'] / n
        row['no_path_endpoints_open_pct'] = (100 * row['no_path_endpoints_open'] / row['pairs_endpoints_open']
                                             if row['pairs_endpoints_open'] else None)
        c = cl[(cl['scenario'] == scenario) & (cl['criterion'] == criterion) & (cl['level'] == level)
               & (cl['repeat'] == repeat)]
        if len(c) != 1:
            problems.append(f'{scenario} {criterion} {level} r{repeat}: {len(c)} rows in closures.csv')
        else:
            c = c.iloc[0]
            size = primal.closure_size(closed)
            if int(c['roads_closed']) != len(chosen):
                problems.append(f'{scenario} {criterion} {level} r{repeat}: roads {len(chosen)} vs {int(c["roads_closed"])}')
            if not math.isclose(size['removed_length_km'], float(c['removed_length_km']), rel_tol=1e-9, abs_tol=1e-9):
                problems.append(f'{scenario} {criterion} {level} r{repeat}: length {size["removed_length_km"]} vs '
                                f'{c["removed_length_km"]}')
            if not math.isclose(row['no_path_pct'], float(c['od_disconnected_pct']), abs_tol=1e-9):
                problems.append(f'{scenario} {criterion} {level} r{repeat}: no path {row["no_path_pct"]} vs '
                                f'{c["od_disconnected_pct"]}')
        rows.append(row)
        print(f"   {scenario:6s} {criterion:16s} {level:>5} r{repeat}: no path {row['no_path_pct']:.2f}% = "
              f"end point without road {row['no_path_endpoint_without_road_pct']:.2f}% + separated "
              f"{row['no_path_separated_pct']:.2f}%; end points open {row['pairs_endpoints_open']} pairs, no path "
              f"{row['no_path_endpoints_open_pct'] if row['no_path_endpoints_open_pct'] is not None else float('nan'):.2f}%")

    sec_path = os.path.join(o.closures, 'section_closures.csv')
    sec_rows = []
    if o.section_roads > 0 and os.path.exists(sec_path):
        sec = pd.read_csv(sec_path, dtype={'road': str}, keep_default_na=False, na_values=[])
        rng_local = np.random.default_rng(o.seed + 1)
        drive_seg = primal.drive_seg
        for rank, i in enumerate(ranks['pagerank_top'][:o.section_roads], 1):
            road = names[i]
            found = primal.road_section(road, o.section_length)
            if found is None or found[0].empty:
                continue
            segs, cx, cy = found
            closed_segs = set(segs['seg'])
            local = primal.local_pairs(cx, cy, o.local_radius, o.local_pairs, rng_local)
            for kind in ('section', 'whole_road'):
                if kind == 'section':
                    is_closed = np.array([sg in closed_segs for sg in drive_seg], dtype=bool)
                    dist, _ = primal.routes(local, closed_segs=closed_segs)
                else:
                    is_closed = np.array([isinstance(k, str) and k == road for k in dkey], dtype=bool)
                    dist, _ = primal.routes(local, {road})
                op = ~is_closed
                out_open = np.bincount(eu[op], minlength=n_nodes)
                in_open = np.bincount(ev[op], minlength=n_nodes)
                on_closed = np.zeros(n_nodes, dtype=bool)
                on_closed[eu[is_closed]] = True
                on_closed[ev[is_closed]] = True
                o_ = np.array([a for a, _ in local], dtype=np.int64)
                d_ = np.array([b for _, b in local], dtype=np.int64)
                nop = np.array([math.isinf(dist[q]) for q in local], dtype=bool)
                no_road = (out_open[o_] == 0) | (in_open[d_] == 0)
                ends_open = ~(on_closed[o_] | on_closed[d_])
                n = max(len(local), 1)
                row = {'road': road, 'pagerank_rank': rank, 'closure': kind, 'local_pairs': len(local),
                       'pairs_no_path': int(nop.sum()), 'no_path_endpoint_without_road': int((nop & no_road).sum()),
                       'no_path_separated': int((nop & ~no_road).sum()), 'pairs_endpoints_open': int(ends_open.sum()),
                       'no_path_endpoints_open': int((nop & ends_open).sum())}
                row['no_path_pct'] = 100 * row['pairs_no_path'] / n
                row['no_path_endpoint_without_road_pct'] = 100 * row['no_path_endpoint_without_road'] / n
                row['no_path_separated_pct'] = 100 * row['no_path_separated'] / n
                row['no_path_endpoints_open_pct'] = (100 * row['no_path_endpoints_open'] / row['pairs_endpoints_open']
                                                     if row['pairs_endpoints_open'] else None)
                c = sec[(sec['road'] == road) & (sec['closure'] == kind)]
                if len(c) != 1 or not math.isclose(row['no_path_pct'], float(c.iloc[0]['local_disconnected_pct']),
                                                   abs_tol=1e-9):
                    problems.append(f"section {kind} {road}: nearby no path {row['no_path_pct']} vs "
                                    f"{c.iloc[0]['local_disconnected_pct'] if len(c) else 'missing'}")
                sec_rows.append(row)
                print(f"   section {kind:10s} #{rank} {road}: nearby no path {row['no_path_pct']:.1f}% = end point "
                      f"without road {row['no_path_endpoint_without_road_pct']:.1f}% + separated "
                      f"{row['no_path_separated_pct']:.1f}%")
        pd.DataFrame(sec_rows).to_csv(os.path.join(o.closures, 'od_endpoint_check_sections.csv'), index=False)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(o.closures, 'od_endpoint_check.csv'), index=False)
    print(f"Check against closures.csv: {'OK, same closed roads, lengths and pairs without a path' if not problems else 'DIFFERENCES'}")
    for pr_ in problems:
        print('   ', pr_)
    print(f"Written {os.path.join(o.closures, 'od_endpoint_check.csv')}")


if __name__ == '__main__':
    main()
