"""
Compare the primal graph built with the cell-based pipeline (fullgraph_nodes.csv,
fullgraph_rels.csv) with the graph built with a single OSMnx request (GraphML file
saved by road_network_single_request.py --output-file).

Checks:
  1. node identities (OSM node ids) and node coordinates
  2. edge identities as directed triples (source, target, OSM way id), i.e. directionality
  3. parallel edges (more than one edge between the same source and target)
  4. edge attributes of the common edges
  5. shortest-path consistency on random origin-destination pairs (length as weight)

Usage:
  python3 compare_with_single_request.py --cell-nodes modena_cell/fullgraph_nodes.csv \
      --cell-rels modena_cell/fullgraph_rels.csv \
      --single-graphml modena_single.graphml --pairs 500 --report modena_comparison.txt
"""
import argparse
import math
import random
import time
from collections import Counter

import networkx as nx
import osmnx as ox
import pandas as pd

EDGE_ATTRIBUTES = ['length', 'highway', 'name', 'oneway', 'reversed', 'maxspeed',
                   'lanes', 'ref', 'access', 'junction', 'driveable']


def norm_id(value):
    """OSM ids as strings without decimals (CSV files may store 123.0)."""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def norm_value(value):
    """Attribute values in a common form: numbers with 3 decimals, missing values as ''."""
    if value is None:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    if isinstance(value, (list, tuple)):
        return ';'.join(norm_value(v) for v in value)
    text = str(value).strip()
    if text.lower() in ('', 'nan', 'none'):
        return ''
    try:
        return f"{float(text):.3f}"
    except ValueError:
        return text


def normalize_columns(df, renames):
    """Accept also the neo4j-admin headers of the Neo4j-based script (e.g. 'lat:FLOAT',
    ':START_ID(RoadJunction)') and its column names (id, lat, lon, osmid)."""
    cols = {}
    for c in df.columns:
        if c.startswith(':START_ID'):
            cols[c] = 'source_id'
        elif c.startswith(':END_ID'):
            cols[c] = 'target_id'
        elif ':' in c and not c.startswith(':'):
            cols[c] = c.split(':')[0]
    df = df.rename(columns=cols).drop(columns=[':TYPE'], errors='ignore')
    return df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})


def load_cell_graph(nodes_csv, rels_csv):
    nodes = pd.read_csv(nodes_csv, dtype=str, low_memory=False)
    rels = pd.read_csv(rels_csv, dtype=str, low_memory=False)
    nodes = normalize_columns(nodes, {'id': 'osmnode_id', 'lon': 'x', 'lat': 'y'})
    rels = normalize_columns(rels, {'osmid': 'osmway_id'})
    G = nx.MultiDiGraph()
    for row in nodes.itertuples(index=False):
        d = row._asdict()
        G.add_node(norm_id(d['osmnode_id']), x=float(d['x']), y=float(d['y']),
                   street_count=norm_value(d.get('street_count')))
    columns = [c for c in rels.columns if c not in ('source_id', 'target_id')]
    for values in zip(rels['source_id'], rels['target_id'], *[rels[c] for c in columns]):
        u, v = norm_id(values[0]), norm_id(values[1])
        attrs = dict(zip(columns, values[2:]))
        attrs['osmway_id'] = norm_id(attrs.get('osmway_id'))
        attrs['length'] = float(attrs['length']) if norm_value(attrs.get('length')) else float('nan')
        G.add_edge(u, v, **attrs)
    return G


def load_single_graph(graphml_path):
    G0 = ox.load_graphml(graphml_path)
    G = nx.MultiDiGraph()
    for n, d in G0.nodes(data=True):
        G.add_node(norm_id(n), x=float(d['x']), y=float(d['y']),
                   street_count=norm_value(d.get('street_count')))
    for u, v, d in G0.edges(data=True):
        attrs = dict(d)
        attrs['osmway_id'] = norm_id(attrs.pop('osmid', ''))
        attrs['length'] = float(attrs.get('length', float('nan')))
        G.add_edge(norm_id(u), norm_id(v), **attrs)
    return G


def edge_index(G):
    """Counter of directed triples and the attributes of the first edge of each triple."""
    triples = Counter()
    attrs = {}
    for u, v, d in G.edges(data=True):
        t = (u, v, d.get('osmway_id', ''))
        triples[t] += 1
        attrs.setdefault(t, d)
    return triples, attrs


def parallel_pairs(G):
    pairs = Counter((u, v) for u, v in G.edges())
    return {p: c for p, c in pairs.items() if c > 1}


def main():
    parser = argparse.ArgumentParser(description='Compare cell-based and single-request primal graphs.')
    parser.add_argument('--cell-nodes', required=True)
    parser.add_argument('--cell-rels', required=True)
    parser.add_argument('--single-graphml', default=None,
                        help='reference graph: GraphML file of a single request')
    parser.add_argument('--ref-nodes', default=None,
                        help='reference graph given as CSV files of another cell-based run (with --ref-rels)')
    parser.add_argument('--ref-rels', default=None)
    parser.add_argument('--pairs', type=int, default=500, help='random OD pairs for shortest paths')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', default='graph_comparison.txt')
    args = parser.parse_args()

    lines = []

    def out(text=''):
        print(text)
        lines.append(text)

    t0 = time.time()
    C = load_cell_graph(args.cell_nodes, args.cell_rels)
    if args.single_graphml:
        S = load_single_graph(args.single_graphml)
    elif args.ref_nodes and args.ref_rels:
        S = load_cell_graph(args.ref_nodes, args.ref_rels)
    else:
        parser.error('give --single-graphml, or --ref-nodes and --ref-rels')
    out(f"Cell-based graph:     {C.number_of_nodes()} nodes, {C.number_of_edges()} edges")
    out(f"Single-request graph: {S.number_of_nodes()} nodes, {S.number_of_edges()} edges")

    cn, sn = set(C.nodes), set(S.nodes)
    common_nodes = cn & sn
    out("\n1. Node identities")
    out(f"   common: {len(common_nodes)} | only cell-based: {len(cn - sn)} | only single request: {len(sn - cn)}")
    max_dx = max((abs(C.nodes[n]['x'] - S.nodes[n]['x']) for n in common_nodes), default=0.0)
    max_dy = max((abs(C.nodes[n]['y'] - S.nodes[n]['y']) for n in common_nodes), default=0.0)
    sc_diff = sorted(n for n in common_nodes if C.nodes[n]['street_count'] != S.nodes[n]['street_count'])
    out(f"   max coordinate difference: lon {max_dx:.2e} deg, lat {max_dy:.2e} deg")
    out(f"   nodes with different street_count: {len(sc_diff)}")
    for n in sc_diff[:5]:
        out(f"      e.g. node {n} (lat {S.nodes[n]['y']:.6f}, lon {S.nodes[n]['x']:.6f}): "
            f"cell={C.nodes[n]['street_count']!r} single={S.nodes[n]['street_count']!r}, "
            f"edges in single graph: {S.in_degree(n)} in / {S.out_degree(n)} out")
    for n in sorted(cn - sn)[:5]:
        out(f"   example only cell-based: {n}")
    for n in sorted(sn - cn)[:5]:
        out(f"   example only single request: {n}")

    ct, ca = edge_index(C)
    st, sa = edge_index(S)
    common_edges = set(ct) & set(st)
    out("\n2. Edge identities (source, target, OSM way id), directed")
    out(f"   common: {len(common_edges)} | only cell-based: {len(set(ct) - set(st))} | "
        f"only single request: {len(set(st) - set(ct))}")
    mult_diff = sum(1 for t in common_edges if ct[t] != st[t])
    out(f"   triples with a different number of edges: {mult_diff}")
    for t in sorted(set(ct) - set(st))[:5]:
        out(f"   example only cell-based: {t}")
    for t in sorted(set(st) - set(ct))[:5]:
        out(f"   example only single request: {t}")

    cp, sp = parallel_pairs(C), parallel_pairs(S)
    out("\n3. Parallel edges (same source and target)")
    out(f"   cell-based: {len(cp)} pairs ({sum(cp.values())} edges) | "
        f"single request: {len(sp)} pairs ({sum(sp.values())} edges) | identical: {cp == sp}")

    out("\n4. Attributes of the common edges")
    for a in EDGE_ATTRIBUTES:
        diff = [t for t in common_edges if norm_value(ca[t].get(a)) != norm_value(sa[t].get(a))]
        out(f"   {a:<10} different: {len(diff)}")
        for t in diff[:3]:
            out(f"      e.g. {t}: cell={norm_value(ca[t].get(a))!r} single={norm_value(sa[t].get(a))!r}")

    out(f"\n5. Shortest paths on {args.pairs} random origin-destination pairs (weight = length)")
    rng = random.Random(args.seed)
    pool = sorted(common_nodes)
    same = both_unreachable = only_one_reachable = different = 0
    max_abs = 0.0
    for _ in range(args.pairs):
        s, t = rng.sample(pool, 2)
        try:
            lc = nx.shortest_path_length(C, s, t, weight='length')
        except nx.NetworkXNoPath:
            lc = None
        try:
            ls = nx.shortest_path_length(S, s, t, weight='length')
        except nx.NetworkXNoPath:
            ls = None
        if lc is None and ls is None:
            both_unreachable += 1
        elif lc is None or ls is None:
            only_one_reachable += 1
        elif abs(lc - ls) <= 1e-6:
            same += 1
            max_abs = max(max_abs, abs(lc - ls))
        else:
            different += 1
            max_abs = max(max_abs, abs(lc - ls))
    out(f"   identical length: {same} | different length: {different} | "
        f"no path in both: {both_unreachable} | path in only one graph: {only_one_reachable}")
    out(f"   largest length difference: {max_abs:.6f} m")

    out(f"\nTime: {time.time() - t0:.1f} s")
    with open(args.report, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {args.report}")


if __name__ == '__main__':
    main()
