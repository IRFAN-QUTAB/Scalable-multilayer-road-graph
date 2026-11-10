"""
Dual graphs of a road network graph (CSV files of the cell-based pipeline).

1. Name-based dual graph (main output, folder --output): one RoadOsm node per road name, and CONNECTED edges
   built with the junction rule (from a road with a segment entering a junction to a road with a segment
   leaving it), the roundabout rule and the unnamed-segment rule.
   Unnamed-segment rule: the chain of unnamed segments can also pass through unnamed roundabout segments.
   Remaining roads: after the three rules, a road that is still not connected to the largest component
   is connected to the nearest road of the largest component that can be reached in the primal graph
   (shortest path, directions ignored); these CONNECTED edges have via = 'path'. They are added only for
   roads outside the largest component, so the rest of the dual graph is not changed.
   Attributes of the RoadOsm nodes:
     distance        length of the road, every road segment counted once
     segment_count   number of road segments
     junction_count  number of junctions (primal nodes) of the road
     lane_km         sum of length x lanes over the segments with a lanes tag (km)
     lanes_known     share of the length of the road with a lanes tag
     span_m          diagonal of the bounding box of the road (m)
     pieces, max_gap_m  connected pieces of the road and largest distance between them (see 2)
2. Roads with the same name that are not connected: for every name the
   segments are grouped into connected pieces and the distance between the pieces is measured
   (name_pieces.csv, name_pieces_summary.txt).
3. Alternative dual graphs built in the same run, with the same CONNECTED rules (folder --output/...):
     name_split/  a road is a name and a connected part of it: pieces of the same name that are more than
                  --split-gap metres apart become different RoadOsm nodes (e.g. two streets with the same
                  name in two different towns).
     strokes/     a road is a stroke (geometric continuity, momepy COINS), named or not; the CONNECTED
                  edges are built with the junction rule (strokes already follow roundabouts and unnamed
                  segments).  Needs momepy (pip install momepy); skipped with --no-strokes.
4. edge_keys.csv: road key of every primal edge in the three dual graphs (name, name_split, stroke).
5. dual_report.txt in every folder: nodes, edges, edges by rule (junction / roundabout / unnamed),
   reciprocity, named and unnamed length of the primal graph, time of each step.
"""
import argparse
import math
import os
import shutil
import time
from collections import defaultdict, Counter, deque
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def is_missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value) in ('', 'nan')


def first_number(value):
    """lanes can be '2', '2;3', '[2, 3]' or missing: first number, or None"""
    if is_missing(value):
        return None
    for token in str(value).replace('[', ' ').replace(']', ' ').replace(';', ' ').replace(',', ' ').split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


class DualGraphGenerator:
    NUMERIC_DOUBLE_FIELDS = {'lat', 'lon', 'x', 'y', 'distance', 'length', 'lane_km', 'lanes_known',
                             'span_m', 'max_gap_m'}
    NUMERIC_LONG_FIELDS = {'street_count', 'segment_count', 'junction_count', 'pieces', 'component_id'}

    def __init__(self, G, output_folder: str, key_attr: str = 'name',
                 roundabout_max_hops: int = 50, unnamed_max_hops: int = 500,
                 use_roundabout_rule: bool = True, use_unnamed_rule: bool = True, label: str = ''):
        self.G = G
        self.output_folder = output_folder
        self.key_attr = key_attr
        self.roundabout_max_hops = roundabout_max_hops
        self.unnamed_max_hops = unnamed_max_hops
        self.use_roundabout_rule = use_roundabout_rule
        self.use_unnamed_rule = use_unnamed_rule
        self.label = label
        self.roadosm_nodes = []
        self.connected_edges = []
        self.times = {}
        self.extra_node_attrs = {}
        os.makedirs(self.output_folder, exist_ok=True)

    def _key(self, data):
        if data.get('status') != 'active':
            return None
        value = data.get(self.key_attr)
        if is_missing(value):
            return None
        return str(value)

    def _build_neo4j_header(self, df, id_col=None, id_space=None, start_id_col=None, end_id_col=None,
                            start_id_space=None, end_id_space=None) -> List[str]:
        headers = []
        for col in df.columns:
            if id_col and col == id_col:
                headers.append(f'id:ID({id_space})' if id_space else 'id:ID')
            elif start_id_col and col == start_id_col:
                headers.append(f':START_ID({start_id_space})' if start_id_space else ':START_ID')
            elif end_id_col and col == end_id_col:
                headers.append(f':END_ID({end_id_space})' if end_id_space else ':END_ID')
            elif col == 'edge_type':
                headers.append(':TYPE')
            elif col in self.NUMERIC_DOUBLE_FIELDS:
                headers.append(f"{col}:DOUBLE")
            elif col in self.NUMERIC_LONG_FIELDS:
                headers.append(f"{col}:LONG")
            else:
                headers.append(f"{col}:STRING")
        return headers

    def _save_csv_with_neo4j_headers(self, df, filepath, **kw):
        headers = self._build_neo4j_header(df, **kw)
        with open(filepath, 'w', newline='') as f:
            f.write(','.join(headers) + '\n')
        df.to_csv(filepath, mode='a', index=False, header=False)

    def create_roadosm_nodes(self):
        """RoadOsm node per road key; every road segment is counted once"""
        t = time.time()
        xs = nx.get_node_attributes(self.G, 'x')
        ys = nx.get_node_attributes(self.G, 'y')
        road = defaultdict(lambda: {'osmway_ids': set(), 'segments': set(), 'distance': 0.0,
                                    'junctions': set(), 'lane_m': 0.0, 'lanes_known_m': 0.0, 'names': Counter()})
        for u, v, k, data in self.G.edges(keys=True, data=True):
            key = self._key(data)
            if key is None:
                continue
            r = road[key]
            r['osmway_ids'].add(str(data.get('osmway_id', '')))
            r['junctions'].update((u, v))
            if not is_missing(data.get('name')):
                r['names'][str(data.get('name'))] += 1
            segment = (min(u, v), max(u, v), str(data.get('osmway_id', '')))
            if segment in r['segments']:
                continue
            r['segments'].add(segment)
            length = data.get('distance') or data.get('length') or 0
            if isinstance(length, (int, float)) and not np.isnan(length):
                r['distance'] += length
                lanes = first_number(data.get('lanes'))
                if lanes is not None:
                    r['lane_m'] += length * lanes
                    r['lanes_known_m'] += length

        ref_lat = float(np.mean(list(ys.values()))) if ys else 0.0
        kx = 111320.0 * math.cos(math.radians(ref_lat))
        for key, r in road.items():
            pts = [(float(xs[n]) * kx, float(ys[n]) * 111320.0) for n in r['junctions'] if n in xs]
            span = 0.0
            if pts:
                a = np.array(pts)
                span = float(np.hypot(a[:, 0].max() - a[:, 0].min(), a[:, 1].max() - a[:, 1].min()))
            node = {
                'id': key,
                'name': r['names'].most_common(1)[0][0] if r['names'] else '',
                'node_type': 'RoadOsm',
                'status': 'active',
                'distance': round(r['distance'], 3),
                'segment_count': len(r['segments']),
                'junction_count': len(r['junctions']),
                'lane_km': round(r['lane_m'] / 1000.0, 4),
                'lanes_known': round(r['lanes_known_m'] / r['distance'], 4) if r['distance'] else 0.0,
                'span_m': round(span, 1),
                'osmway_ids': ';'.join(sorted(r['osmway_ids'])),
            }
            node.update(self.extra_node_attrs.get(key, {}))
            self.roadosm_nodes.append(node)
        self.times['RoadOsm nodes'] = time.time() - t
        print(f"[{self.label}] {len(self.roadosm_nodes)} RoadOsm nodes")

    def create_connected_relationships(self):
        """For junction m: incoming road A (x -A-> m) and outgoing road B (m -B-> n), A != B: A -> B"""
        t = time.time()
        valid = set(n['id'] for n in self.roadosm_nodes)
        outgoing, incoming = defaultdict(set), defaultdict(set)
        for u, v, k, data in self.G.edges(keys=True, data=True):
            key = self._key(data)
            if key is None or key not in valid:
                continue
            outgoing[u].add(key)
            incoming[v].add(key)
        connection = defaultdict(lambda: {'junction_ids': set(), 'locations': set()})
        for m in set(outgoing) & set(incoming):
            loc = self.G.nodes.get(m, {}).get('location', '')
            for target in outgoing[m]:
                for source in incoming[m]:
                    if source == target:
                        continue
                    connection[(source, target)]['junction_ids'].add(str(m))
                    if loc and not is_missing(loc):
                        connection[(source, target)]['locations'].add(str(loc))
        for (source, target), info in connection.items():
            self.connected_edges.append({'source_id': source, 'target_id': target,
                                         'junctions': ';'.join(info['junction_ids']),
                                         'locations': ';'.join(info['locations']),
                                         'edge_type': 'CONNECTED', 'via': ''})
        self.times['CONNECTED (junctions)'] = time.time() - t

    def _through_subgraph(self, sub_edges, max_hops, via, entry_ok):
        """Connect roads that enter and leave a chain of sub_edges (roundabout or unnamed), both directions"""
        valid = set(n['id'] for n in self.roadosm_nodes)
        existing = set((e['source_id'], e['target_id']) for e in self.connected_edges)
        H = nx.DiGraph()
        H.add_edges_from(sub_edges)
        if H.number_of_nodes() == 0:
            return 0
        sub_nodes = set(H.nodes())
        incoming, outgoing = defaultdict(set), defaultdict(set)
        for u, v, k, data in self.G.edges(keys=True, data=True):
            if not entry_ok(data):
                continue
            key = self._key(data)
            if key is None or key not in valid:
                continue
            if v in sub_nodes:
                incoming[v].add(key)
            if u in sub_nodes:
                outgoing[u].add(key)
        pairs = set()
        for entry, roads in incoming.items():
            visited, queue = set(), [(entry, 0)]
            while queue:
                cur, depth = queue.pop(0)
                if depth >= max_hops:
                    continue
                for nb in H.successors(cur):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    queue.append((nb, depth + 1))
                    for r1 in roads:
                        for r2 in outgoing.get(nb, ()):
                            if r1 != r2:
                                pairs.add((r1, r2))
                                pairs.add((r2, r1))
        added = 0
        for s, t in pairs:
            if (s, t) not in existing:
                existing.add((s, t))
                self.connected_edges.append({'source_id': s, 'target_id': t, 'junctions': '', 'locations': '',
                                             'edge_type': 'CONNECTED', 'via': via})
                added += 1
        return added

    def create_roundabout_connections(self):
        t = time.time()
        rb = [(u, v) for u, v, k, d in self.G.edges(keys=True, data=True)
              if d.get('junction') == 'roundabout' and d.get('status') == 'active']
        n = self._through_subgraph(rb, self.roundabout_max_hops, 'roundabout',
                                   lambda d: d.get('status') == 'active' and d.get('junction') != 'roundabout')
        self.times['CONNECTED (roundabouts)'] = time.time() - t
        print(f"[{self.label}] {n} connections through roundabouts")

    def create_unnamed_connections(self):
        t = time.time()
        un = [(u, v) for u, v, k, d in self.G.edges(keys=True, data=True)
              if d.get('status') == 'active' and is_missing(d.get(self.key_attr))]
        n = self._through_subgraph(un, self.unnamed_max_hops, 'unnamed',
                                   lambda d: d.get('status') == 'active' and d.get('junction') != 'roundabout')
        self.times['CONNECTED (unnamed segments)'] = time.time() - t
        print(f"[{self.label}] {n} connections through unnamed segments")

    def export_to_csv(self):
        t = time.time()
        if self.roadosm_nodes:
            self._save_csv_with_neo4j_headers(pd.DataFrame(self.roadosm_nodes),
                                              os.path.join(self.output_folder, 'dual_nodes.csv'),
                                              id_col='id', id_space='RoadOsm')
        if self.connected_edges:
            self._save_csv_with_neo4j_headers(pd.DataFrame(self.connected_edges),
                                              os.path.join(self.output_folder, 'dual_edges.csv'),
                                              start_id_col='source_id', end_id_col='target_id',
                                              start_id_space='RoadOsm', end_id_space='RoadOsm')
        self.times['writing CSV files'] = time.time() - t

    def report(self, extra_lines=()):
        pairs = set((e['source_id'], e['target_id']) for e in self.connected_edges)
        reciprocal = sum(1 for s, t in pairs if (t, s) in pairs)
        via = Counter(e['via'] or 'junction' for e in self.connected_edges)
        D = nx.DiGraph()
        D.add_nodes_from(n['id'] for n in self.roadosm_nodes)
        D.add_edges_from(pairs)
        wcc = [len(c) for c in nx.weakly_connected_components(D)] if D.number_of_nodes() else [0]
        scc = [len(c) for c in nx.strongly_connected_components(D)] if D.number_of_nodes() else [0]
        lines = [f"Dual graph: {self.label}",
                 f"RoadOsm nodes: {len(self.roadosm_nodes)}",
                 f"CONNECTED edges: {len(self.connected_edges)} "
                 f"(junction {via.get('junction', 0)}, roundabout {via.get('roundabout', 0)}, unnamed {via.get('unnamed', 0)}, "
                 f"remaining roads (path) {via.get('path', 0)})",
                 f"Reciprocity (share of edges whose reverse edge exists): {100.0 * reciprocal / max(len(pairs), 1):.1f}%",
                 f"Weakly connected components: {len(wcc)} (largest {max(wcc)} nodes)",
                 f"Strongly connected components: {len(scc)} (largest {max(scc)} nodes)",
                 f"Mean RoadOsm distance: {np.mean([n['distance'] for n in self.roadosm_nodes]):.1f} m; "
                 f"mean segments {np.mean([n['segment_count'] for n in self.roadosm_nodes]):.1f}; "
                 f"mean junctions {np.mean([n['junction_count'] for n in self.roadosm_nodes]):.1f}"]
        lines += list(extra_lines)
        lines += [f"Time - {k}: {v:.2f} s" for k, v in self.times.items()]
        with open(os.path.join(self.output_folder, 'dual_report.txt'), 'w') as f:
            f.write("\n".join(lines) + "\n")
        for line in lines:
            print(f"[{self.label}] {line}")
        return lines

    def connect_remaining_roads(self):
        """Roads outside the largest weakly connected component: connected to the nearest road of the
        largest component along the shortest path of the primal graph (directions ignored), both directions"""
        t = time.time()
        valid = set(n['id'] for n in self.roadosm_nodes)
        node_roads = defaultdict(set)
        adj = defaultdict(set)
        for u, v, k, data in self.G.edges(keys=True, data=True):
            if data.get('status') != 'active':
                continue
            adj[u].add(v)
            adj[v].add(u)
            key = self._key(data)
            if key is not None and key in valid:
                node_roads[u].add(key)
                node_roads[v].add(key)
        road_nodes = defaultdict(set)
        for n, keys in node_roads.items():
            for key in keys:
                road_nodes[key].add(n)
        added = 0
        unreachable = set()
        while True:
            D = nx.Graph()
            D.add_nodes_from(valid)
            D.add_edges_from((e['source_id'], e['target_id']) for e in self.connected_edges)
            comps = sorted(nx.connected_components(D), key=len, reverse=True)
            largest = comps[0]
            todo = [c for c in comps[1:] if not (c & unreachable)]
            if not todo:
                break
            small = todo[0]
            start = {n: key for key in small for n in road_nodes[key]}
            prev = {n: None for n in start}
            queue = deque(start)
            found = None
            while queue:
                cur = queue.popleft()
                hit = node_roads[cur] & largest
                if hit:
                    found = (cur, sorted(hit)[0])
                    break
                for nb in adj[cur]:
                    if nb not in prev:
                        prev[nb] = cur
                        queue.append(nb)
            if found is None:
                print(f"[{self.label}] no primal path from {sorted(small)} to the largest component")
                unreachable |= small
                continue
            node = found[0]
            while prev[node] is not None:
                node = prev[node]
            source_road = start[node]
            for s, t2 in ((source_road, found[1]), (found[1], source_road)):
                self.connected_edges.append({'source_id': s, 'target_id': t2, 'junctions': '', 'locations': '',
                                             'edge_type': 'CONNECTED', 'via': 'path'})
                added += 1
            print(f"[{self.label}] connected {source_road} <-> {found[1]} (component of {len(small)} roads)")
        self.times['CONNECTED (remaining roads, shortest path)'] = time.time() - t
        print(f"[{self.label}] {added} connections for roads outside the largest component")

    def build(self):
        self.create_roadosm_nodes()
        self.create_connected_relationships()
        if self.use_roundabout_rule:
            self.create_roundabout_connections()
        if self.use_unnamed_rule:
            self.create_unnamed_connections()
        self.connect_remaining_roads()
        self.assign_component_ids()
        self.export_to_csv()

    def assign_component_ids(self):
        """ComponentId of every RoadOsm node: weakly connected component of the final dual graph
        (0 = largest component, then by decreasing size)"""
        D = nx.Graph()
        D.add_nodes_from(n['id'] for n in self.roadosm_nodes)
        D.add_edges_from((e['source_id'], e['target_id']) for e in self.connected_edges)
        comps = sorted(nx.connected_components(D), key=lambda c: (-len(c), min(c)))
        comp_of = {road: i for i, c in enumerate(comps) for road in c}
        for n in self.roadosm_nodes:
            n['component_id'] = comp_of[n['id']]


def load_primal_graph(nodes_csv, rels_csv):
    t = time.time()
    nodes_df = pd.read_csv(nodes_csv, low_memory=False)
    rels_df = pd.read_csv(rels_csv, low_memory=False)
    G = nx.MultiDiGraph()
    node_cols = [c for c in nodes_df.columns if c != 'osmnode_id']
    for values in zip(nodes_df['osmnode_id'], *[nodes_df[c] for c in node_cols]):
        G.add_node(int(values[0]), **dict(zip(node_cols, values[1:])))
    edge_cols = [c for c in rels_df.columns if c not in ('source_id', 'target_id')]
    for values in zip(rels_df['source_id'], rels_df['target_id'], *[rels_df[c] for c in edge_cols]):
        G.add_edge(int(values[0]), int(values[1]), **dict(zip(edge_cols, values[2:])))
    print(f"Primal graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({time.time() - t:.1f} s)")
    return G, time.time() - t


def metric_xy(G):
    xs = nx.get_node_attributes(G, 'x')
    ys = nx.get_node_attributes(G, 'y')
    ref_lat = float(np.mean(list(ys.values())))
    kx = 111320.0 * math.cos(math.radians(ref_lat))
    return {n: (float(xs[n]) * kx, float(ys[n]) * 111320.0) for n in xs}


def primal_length_shares(G):
    named = unnamed = 0.0
    seen = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        seg = (min(u, v), max(u, v), str(d.get('osmway_id', '')))
        if seg in seen:
            continue
        seen.add(seg)
        length = d.get('length') or 0
        if not isinstance(length, (int, float)) or np.isnan(length):
            continue
        if is_missing(d.get('name')):
            unnamed += length
        else:
            named += length
    total = named + unnamed
    return [f"Primal graph: {len(seen)} road segments, {total / 1000:.1f} km; named {named / 1000:.1f} km "
            f"({100 * named / max(total, 1e-9):.1f}%), unnamed {unnamed / 1000:.1f} km ({100 * unnamed / max(total, 1e-9):.1f}%)"]


def name_pieces(G, xy, output_folder, split_gap):
    """Connected pieces of every name, distance between pieces, and groups of pieces within split_gap.
    Returns piece_key (edge (u, v, osmway) -> name or 'name #k') and attributes per road key."""
    t = time.time()
    name_edges = defaultdict(set)
    name_length = defaultdict(float)
    seen = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get('status') != 'active' or is_missing(d.get('name')):
            continue
        name = str(d.get('name'))
        name_edges[name].add((u, v))
        seg = (name, min(u, v), max(u, v), str(d.get('osmway_id', '')))
        if seg not in seen:
            seen.add(seg)
            length = d.get('length') or 0
            if isinstance(length, (int, float)) and not np.isnan(length):
                name_length[name] += length

    rows, node_group, attrs_name, attrs_split = [], {}, {}, {}
    for name, edges in name_edges.items():
        H = nx.Graph()
        H.add_edges_from(edges)
        pieces = sorted((sorted(c) for c in nx.connected_components(H)), key=lambda p: p[0])
        coords = [np.array([xy[n] for n in p]) for p in pieces]
        max_gap, span = 0.0, 0.0
        parent = list(range(len(pieces)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        if len(pieces) > 1:
            trees = [cKDTree(c) for c in coords]
            for i in range(len(pieces)):
                gap_i = float('inf')
                for j in range(len(pieces)):
                    if i == j:
                        continue
                    dist = float(trees[j].query(coords[i], k=1)[0].min())
                    gap_i = min(gap_i, dist)
                    if dist <= split_gap:
                        parent[find(i)] = find(j)
                max_gap = max(max_gap, gap_i)
            allc = np.vstack(coords)
            span = float(np.hypot(allc[:, 0].max() - allc[:, 0].min(), allc[:, 1].max() - allc[:, 1].min()))
        groups = sorted(set(find(i) for i in range(len(pieces))), key=lambda g: pieces[g][0])
        group_no = {g: k + 1 for k, g in enumerate(groups)}
        for i, p in enumerate(pieces):
            key = name if len(groups) == 1 else f"{name} #{group_no[find(i)]}"
            for n in p:
                node_group[(name, n)] = key
        attrs_name[name] = {'pieces': len(pieces), 'max_gap_m': round(max_gap, 1)}
        for g in groups:
            key = name if len(groups) == 1 else f"{name} #{group_no[g]}"
            attrs_split[key] = {'pieces': sum(1 for i in range(len(pieces)) if find(i) == g)}
        rows.append({'name': name, 'pieces': len(pieces), 'groups_split': len(groups),
                     'length_m': round(name_length[name], 1), 'max_gap_m': round(max_gap, 1), 'span_m': round(span, 1)})

    df = pd.DataFrame(rows).sort_values(['max_gap_m', 'length_m'], ascending=False)
    df.to_csv(os.path.join(output_folder, 'name_pieces.csv'), index=False)
    total = df['length_m'].sum()
    lines = [f"Road names: {len(df)} (named length {total / 1000:.1f} km)",
             f"Names with one connected piece: {(df['pieces'] == 1).sum()}",
             f"Names with more than one piece: {(df['pieces'] > 1).sum()} "
             f"({100.0 * df.loc[df['pieces'] > 1, 'length_m'].sum() / max(total, 1e-9):.1f}% of the named length)"]
    for g in (50, 100, 500, 1000, 5000):
        sel = df['max_gap_m'] > g
        lines.append(f"Names with pieces more than {g} m apart: {sel.sum()} "
                     f"({100.0 * df.loc[sel, 'length_m'].sum() / max(total, 1e-9):.1f}% of the named length)")
    lines.append(f"Split gap used for name_split: {split_gap} m -> names split: {(df['groups_split'] > 1).sum()}, "
                 f"RoadOsm nodes {len(df)} -> {int(df['groups_split'].sum())}")
    lines.append("Largest gaps (name, pieces, max gap m, span m): " + "; ".join(
        f"{r.name} ({r.pieces}, {r.max_gap_m:.0f}, {r.span_m:.0f})" for r in df.head(10).itertuples()))
    lines.append(f"Time - name pieces: {time.time() - t:.2f} s")
    with open(os.path.join(output_folder, 'name_pieces_summary.txt'), 'w') as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print("Name pieces - " + line)

    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get('status') != 'active' or is_missing(d.get('name')):
            continue
        name = str(d.get('name'))
        d['name_split'] = node_group.get((name, u), node_group.get((name, v), name))
    return attrs_name, attrs_split


def assign_strokes(G, xy, angle_threshold):
    """Stroke (momepy COINS) of every edge; written as edge attribute 'stroke'"""
    import geopandas as gpd
    import momepy
    from shapely.geometry import LineString
    t = time.time()
    segs = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get('status') != 'active':
            continue
        seg = (min(u, v), max(u, v), str(d.get('osmway_id', '')))
        if seg not in segs and seg[0] in xy and seg[1] in xy and seg[0] != seg[1]:
            segs[seg] = LineString([xy[seg[0]], xy[seg[1]]])
    keys = list(segs)
    gdf = gpd.GeoDataFrame({'seg': range(len(keys))}, geometry=[segs[s] for s in keys])
    coins = momepy.COINS(gdf, angle_threshold=angle_threshold)
    stroke = coins.stroke_attribute()
    seg_stroke = {s: f"stroke_{int(stroke.iloc[i])}" for i, s in enumerate(keys)}
    for u, v, k, d in G.edges(keys=True, data=True):
        d['stroke'] = seg_stroke.get((min(u, v), max(u, v), str(d.get('osmway_id', ''))))
    print(f"Strokes: {len(set(seg_stroke.values()))} strokes from {len(keys)} segments ({time.time() - t:.1f} s)")
    return time.time() - t


def main(args=None):
    start = time.time()
    p = argparse.ArgumentParser(description='Dual graphs (name-based, name_split, strokes) from the primal graph CSV files.')
    p.add_argument('--nodes-csv', '-n', required=True)
    p.add_argument('--rels-csv', '-r', required=True)
    p.add_argument('--output', '-o', required=True)
    p.add_argument('--roundabout-hops', type=int, default=50)
    p.add_argument('--unnamed-hops', type=int, default=500)
    p.add_argument('--split-gap', type=float, default=500.0,
                   help='pieces of the same name more than this distance (m) apart become different roads (name_split)')
    p.add_argument('--no-split', action='store_true', help='do not build the name_split dual graph')
    p.add_argument('--no-strokes', action='store_true', help='do not build the stroke-based dual graph')
    p.add_argument('--stroke-angle', type=float, default=0.0, help='angle_threshold of momepy COINS (default 0)')
    o = p.parse_args(args=args)
    os.makedirs(o.output, exist_ok=True)
    for name in ('dual_nodes.csv', 'dual_edges.csv', 'dual_report.txt', 'edge_keys.csv', 'name_pieces.csv',
                 'name_pieces_summary.txt', 'dual_paper_results.txt'):
        path = os.path.join(o.output, name)
        if os.path.exists(path):
            os.remove(path)
    for sub in ('name_split', 'strokes'):
        if os.path.isdir(os.path.join(o.output, sub)):
            shutil.rmtree(os.path.join(o.output, sub))
    paper = [f"Dual graphs - results for the paper ({time.strftime('%Y-%m-%d %H:%M')})",
             f"Input: {o.nodes_csv}, {o.rels_csv}",
             f"Parameters: roundabout hops {o.roundabout_hops}, unnamed hops {o.unnamed_hops}, "
             f"split gap {o.split_gap:.0f} m, strokes {'no' if o.no_strokes else f'yes (COINS angle_threshold {o.stroke_angle})'}",
             ""]

    G, t_load = load_primal_graph(o.nodes_csv, o.rels_csv)
    xy = metric_xy(G)
    shares = primal_length_shares(G)
    attrs_name, attrs_split = name_pieces(G, xy, o.output, o.split_gap)

    main_dual = DualGraphGenerator(G, o.output, 'name', o.roundabout_hops, o.unnamed_hops, label='name-based')
    main_dual.extra_node_attrs = attrs_name
    main_dual.build()
    paper += main_dual.report(shares + [f"Time - loading the primal graph: {t_load:.2f} s"]) + [""]

    if not o.no_split:
        split = DualGraphGenerator(G, os.path.join(o.output, 'name_split'), 'name_split', o.roundabout_hops,
                                   o.unnamed_hops, label=f'name_split ({o.split_gap:.0f} m)')
        split.extra_node_attrs = attrs_split
        split.build()
        paper += split.report(shares) + [""]

    if not o.no_strokes:
        try:
            t_strokes = assign_strokes(G, xy, o.stroke_angle)
            strokes = DualGraphGenerator(G, os.path.join(o.output, 'strokes'), 'stroke', label='strokes',
                                         use_roundabout_rule=False, use_unnamed_rule=False)
            strokes.build()
            paper += strokes.report(shares + [f"Time - strokes (momepy COINS, angle_threshold {o.stroke_angle}): {t_strokes:.2f} s"]) + [""]
        except ImportError as e:
            print(f"Stroke-based dual graph skipped: {e} (pip install momepy)")

    rows = []
    for u, v, k, d in G.edges(keys=True, data=True):
        rows.append((u, v, d.get('osmway_id'), None if is_missing(d.get('name')) else str(d.get('name')),
                     d.get('name_split'), d.get('stroke')))
    pd.DataFrame(rows, columns=['source_id', 'target_id', 'osmway_id', 'name', 'name_split', 'stroke']).to_csv(
        os.path.join(o.output, 'edge_keys.csv'), index=False)
    print(f"Edge keys written: {len(rows)} primal edges -> edge_keys.csv")

    with open(os.path.join(o.output, 'name_pieces_summary.txt')) as f:
        paper += ["Name pieces (roads with the same name that are not connected):"] + f.read().splitlines() + [""]
    paper.append(f"Total execution time: {time.time() - start:.2f} seconds")
    with open(os.path.join(o.output, 'dual_paper_results.txt'), 'w') as f:
        f.write("\n".join(paper) + "\n")
    print(f"Results for the paper written to {os.path.join(o.output, 'dual_paper_results.txt')}")
    print(f"\nTotal execution time: {time.time() - start:.2f} seconds")


if __name__ == '__main__':
    main()
