import argparse
import json
import math
import os
import time

import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd

from poi_long_links import load_area, RULE

DEFAULT_TYPES = ['fence', 'wall', 'hedge', 'retaining_wall', 'city_wall']


def coordinates(nodes):
    """(lon, lat) of every node: POI rows have lat/lon, road junction rows have y/x"""
    lat = pd.to_numeric(nodes['lat'], errors='coerce').fillna(pd.to_numeric(nodes['y'], errors='coerce'))
    lon = pd.to_numeric(nodes['lon'], errors='coerce').fillna(pd.to_numeric(nodes['x'], errors='coerce'))
    xy = pd.DataFrame({'lon': lon.values, 'lat': lat.values}, index=nodes['osmnode_id'].values)
    return xy[~xy.index.duplicated()]


def build_links(nodes, edges):
    """Every NEAR link with the POI it belongs to (same rules as poi_long_links.py)"""
    pois = nodes[nodes['poi_type'].isin(['OSMNode', 'OSMWay'])]
    poi_ids = set(pois['osmnode_id'])
    way_ids = set(pois.loc[pois['poi_type'] == 'OSMWay', 'osmnode_id'])
    near = edges[edges['type'] == 'NEAR'].copy()
    near['distance'] = pd.to_numeric(near['distance'], errors='coerce')
    part_of = edges[edges['type'] == 'PART_OF'][['source', 'target']].rename(
        columns={'source': 'member', 'target': 'poi'})
    direct = near[near['source'].isin(poi_ids)].assign(poi=lambda d: d['source'])
    direct['kind'] = direct['source'].map(lambda s: 'building centroid' if s in way_ids else 'POI node')
    member = near[~near['source'].isin(poi_ids)].merge(part_of, left_on='source', right_on='member', how='inner')
    member = member.drop(columns='member')
    member['kind'] = member['from_entrance'].map(lambda v: 'entrance node' if str(v) == 'True' else 'outline node')
    links = pd.concat([direct, member], ignore_index=True)
    links['rule'] = links['status'].map(RULE).fillna(links['status'])
    return links, poi_ids


def overpass_url(url):
    url = url.rstrip('/')
    return url if url.endswith('/interpreter') else url + '/interpreter'


def download_barriers(url, bbox, types, tile_km, retries=5):
    """Barrier ways inside bbox (min_lat, min_lon, max_lat, max_lon), downloaded tile by tile.
    Returns {way id: (barrier type, [(lon, lat), ...])}."""
    min_lat, min_lon, max_lat, max_lon = bbox
    dlat = tile_km / 111.195
    dlon = tile_km / (111.195 * math.cos(math.radians((min_lat + max_lat) / 2)))
    n_lat = max(1, math.ceil((max_lat - min_lat) / dlat))
    n_lon = max(1, math.ceil((max_lon - min_lon) / dlon))
    regex = '^(' + '|'.join(types) + ')$'
    ways = {}
    for i in range(n_lat):
        for j in range(n_lon):
            s, w = min_lat + i * dlat, min_lon + j * dlon
            n, e = min(max_lat, s + dlat), min(max_lon, w + dlon)
            query = f'[out:json][timeout:900];way["barrier"~"{regex}"]({s},{w},{n},{e});out geom;'
            for attempt in range(1, retries + 1):
                try:
                    body = urllib.parse.urlencode({'data': query}).encode()
                    with urllib.request.urlopen(overpass_url(url), data=body, timeout=1000) as r:
                        elements = json.loads(r.read().decode())['elements']
                    break
                except Exception as ex:
                    if attempt == retries:
                        raise RuntimeError(f"tile {i},{j} failed after {retries} attempts: {ex}")
                    time.sleep(5)
            for el in elements:
                geom = el.get('geometry') or []
                if el.get('type') == 'way' and len(geom) >= 2:
                    ways[el['id']] = (el.get('tags', {}).get('barrier'), [(g['lon'], g['lat']) for g in geom])
    return ways, n_lat * n_lon


def barrier_segments(ways):
    """All segments of the barrier lines as arrays x1, y1, x2, y2 (lon, lat) and the barrier type of each"""
    x1, y1, x2, y2, kind = [], [], [], [], []
    for btype, pts in ways.values():
        for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
            if (a, b) != (c, d):
                x1.append(a); y1.append(b); x2.append(c); y2.append(d); kind.append(btype)
    return (np.array(x1), np.array(y1), np.array(x2), np.array(y2)), np.array(kind, dtype=object)


def grid_index(seg, cell):
    """Grid of square cells (degrees): cell -> indices of the segments whose bounding box touches the cell"""
    x1, y1, x2, y2 = seg
    grid = defaultdict(list)
    i0 = np.floor(np.minimum(x1, x2) / cell).astype(int); i1 = np.floor(np.maximum(x1, x2) / cell).astype(int)
    j0 = np.floor(np.minimum(y1, y2) / cell).astype(int); j1 = np.floor(np.maximum(y1, y2) / cell).astype(int)
    for k in range(len(x1)):
        for i in range(i0[k], i1[k] + 1):
            for j in range(j0[k], j1[k] + 1):
                grid[(i, j)].append(k)
    return grid


def proper_crossings(px, py, qx, qy, seg, idx):
    """Indices (in idx) of the segments that the segment p-q crosses at an inner point of both segments"""
    x1, y1, x2, y2 = (a[idx] for a in seg)
    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d1 = orient(x1, y1, x2, y2, px, py)
    d2 = orient(x1, y1, x2, y2, qx, qy)
    d3 = orient(px, py, qx, qy, x1, y1)
    d4 = orient(px, py, qx, qy, x2, y2)
    hit = (d1 * d2 < 0) & (d3 * d4 < 0)
    return idx[hit]


def pct(a, b):
    return f"{a} ({100.0 * a / b:.2f}%)" if b else f"{a}"


def main():
    p = argparse.ArgumentParser(description='Barriers between POIs and their road junctions.')
    p.add_argument('--area', action='append', required=True, help='name=folder of the POI output (repeat)')
    p.add_argument('--overpass-url', required=True, help='Overpass server, e.g. http://localhost:12351/api/interpreter')
    p.add_argument('--output', required=True)
    p.add_argument('--barrier-types', nargs='+', default=DEFAULT_TYPES, help='values of the OSM tag barrier=*')
    p.add_argument('--tile-km', type=float, default=10.0, help='size of the download tiles (km)')
    o = p.parse_args()
    os.makedirs(o.output, exist_ok=True)

    summary, rows, features = [], [], []
    summary.append(f"Barrier types (OSM barrier=*, ways): {', '.join(o.barrier_types)}")
    summary.append("A link crosses a barrier when its straight line crosses a mapped barrier line "
                   "(touching at one end is not counted).")
    for item in o.area:
        area, folder = item.split('=', 1)
        print(f"=== {area}: {folder}")
        nodes, edges = load_area(folder)
        xy = coordinates(nodes)
        links, poi_ids = build_links(nodes, edges)
        links = links.join(xy.rename(columns={'lon': 'lon1', 'lat': 'lat1'}), on='source')
        links = links.join(xy.rename(columns={'lon': 'lon2', 'lat': 'lat2'}), on='target')
        missing = int(links[['lon1', 'lat1', 'lon2', 'lat2']].isna().any(axis=1).sum())
        links = links.dropna(subset=['lon1', 'lat1', 'lon2', 'lat2']).reset_index(drop=True)

        margin = 0.001
        bbox = (min(links['lat1'].min(), links['lat2'].min()) - margin, min(links['lon1'].min(), links['lon2'].min()) - margin,
                max(links['lat1'].max(), links['lat2'].max()) + margin, max(links['lon1'].max(), links['lon2'].max()) + margin)
        t0 = time.time()
        ways, n_tiles = download_barriers(o.overpass_url, bbox, o.barrier_types, o.tile_km)
        print(f"{len(ways)} barrier lines in {n_tiles} tiles ({time.time() - t0:.1f} s)")
        seg, seg_type = barrier_segments(ways)
        cell = 0.002
        grid = grid_index(seg, cell) if len(seg_type) else {}
        crosses = np.zeros(len(links), dtype=bool)
        first_type = np.array([None] * len(links), dtype=object)
        ax, ay = links['lon1'].to_numpy(), links['lat1'].to_numpy()
        bx, by = links['lon2'].to_numpy(), links['lat2'].to_numpy()
        for k in range(len(links)):
            if not grid or (ax[k] == bx[k] and ay[k] == by[k]):
                continue
            i0, i1 = sorted((math.floor(ax[k] / cell), math.floor(bx[k] / cell)))
            j0, j1 = sorted((math.floor(ay[k] / cell), math.floor(by[k] / cell)))
            cand = set()
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    cand.update(grid.get((i, j), ()))
            if not cand:
                continue
            hit = proper_crossings(ax[k], ay[k], bx[k], by[k], seg, np.fromiter(cand, dtype=int))
            if len(hit):
                crosses[k] = True
                first_type[k] = seg_type[hit.min()]
        barrier_type = [ways[w][0] for w in ways]
        links['crosses_barrier'] = crosses
        links['barrier_type'] = first_type

        n = len(links)
        summary.append(f"\n=== {area} ({folder})")
        summary.append(f"Barrier lines downloaded: {len(ways)} (" + ", ".join(
            f"{t} {c}" for t, c in pd.Series(barrier_type).value_counts().items()) + ")")
        summary.append(f"NEAR links: {n} (links without coordinates skipped: {missing}); "
                       f"crossing a barrier: {pct(int(crosses.sum()), n)}")
        for rule, g in links.groupby('rule'):
            summary.append(f"   {rule}: {pct(int(g['crosses_barrier'].sum()), len(g))} of {len(g)} links")
        for kind, g in links.groupby('kind'):
            summary.append(f"   from {kind}: {pct(int(g['crosses_barrier'].sum()), len(g))} of {len(g)} links")
        summary.append("   by barrier type: " + ", ".join(
            f"{t} {c}" for t, c in links.loc[links['crosses_barrier'], 'barrier_type'].value_counts().items()))

        n_pois = len(poi_ids)
        free = links[~links['crosses_barrier']]
        drive = links['rule'].astype(str).str.startswith(('R1', 'R3'))
        free_drive = links[drive & ~links['crosses_barrier']]
        with_links = links['poi'].nunique()
        summary.append(f"POIs: {n_pois}; with at least one link: {with_links}")
        summary.append(f"POIs with at least one link that crosses no barrier: {pct(free['poi'].nunique(), n_pois)}")
        summary.append(f"POIs with at least one driveable link (R1 or R3) that crosses no barrier: "
                       f"{pct(free_drive['poi'].nunique(), n_pois)}")
        summary.append(f"POIs whose links all cross a barrier: {pct(with_links - free['poi'].nunique(), n_pois)}")
        nearest = links.loc[links.groupby('poi')['distance'].idxmin()]
        summary.append(f"POIs whose nearest link crosses a barrier: {pct(int(nearest['crosses_barrier'].sum()), n_pois)}")

        cross = links[links['crosses_barrier']]
        info = nodes.set_index('osmnode_id')
        info = info[~info.index.duplicated()]
        for r in cross.itertuples():
            rows.append({'area': area, 'poi': r.poi, 'amenity': info['amenity'].get(r.poi),
                         'name': info['name'].get(r.poi), 'from_node': r.source, 'kind': r.kind, 'rule': r.rule,
                         'junction': r.target, 'distance_m': round(float(r.distance), 1),
                         'barrier_type': r.barrier_type})
            features.append({'type': 'Feature',
                             'geometry': {'type': 'LineString', 'coordinates': [[r.lon1, r.lat1], [r.lon2, r.lat2]]},
                             'properties': {'area': area, 'poi': r.poi, 'kind': r.kind, 'rule': r.rule,
                                            'distance_m': round(float(r.distance), 1), 'barrier_type': r.barrier_type}})
        print("\n".join(summary[-12:]))

    pd.DataFrame(rows).to_csv(os.path.join(o.output, 'crossing_links.csv'), index=False)
    with open(os.path.join(o.output, 'crossing_links.geojson'), 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': features}, f)
    with open(os.path.join(o.output, 'barrier_summary.txt'), 'w') as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nWritten: {o.output}/barrier_summary.txt, crossing_links.csv, crossing_links.geojson")


if __name__ == '__main__':
    main()
