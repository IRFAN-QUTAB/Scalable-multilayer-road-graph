import argparse
import json
import os

import pandas as pd

RULE = {'driveable_within_100m': 'R1', 'nearest_non_driveable': 'R2',
        'nearest_driveable_from_non_driveable': 'R3', 'nearest_driveable': 'R1 (no non-driveable junction)'}


def clean(df):
    cols = {}
    for c in df.columns:
        if c.startswith(':START_ID'):
            cols[c] = 'source'
        elif c.startswith(':END_ID'):
            cols[c] = 'target'
        elif c == ':TYPE':
            cols[c] = 'type'
        else:
            base, _, suffix = c.rpartition(':')
            if base and (suffix in ('STRING', 'DOUBLE', 'LONG', 'INT', 'FLOAT', 'BOOLEAN') or suffix.startswith('ID(')):
                cols[c] = base
            else:
                cols[c] = c
    df = df.rename(columns=cols)
    return df.loc[:, ~df.columns.duplicated()]


def norm(s):
    s = s.astype(str).str.strip()
    return s.where(~s.str.endswith('.0'), s.str[:-2])


def load_area(folder):
    nodes = clean(pd.read_csv(os.path.join(folder, 'combined_nodes.csv'), dtype=str, low_memory=False))
    edges = clean(pd.read_csv(os.path.join(folder, 'combined_edges.csv'), dtype=str, low_memory=False,
                              usecols=lambda c: c.startswith(':') or c.split(':')[0] in
                              ('distance', 'status', 'from_entrance')))
    for c in ('name', 'amenity', 'access', 'entrance', 'poi_type', 'node_type', 'lat', 'lon', 'x', 'y'):
        if c not in nodes.columns:
            nodes[c] = None
    if 'from_entrance' not in edges.columns:
        edges['from_entrance'] = None
    nodes['osmnode_id'] = norm(nodes['osmnode_id'])
    edges['source'] = norm(edges['source'])
    edges['target'] = norm(edges['target'])
    return nodes, edges


def main():
    p = argparse.ArgumentParser(description='Long POI links.')
    p.add_argument('--area', action='append', required=True, help='name=folder of the POI output (repeat)')
    p.add_argument('--output', required=True)
    p.add_argument('--min-distance', type=float, default=1000.0, help='links listed from this length (m)')
    p.add_argument('--report-distances', type=float, nargs='+', default=[100, 500, 1000, 2000],
                   help='distances (m) reported in the summary')
    o = p.parse_args()
    os.makedirs(o.output, exist_ok=True)

    all_links, all_far, summary, features = [], [], [], []
    for item in o.area:
        area, folder = item.split('=', 1)
        print(f"=== {area}: {folder}")
        nodes, edges = load_area(folder)
        info = nodes.set_index('osmnode_id')
        info = info[~info.index.duplicated()]
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

        nearest = links.groupby('poi')['distance'].min()
        nearest_drive = links[links['rule'].astype(str).str.startswith(('R1', 'R3'))].groupby('poi')['distance'].min()

        lines = [f"{area}: {len(pois)} POIs, {len(near)} NEAR links ({folder})"]
        for t in o.report_distances:
            n_links = int((near['distance'] >= t).sum())
            n_any = int(links.loc[links['distance'] >= t, 'poi'].nunique())
            n_far = int((nearest >= t).sum())
            n_far_d = int((nearest_drive >= t).sum())
            lines.append(f"   >= {t:g} m: {n_links} links ({100.0 * n_links / max(len(near), 1):.2f}%); POIs with at least "
                         f"one such link {n_any}; POIs whose nearest link is that long {n_far} "
                         f"({100.0 * n_far / max(len(pois), 1):.2f}%); POIs whose nearest DRIVEABLE link is that long "
                         f"{n_far_d} ({100.0 * n_far_d / max(len(pois), 1):.2f}%)")
        long = links[links['distance'] >= o.min_distance].copy()
        lines.append(f"   Links >= {o.min_distance:g} m by rule: " +
                     ", ".join(f"{k} {v}" for k, v in long['rule'].value_counts().items()))
        lines.append(f"   Links >= {o.min_distance:g} m by node: " +
                     ", ".join(f"{k} {v}" for k, v in long['kind'].value_counts().items()))
        far = nearest[nearest >= o.min_distance]
        far_info = info.reindex(far.index)
        lines.append(f"   POIs whose nearest link is >= {o.min_distance:g} m, by amenity: " +
                     ", ".join(f"{k} {v}" for k, v in far_info['amenity'].fillna('?').value_counts().head(15).items()))
        lines.append(f"   ... by access tag: " +
                     ", ".join(f"{k} {v}" for k, v in far_info['access'].fillna('(none)').value_counts().head(8).items()))
        summary += lines + [""]
        print("\n".join(lines))

        def coord(ids, a, b):
            return ids.map(lambda i: info.at[i, a] if i in info.index else None)
        long['area'] = area
        long['poi_type'] = long['poi'].map(lambda i: info.at[i, 'poi_type'] if i in info.index else None)
        long['name'] = long['poi'].map(lambda i: info.at[i, 'name'] if i in info.index else None)
        long['amenity'] = long['poi'].map(lambda i: info.at[i, 'amenity'] if i in info.index else None)
        long['access'] = long['poi'].map(lambda i: info.at[i, 'access'] if i in info.index else None)
        long['from_lat'] = coord(long['source'], 'lat', None)
        long['from_lon'] = coord(long['source'], 'lon', None)
        long['junction_lat'] = coord(long['target'], 'y', None)
        long['junction_lon'] = coord(long['target'], 'x', None)
        long['poi_nearest_link_m'] = long['poi'].map(nearest)
        long['poi_nearest_driveable_link_m'] = long['poi'].map(nearest_drive)
        cols = ['area', 'poi', 'poi_type', 'name', 'amenity', 'access', 'source', 'kind', 'rule', 'distance',
                'poi_nearest_link_m', 'poi_nearest_driveable_link_m', 'from_lat', 'from_lon', 'target', 'junction_lat', 'junction_lon']
        all_links.append(long[cols].sort_values('distance', ascending=False))
        fp = pd.DataFrame({'area': area, 'poi': far.index, 'nearest_link_m': far.values,
                           'nearest_driveable_link_m': nearest_drive.reindex(far.index).values})
        fp = fp.join(far_info[['poi_type', 'name', 'amenity', 'access', 'lat', 'lon']].reset_index(drop=True))
        all_far.append(fp.sort_values('nearest_link_m', ascending=False))
        for r in long.itertuples():
            try:
                line = [[float(r.from_lon), float(r.from_lat)], [float(r.junction_lon), float(r.junction_lat)]]
            except (TypeError, ValueError):
                continue
            features.append({'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': line},
                             'properties': {'area': area, 'poi': r.poi, 'name': r.name if isinstance(r.name, str) else None,
                                            'amenity': r.amenity if isinstance(r.amenity, str) else None,
                                            'kind': r.kind, 'rule': r.rule, 'distance_m': round(float(r.distance), 1),
                                            'poi_nearest_link_m': round(float(r.poi_nearest_link_m), 1)}})

    pd.concat(all_links, ignore_index=True).to_csv(os.path.join(o.output, 'long_links.csv'), index=False)
    pd.concat(all_far, ignore_index=True).to_csv(os.path.join(o.output, 'far_pois.csv'), index=False)
    with open(os.path.join(o.output, 'long_links.geojson'), 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': features}, f)
    with open(os.path.join(o.output, 'long_links_summary.txt'), 'w') as f:
        f.write(f"Long POI links (NEAR) - minimum distance for the lists: {o.min_distance:g} m\n\n" + "\n".join(summary))
    print(f"Written to {o.output}: long_links.csv, far_pois.csv, long_links.geojson, long_links_summary.txt")


if __name__ == '__main__':
    main()
