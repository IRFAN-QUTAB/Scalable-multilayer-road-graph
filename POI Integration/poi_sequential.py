import overpy
import json
import argparse
import os
import time
import math
import multiprocessing
import pandas as pd
from typing import List, Tuple, Dict
import threading
from scipy.spatial import KDTree
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import psutil
except ImportError:
    psutil = None


KM_PER_DEGREE = math.pi * 6371.009 / 180

_GENERATOR = None


def _process_cell(args):
    lat, lon, section_num, area_size_km = args
    return _GENERATOR.process_single_section(lat, lon, section_num, area_size_km)


_ROAD_TEXT = None


def _road_chunk_text(bounds):
    start, end = bounds
    table, suffix = _ROAD_TEXT
    text = table.iloc[start:end].to_csv(index=False, header=False, lineterminator='\n')
    return os.linesep.join(line + suffix for line in text.split('\n')[:-1]) + os.linesep


def start_memory_monitor(interval=0.5):
    """Peak memory (resident set size) of this process and its child processes, sampled every interval s:
    [sum over all processes, largest single process] in bytes (same method as the road network script)."""
    peak = [0, 0]
    if psutil is None:
        return peak, None
    stop = threading.Event()
    proc = psutil.Process(os.getpid())

    def sample():
        while not stop.is_set():
            try:
                sizes = [proc.memory_info().rss]
                for c in proc.children(recursive=True):
                    try:
                        sizes.append(c.memory_info().rss)
                    except psutil.Error:
                        pass
                peak[0] = max(peak[0], sum(sizes))
                peak[1] = max(peak[1], max(sizes))
            except psutil.Error:
                pass
            stop.wait(interval)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    return peak, stop

class NetworkXPOIGenerator:
    def __init__(self, nodes_csv: str, rels_csv: str, output_folder: str, overpass_url: str = None,
                 max_workers: int = 4, driveable_scc: bool = True):
        self.nodes_csv = nodes_csv
        self.rels_csv = rels_csv
        self.output_folder = output_folder

        if overpass_url:
            self.overpass_api = overpy.Overpass(url=overpass_url)
            print(f"Using custom Overpass API at: {overpass_url}")
        else:
            self.overpass_api = overpy.Overpass()
            print("Using default Overpass API")

        self.max_workers = max_workers
        self.overpass_url = overpass_url
        self.driveable_scc = driveable_scc
        self.use_rate_limit = overpass_url is None
        self.mp_context = multiprocessing.get_context('fork')
        self.rate_limit_lock = self.mp_context.Lock()
        self.last_api_call = self.mp_context.Value('d', 0.0)
        self.data_lock = threading.Lock()
        self.min_api_interval = 5.0
        self.max_retries = 5
        self.node_driveable = {}
        self.phase_times = []

        self.osmnode_poi_nodes = []
        self.osmnode_way_nodes = []
        self.osmway_poi_nodes = []
        self.tag_nodes = []

        self.near_edges = []
        self.part_of_edges = []
        self.tags_edges = []

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _flatten_value(self, value):
        """Flatten list values and clean problematic characters for CSV"""
        if isinstance(value, list):
            value = ";".join(str(v) for v in value)
        if isinstance(value, str):
            value = value.replace('\n', ' ').replace('\r', ' ')
            value = value.replace('"', "'")
        return value

    NUMERIC_DOUBLE_FIELDS = {'lat', 'lon', 'x', 'y', 'distance', 'length', 'street_count'}
    NUMERIC_LONG_FIELDS = set()

    def _build_neo4j_header(self, df, id_col=None, id_space=None,
                            start_id_col=None, end_id_col=None,
                            start_id_space=None, end_id_space=None):
        """Build neo4j-admin import compatible header from DataFrame columns"""
        headers = []
        for col in df.columns:
            if id_col and col == id_col:
                if id_space:
                    headers.append(f'{col}:ID({id_space})')
                else:
                    headers.append('id:ID')
            elif start_id_col and col == start_id_col:
                if start_id_space:
                    headers.append(f':START_ID({start_id_space})')
                else:
                    headers.append(':START_ID')
            elif end_id_col and col == end_id_col:
                if end_id_space:
                    headers.append(f':END_ID({end_id_space})')
                else:
                    headers.append(':END_ID')
            elif col == 'edge_type':
                headers.append(':TYPE')
            elif col in self.NUMERIC_DOUBLE_FIELDS:
                headers.append(f"{col}:DOUBLE")
            elif col in self.NUMERIC_LONG_FIELDS:
                headers.append(f"{col}:LONG")
            else:
                headers.append(f"{col}:STRING")
        return headers

    def _save_csv_with_neo4j_headers(self, df, filepath,
                                     id_col=None, id_space=None,
                                     start_id_col=None, end_id_col=None,
                                     start_id_space=None, end_id_space=None):
        """Save DataFrame as CSV with neo4j-admin import headers"""
        headers = self._build_neo4j_header(df, id_col=id_col, id_space=id_space,
                                           start_id_col=start_id_col, end_id_col=end_id_col,
                                           start_id_space=start_id_space, end_id_space=end_id_space)
        with open(filepath, 'w', newline='') as f:
            f.write(','.join(headers) + '\n')

        for col in df.columns:
            df[col] = df[col].apply(
                lambda v: str(v).replace('\n', ' ').replace('\r', ' ').replace('"', "'") if isinstance(v, str) else v)

        df.to_csv(filepath, mode='a', index=False, header=False)

    @staticmethod
    def _clean_value(v):
        """text values without line breaks and double quotes (as in the CSV files of the road network)"""
        return str(v).replace('\n', ' ').replace('\r', ' ').replace('"', "'") if isinstance(v, str) else v

    def _clean_column(self, s):
        """Same result as s.apply(self._clean_value), computed only on the non-empty values: only the text
        values that contain a line break or a double quote are changed. A text column keeps its type; a column
        without text values is passed to apply, so that pandas gives it the same type as before."""
        if not (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
            return s
        vals = s.to_numpy(dtype=object)
        changes, has_text = [], False
        for i in np.flatnonzero(pd.notna(vals)):
            v = vals[i]
            if isinstance(v, str):
                has_text = True
                if '\n' in v or '\r' in v or '"' in v:
                    changes.append((i, self._clean_value(v)))
        if not has_text:
            return s.apply(self._clean_value)
        if changes:
            vals = vals.copy()
            for i, v in changes:
                vals[i] = v
            s = pd.Series(vals, index=s.index, name=s.name, dtype=s.dtype)
        return s

    @staticmethod
    def _csv_fields(s):
        """the values of a column written as CSV fields, exactly as DataFrame.to_csv writes them"""
        if len(s) == 0:
            return []
        tmp = pd.DataFrame({'_': np.zeros(len(s), dtype=np.int8), 'v': s.reset_index(drop=True)})
        lines = tmp.to_csv(index=False, header=False, lineterminator='\n').split('\n')[:-1]
        return [line[2:] for line in lines]

    def _write_combined_csv(self, road_df, poi_df, filepath, transforms=None, **header_kwargs):
        """Write the road rows followed by the POI rows (the same file as pd.concat([road_df, poi_df]) written with
        _save_csv_with_neo4j_headers after the transforms), without building the table of all rows and all OSM
        tag columns: the road rows are written with the columns of the road network, the POI rows only with
        their non-empty values. transforms: {column: function applied to the whole column after the concat}."""
        transforms = transforms or {}
        n0 = len(road_df)
        if poi_df is None:
            poi_df = road_df.iloc[:0]
        n1 = len(poi_df)
        columns = list(road_df.columns) + [c for c in poi_df.columns if c not in road_df.columns]
        road_cols = set(road_df.columns)

        def column(c):
            right = poi_df[[c]] if c in poi_df.columns else poi_df.iloc[:, :0]
            if c in road_cols or c in transforms or n0 == 0:
                left = road_df[[c]] if c in road_cols else road_df.iloc[:, :0]
                col = pd.concat([left, right], ignore_index=True)[c]
                if c in transforms:
                    col = transforms[c](col)
                return col.apply(self._clean_value), n0
            col = pd.concat([road_df.iloc[:1, :0], right], ignore_index=True)[c]
            return self._clean_column(col), 1

        header = self._build_neo4j_header(pd.DataFrame(columns=columns), **header_kwargs)
        n_cols = len(columns)
        k = len(road_df.columns)
        road_part, suffix = {}, ''
        poi_rows = [[] for _ in range(n1)]
        for pos, c in enumerate(columns):
            col, n_road = column(c)
            if c in road_cols:
                road_part[c] = col.iloc[:n0].reset_index(drop=True)
            elif n0:
                road_vals = col.iloc[:n_road]
                if road_vals.isna().all():
                    suffix += ','
                else:
                    fields = set(self._csv_fields(road_vals.drop_duplicates()))
                    if len(fields) != 1 or road_vals.isna().any():
                        raise ValueError(f"column {c}: road rows with different values")
                    suffix += ',' + fields.pop()
            tail = col.iloc[n_road:]
            mask = tail.notna().to_numpy()
            if mask.any():
                for i, f in zip(np.flatnonzero(mask), self._csv_fields(tail[mask])):
                    poi_rows[i].append((pos, f))
            del col

        with open(filepath, 'w', newline='') as f:
            f.write(','.join(header) + '\n')
            if n0:
                global _ROAD_TEXT
                _ROAD_TEXT = (pd.DataFrame(road_part, columns=list(road_df.columns)), suffix)
                del road_part
                n_blocks = max(1, self.max_workers) * 4
                size = -(-n0 // n_blocks)
                blocks = [(i, min(i + size, n0)) for i in range(0, n0, size)]
                if self.max_workers > 1 and len(blocks) > 1:
                    with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=self.mp_context) as ex:
                        for text in ex.map(_road_chunk_text, blocks):
                            f.write(text)
                else:
                    for b in blocks:
                        f.write(_road_chunk_text(b))
                _ROAD_TEXT = None
            out = []
            for fields in poi_rows:
                text, cur = [], 0
                for pos, value in fields:
                    text.append(',' * (pos - cur))
                    text.append(value)
                    cur = pos
                text.append(',' * (n_cols - 1 - cur))
                out.append(''.join(text))
                if len(out) >= 100000:
                    f.write(os.linesep.join(out) + os.linesep)
                    out = []
            if out:
                f.write(os.linesep.join(out) + os.linesep)
        return n0 + n1

    def load_graph_from_csv(self):
        """Read the road junctions (id, coordinates) and the road segments (start, end, driveable) from the CSV
        files of the road network. Only these columns are needed to link the POIs."""
        print("Loading road graph from CSV files...")

        nodes_df = pd.read_csv(self.nodes_csv, low_memory=False)
        print(f"Loaded {len(nodes_df)} road nodes")

        rels_df = pd.read_csv(self.rels_csv, low_memory=False)
        print(f"Loaded {len(rels_df)} road relationships")

        lat_col = 'lat' if 'lat' in nodes_df.columns else ('y' if 'y' in nodes_df.columns else None)
        lon_col = 'lon' if 'lon' in nodes_df.columns else ('x' if 'x' in nodes_df.columns else None)
        ids = nodes_df['osmnode_id'].tolist()
        lats = nodes_df[lat_col].tolist() if lat_col else [None] * len(ids)
        lons = nodes_df[lon_col].tolist() if lon_col else [None] * len(ids)
        self.junction_coords = list(zip(ids, lats, lons))

        self.node_driveable = dict.fromkeys(ids, False)
        sources = rels_df['source_id'].tolist()
        targets = rels_df['target_id'].tolist()
        for u, v in zip(sources, targets):
            if u not in self.node_driveable:
                self.node_driveable[u] = False
            if v not in self.node_driveable:
                self.node_driveable[v] = False
        if 'driveable' in rels_df.columns:
            flags = [d is True or d == 'True' or d == 'true' for d in rels_df['driveable'].tolist()]
        else:
            flags = [False] * len(sources)
        self.drive_edges = [(u, v) for u, v, f in zip(sources, targets, flags) if f]

        print(f"Graph loaded: {len(self.node_driveable)} nodes, {len(rels_df)} edges")
        del nodes_df, rels_df
        self.mark_driveable_roadjunctions()

    def mark_driveable_roadjunctions(self):
        """Mark which road junctions are driveable.
        A junction touched by a driveable edge is driveable. With driveable_scc (default), only the junctions
        of the largest strongly connected component of the driveable network are kept as driveable, so that
        from every POI linked to a driveable junction a car can reach every other driveable junction and come
        back. The other junctions are treated as non-driveable (rules R2 and R3)."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        print("Marking driveable road junctions...")

        drive_edges = self.drive_edges
        for u, v in drive_edges:
            self.node_driveable[u] = True
            self.node_driveable[v] = True

        touched = [n for n, d in self.node_driveable.items() if d]
        self.driveable_stats = {'touched': len(touched), 'largest_scc': len(touched), 'scc_count': None}
        print(f"Junctions on driveable edges: {len(touched)}")

        if self.driveable_scc and drive_edges:
            index = {n: i for i, n in enumerate(touched)}
            rows = [index[u] for u, v in drive_edges]
            cols = [index[v] for u, v in drive_edges]
            A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(touched), len(touched)))
            n_comp, labels = connected_components(A, directed=True, connection='strong')
            largest = np.bincount(labels).argmax()
            for n, i in index.items():
                if labels[i] != largest:
                    self.node_driveable[n] = False
            kept = int((labels == largest).sum())
            self.driveable_stats.update({'largest_scc': kept, 'scc_count': int(n_comp)})
            print(f"Largest strongly connected component of the driveable network: {kept} junctions "
                  f"({len(touched) - kept} junctions on driveable edges outside it are treated as non-driveable)")

        self.drive_edges = None
        driveable_count = sum(1 for d in self.node_driveable.values() if d)
        print(f"Marked {driveable_count} driveable road junctions")

    def _xy(self, lat, lon):
        """Local coordinates in meters, so that KD-tree distances are in meters."""
        return [lat * 111320.0, lon * 111320.0 * math.cos(math.radians(self.ref_lat))]

    def build_spatial_index(self):
        """Build KD-tree for efficient spatial queries"""
        print("Building spatial index...")

        self.driveable_nodes = []
        self.driveable_coords = []
        self.non_driveable_nodes = []
        self.non_driveable_coords = []

        for node, lat, lon in self.junction_coords:
            if lat is not None and lon is not None:
                try:
                    lat = float(lat)
                    lon = float(lon)
                except (ValueError, TypeError):
                    continue
                if self.node_driveable.get(node):
                    self.driveable_nodes.append(node)
                    self.driveable_coords.append([lat, lon])
                else:
                    self.non_driveable_nodes.append(node)
                    self.non_driveable_coords.append([lat, lon])

        all_lats = [c[0] for c in self.driveable_coords + self.non_driveable_coords]
        self.ref_lat = sum(all_lats) / len(all_lats) if all_lats else 0.0

        if self.driveable_coords:
            self.driveable_tree = KDTree([self._xy(a, b) for a, b in self.driveable_coords])
            print(f"Built spatial index for {len(self.driveable_coords)} driveable nodes")
        else:
            self.driveable_tree = None
            print("No driveable nodes found")

        if self.non_driveable_coords:
            self.non_driveable_tree = KDTree([self._xy(a, b) for a, b in self.non_driveable_coords])
            print(f"Built spatial index for {len(self.non_driveable_coords)} non-driveable nodes")
        else:
            self.non_driveable_tree = None

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def way_centroid(coords):
        """Centroid of an OSM way from its (lat, lon) node coordinates.
        Closed way (building, area): centroid of the polygon (area-weighted, shoelace formula), computed in
        local metric coordinates. Open way or polygon without area: mean of the distinct nodes.
        Returns (lat, lon) or None if the way has no nodes."""
        if not coords:
            return None
        lat0 = sum(c[0] for c in coords) / len(coords)
        lon0 = sum(c[1] for c in coords) / len(coords)
        k = math.cos(math.radians(lat0))
        pts = [((lon - lon0) * k, lat - lat0) for lat, lon in coords]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            a = cx = cy = 0.0
            for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                cross = x1 * y2 - x2 * y1
                a += cross
                cx += (x1 + x2) * cross
                cy += (y1 + y2) * cross
            if abs(a) > 1e-14:
                cx /= 3 * a
                cy /= 3 * a
                return (lat0 + cy, lon0 + cx / k)
        distinct = list(dict.fromkeys(coords))
        return (sum(c[0] for c in distinct) / len(distinct), sum(c[1] for c in distinct) / len(distinct))

    def find_nearest_roadjunction(self, poi_lat, poi_lon):
        """Find nearest road junction for a node using spatial index"""
        connections = []

        if self.driveable_tree:
            indices = self.driveable_tree.query_ball_point(self._xy(poi_lat, poi_lon), r=100)
            if indices:
                candidates = []
                for idx in indices:
                    rj_node = self.driveable_nodes[idx]
                    rj_lat, rj_lon = self.driveable_coords[idx]
                    dist = self.haversine_distance(poi_lat, poi_lon, rj_lat, rj_lon)
                    if dist <= 100:
                        candidates.append((rj_node, dist, 'driveable_within_100m'))
                if candidates:
                    nearest = min(candidates, key=lambda x: x[1])
                    connections.append(nearest)

        if not connections and self.non_driveable_tree:
            dist, idx = self.non_driveable_tree.query(self._xy(poi_lat, poi_lon))
            if idx < len(self.non_driveable_nodes):
                rj_node = self.non_driveable_nodes[idx]
                rj_lat, rj_lon = self.non_driveable_coords[idx]
                actual_dist = self.haversine_distance(poi_lat, poi_lon, rj_lat, rj_lon)
                connections.append((rj_node, actual_dist, 'nearest_non_driveable'))

        if not connections and self.driveable_tree:
            dist, idx = self.driveable_tree.query(self._xy(poi_lat, poi_lon))
            if idx < len(self.driveable_nodes):
                rj_node = self.driveable_nodes[idx]
                rj_lat, rj_lon = self.driveable_coords[idx]
                actual_dist = self.haversine_distance(poi_lat, poi_lon, rj_lat, rj_lon)
                connections.append((rj_node, actual_dist, 'nearest_driveable'))

        return connections

    @staticmethod
    def km_to_degrees(km_lat, km_lon, reference_lat):
        degrees_lat = km_lat / KM_PER_DEGREE
        degrees_lon = km_lon / (KM_PER_DEGREE * math.cos(math.radians(reference_lat)))
        return degrees_lat, degrees_lon

    def get_processed_sections(self):
        processed_file = os.path.join(self.output_folder, "processed_poi_sections.txt")
        processed_sections = set()
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                for line in f:
                    try:
                        processed_sections.add(int(line.strip()))
                    except ValueError:
                        continue
        return processed_sections

    def save_processed_section(self, section_num):
        processed_file = os.path.join(self.output_folder, "processed_poi_sections.txt")
        with self.data_lock:
            with open(processed_file, 'a') as f:
                f.write(f"{section_num}\n")

    @staticmethod
    def study_area_bbox(center_lat, center_lon, total_distance_km):
        """Study area as in OSMnx bbox_from_point (Earth radius 6,371,009 m), the same square as the
        road network graph: (min_lat, min_lon, max_lat, max_lon)"""
        d_lat = math.degrees(total_distance_km * 1000 / 6371009)
        d_lon = d_lat / math.cos(math.radians(center_lat))
        return (center_lat - d_lat, center_lon - d_lon, center_lat + d_lat, center_lon + d_lon)

    def calculate_grid_points(self, center_lat, center_lon, total_distance_km, area_size_km):
        points = []
        step_size_km = area_size_km
        steps_needed = max(1, math.ceil(2 * total_distance_km / step_size_km))
        start_offset_km = -(steps_needed - 1) * step_size_km / 2
        for i in range(steps_needed):
            for j in range(steps_needed):
                offset_lat_km = start_offset_km + i * step_size_km
                offset_lon_km = start_offset_km + j * step_size_km
                deg_lat, deg_lon = self.km_to_degrees(offset_lat_km, offset_lon_km, center_lat)
                new_lat = center_lat + deg_lat
                new_lon = center_lon + deg_lon
                points.append((new_lat, new_lon))
        return points

    def rate_limited_overpass_query(self, query):
        if not self.use_rate_limit:
            return self.overpass_api.query(query)
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_api_call.value
            if time_since_last < self.min_api_interval:
                time.sleep(self.min_api_interval - time_since_last)
            try:
                return self.overpass_api.query(query)
            finally:
                self.last_api_call.value = time.time()

    def query_section_data(self, lat, lon, area_size_km, section_num):
        delta_lat, delta_lon = self.km_to_degrees(area_size_km / 2, area_size_km / 2, lat)
        min_lat = lat - delta_lat
        max_lat = lat + delta_lat
        min_lon = lon - delta_lon
        max_lon = lon + delta_lon
        if getattr(self, 'study_bbox', None):
            min_lat = max(min_lat, self.study_bbox[0])
            min_lon = max(min_lon, self.study_bbox[1])
            max_lat = min(max_lat, self.study_bbox[2])
            max_lon = min(max_lon, self.study_bbox[3])

        query = f"""
            [out:json][timeout:90];
            (
              node["amenity"]({min_lat},{min_lon},{max_lat},{max_lon});
              way["amenity"]({min_lat},{min_lon},{max_lat},{max_lon});
            );
            (._;>;);
            out body;
        """

        for attempt in range(self.max_retries):
            try:
                return self.rate_limited_overpass_query(query)
            except Exception as e:
                error_msg = str(e)
                if "Server load too high" in error_msg or "timeout" in error_msg.lower():
                    wait_time = min(300, (2 ** attempt) * 10)
                    print(f"Section {section_num} - Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}")
                    if attempt < self.max_retries - 1:
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

    def process_single_section(self, lat, lon, section_num, area_size_km):
        """Download and link the POIs of one cell. Runs in a worker process: the results are returned (not stored
        in self), with the waiting time for the Overpass server and the linking time of the cell.
        Returns (section_num, ok, data or error message, overpass_s, linking_s)."""
        t_start = time.time()
        try:
            result = self.query_section_data(lat, lon, area_size_km, section_num)
            t_query = time.time()

            section_osmnode_poi = []
            section_osmnode_way = []
            section_osmway_poi = []
            section_tag_nodes = []
            section_near_edges = []
            section_part_of_edges = []
            section_tags_edges = []

            tag_counter = [0]

            way_node_ids = set()
            for way in result.ways:
                way_osm_id = way.id
                way_node_id = f"way_{way_osm_id}"

                way_nodes = way.get_nodes(resolve_missing=False)
                centroid = self.way_centroid([(float(n.lat), float(n.lon)) for n in way_nodes])
                osmway_data = {
                    'osmnode_id': way_node_id,
                    'osm_id': way_osm_id,
                    'node_type': 'OSMWay',
                    'poi_type': 'OSMWay',
                    'name': way.tags.get('name'),
                    'amenity': way.tags.get('amenity'),
                }
                if way.tags.get('access'):
                    osmway_data['access'] = self._flatten_value(way.tags.get('access'))
                if centroid is not None:
                    osmway_data['lat'] = centroid[0]
                    osmway_data['lon'] = centroid[1]
                    osmway_data['geometry'] = f'POINT({centroid[0]} {centroid[1]})'
                section_osmway_poi.append(osmway_data)

                if centroid is not None:
                    for rj_node, distance, status in self.find_nearest_roadjunction(centroid[0], centroid[1]):
                        section_near_edges.append({
                            'source_id': way_node_id,
                            'target_id': rj_node,
                            'edge_type': 'NEAR',
                            'distance': distance,
                            'status': status,
                        })

                tag_id = f"tag_way_{way_osm_id}"
                tag_data = {'osmnode_id': tag_id, 'node_type': 'Tag'}
                for tag_key, tag_value in way.tags.items():
                    tag_data[tag_key] = self._flatten_value(tag_value)
                section_tag_nodes.append(tag_data)

                section_tags_edges.append({
                    'source_id': way_node_id,
                    'target_id': tag_id,
                    'edge_type': 'TAGS',
                })

                entrance_count = 0
                for node in way_nodes:
                    node_osm_id = node.id
                    way_node_ids.add(node_osm_id)
                    member_node_id = f"node_{node_osm_id}"

                    osmnode_data = {
                        'osmnode_id': member_node_id,
                        'osm_id': node_osm_id,
                        'node_type': 'OSMNode',
                        'poi_type': '',
                        'lat': float(node.lat),
                        'lon': float(node.lon),
                        'geometry': f'POINT({node.lat} {node.lon})',
                    }
                    entrance = node.tags.get('entrance')
                    if entrance and entrance != 'no':
                        osmnode_data['entrance'] = self._flatten_value(entrance)
                        if node.tags.get('access'):
                            osmnode_data['entrance_access'] = self._flatten_value(node.tags.get('access'))
                        entrance_count += 1
                    section_osmnode_way.append(osmnode_data)

                    section_part_of_edges.append({
                        'source_id': member_node_id,
                        'target_id': way_node_id,
                        'edge_type': 'PART_OF',
                    })

                    connections = self.find_nearest_roadjunction(float(node.lat), float(node.lon))
                    for rj_node, distance, status in connections:
                        section_near_edges.append({
                            'source_id': member_node_id,
                            'target_id': rj_node,
                            'edge_type': 'NEAR',
                            'distance': distance,
                            'status': status,
                        })
                osmway_data['entrance_count'] = entrance_count

            for node in result.nodes:
                if node.id not in way_node_ids and node.tags.get('amenity'):
                    node_osm_id = node.id
                    poi_node_id = f"node_{node_osm_id}"

                    osmnode_poi_data = {
                        'osmnode_id': poi_node_id,
                        'osm_id': node_osm_id,
                        'node_type': 'OSMNode',
                        'poi_type': 'OSMNode',
                        'lat': float(node.lat),
                        'lon': float(node.lon),
                        'geometry': f'POINT({node.lat} {node.lon})',
                        'name': node.tags.get('name'),
                        'amenity': node.tags.get('amenity'),
                    }
                    if node.tags.get('access'):
                        osmnode_poi_data['access'] = self._flatten_value(node.tags.get('access'))
                    section_osmnode_poi.append(osmnode_poi_data)

                    tag_id = f"tag_node_{node_osm_id}"
                    tag_data = {'osmnode_id': tag_id, 'node_type': 'Tag'}
                    for tag_key, tag_value in node.tags.items():
                        tag_data[tag_key] = self._flatten_value(tag_value)
                    section_tag_nodes.append(tag_data)

                    section_tags_edges.append({
                        'source_id': poi_node_id,
                        'target_id': tag_id,
                        'edge_type': 'TAGS',
                    })

                    connections = self.find_nearest_roadjunction(float(node.lat), float(node.lon))
                    for rj_node, distance, status in connections:
                        section_near_edges.append({
                            'source_id': poi_node_id,
                            'target_id': rj_node,
                            'edge_type': 'NEAR',
                            'distance': distance,
                            'status': status,
                        })

            data = {'osmnode_poi': section_osmnode_poi, 'osmnode_way': section_osmnode_way,
                    'osmway_poi': section_osmway_poi, 'tag_nodes': section_tag_nodes,
                    'near': section_near_edges, 'part_of': section_part_of_edges, 'tags': section_tags_edges}
            return section_num, True, data, t_query - t_start, time.time() - t_query

        except Exception as e:
            return section_num, False, f"{type(e).__name__}: {e}", time.time() - t_start, 0.0

    def add_section_data(self, data):
        """Add the results of one cell to the lists of the whole area"""
        self.osmnode_poi_nodes.extend(data['osmnode_poi'])
        self.osmnode_way_nodes.extend(data['osmnode_way'])
        self.osmway_poi_nodes.extend(data['osmway_poi'])
        self.tag_nodes.extend(data['tag_nodes'])
        self.near_edges.extend(data['near'])
        self.part_of_edges.extend(data['part_of'])
        self.tags_edges.extend(data['tags'])

    @staticmethod
    def section_summary(section_num, data):
        total_pois = len(data['osmnode_poi']) + len(data['osmway_poi'])
        total_edges = len(data['near']) + len(data['part_of']) + len(data['tags'])
        return (f"Section {section_num}: {total_pois} POIs, {len(data['osmnode_way'])} way-member nodes, "
                f"{total_edges} relationships")

    def apply_connection_strategies(self):
        """Apply connection strategies for OSMNodes without NEAR connections"""
        print("\nApplying connection strategies...")

        if not self.driveable_tree or not self.driveable_nodes:
            print("No driveable junctions available; skipping")
            return

        all_osmnodes = self.osmnode_poi_nodes + self.osmnode_way_nodes + self.osmway_poi_nodes
        connected = set(e['source_id'] for e in self.near_edges if e.get('edge_type') == 'NEAR')

        step1 = step2 = step3 = 0

        for node in all_osmnodes:
            nid = node['osmnode_id']
            if nid in connected:
                continue
            lat, lon = node.get('lat'), node.get('lon')
            if lat is None or lon is None:
                continue
            if self.driveable_tree:
                indices = self.driveable_tree.query_ball_point(self._xy(lat, lon), r=100)
                if indices:
                    candidates = []
                    for idx in indices:
                        rj = self.driveable_nodes[idx]
                        rlat, rlon = self.driveable_coords[idx]
                        d = self.haversine_distance(lat, lon, rlat, rlon)
                        if d <= 100:
                            candidates.append((rj, d))
                    if candidates:
                        nearest = min(candidates, key=lambda x: x[1])
                        self.near_edges.append({
                            'source_id': nid, 'target_id': nearest[0],
                            'edge_type': 'NEAR', 'distance': nearest[1],
                            'status': 'driveable_within_100m',
                        })
                        connected.add(nid)
                        step1 += 1

        if self.non_driveable_tree:
            for node in all_osmnodes:
                nid = node['osmnode_id']
                if nid in connected:
                    continue
                lat, lon = node.get('lat'), node.get('lon')
                if lat is None or lon is None:
                    continue
                dist, idx = self.non_driveable_tree.query(self._xy(lat, lon))
                if idx < len(self.non_driveable_nodes):
                    rj = self.non_driveable_nodes[idx]
                    rlat, rlon = self.non_driveable_coords[idx]
                    d = self.haversine_distance(lat, lon, rlat, rlon)
                    self.near_edges.append({
                        'source_id': nid, 'target_id': rj,
                        'edge_type': 'NEAR', 'distance': d,
                        'status': 'nearest_non_driveable',
                    })
                    connected.add(nid)
                    step2 += 1

        edges_by_src = {}
        for e in self.near_edges:
            if e.get('edge_type') == 'NEAR':
                edges_by_src.setdefault(e['source_id'], []).append(e)

        for node in all_osmnodes:
            nid = node['osmnode_id']
            edges = edges_by_src.get(nid, [])
            if not edges:
                continue
            has_drv = any(self.node_driveable[e['target_id']] for e in edges if e['target_id'] in self.node_driveable)
            has_nondrv = any(not self.node_driveable[e['target_id']] for e in edges if e['target_id'] in self.node_driveable)
            if has_nondrv and not has_drv:
                lat, lon = node.get('lat'), node.get('lon')
                if lat is None or lon is None:
                    continue
                dist, idx = self.driveable_tree.query(self._xy(lat, lon))
                if idx < len(self.driveable_nodes):
                    rj = self.driveable_nodes[idx]
                    rlat, rlon = self.driveable_coords[idx]
                    d = self.haversine_distance(lat, lon, rlat, rlon)
                    self.near_edges.append({
                        'source_id': nid, 'target_id': rj,
                        'edge_type': 'NEAR', 'distance': d,
                        'status': 'nearest_driveable_from_non_driveable',
                    })
                    step3 += 1

        print(f"Step 1: {step1}, Step 2: {step2}, Step 3: {step3}")
        print(f"Total connections: {step1 + step2 + step3}")

    def export_to_csv(self):
        """Export all data to CSV files with neo4j-admin headers"""
        print("\nExporting data to CSV...")

        def dedup(lst, key):
            seen = set()
            result = []
            for item in lst:
                k = item[key]
                if k not in seen:
                    seen.add(k)
                    result.append(item)
            return result

        self.osmnode_poi_nodes = dedup(self.osmnode_poi_nodes, 'osmnode_id')
        self.osmnode_way_nodes = dedup(self.osmnode_way_nodes, 'osmnode_id')
        self.osmway_poi_nodes = dedup(self.osmway_poi_nodes, 'osmnode_id')
        self.tag_nodes = dedup(self.tag_nodes, 'osmnode_id')

        def dedup_edges(lst):
            seen = set()
            result = []
            for item in lst:
                k = (item['source_id'], item['target_id'], item.get('edge_type', ''))
                if k not in seen:
                    seen.add(k)
                    result.append(item)
            return result

        self.near_edges = dedup_edges(self.near_edges)
        self.part_of_edges = dedup_edges(self.part_of_edges)
        self.tags_edges = dedup_edges(self.tags_edges)

        print("Creating combined nodes file...")
        original_nodes_df = pd.read_csv(self.nodes_csv)

        all_poi_nodes = self.osmnode_poi_nodes + self.osmnode_way_nodes + self.osmway_poi_nodes + self.tag_nodes

        if all_poi_nodes:
            poi_rows = []
            for poi in all_poi_nodes:
                row = {'osmnode_id': poi['osmnode_id']}
                for col in original_nodes_df.columns:
                    if col == 'osmnode_id':
                        continue
                    row[col] = poi.get(col, None)
                for key in ['node_type', 'poi_type', 'name', 'amenity', 'osm_id']:
                    if key in poi:
                        row[key] = poi[key]
                for key, value in poi.items():
                    if key not in row:
                        row[key] = value
                poi_rows.append(row)

            poi_df = pd.DataFrame(poi_rows)
            del poi_rows
        else:
            poi_df = None

        def as_integer_text(col):
            return col.map(lambda v: str(int(v)) if isinstance(v, float) and not math.isnan(v) and v == int(v) else v)
        node_columns = set(original_nodes_df.columns) | (set(poi_df.columns) if poi_df is not None else set())
        transforms = {c: as_integer_text for c in ('osm_id', 'entrance_count') if c in node_columns}

        combined_nodes_file = os.path.join(self.output_folder, 'combined_nodes.csv')
        n_nodes = self._write_combined_csv(original_nodes_df, poi_df, combined_nodes_file, transforms,
                                           id_col='osmnode_id', id_space='RoadJunction')
        del poi_df, original_nodes_df
        print(f"Exported {n_nodes} combined nodes")

        print("Creating combined edges file...")
        original_rels_df = pd.read_csv(self.rels_csv)

        all_poi_edges = self.near_edges + self.part_of_edges + self.tags_edges
        entrance_ids = set(n['osmnode_id'] for n in self.osmnode_way_nodes if n.get('entrance'))

        if all_poi_edges:
            edge_rows = []
            for edge in all_poi_edges:
                row = {'source_id': edge['source_id'], 'target_id': edge['target_id']}
                for col in original_rels_df.columns:
                    if col in ('source_id', 'target_id'):
                        continue
                    row[col] = edge.get(col, None)
                row['edge_type'] = edge.get('edge_type')
                row['distance'] = edge.get('distance')
                row['status'] = edge.get('status')
                if edge.get('edge_type') == 'NEAR':
                    row['from_entrance'] = 'True' if edge['source_id'] in entrance_ids else 'False'
                edge_rows.append(row)

            poi_edges_df = pd.DataFrame(edge_rows)
            del edge_rows
        else:
            poi_edges_df = None

        edge_columns = set(original_rels_df.columns) | (set(poi_edges_df.columns) if poi_edges_df is not None else set())
        transforms = {'edge_type': lambda col: col.fillna('ROUTE')} if 'edge_type' in edge_columns else {}

        combined_edges_file = os.path.join(self.output_folder, 'combined_edges.csv')
        n_edges = self._write_combined_csv(original_rels_df, poi_edges_df, combined_edges_file, transforms,
                                           start_id_col='source_id', end_id_col='target_id',
                                           start_id_space='RoadJunction', end_id_space='RoadJunction')
        del poi_edges_df, original_rels_df
        print(f"Exported {n_edges} combined edges")
        self.combined_counts = (n_nodes, n_edges)

        self.paper_stats = self.link_statistics()

        print(f"\nSummary:")
        print(f"  OSMNode:POI nodes: {len(self.osmnode_poi_nodes)}")
        print(f"  OSMNode (way members): {len(self.osmnode_way_nodes)}")
        print(f"  OSMWay:POI nodes: {len(self.osmway_poi_nodes)}")
        print(f"  Tag nodes: {len(self.tag_nodes)}")
        print(f"  NEAR edges: {len(self.near_edges)}")
        way_ids = set(w['osmnode_id'] for w in self.osmway_poi_nodes)
        print(f"  NEAR edges from OSMWay centroids: {sum(1 for e in self.near_edges if e['source_id'] in way_ids)}")
        print(f"  PART_OF edges: {len(self.part_of_edges)}")
        print(f"  TAGS edges: {len(self.tags_edges)}")

    def link_statistics(self):
        """Counts and NEAR distances for the paper (after duplicate removal)"""
        import statistics
        way_ids = set(w['osmnode_id'] for w in self.osmway_poi_nodes)
        poi_ids = set(n['osmnode_id'] for n in self.osmnode_poi_nodes) | way_ids
        rule = {'driveable_within_100m': 'R1 (driveable within 100 m)',
                'nearest_non_driveable': 'R2 (nearest non-driveable)',
                'nearest_driveable_from_non_driveable': 'R3 (extra link to the nearest driveable)',
                'nearest_driveable': 'nearest driveable (no non-driveable junction)'}

        def dist_stats(values):
            if not values:
                return "none"
            v = sorted(values)
            p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
            return (f"{len(v)} links, median {statistics.median(v):.1f} m, 95th percentile {p95:.1f} m, "
                    f"max {v[-1]:.1f} m")

        lines = []
        for label, ids in (("POIs (OSMNode POIs and OSMWay centroids)", poi_ids),
                           ("OSMWay centroids only", way_ids),
                           ("way-member nodes", None)):
            edges = [e for e in self.near_edges
                     if (e['source_id'] in ids if ids is not None else e['source_id'] not in poi_ids)]
            lines.append(f"NEAR from {label}: {dist_stats([float(e['distance']) for e in edges])}")
            for status, name in rule.items():
                sel = [float(e['distance']) for e in edges if e.get('status') == status]
                if sel:
                    lines.append(f"   {name}: {dist_stats(sel)}")
        by_poi = {}
        for e in self.near_edges:
            if e['source_id'] in poi_ids:
                by_poi.setdefault(e['source_id'], set()).add(e.get('status'))
        lines.append(f"POIs with a link: {len(by_poi)} of {len(poi_ids)}; linked by R1: "
                     f"{sum(1 for s in by_poi.values() if 'driveable_within_100m' in s)}, by R2: "
                     f"{sum(1 for s in by_poi.values() if 'nearest_non_driveable' in s)}, with the R3 extra link: "
                     f"{sum(1 for s in by_poi.values() if 'nearest_driveable_from_non_driveable' in s)}")
        lines.append(f"POIs without a link: {len(poi_ids) - len(by_poi)}")
        poi_d = [float(e['distance']) for e in self.near_edges if e['source_id'] in poi_ids]
        if poi_d:
            lines.append("POI links longer than 100 m / 500 m / 1 km: " + " / ".join(
                f"{sum(1 for d in poi_d if d > t)} ({100.0 * sum(1 for d in poi_d if d > t) / len(poi_d):.1f}%)"
                for t in (100, 500, 1000)))
        from collections import Counter
        pois = self.osmnode_poi_nodes + self.osmway_poi_nodes
        acc = Counter(str(p.get('access')) for p in pois if p.get('access'))
        n_pois = max(len(pois), 1)
        lines.append(f"POIs with an access tag: {sum(acc.values())} of {len(pois)} "
                     f"({100.0 * sum(acc.values()) / n_pois:.1f}%); private or no: "
                     f"{acc.get('private', 0) + acc.get('no', 0)} ({100.0 * (acc.get('private', 0) + acc.get('no', 0)) / n_pois:.1f}%)")
        lines.append("   access values: " + ", ".join(f"{k} {v}" for k, v in acc.most_common(10)))
        ds = getattr(self, 'driveable_stats', {})
        if ds:
            lines.append(f"Driveable junctions: {ds['touched']} on driveable edges; used as driveable: {ds['largest_scc']}"
                         + (f" (largest strongly connected component of the driveable network, {ds['scc_count']} "
                            f"components; {ds['touched'] - ds['largest_scc']} junctions outside it treated as "
                            f"non-driveable)" if ds.get('scc_count') is not None else " (all)"))
        entrance_nodes = [n for n in self.osmnode_way_nodes if n.get('entrance')]
        entrance_ids = set(n['osmnode_id'] for n in entrance_nodes)
        ways_with = sum(1 for w in self.osmway_poi_nodes if w.get('entrance_count', 0) > 0)
        lines.append(f"OSMWay POIs with at least one entrance node (entrance=*): {ways_with} of "
                     f"{len(self.osmway_poi_nodes)} ({100.0 * ways_with / max(len(self.osmway_poi_nodes), 1):.1f}%); "
                     f"entrance nodes: {len(entrance_nodes)}")
        lines.append(f"NEAR from entrance nodes: "
                     f"{dist_stats([float(e['distance']) for e in self.near_edges if e['source_id'] in entrance_ids])}")
        return lines

    def write_paper_results(self, total_time, peak_mem=None):
        """Values needed for the paper, in one text file"""
        bbox = self.study_bbox
        side_km = 2 * self.run_params['total_distance_km']
        lines = [f"POIs - results for the paper ({time.strftime('%Y-%m-%d %H:%M')})",
                 f"Script: {os.path.basename(__file__)}; workers: {self.max_workers}; "
                 f"Overpass: {self.run_params['overpass_url'] or 'public server'}",
                 f"Road network input: {self.nodes_csv}",
                 f"Centre: {self.run_params['center_lat']}, {self.run_params['center_lon']}; "
                 f"D = {self.run_params['total_distance_km']} km",
                 f"Study area: lat {bbox[0]:.6f} to {bbox[2]:.6f}, lon {bbox[1]:.6f} to {bbox[3]:.6f}; "
                 f"{side_km:g} x {side_km:g} km = {side_km * side_km:g} km2",
                 f"POI cells: {self.run_params['n_cells']} (cell size {self.run_params['area_size_km']} km)",
                 "",
                 f"OSMNode:POI: {len(self.osmnode_poi_nodes)}",
                 f"OSMWay:POI: {len(self.osmway_poi_nodes)}",
                 f"POIs in total: {len(self.osmnode_poi_nodes) + len(self.osmway_poi_nodes)}",
                 f"Way-member OSMNodes: {len(self.osmnode_way_nodes)}",
                 f"Tag nodes: {len(self.tag_nodes)}",
                 f"NEAR: {len(self.near_edges)}; PART_OF: {len(self.part_of_edges)}; TAGS: {len(self.tags_edges)}",
                 f"Combined nodes: {self.combined_counts[0]}; combined edges: {self.combined_counts[1]}",
                 ""] + getattr(self, 'paper_stats', []) + [""] + self.timing_lines(peak_mem) + \
                ["", f"Total execution time: {total_time:.2f} seconds"]
        path = os.path.join(self.output_folder, 'poi_paper_results.txt')
        with open(path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"Results for the paper written to {path}")

    def generate_large_area_poi(self, center_lat, center_lon, total_distance_km, area_size_km):
        """Main method"""
        print("Large area POI generation...")
        print(f"Center: ({center_lat}, {center_lon})")
        print(f"Total area: {total_distance_km}km, Section size: {area_size_km}km")
        self.study_bbox = self.study_area_bbox(center_lat, center_lon, total_distance_km)
        print(f"Study area: lat [{self.study_bbox[0]:.6f}, {self.study_bbox[2]:.6f}], "
              f"lon [{self.study_bbox[1]:.6f}, {self.study_bbox[3]:.6f}]")
        print(f"Worker processes: {self.max_workers}")

        def phase(name, t0):
            seconds = time.time() - t0
            self.phase_times.append((name, seconds))
            print(f"Phase time - {name}: {seconds:.2f} seconds")

        t0 = time.time()
        self.load_graph_from_csv()
        phase("reading the road network and marking the driveable junctions", t0)
        t0 = time.time()
        self.build_spatial_index()
        phase("spatial index of the junctions", t0)

        grid_points = self.calculate_grid_points(center_lat, center_lon, total_distance_km, area_size_km)
        self.run_params = {'center_lat': center_lat, 'center_lon': center_lon, 'total_distance_km': total_distance_km,
                           'area_size_km': area_size_km, 'n_cells': len(grid_points),
                           'overpass_url': getattr(self, 'overpass_url', None)}
        processed_sections = self.get_processed_sections()
        print(f"Found {len(processed_sections)} previously processed sections")

        sections_to_process = [
            (i, lat, lon) for i, (lat, lon) in enumerate(grid_points, 1)
            if i not in processed_sections
        ]
        print(f"Will process {len(sections_to_process)} new sections out of {len(grid_points)} total")

        try:
            t0 = time.time()
            results = {}
            cell_times = []
            failed_sections = []
            position = {i: (i, lat, lon) for i, lat, lon in sections_to_process}

            def collect(out):
                section_num, ok, data, overpass_s, linking_s = out
                if ok:
                    results[section_num] = data
                    cell_times.append((overpass_s, linking_s))
                    self.save_processed_section(section_num)
                    print(self.section_summary(section_num, data))
                else:
                    print(f"Section {section_num} failed with error: {data}")
                    failed_sections.append(position[section_num])

            tasks = [(lat, lon, i, area_size_km) for i, lat, lon in sections_to_process]
            if self.max_workers <= 1:
                for task in tasks:
                    collect(self.process_single_section(*task))
            elif tasks:
                global _GENERATOR
                _GENERATOR = self
                with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=self.mp_context) as executor:
                    futures = [executor.submit(_process_cell, task) for task in tasks]
                    for future in as_completed(futures):
                        collect(future.result())
                _GENERATOR = None

            for attempt in range(1, 11):
                if not failed_sections:
                    break
                print(f"\n{len(failed_sections)} sections failed: {sorted(x[0] for x in failed_sections)}. "
                      f"Retry {attempt}/10 in 5 seconds...")
                time.sleep(5)
                retry, failed_sections = sorted(failed_sections), []
                for i, lat, lon in retry:
                    collect(self.process_single_section(lat, lon, i, area_size_km))
            if failed_sections:
                raise RuntimeError(f"{len(failed_sections)} sections failed after 10 retries: "
                                   f"{sorted(x[0] for x in failed_sections)}")
            print("Failed sections after retries: none")
            phase("download and linking of the cells", t0)
            self.cell_times = cell_times

            for section_num in sorted(results):
                self.add_section_data(results[section_num])
            results.clear()

            print("\n=== Final post-processing ===")
            t0 = time.time()
            self.apply_connection_strategies()
            phase("connection strategies (R3)", t0)

            total_pois = len(self.osmnode_poi_nodes) + len(self.osmway_poi_nodes)
            print(f"\n=== POI Collection Completed ===")
            print(f"Total POIs: {total_pois}")
            print(f"Total way-member nodes: {len(self.osmnode_way_nodes)}")
            print(f"Total Tag nodes: {len(self.tag_nodes)}")

            t0 = time.time()
            self.export_to_csv()
            phase("writing the combined CSV files", t0)
            print(f"\n=== Export Completed Successfully ===")
            print(f"Output folder: {self.output_folder}")

        except Exception as e:
            print(f"\n=== POI Generation Failed: {e} ===")
            self.export_to_csv()
            raise

    def timing_lines(self, peak_mem):
        """Phase times, waiting for the Overpass server and linking per cell, peak memory"""
        lines = [f"Phase time - {name}: {seconds:.2f} seconds" for name, seconds in self.phase_times]
        ct = getattr(self, 'cell_times', [])
        if ct:
            wait = sum(c[0] for c in ct)
            link = sum(c[1] for c in ct)
            lines.append(f"Cells: {len(ct)}; waiting for the Overpass server (request and response): sum {wait:.2f} s, "
                         f"mean {wait / len(ct):.3f} s per cell; linking of the POIs in the worker: sum {link:.2f} s, "
                         f"mean {link / len(ct):.3f} s per cell; share of the cell time spent waiting: "
                         f"{100.0 * wait / max(wait + link, 1e-9):.1f}%")
        if peak_mem is not None and peak_mem[0]:
            lines.append(f"Peak memory (all processes): {peak_mem[0] / 1024**3:.2f} GB")
            lines.append(f"Peak memory (largest single process): {peak_mem[1] / 1024**3:.2f} GB")
        else:
            lines.append("Peak memory: not measured (psutil not installed)")
        return lines


def add_options():
    parser = argparse.ArgumentParser(description='Insertion of POI using NetworkX, one cell after the other.')
    parser.add_argument('--nodes-csv', '-n', dest='nodes_csv', type=str, required=True)
    parser.add_argument('--rels-csv', '-r', dest='rels_csv', type=str, required=True)
    parser.add_argument('--output', '-o', dest='output', type=str, required=True)
    parser.add_argument('--latitude', '-x', dest='lat', type=float, required=True)
    parser.add_argument('--longitude', '-y', dest='lon', type=float, required=True)
    parser.add_argument('--distance', '-d', dest='dist', type=float, required=True)
    parser.add_argument('--area-size', '-a', dest='area_size', type=float, required=True)
    parser.add_argument('--overpass-url', '-u', dest='overpass_url', type=str, default=None)
    parser.add_argument('--all-driveable-junctions', action='store_true',
                        help='use every junction on a driveable edge as driveable (without it, only the junctions of '
                             'the largest strongly connected component of the driveable network)')
    return parser


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)

    for name in ('processed_poi_sections.txt', 'combined_nodes.csv', 'combined_edges.csv',
                 'poi_paper_results.txt'):
        path = os.path.join(options.output, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed old file {path}")

    peak_mem, stop_monitor = start_memory_monitor()
    generator = NetworkXPOIGenerator(
        nodes_csv=options.nodes_csv,
        rels_csv=options.rels_csv,
        output_folder=options.output,
        overpass_url=options.overpass_url,
        driveable_scc=not options.all_driveable_junctions,
        max_workers=1,
    )

    try:
        generator.generate_large_area_poi(
            center_lat=options.lat,
            center_lon=options.lon,
            total_distance_km=options.dist,
            area_size_km=options.area_size,
        )
    except Exception as e:
        print(f"Error: {e}")
        raise

    if stop_monitor is not None:
        stop_monitor.set()
    for line in generator.timing_lines(peak_mem):
        print(line)
    generator.write_paper_results(time.time() - start_time, peak_mem)
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    return 0


if __name__ == "__main__":
    main()
