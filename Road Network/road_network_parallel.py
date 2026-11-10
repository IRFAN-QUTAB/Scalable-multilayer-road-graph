import osmnx as ox
import argparse
import os
import time
import pandas as pd
import math
import networkx as nx
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import csv
import statistics
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import multiprocessing

import threading
import psutil

ox.settings.use_cache = False


from osmnx import _overpass as _ox_overpass
_original_overpass_request = _ox_overpass._overpass_request


_REQUEST_TIME = [0.0, 0]


def _overpass_request_drop_incomplete_ways(data, **kwargs):
    t_request = time.time()
    response = _original_overpass_request(data, **kwargs)
    _REQUEST_TIME[0] += time.time() - t_request
    _REQUEST_TIME[1] += 1
    elements = response.get("elements", [])
    node_ids = {e["id"] for e in elements if e.get("type") == "node"}
    bad_ways = {e["id"] for e in elements
                if e.get("type") == "way" and any(n not in node_ids for n in e.get("nodes", []))}
    if bad_ways:
        print(f"Dropped {len(bad_ways)} way(s) that reference nodes missing from the "
              f"Overpass response: {sorted(bad_ways)}")
        response = dict(response)
        response["elements"] = [e for e in elements
                                if not (e.get("type") == "way" and e["id"] in bad_ways)]
    return response


_ox_overpass._overpass_request = _overpass_request_drop_incomplete_ways


PHASE = ['start']
OVERPASS_PROCESS_NAMES = ('interpreter', 'osm3s_query')


def start_memory_monitor(interval=0.5, samples_path=None):
    """Record during the run:
    peak[0] = peak RAM of all processes together (main + workers), peak[1] = largest single process,
    and, every second, a sample with the phase, the RAM, the system CPU use, the 1-minute load
    average and the number of running Overpass server processes (written to samples_path)."""
    proc = psutil.Process(os.getpid())
    peak = [0, 0]
    samples = []
    stop = threading.Event()
    psutil.cpu_percent(None)

    def count_overpass_processes():
        n = 0
        for p in psutil.process_iter(['name']):
            if (p.info.get('name') or '') in OVERPASS_PROCESS_NAMES:
                n += 1
        return n

    def run():
        tick = 0
        while not stop.is_set():
            sizes = [proc.memory_info().rss]
            for c in proc.children(recursive=True):
                try:
                    sizes.append(c.memory_info().rss)
                except psutil.Error:
                    pass
            peak[0] = max(peak[0], sum(sizes))
            peak[1] = max(peak[1], max(sizes))
            if tick % 2 == 0:
                try:
                    n_overpass = count_overpass_processes()
                except Exception:
                    n_overpass = -1
                samples.append((round(time.time(), 2), PHASE[0], round(sum(sizes) / 1024**3, 3),
                                round(max(sizes) / 1024**3, 3), psutil.cpu_percent(None),
                                round(os.getloadavg()[0], 2), len(sizes) - 1, n_overpass))
            tick += 1
            stop.wait(interval)

    def write_samples():
        if samples_path:
            with open(samples_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time', 'phase', 'mem_all_gb', 'mem_max_gb', 'system_cpu_percent',
                            'load_1min', 'worker_processes', 'overpass_server_processes'])
                w.writerows(samples)

    threading.Thread(target=run, daemon=True).start()
    return stop, peak, samples, write_samples


def use_overpass_server(url, timeout_s=None, memory_mb=None):
    """Use a given Overpass server (e.g. a local instance) instead of the public one.
    timeout_s / memory_mb set the [timeout:] and [maxsize:] values sent with each query."""
    if url:
        url = url.rstrip('/')
        if url.endswith('/interpreter'):
            url = url[:-len('/interpreter')]
        ox.settings.overpass_url = url
        ox.settings.overpass_rate_limit = False
    if timeout_s:
        ox.settings.requests_timeout = int(timeout_s)
    if memory_mb:
        ox.settings.overpass_memory = int(memory_mb) * 1024 * 1024


class LargeAreaGraphGenerator:
    def __init__(self, import_folder_path: str, overpass_url: str = None,
                 overpass_timeout: int = None, overpass_memory: int = None,
                 max_retries: int = 10, retry_pause: int = 5, workers: int = None):
        self.import_folder_path = import_folder_path
        self.workers = workers
        self.read_threads = workers if workers else cpu_count()
        self.substeps = {}
        self.cell_timings = []
        self.download_cpu = None
        self.download_wall = None
        self.overpass_url = overpass_url
        self.overpass_timeout = overpass_timeout
        self.overpass_memory = overpass_memory
        self.max_retries = max_retries
        self.retry_pause = retry_pause
        if not os.path.exists(self.import_folder_path):
            os.makedirs(self.import_folder_path)

    def get_processed_sections(self):
        """Read or create a file to track processed section numbers"""
        processed_file = os.path.join(self.import_folder_path, "processed_graph_sections.txt")
        processed_sections = set()
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                for line in f:
                    try:
                        processed_sections.add(int(line.strip()))
                    except ValueError:
                        continue
        return processed_sections

    def save_processed_section(self, section_num: int):
        """Save a processed section number to the tracking file"""
        processed_file = os.path.join(self.import_folder_path, "processed_graph_sections.txt")
        with open(processed_file, 'a') as f:
            f.write(f"{section_num}\n")

    def section_files_exist(self, section_num: int) -> bool:
        """Check if both nodes and rels CSV files exist for a section"""
        nodes_csv_name = f"nodes_{section_num}.csv"
        rels_csv_name = f"rels_{section_num}.csv"
        nodes_path = os.path.join(self.import_folder_path, nodes_csv_name)
        rels_path = os.path.join(self.import_folder_path, rels_csv_name)
        return os.path.exists(nodes_path) and os.path.exists(rels_path)

    @staticmethod
    def km_to_degrees(km_lat: float, km_lon: float, reference_lat: float) -> Tuple[float, float]:
        """Convert kilometers to degrees (same Earth radius as OSMnx, 6,371,009 m)"""
        earth_radius_km = 6371.009
        degrees_lat = math.degrees(km_lat / earth_radius_km)
        degrees_lon = math.degrees(km_lon / earth_radius_km) / math.cos(math.radians(reference_lat))
        return degrees_lat, degrees_lon

    def calculate_grid_points(self, center_lat: float, center_lon: float,
                              total_distance_km: float, area_size_km: float,
                              overlap_km: float) -> List[Tuple[float, float]]:
        """Calculate all grid center points for the large area"""
        points = []
        step_size_km = area_size_km - overlap_km
        study_width_km = 2 * total_distance_km
        if area_size_km >= study_width_km:
            steps_needed = 1
        else:
            steps_needed = math.ceil((study_width_km - area_size_km) / step_size_km) + 1
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

    def generate_single_graph(self, lat: float, lon: float, distance_km: float,
                              graph_name: str) -> nx.MultiDiGraph:
        """Generate a single graph and return NetworkX graph"""
        distance_m = distance_km * 1000
        use_overpass_server(self.overpass_url, self.overpass_timeout, self.overpass_memory)
        print(f"Generating graph {graph_name} at ({lat:.6f}, {lon:.6f}) with half-side {distance_km}km")

        try:
            G = ox.graph_from_point(
                (lat, lon),
                dist=int(distance_m),
                dist_type='bbox',
                simplify=False,
                retain_all=True,
                truncate_by_edge=True,
                custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service"]'
            )

            for u, v, k, data in G.edges(keys=True, data=True):
                highway = data.get('highway')
                access = data.get('access')
                data['driveable'] = "True"

                if access in {"no", "private"}:
                    data['driveable'] = "False"
                elif highway == 'service':
                    data['driveable'] = "False"
                elif highway == 'residential' and access == 'private':
                    data['driveable'] = "False"

            return G
        except Exception as e:
            msg = str(e).lower()
            if 'no data elements' in msg or 'no graph nodes' in msg:
                print(f"Section {graph_name}: no road data in this cell")
                return None
            raise

    def filter_graph_by_boundary(self, nodes_df: pd.DataFrame, rels_df: pd.DataFrame,
                                 center_lat: float, center_lon: float,
                                 total_distance_km: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter merged nodes and rels DataFrames by square boundary around the center"""
        min_lon, min_lat, max_lon, max_lat = ox.utils_geo.bbox_from_point(
            (center_lat, center_lon), dist=total_distance_km * 1000)

        print(f"Applying boundary filter: lat [{min_lat:.6f}, {max_lat:.6f}], lon [{min_lon:.6f}, {max_lon:.6f}]")
        original_node_count = len(nodes_df)
        original_rel_count = len(rels_df)

        nodes_df = nodes_df[
            (nodes_df['y'] >= min_lat) & (nodes_df['y'] <= max_lat) &
            (nodes_df['x'] >= min_lon) & (nodes_df['x'] <= max_lon)
        ].copy()

        valid_node_ids = set(nodes_df['osmnode_id'].astype(str))
        rels_df = rels_df[
            rels_df['source_id'].astype(str).isin(valid_node_ids) &
            rels_df['target_id'].astype(str).isin(valid_node_ids)
        ].copy()

        print(f"Boundary filter removed {original_node_count - len(nodes_df)} nodes and {original_rel_count - len(rels_df)} rels")
        print(f"After boundary filter: {len(nodes_df)} nodes, {len(rels_df)} rels")

        return nodes_df, rels_df

    def _flatten_value(self, value):
        """Flatten list values to semicolon-separated strings for CSV"""
        if isinstance(value, list):
            return ";".join(str(v) for v in value)
        return value

    def extract_nodes_and_relationships(self, G: nx.MultiDiGraph) -> Tuple[List[dict], List[dict]]:
        """Extract ALL node and relationship properties from NetworkX graph"""
        nodes_data = []
        rels_data = []

        for node, data in G.nodes(data=True):
            node_dict = {'osmnode_id': str(node)}
            if data.get('y') and data.get('x'):
                node_dict['geometry'] = f"POINT({data.get('y')} {data.get('x')})"
            else:
                node_dict['geometry'] = None
            for key, value in data.items():
                node_dict[key] = self._flatten_value(value)
            nodes_data.append(node_dict)

        for u, v, k, data in G.edges(keys=True, data=True):
            rel_dict = {
                'source_id': str(u),
                'target_id': str(v),
            }
            for key, value in data.items():
                if key == 'osmid':
                    rel_dict['osmway_id'] = self._flatten_value(value)
                else:
                    rel_dict[key] = self._flatten_value(value)
            if 'status' not in rel_dict:
                rel_dict['status'] = 'active'
            rels_data.append(rel_dict)

        return nodes_data, rels_data

    def process_single_section(self, lat: float, lon: float, distance_km: float,
                               section_num: int, center_lat: float, center_lon: float,
                               total_distance_km: float) -> Tuple[str, str]:
        """Process a single section using NetworkX and return CSV filenames.
        No boundary filtering here -- that happens after merging all sections."""
        graph_name = f"g{section_num}"
        nodes_csv_name = f"nodes_{section_num}.csv"
        rels_csv_name = f"rels_{section_num}.csv"

        print(f"Processing section {section_num} on process {os.getpid()}")
        t_cell = time.time()
        _REQUEST_TIME[0], _REQUEST_TIME[1] = 0.0, 0

        G = self.generate_single_graph(lat, lon, distance_km, graph_name)
        t_graph = time.time()
        timing = {'section': section_num, 'pid': os.getpid(), 'start': round(t_cell, 3),
                  'overpass_s': round(_REQUEST_TIME[0], 3), 'requests': _REQUEST_TIME[1],
                  'graph_total_s': round(t_graph - t_cell, 3),
                  'osmnx_build_s': round(t_graph - t_cell - _REQUEST_TIME[0], 3)}
        
        if G is None or G.number_of_nodes() == 0:
            print(f"Section {section_num} is empty, creating empty CSV files")
            empty_nodes_df = pd.DataFrame(columns=['osmnode_id'])
            empty_rels_df = pd.DataFrame(columns=['source_id', 'target_id'])
            
            nodes_csv_path = os.path.join(self.import_folder_path, nodes_csv_name)
            rels_csv_path = os.path.join(self.import_folder_path, rels_csv_name)
            empty_nodes_df.to_csv(nodes_csv_path, index=False)
            empty_rels_df.to_csv(rels_csv_path, index=False)
            timing.update({'write_s': round(time.time() - t_graph, 3), 'end': round(time.time(), 3),
                           'nodes': 0, 'edges': 0})
            return nodes_csv_name, rels_csv_name, timing

        nodes_data, rels_data = self.extract_nodes_and_relationships(G)

        nodes_csv_path = os.path.join(self.import_folder_path, nodes_csv_name)
        rels_csv_path = os.path.join(self.import_folder_path, rels_csv_name)

        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(nodes_csv_path, index=False)
        rels_df = pd.DataFrame(rels_data)
        rels_df.to_csv(rels_csv_path, index=False)
        
        print(f"Section {section_num} completed: {len(nodes_data)} nodes, {len(rels_data)} relationships")
        timing.update({'write_s': round(time.time() - t_graph, 3), 'end': round(time.time(), 3),
                       'nodes': len(nodes_data), 'edges': len(rels_data)})
        return nodes_csv_name, rels_csv_name, timing

    def merge_csv_files(self, nodes_csv_files: List[str], rels_csv_files: List[str],
                        center_lat: float, center_lon: float,
                        total_distance_km: float) -> Tuple[str, str]:
        """Merge all nodes and relationships CSV files, deduplicate, and apply boundary filter.
        The merged tables are returned (kept in memory); they are written after the largest component."""
        print(f"Merging {len(nodes_csv_files)} nodes CSV files and {len(rels_csv_files)} relationships CSV files...")
        t_read = time.time()

        def read(name):
            path = os.path.join(self.import_folder_path, name)
            if os.path.exists(path):
                df = pd.read_csv(path)
                if len(df) > 0:
                    return df
            return None

        with ThreadPoolExecutor(max_workers=self.read_threads) as ex:
            all_nodes_data = [df for df in ex.map(read, nodes_csv_files) if df is not None]
            all_rels_data = [df for df in ex.map(read, rels_csv_files) if df is not None]
        self.substeps['merge: read cell CSV files'] = time.time() - t_read
        t_dedup = time.time()

        if all_nodes_data:
            combined_nodes_df = pd.concat(all_nodes_data, ignore_index=True, sort=False)
            if 'street_count' in combined_nodes_df.columns:
                street_count = pd.to_numeric(combined_nodes_df['street_count'], errors='coerce')
                combined_nodes_df['street_count'] = (street_count.groupby(combined_nodes_df['osmnode_id'])
                                                     .transform('max').astype('Int64'))
            combined_nodes_df = combined_nodes_df.drop_duplicates(subset=['osmnode_id'], keep='first')
        else:
            combined_nodes_df = pd.DataFrame(columns=['osmnode_id'])

        if all_rels_data:
            combined_rels_df = pd.concat(all_rels_data, ignore_index=True, sort=False)
            dedup_cols = ['source_id', 'target_id']
            if 'osmway_id' in combined_rels_df.columns:
                dedup_cols.append('osmway_id')
            combined_rels_df = combined_rels_df.drop_duplicates(subset=dedup_cols, keep='first')
        else:
            combined_rels_df = pd.DataFrame(columns=['source_id', 'target_id'])

        print(f"After deduplication: {len(combined_nodes_df)} unique nodes, {len(combined_rels_df)} unique relationships")
        self.substeps['merge: concatenation, street_count, deduplication'] = time.time() - t_dedup

        t_boundary = time.time()
        print(f"\n--- Applying boundary filter on merged data ---")
        combined_nodes_df, combined_rels_df = self.filter_graph_by_boundary(
            combined_nodes_df, combined_rels_df,
            center_lat, center_lon, total_distance_km
        )
        self.substeps['merge: boundary filter'] = time.time() - t_boundary
        return combined_nodes_df, combined_rels_df

    def generate_large_area_graph(self, center_lat: float, center_lon: float,
                                  total_distance_km: float, area_size_km: float,
                                  overlap_km: float):
        """Main method to generate large area graph with parallel processing"""
        print(f"Large area graph generation...")
        print(f"Center: ({center_lat}, {center_lon})")
        print(f"Total area: {total_distance_km}km")
        print(f"Section size: {area_size_km}km")
        print(f"Overlap: {overlap_km}km")

        available_cores = self.workers if self.workers else cpu_count()
        print(f"\nAvailable CPU cores: {cpu_count()}")
        print(f"Will use {available_cores} parallel processes")

        grid_points = self.calculate_grid_points(center_lat, center_lon,
                                                 total_distance_km, area_size_km, overlap_km)
        t_phase_start = time.time()
        PHASE[0] = 'download'
        cpu0 = psutil.Process(os.getpid()).cpu_times()

        processed_sections = self.get_processed_sections()
        print(f"Found {len(processed_sections)} previously processed sections: {processed_sections}")

        print(f"Will generate {len(grid_points)} sections")

        nodes_csv_files = []
        rels_csv_files = []
        
        tasks = []
        for i, (lat, lon) in enumerate(grid_points, 1):
            nodes_csv_name = f"nodes_{i}.csv"
            rels_csv_name = f"rels_{i}.csv"
            
            if i in processed_sections and self.section_files_exist(i):
                print(f"Skipping section {i}/{len(grid_points)} (already processed)")
                nodes_csv_files.append(nodes_csv_name)
                rels_csv_files.append(rels_csv_name)
                continue
            
            tasks.append((lat, lon, area_size_km / 2 + 0.01, i, center_lat, center_lon, total_distance_km))
        
        if tasks:
            print(f"\n{len(tasks)} sections to process in parallel...")
            
            try:
                task_by_section = {task[3]: task for task in tasks}
                pending_tasks = tasks
                failed_sections = []
                for attempt in range(self.max_retries + 1):
                    failed_sections = []
                    with Pool(processes=available_cores) as pool:
                        for section_num, result in pool.imap_unordered(self._process_section_star, pending_tasks):
                            if result is None:
                                failed_sections.append(section_num)
                                continue
                            nodes_csv_name, rels_csv_name, timing = result
                            self.cell_timings.append(timing)
                            nodes_csv_files.append(nodes_csv_name)
                            rels_csv_files.append(rels_csv_name)
                            self.save_processed_section(section_num)
                            print(f"Section {section_num} marked as complete")

                    if not failed_sections:
                        break
                    if attempt < self.max_retries:
                        print(f"\n{len(failed_sections)} sections failed: {sorted(failed_sections)}. "
                              f"Retry {attempt + 1}/{self.max_retries} in {self.retry_pause} seconds...")
                        time.sleep(self.retry_pause)
                        pending_tasks = [task_by_section[s] for s in failed_sections]

                if failed_sections:
                    raise RuntimeError(f"{len(failed_sections)} sections failed after {self.max_retries} retries: "
                                       f"{sorted(failed_sections)}. Run the script again to retry only these sections.")
                        
            except Exception as e:
                print(f"\n=== Graph generation interrupted ===")
                print(f"Error: {e}")
                print(f"Progress saved. You can restart to resume.")
                raise
        else:
            print("\nAll sections already processed!")

        t_download_end = time.time()
        cpu1 = psutil.Process(os.getpid()).cpu_times()
        self.download_cpu = sum(cpu1[:4]) - sum(cpu0[:4])
        self.download_wall = t_download_end - t_phase_start
        PHASE[0] = 'merge'
        print(f"\nPhase time - download and processing of the sections: {t_download_end - t_phase_start:.2f} seconds")
        print(f"\n--- Merging all CSV files ---")
        section_order = lambda f: int(f.split('_')[1].split('.')[0])
        nodes_csv_files.sort(key=section_order)
        rels_csv_files.sort(key=section_order)
        merged_nodes, merged_rels = self.merge_csv_files(
            nodes_csv_files, rels_csv_files,
            center_lat, center_lon, total_distance_km
        )
        t_merge_end = time.time()
        PHASE[0] = 'largest_component'
        print(f"Phase time - merge, deduplication and boundary filter: {t_merge_end - t_download_end:.2f} seconds")

        print(f"\n--- Validating connected components ---")
        nodes_csv, rels_csv = self.validate_connected_component(merged_nodes, merged_rels)
        print(f"Phase time - largest connected component and connectivity report: {time.time() - t_merge_end:.2f} seconds")
        PHASE[0] = 'end'

        print(f"\nfullgraph_nodes.csv and fullgraph_rels.csv completed successfully!")
        print(f"Nodes CSV: {nodes_csv}")
        print(f"Relationships CSV: {rels_csv}")
        
        processed_file = os.path.join(self.import_folder_path, "processed_graph_sections.txt")
        if os.path.exists(processed_file):
            os.remove(processed_file)
            print("Cleaned up progress tracking file.")

    @staticmethod
    def _components(nodes_df, rels_df, connection):
        """Number of components and component label of every node (scipy.sparse.csgraph)"""
        ids = pd.Index(nodes_df['osmnode_id'].astype(str))
        src = ids.get_indexer(rels_df['source_id'].astype(str))
        dst = ids.get_indexer(rels_df['target_id'].astype(str))
        n = len(ids)
        A = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n)).tocsr()
        return connected_components(A, directed=True, connection=connection)

    def report_connectivity(self, nodes_df, rels_df, removed_components, removed_nodes):
        """Write weak and strong connectivity of the final graph to connectivity_report.txt"""
        wcc, _ = self._components(nodes_df, rels_df, 'weak')
        n_scc, scc_labels = self._components(nodes_df, rels_df, 'strong')
        largest_scc = int(np.bincount(scc_labels).max()) if len(scc_labels) else 0
        n_nodes = len(nodes_df)
        lines = [
            f"Nodes: {n_nodes}",
            f"Edges: {len(rels_df)}",
            f"Smaller components removed: {removed_components} ({removed_nodes} nodes)",
            f"Weakly connected components: {wcc}",
            f"Strongly connected components: {n_scc}",
            f"Largest strongly connected component: {largest_scc} nodes "
            f"({100.0 * largest_scc / max(n_nodes, 1):.2f}%)",
        ]
        with open(os.path.join(self.import_folder_path, "connectivity_report.txt"), 'w') as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))

    def validate_connected_component(self, nodes_df, rels_df) -> Tuple[str, str]:
        """Keep only the largest (weakly) connected component and write the final CSV files"""
        t_lcc = time.time()
        n_comp, labels = self._components(nodes_df, rels_df, 'weak')
        print(f"Found {n_comp} connected components")
        removed_components, removed_nodes = 0, 0
        if n_comp > 1:
            sizes = np.bincount(labels)
            largest = int(np.argmax(sizes))
            keep = labels == largest
            removed_components, removed_nodes = n_comp - 1, int((~keep).sum())
            print(f"Largest component: {int(sizes[largest])} nodes")
            print(f"Removing {removed_nodes} isolated nodes from {removed_components} smaller components")
            nodes_df = nodes_df[keep]
            kept_ids = set(nodes_df['osmnode_id'].astype(str))
            rels_df = rels_df[rels_df['source_id'].astype(str).isin(kept_ids) &
                              rels_df['target_id'].astype(str).isin(kept_ids)]
            print(f"Validated graph: {len(nodes_df)} nodes, {len(rels_df)} edges (single connected component)")
        else:
            print("Graph is already a single connected component. No changes needed.")
        self.substeps['largest component: weak components and filtering'] = time.time() - t_lcc

        t_report = time.time()
        self.report_connectivity(nodes_df, rels_df, removed_components, removed_nodes)
        self.substeps['largest component: connectivity report (weak and strong components)'] = time.time() - t_report

        t_write = time.time()
        nodes_csv, rels_csv = "fullgraph_nodes.csv", "fullgraph_rels.csv"
        nodes_df.to_csv(os.path.join(self.import_folder_path, nodes_csv), index=False)
        rels_df.to_csv(os.path.join(self.import_folder_path, rels_csv), index=False)
        self.substeps['largest component: writing the final CSV files'] = time.time() - t_write
        return nodes_csv, rels_csv

    def write_diagnostics(self, samples, workers_used):
        """Write cell_timings.csv and diagnostics.txt (measurements that explain the run time)"""
        out = []
        rows = sorted(self.cell_timings, key=lambda r: r['section'])
        keys = ['section', 'pid', 'start', 'end', 'requests', 'overpass_s', 'osmnx_build_s',
                'graph_total_s', 'write_s', 'nodes', 'edges']
        with open(os.path.join(self.import_folder_path, 'cell_timings.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)

        def stats(values):
            values = sorted(values)
            if not values:
                return "n/a"
            p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
            return (f"sum {sum(values):.1f} s, mean {statistics.mean(values):.3f} s, "
                    f"median {statistics.median(values):.3f} s, p95 {p95:.3f} s, max {values[-1]:.3f} s")

        cores = cpu_count()
        out.append(f"Workers: {workers_used} | CPU cores: {cores}")
        out.append(f"Cells measured: {len(rows)} (for a retried cell only the successful attempt is measured)")
        out.append(f"Overpass requests: {sum(r['requests'] for r in rows)}")
        out.append("Per cell - waiting for the Overpass server (request and response): "
                   + stats([r['overpass_s'] for r in rows]))
        out.append("Per cell - OSMnx graph construction in the worker: " + stats([r['osmnx_build_s'] for r in rows]))
        out.append("Per cell - writing the cell CSV files: " + stats([r['write_s'] for r in rows]))
        total_cell = sum(r['end'] - r['start'] for r in rows)
        total_overpass = sum(r['overpass_s'] for r in rows)
        if rows and self.download_wall:
            out.append(f"Download phase: {self.download_wall:.2f} s wall time; sum of the cell times {total_cell:.1f} s; "
                       f"workers busy {100.0 * total_cell / (workers_used * self.download_wall):.1f}% of the phase")
            out.append(f"Share of the cell time spent waiting for the Overpass server: "
                       f"{100.0 * total_overpass / max(total_cell, 1e-9):.1f}%")
        if self.download_cpu is not None and self.download_wall:
            out.append(f"CPU used by this run in the download phase: {self.download_cpu:.1f} CPU-seconds = "
                       f"{self.download_cpu / self.download_wall:.2f} cores on average out of {cores}")
        for phase in ('download', 'merge', 'largest_component'):
            ph = [x for x in samples if x[1] == phase]
            if not ph:
                continue
            cpu = [x[4] for x in ph]
            load = [x[5] for x in ph]
            over = [x[7] for x in ph if x[7] >= 0]
            line = (f"Phase {phase}: samples {len(ph)}, system CPU mean {statistics.mean(cpu):.1f}% / max {max(cpu):.1f}%, "
                    f"load average mean {statistics.mean(load):.2f} / max {max(load):.2f}, "
                    f"peak RAM all {max(x[2] for x in ph):.2f} GB / largest process {max(x[3] for x in ph):.2f} GB")
            if over and max(over) > 0:
                line += f", running Overpass server processes mean {statistics.mean(over):.1f} / max {max(over)}"
            else:
                line += ", Overpass server processes not visible from this user"
            out.append(line)
        for name, value in self.substeps.items():
            out.append(f"Sub-step time - {name}: {value:.2f} seconds")
        with open(os.path.join(self.import_folder_path, 'diagnostics.txt'), 'w') as f:
            f.write("\n".join(out) + "\n")
        print("\n--- Diagnostics ---")
        for line in out:
            print("Diagnostics - " + line)

    def process_section_wrapper(self, lat: float, lon: float, distance_km: float,
                                section_num: int, center_lat: float, center_lon: float,
                                total_distance_km: float) -> Tuple[int, Tuple[str, str]]:
        """Wrapper function for parallel processing"""
        try:
            print(f"\n--- Processing section {section_num} ---")
            result = self.process_single_section(lat, lon, distance_km, section_num, 
                                                 center_lat, center_lon, total_distance_km)
            return (section_num, result)
        except Exception as e:
            print(f"Error processing section {section_num}: {e}")
            return (section_num, None)

    def _process_section_star(self, task):
        return self.process_section_wrapper(*task)


def add_options():
    parser = argparse.ArgumentParser(description='Creation of routing graph for large areas.')
    parser.add_argument('--latitude', '-x', dest='lat', type=float, required=True)
    parser.add_argument('--longitude', '-y', dest='lon', type=float, required=True)
    parser.add_argument('--distance', '-d', dest='dist', type=float, required=True)
    parser.add_argument('--output', '-o', dest='output', type=str, required=True)
    parser.add_argument('--overpass-url', '-u', dest='overpass_url', type=str, default=None,
                        help='Overpass API URL, e.g. a local instance (default: public server)')
    parser.add_argument('--overpass-timeout', dest='overpass_timeout', type=int, default=None,
                        help='Overpass query timeout in seconds (default: OSMnx default, 180)')
    parser.add_argument('--overpass-memory', dest='overpass_memory', type=int, default=None,
                        help='Overpass query memory limit in MB (default: server default)')
    parser.add_argument('--max-retries', dest='max_retries', type=int, default=10,
                        help='Automatic retry rounds for failed sections (default: 10)')
    parser.add_argument('--retry-pause', dest='retry_pause', type=int, default=5,
                        help='Seconds to wait before each retry round (default: 5)')
    parser.add_argument('--workers', '-w', dest='workers', type=int, default=None,
                        help='Number of parallel processes (default: all CPU cores)')
    return parser


def get_user_input():
    """Get area size and overlap from user"""
    print("\n=== Large Area Graph Generator===")

    while True:
        try:
            area_size = float(input("Enter the area size for each section (in km): "))
            if area_size <= 0:
                print("Area size must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            overlap = float(input("Enter the overlap between sections (in km): "))
            if overlap < 0:
                print("Overlap cannot be negative. Please try again.")
                continue
            if overlap >= area_size:
                print("Overlap must be less than area size. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    return area_size, overlap


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)

    area_size_km, overlap_km = get_user_input()
    construction_start = time.time()
    os.makedirs(options.output, exist_ok=True)
    print(f"Load average before the run (1, 5, 15 min): {os.getloadavg()}")
    stop_mem, peak_mem, samples, write_samples = start_memory_monitor(
        samples_path=os.path.join(options.output, 'monitor_samples.csv'))

    use_overpass_server(options.overpass_url, options.overpass_timeout, options.overpass_memory)
    generator = LargeAreaGraphGenerator(options.output, overpass_url=options.overpass_url,
                                        overpass_timeout=options.overpass_timeout,
                                        overpass_memory=options.overpass_memory,
                                        max_retries=options.max_retries,
                                        retry_pause=options.retry_pause,
                                        workers=options.workers)

    try:
        generator.generate_large_area_graph(
            center_lat=options.lat,
            center_lon=options.lon,
            total_distance_km=options.dist,
            area_size_km=area_size_km,
            overlap_km=overlap_km
        )

    except Exception as e:
        print(f"Error during graph generation: {e}")
        raise

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    stop_mem.set()
    print(f"Construction time (excluding user input): {end_time - construction_start:.2f} seconds")
    print(f"Peak memory (all processes): {peak_mem[0] / 1024**3:.2f} GB")
    print(f"Peak memory (largest single process): {peak_mem[1] / 1024**3:.2f} GB")
    write_samples()
    generator.write_diagnostics(samples, options.workers if options.workers else cpu_count())
    return 0


if __name__ == "__main__":
    main()
