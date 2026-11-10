import osmnx as ox
import argparse
from neo4j import GraphDatabase
import os
import time
import pandas as pd
import math
from typing import List, Tuple
import networkx as nx
import threading
import psutil

ox.settings.use_cache = False


from osmnx import _overpass as _ox_overpass
_original_overpass_request = _ox_overpass._overpass_request


def _overpass_request_drop_incomplete_ways(data, **kwargs):
    response = _original_overpass_request(data, **kwargs)
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


def start_memory_monitor(interval=0.5):
    """Record the peak RAM during the run:
    peak[0] = all processes together (main + workers), peak[1] = largest single process."""
    proc = psutil.Process(os.getpid())
    peak = [0, 0]
    stop = threading.Event()

    def run():
        while not stop.is_set():
            sizes = [proc.memory_info().rss]
            for c in proc.children(recursive=True):
                try:
                    sizes.append(c.memory_info().rss)
                except psutil.Error:
                    pass
            peak[0] = max(peak[0], sum(sizes))
            peak[1] = max(peak[1], max(sizes))
            stop.wait(interval)

    threading.Thread(target=run, daemon=True).start()
    return stop, peak


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


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def creation_graph(self, file):
        with self.driver.session() as session:
            result = session.execute_write(self._creation_graph, file)
            return result

    @staticmethod
    def _creation_graph(tx, file):
        result = tx.run("""
                        CALL apoc.import.graphml($file, {storeNodeIds: true, defaultRelationshipType: 'ROUTE', batchSize:2000});
                    """, file=file)
        return result.values()

    def get_path(self):
        with self.driver.session() as session:
            result = session.execute_write(self._get_path)
            return result

    @staticmethod
    def _get_path(tx):
        result = tx.run("""
                            Call dbms.listConfig() yield name,value where name = 'dbms.directories.neo4j_home' return value;
                        """)
        return result.values()

    def get_import_folder_name(self):
        with self.driver.session() as session:
            result = session.execute_write(self._get_import_folder_name)
            return result

    @staticmethod
    def _get_import_folder_name(tx):
        result = tx.run("""
                        Call dbms.listConfig() yield name,value where name = 'dbms.directories.import' return value;
                    """)
        return result.values()

    def set_label(self):
        with self.driver.session() as session:
            result = session.execute_write(self._creation_label)
            return result

    @staticmethod
    def _creation_label(tx):
        result = tx.run("""
                        MATCH (n) SET n:RoadJunction;
                    """)
        return result.values()

    def set_location(self):
        with self.driver.session() as session:
            result = session.execute_write(self._creation_location)
            return result

    @staticmethod
    def _creation_location(tx):
        result = tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.y IS NOT NULL AND n.x IS NOT NULL
            SET n.location = point({latitude: toFloat(n.y), longitude: toFloat(n.x)}),
                n.lat = toFloat(n.y),
                n.lon = toFloat(n.x),
                n.geometry = CASE
                    WHEN n.y IS NOT NULL AND n.x IS NOT NULL
                    THEN 'POINT(' + toString(toFloat(n.y)) + ' ' + toString(toFloat(n.x)) + ')'
                    ELSE null
                END
        """)
        return result.values()

    def set_geometry(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_geometry)
            return result

    @staticmethod
    def _set_geometry(tx):
        result = tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.lon IS NOT NULL AND n.lat IS NOT NULL AND n.geometry IS NULL
            SET n.geometry = 'POINT(' + toString(n.lat) + ' ' + toString(n.lon) + ')'
        """)
        return result.values()

    def set_distance(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_distance)
            return result

    @staticmethod
    def _set_distance(tx):
        result = tx.run("""
                           MATCH (n:RoadJunction)-[r:ROUTE]-() 
                           SET r.distance=tofloat(r.length), r.status='active'
                       """)
        return result.values()

    def export_to_csv(self, nodes_filename, rels_filename, center_lat, center_lon, total_distance_km):
        """Export nodes and relationships to separate CSVs with individual property headers, filtered by square boundary"""
        with self.driver.session() as session:
            result = session.execute_write(self._export_to_csv, nodes_filename, rels_filename, center_lat,
                                               center_lon,
                                               total_distance_km)
            return result

    @staticmethod
    def _export_to_csv(tx, nodes_filename, rels_filename, center_lat, center_lon, total_distance_km):
        min_lon, min_lat, max_lon, max_lat = ox.utils_geo.bbox_from_point(
            (center_lat, center_lon), dist=total_distance_km * 1000)

        nodes_result = tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.lat >= $min_lat AND n.lat <= $max_lat
            AND n.lon >= $min_lon AND n.lon <= $max_lon
            RETURN n.id AS id, 
                   n.lat AS lat, 
                   n.lon AS lon, 
                   n.highway AS highway, 
                   n.osmid AS osmid,
                   n.geometry AS geometry,
                   n.street_count AS street_count,
                   n.unit AS unit
        """, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

        nodes_records = []
        for record in nodes_result:
            nodes_records.append({
                'id': record['id'],
                'lat': record['lat'],
                'lon': record['lon'],
                'highway': record['highway'],
                'osmid': record['osmid'],
                'geometry': record['geometry'],
                'street_count': record['street_count'],
                'unit': record['unit']
            })

        rels_result = tx.run("""
            MATCH (n:RoadJunction)-[r:ROUTE]->(m:RoadJunction)
            WHERE n.lat >= $min_lat AND n.lat <= $max_lat
            AND n.lon >= $min_lon AND n.lon <= $max_lon
            AND m.lat >= $min_lat AND m.lat <= $max_lat
            AND m.lon >= $min_lon AND m.lon <= $max_lon
            RETURN n.id AS source_id, 
                   m.id AS target_id, 
                   r.distance AS distance, 
                   r.length AS length, 
                   r.status AS status, 
                   r.highway AS highway, 
                   r.access AS access, 
                   r.driveable AS driveable, 
                   r.name AS name, 
                   r.maxspeed AS maxspeed,
                   r.oneway AS oneway,
                   r.osmid AS osmid,
                   r.reversed AS reversed,
                   r.lanes AS lanes,
                   r.ref AS ref,
                   r.junction AS junction
        """, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

        rels_records = []
        for record in rels_result:
            rels_records.append({
                'source_id': record['source_id'],
                'target_id': record['target_id'],
                'distance': record['distance'],
                'length': record['length'],
                'status': record['status'],
                'highway': record['highway'],
                'access': record['access'],
                'driveable': record['driveable'],
                'name': record['name'],
                'maxspeed': record['maxspeed'],
                'oneway': record['oneway'],
                'osmid': record['osmid'],
                'reversed': record['reversed'],
                'lanes': record['lanes'],
                'ref': record['ref'],
                'junction': record['junction']
            })

        return nodes_records, rels_records

    @staticmethod
    def km_to_degrees(km_lat: float, km_lon: float, reference_lat: float) -> Tuple[float, float]:
        """Convert kilometers to degrees (same Earth radius as OSMnx, 6,371,009 m)"""
        earth_radius_km = 6371.009
        degrees_lat = math.degrees(km_lat / earth_radius_km)
        degrees_lon = math.degrees(km_lon / earth_radius_km) / math.cos(math.radians(reference_lat))
        return degrees_lat, degrees_lon

    def clear_database(self):
        """Clear all nodes and relationships from Neo4j"""
        with self.driver.session() as session:
            result = session.execute_write(self._clear_database)
            return result

    @staticmethod
    def _clear_database(tx):
        tx.run("MATCH (n:RoadJunction) DETACH DELETE n")
        return True

    def drop_index_if_exists(self):
        """Drop the index on :RoadJunction(id) if it exists"""
        with self.driver.session() as session:
            session.execute_write(self._drop_index_if_exists)

    @staticmethod
    def _drop_index_if_exists(tx):
        tx.run("""
            CALL db.indexes() YIELD name, labelsOrTypes, properties
            WHERE labelsOrTypes = ['RoadJunction'] AND properties = ['id']
            WITH 'DROP INDEX ' + name AS dropStatement
            CALL apoc.cypher.run(dropStatement, {}) YIELD value
            RETURN value
        """)

    def set_index(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_index)
            return result

    @staticmethod
    def _set_index(tx):
        result = tx.run("""
                       CREATE INDEX ON :RoadJunction(id)
                   """)
        return result.values()

    def generate_spatial_layer(self):
        with self.driver.session() as session:
            result = session.execute_write(self._generate_spatial_layer)
            return result

    @staticmethod
    def _generate_spatial_layer(tx):
        result = tx.run("""
                    CALL spatial.layers()
                    """)
        if len(result.values()) == 0:
            result = tx.run("""
                    CALL spatial.addWKTLayer('spatial', 'geometry')
                    """)
        return result.values()

    def import_nodes_in_spatial_layer(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_nodes_in_spatial_layer)
            return result

    @staticmethod
    def _import_nodes_in_spatial_layer(tx):
        tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.geometry IS NULL AND n.lat IS NOT NULL AND n.lon IS NOT NULL
            SET n.geometry = 'POINT(' + toString(n.lat) + ' ' + toString(n.lon) + ')'
        """)
        result = tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.geometry IS NOT NULL
            CALL spatial.addNode('spatial', n) YIELD node
            RETURN node
        """)
        return result.values()

    def import_csv_to_neo4j(self, csv_filename):
        """Import CSV data back to Neo4j in batches using apoc.periodic.iterate"""
        with self.driver.session() as session:
            result = session.execute_write(self._import_csv_to_neo4j, csv_filename)
            return result

    @staticmethod
    def _import_csv_to_neo4j(tx, csv_filename):
        result = tx.run("""
            CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM $file AS row RETURN row',
                'MERGE (source:RoadJunction {id: row.source_id})
                 SET source.lat = toFloat(row.source_lat),
                     source.lon = toFloat(row.source_lon),
                     source.highway = row.source_highway,
                     source.osmid = row.source_osmid,
                     source.geometry = row.source_geometry,
                     source.street_count = toInteger(row.source_street_count),
                     source.unit = toFloat(row.source_unit)
                 MERGE (target:RoadJunction {id: row.target_id})
                 SET target.lat = toFloat(row.target_lat),
                     target.lon = toFloat(row.target_lon),
                     target.highway = row.target_highway,
                     target.osmid = row.target_osmid,
                     target.geometry = row.target_geometry,
                     target.street_count = toInteger(row.target_street_count),
                     target.unit = toFloat(row.target_unit)
                 MERGE (source)-[r:ROUTE]->(target)
                 SET r.distance = toFloat(row.route_distance),
                     r.length = toFloat(row.route_length),
                     r.status = row.route_status,
                     r.highway = row.route_highway,
                     r.access = row.route_access,
                     r.driveable = row.route_driveable,
                     r.name = row.route_name,
                     r.maxspeed = row.route_maxspeed,
                     r.oneway = row.route_oneway,
                     r.osmid = row.route_osmid,
                     r.reversed = row.route_reversed',
                {batchSize: 2000, parallel: false, params: {file: $file}}
            )
            YIELD batches, total RETURN batches, total
        """, file=f'file:///{csv_filename}')
        return result.values()

    def mark_driveable_roadjunctions(self):
        with self.driver.session() as session:
            session.execute_write(self._mark_driveable_roadjunctions)

    @staticmethod
    def _mark_driveable_roadjunctions(tx):
        tx.run("""
               MATCH ()-[r:ROUTE]-()
               WHERE r.driveable = 'True' OR r.driveable = 'true' OR r.driveable = true
               SET r.driveable = true
           """).consume()

        tx.run("""
               MATCH (n:RoadJunction)
               SET n.driveable = false
           """).consume()

        tx.run("""
               MATCH (n:RoadJunction)-[r:ROUTE]-()
               WHERE r.driveable = true
               SET n.driveable = true
           """).consume()

class LargeAreaGraphGenerator:
    def __init__(self, greeter: App, import_folder_path: str = "/var/lib/neo4j/import/",
                 overpass_url: str = None, overpass_timeout: int = None, overpass_memory: int = None,
                 max_retries: int = 10, retry_pause: int = 5):
        self.greeter = greeter
        self.import_folder_path = import_folder_path
        self.overpass_url = overpass_url
        self.overpass_timeout = overpass_timeout
        self.overpass_memory = overpass_memory
        self.max_retries = max_retries
        self.retry_pause = retry_pause

    def get_import_folder_path(self):
        """Get the Neo4j import folder path"""
        if not self.import_folder_path:
            neo4j_home = self.greeter.get_path()[0][0]
            import_folder = self.greeter.get_import_folder_name()[0][0]
            self.import_folder_path = os.path.join(neo4j_home, import_folder)
        return self.import_folder_path

    def get_processed_sections(self):
        """Read or create a file to track processed section numbers"""
        processed_file = os.path.join(self.get_import_folder_path(), "processed_graph_sections.txt")
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
        processed_file = os.path.join(self.get_import_folder_path(), "processed_graph_sections.txt")
        with open(processed_file, 'a') as f:
            f.write(f"{section_num}\n")

    def section_files_exist(self, section_num: int) -> bool:
        """Check if both nodes and rels CSV files exist for a section"""
        nodes_csv_name = f"nodes_{section_num}.csv"
        rels_csv_name = f"rels_{section_num}.csv"
        nodes_path = os.path.join(self.get_import_folder_path(), nodes_csv_name)
        rels_path = os.path.join(self.get_import_folder_path(), rels_csv_name)
        return os.path.exists(nodes_path) and os.path.exists(rels_path)

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
                deg_lat, deg_lon = App.km_to_degrees(offset_lat_km, offset_lon_km, center_lat)
                new_lat = center_lat + deg_lat
                new_lon = center_lon + deg_lon
                points.append((new_lat, new_lon))

        return points

    def generate_single_graph(self, lat: float, lon: float, distance_km: float,
                              graph_name: str) -> str:
        """Generate a single graph and return the GraphML filename (None for a cell without roads)"""
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
        except Exception as e:
            msg = str(e).lower()
            if 'no data elements' in msg or 'no graph nodes' in msg:
                print(f"Section {graph_name}: no road data in this cell")
                return None
            raise

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

        graphml_filename = f"{graph_name}.graphml"
        graphml_path = os.path.join(self.get_import_folder_path(), graphml_filename)
        ox.save_graphml(G, graphml_path)

        return graphml_filename

    def process_single_section(self, lat: float, lon: float, distance_km: float,
                               section_num: int, center_lat: float, center_lon: float,
                               total_distance_km: float) -> Tuple[str, str]:
        """Process a single section and return nodes and relationships CSV filenames"""
        graph_name = f"g{section_num}"
        nodes_csv_name = f"nodes_{section_num}.csv"
        rels_csv_name = f"rels_{section_num}.csv"

        graphml_filename = self.generate_single_graph(lat, lon, distance_km, graph_name)

        nodes_csv_path = os.path.join(self.get_import_folder_path(), nodes_csv_name)
        rels_csv_path = os.path.join(self.get_import_folder_path(), rels_csv_name)
        if graphml_filename is None:
            print(f"Section {section_num} is empty, creating empty CSV files")
            pd.DataFrame(columns=['id']).to_csv(nodes_csv_path, index=False)
            pd.DataFrame(columns=['source_id', 'target_id']).to_csv(rels_csv_path, index=False)
            return nodes_csv_name, rels_csv_name

        self.greeter.clear_database()

        print(f"Importing {graph_name} to Neo4j...")
        self.greeter.generate_spatial_layer()
        self.greeter.creation_graph(graphml_filename)
        self.greeter.set_label()
        self.greeter.set_location()
        self.greeter.set_distance()
        self.greeter.mark_driveable_roadjunctions()
        self.greeter.import_nodes_in_spatial_layer()

        print(f"Exporting {graph_name} to {nodes_csv_name} and {rels_csv_name}...")
        nodes_data, rels_data = self.greeter.export_to_csv(nodes_csv_name, rels_csv_name, center_lat, center_lon, total_distance_km)

        nodes_df = pd.DataFrame(nodes_data, columns=['id', 'lat', 'lon', 'highway', 'osmid', 'geometry',
                                                     'street_count', 'unit'])
        nodes_df.to_csv(nodes_csv_path, index=False)
        rels_df = pd.DataFrame(rels_data, columns=['source_id', 'target_id', 'distance', 'length', 'status',
                                                   'highway', 'access', 'driveable', 'name', 'maxspeed',
                                                   'oneway', 'osmid', 'reversed', 'lanes', 'ref', 'junction'])
        rels_df.to_csv(rels_csv_path, index=False)
        print(f"Saved {len(nodes_data)} nodes to {nodes_csv_name} and {len(rels_data)} relationships to {rels_csv_name}")

        print(f"Clearing Neo4j database...")
        self.greeter.clear_database()

        return nodes_csv_name, rels_csv_name

    def merge_csv_files(self, nodes_csv_files: List[str], rels_csv_files: List[str]) -> Tuple[str, str]:
        """Merge all nodes and relationships CSV files into two separate files with neo4j-admin import compatible headers"""
        print(f"Merging {len(nodes_csv_files)} nodes CSV files and {len(rels_csv_files)} relationships CSV files...")

        all_nodes_data = []
        for nodes_csv_file in nodes_csv_files:
            nodes_csv_path = os.path.join(self.get_import_folder_path(), nodes_csv_file)
            if os.path.exists(nodes_csv_path):
                df = pd.read_csv(nodes_csv_path)
                if len(df) > 0:
                    all_nodes_data.append(df)
                    print(f"Loaded {len(df)} nodes from {nodes_csv_file}")

        all_rels_data = []
        for rels_csv_file in rels_csv_files:
            rels_csv_path = os.path.join(self.get_import_folder_path(), rels_csv_file)
            if os.path.exists(rels_csv_path):
                df = pd.read_csv(rels_csv_path)
                if len(df) > 0:
                    all_rels_data.append(df)
                    print(f"Loaded {len(df)} relationships from {rels_csv_file}")

        combined_nodes_df = pd.concat(all_nodes_data, ignore_index=True, sort=False)
        if 'street_count' in combined_nodes_df.columns:
            street_count = pd.to_numeric(combined_nodes_df['street_count'], errors='coerce')
            combined_nodes_df['street_count'] = (street_count.groupby(combined_nodes_df['id'])
                                                 .transform('max').astype('Int64'))
        combined_nodes_df = combined_nodes_df.drop_duplicates(subset=['id'], keep='first')

        combined_rels_df = pd.concat(all_rels_data, ignore_index=True, sort=False)
        combined_rels_df = combined_rels_df.drop_duplicates(subset=['source_id', 'target_id', 'osmid'], keep='first')
        print(f"After deduplication: {len(combined_nodes_df)} unique nodes, {len(combined_rels_df)} unique relationships")

        combined_nodes_df, combined_rels_df = self.keep_largest_component(combined_nodes_df, combined_rels_df)

        available_node_columns = combined_nodes_df.columns.tolist()
        print(f"Available node columns: {available_node_columns}")
        
        node_column_mapping = {
            'id': 'id:ID(RoadJunction)',
            'lat': 'lat:FLOAT',
            'lon': 'lon:FLOAT',
            'highway': 'highway:STRING',
            'osmid': 'osmid:STRING',
            'geometry': 'geometry:STRING',
            'street_count': 'street_count:INT',
            'unit': 'unit:FLOAT'
        }
        
        columns_to_select = [col for col in node_column_mapping.keys() if col in available_node_columns]
        rename_mapping = {col: node_column_mapping[col] for col in columns_to_select}
        
        nodes_df = combined_nodes_df[columns_to_select].rename(columns=rename_mapping)

        available_rel_columns = combined_rels_df.columns.tolist()
        print(f"Available relationship columns: {available_rel_columns}")
        
        rel_column_mapping = {
            'source_id': ':START_ID(RoadJunction)',
            'target_id': ':END_ID(RoadJunction)',
            'distance': 'distance:FLOAT',
            'length': 'length:FLOAT',
            'status': 'status:STRING',
            'highway': 'highway:STRING',
            'access': 'access:STRING',
            'driveable': 'driveable:STRING',
            'name': 'name:STRING',
            'maxspeed': 'maxspeed:STRING',
            'oneway': 'oneway:STRING',
            'osmid': 'osmid:STRING',
            'reversed': 'reversed:STRING',
            'lanes': 'lanes:STRING',
            'ref': 'ref:STRING',
            'junction': 'junction:STRING'
        }
        
        rel_columns_to_select = [col for col in rel_column_mapping.keys() if col in available_rel_columns]
        rel_rename_mapping = {col: rel_column_mapping[col] for col in rel_columns_to_select}
        
        rels_df = combined_rels_df[rel_columns_to_select].rename(columns=rel_rename_mapping)
        rels_df[':TYPE'] = 'ROUTE'

        nodes_filename = "fullgraph_nodes.csv"
        rels_filename = "fullgraph_rels.csv"
        nodes_path = os.path.join(self.get_import_folder_path(), nodes_filename)
        rels_path = os.path.join(self.get_import_folder_path(), rels_filename)
        nodes_df.to_csv(nodes_path, index=False)
        rels_df.to_csv(rels_path, index=False)

        print(f"Merged nodes CSV saved as {nodes_filename} with {len(nodes_df)} unique nodes")
        print(f"Merged relationships CSV saved as {rels_filename} with {len(rels_df)} unique relationships")
        return nodes_filename, rels_filename

    def report_connectivity(self, nodes_df, rels_df, removed_components, removed_nodes):
        """Write weak and strong connectivity of the final graph to connectivity_report.txt"""
        DG = nx.DiGraph()
        DG.add_nodes_from(nodes_df['id'].astype(str))
        DG.add_edges_from(zip(rels_df['source_id'].astype(str), rels_df['target_id'].astype(str)))
        wcc = nx.number_weakly_connected_components(DG)
        scc_sizes = [len(c) for c in nx.strongly_connected_components(DG)]
        largest_scc = max(scc_sizes) if scc_sizes else 0
        lines = [
            f"Nodes: {DG.number_of_nodes()}",
            f"Edges: {len(rels_df)}",
            f"Smaller components removed: {removed_components} ({removed_nodes} nodes)",
            f"Weakly connected components: {wcc}",
            f"Strongly connected components: {len(scc_sizes)}",
            f"Largest strongly connected component: {largest_scc} nodes "
            f"({100.0 * largest_scc / max(DG.number_of_nodes(), 1):.2f}%)",
        ]
        with open(os.path.join(self.get_import_folder_path(), "connectivity_report.txt"), 'w') as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))

    def keep_largest_component(self, nodes_df, rels_df):
        """Keep only the largest (weakly) connected component of the merged graph"""
        G = nx.Graph()
        G.add_nodes_from(nodes_df['id'].astype(str))
        G.add_edges_from(zip(rels_df['source_id'].astype(str), rels_df['target_id'].astype(str)))
        components = list(nx.connected_components(G))
        print(f"Found {len(components)} connected components")
        if len(components) <= 1:
            self.report_connectivity(nodes_df, rels_df, 0, 0)
            return nodes_df, rels_df
        largest = max(components, key=len)
        removed_nodes = G.number_of_nodes() - len(largest)
        print(f"Removing {removed_nodes} nodes from {len(components) - 1} smaller components")
        nodes_df = nodes_df[nodes_df['id'].astype(str).isin(largest)]
        rels_df = rels_df[rels_df['source_id'].astype(str).isin(largest) &
                          rels_df['target_id'].astype(str).isin(largest)]
        self.report_connectivity(nodes_df, rels_df, len(components) - 1, removed_nodes)
        return nodes_df, rels_df

    def generate_large_area_graph(self, center_lat: float, center_lon: float,
                                  total_distance_km: float, area_size_km: float,
                                  overlap_km: float):
        """Main method to generate large area graph"""
        print(f"Large area graph generation...")
        print(f"Center: ({center_lat}, {center_lon})")
        print(f"Total area: {total_distance_km}km")
        print(f"Section size: {area_size_km}km")
        print(f"Overlap: {overlap_km}km")

        grid_points = self.calculate_grid_points(center_lat, center_lon,
                                                 total_distance_km, area_size_km, overlap_km)
        t_phase_start = time.time()

        processed_sections = self.get_processed_sections()
        print(f"Found {len(processed_sections)} previously processed sections: {processed_sections}")

        print(f"Will generate {len(grid_points)} sections")

        nodes_csv_files = []
        rels_csv_files = []
        
        try:
            pending = []
            for i, (lat, lon) in enumerate(grid_points, 1):
                if i in processed_sections and self.section_files_exist(i):
                    print(f"\n--- Skipping section {i}/{len(grid_points)} (already processed) ---")
                    nodes_csv_files.append(f"nodes_{i}.csv")
                    rels_csv_files.append(f"rels_{i}.csv")
                    continue
                pending.append((i, lat, lon))

            failed = []
            for attempt in range(self.max_retries + 1):
                failed = []
                for i, lat, lon in pending:
                    print(f"\n--- Processing section {i}/{len(grid_points)} ---")
                    try:
                        nodes_csv_filename, rels_csv_filename = self.process_single_section(
                            lat, lon, area_size_km / 2 + 0.01, i, center_lat, center_lon, total_distance_km
                        )
                    except Exception as e:
                        print(f"Error processing section {i}: {e}")
                        failed.append((i, lat, lon))
                        continue
                    nodes_csv_files.append(nodes_csv_filename)
                    rels_csv_files.append(rels_csv_filename)
                    self.save_processed_section(i)

                if not failed:
                    break
                if attempt < self.max_retries:
                    print(f"\n{len(failed)} sections failed: {sorted(f[0] for f in failed)}. "
                          f"Retry {attempt + 1}/{self.max_retries} in {self.retry_pause} seconds...")
                    time.sleep(self.retry_pause)
                    pending = failed

            if failed:
                raise RuntimeError(f"{len(failed)} sections failed after {self.max_retries} retries: "
                                   f"{sorted(f[0] for f in failed)}. Run the script again to retry only these sections.")

            t_download_end = time.time()
            print(f"\nPhase time - download, Neo4j import and export of the sections: {t_download_end - t_phase_start:.2f} seconds")
            print(f"\n--- Merging all CSV files ---")
            section_order = lambda f: int(f.split('_')[1].split('.')[0])
            nodes_csv_files.sort(key=section_order)
            rels_csv_files.sort(key=section_order)
            nodes_csv, rels_csv = self.merge_csv_files(nodes_csv_files, rels_csv_files)
            print(f"Phase time - merge, deduplication, largest connected component and CSV headers: {time.time() - t_download_end:.2f} seconds")

            print(f"\nfullgraph_nodes.csv and fullgraph_rels.csv completed successfully!")
            print(f"Files are ready to import in Neo4j using neo4j-admin import in offline mode.")
            print(f"Nodes CSV: {nodes_csv}")
            print(f"Relationships CSV: {rels_csv}")
            
            processed_file = os.path.join(self.get_import_folder_path(), "processed_graph_sections.txt")
            if os.path.exists(processed_file):
                os.remove(processed_file)
                print("Cleaned up progress tracking file.")
                
        except Exception as e:
            print(f"\n=== Graph generation interrupted ===")
            print(f"Progress saved. You can restart to resume from section {len(processed_sections) + 1}")
            raise


def add_options():
    parser = argparse.ArgumentParser(description='Creation of routing graph for large areas.')
    parser.add_argument('--latitude', '-x', dest='lat', type=float, required=True)
    parser.add_argument('--longitude', '-y', dest='lon', type=float, required=True)
    parser.add_argument('--distance', '-d', dest='dist', type=float, required=True)
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str, required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str, required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str, required=True)
    parser.add_argument('--import-folder', dest='import_folder', type=str,
                        default="/var/lib/neo4j/import/",
                        help='Neo4j import folder (GraphML and CSV files are written here)')
    parser.add_argument('--overpass-url', dest='overpass_url', type=str, default=None,
                        help='Overpass API URL, e.g. a local instance (default: public server)')
    parser.add_argument('--overpass-timeout', dest='overpass_timeout', type=int, default=None,
                        help='Overpass query timeout in seconds (default: OSMnx default, 180)')
    parser.add_argument('--overpass-memory', dest='overpass_memory', type=int, default=None,
                        help='Overpass query memory limit in MB (default: server default)')
    parser.add_argument('--max-retries', dest='max_retries', type=int, default=10,
                        help='Automatic retry rounds for failed sections (default: 10)')
    parser.add_argument('--retry-pause', dest='retry_pause', type=int, default=5,
                        help='Seconds to wait before each retry round (default: 5)')
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
    stop_mem, peak_mem = start_memory_monitor()

    use_overpass_server(options.overpass_url, options.overpass_timeout, options.overpass_memory)
    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)
    generator = LargeAreaGraphGenerator(greeter, import_folder_path=options.import_folder,
                                        overpass_url=options.overpass_url,
                                        overpass_timeout=options.overpass_timeout,
                                        overpass_memory=options.overpass_memory,
                                        max_retries=options.max_retries,
                                        retry_pause=options.retry_pause)

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
    finally:
       greeter.close()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    stop_mem.set()
    print(f"Construction time (excluding user input): {end_time - construction_start:.2f} seconds")
    print(f"Peak memory of the Python script: {peak_mem[1] / 1024**3:.2f} GB (Neo4j server not included)")
    return 0


if __name__ == "__main__":
    main()