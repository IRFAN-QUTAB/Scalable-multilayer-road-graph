import overpy
import json
from neo4j import GraphDatabase
import argparse
import os
import time
import math
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


KM_PER_DEGREE = math.pi * 6371.009 / 180


def way_centroid(coords):
    """Centroid of an OSM way from its (lat, lon) node coordinates (same function as in poi_layer_parallel.py).
    Closed way (building, area): centroid of the polygon (area-weighted, shoelace formula), computed in local
    metric coordinates. Open way or polygon without area: mean of the distinct nodes. None if no nodes."""
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


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

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

    def import_node(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_node)
            return result

    @staticmethod
    def _import_node(tx):
        result = tx.run("""
            CALL apoc.periodic.iterate(
                'CALL apoc.load.json("nodefile.json") YIELD value AS value 
                 WITH value.elements AS elements
                 UNWIND elements AS nodo
                 RETURN nodo',
                'MERGE (wn:OSMNode {osm_id: nodo.id})
                  ON CREATE SET wn.lat=tofloat(nodo.lat), 
                                wn.lon=tofloat(nodo.lon), 
                                wn.geometry="POINT(" + nodo.lat + " " + nodo.lon +")"
                MERGE (wn)-[:PART_OF]->(n:PointOfInterest {osm_id: nodo.id})
                  ON CREATE SET n.name=nodo.tags.name
                MERGE (n)-[:TAGS]->(t:Tag)
                  ON CREATE SET t += nodo.tags
                MERGE (wn)-[:TAGS]->(t)',
                {batchSize: 1000, parallel: true}
            ) YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """)
        return result.values()

    def import_node_way(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_node_way)
            return result

    @staticmethod
    def _import_node_way(tx):
        result = tx.run("""
            CALL apoc.periodic.iterate(
                'CALL apoc.load.json("nodeway.json") YIELD value AS value 
                 WITH value.elements AS elements
                 UNWIND elements AS nodo
                 RETURN nodo',
                'MERGE (wn:OSMNode {osm_id: nodo.id})
                  ON CREATE SET wn.lat=tofloat(nodo.lat), 
                                wn.lon=tofloat(nodo.lon), 
                                wn.geometry="POINT(" + nodo.lat + " " + nodo.lon +")"',
                {batchSize: 1000, parallel: true}
            ) YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """)
        return result.values()

    def import_way(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_way)
            return result

    @staticmethod
    def _import_way(tx):
        result = tx.run("""
            CALL apoc.periodic.iterate(
                'CALL apoc.load.json("wayfile.json") YIELD value 
                 WITH value.elements AS elements
                 UNWIND elements AS way
                 RETURN way',
                'MERGE (w:OSMWay:PointOfInterest {osm_id: way.id}) 
                  ON CREATE SET w.name = way.tags.name,
                                w.lat = way.lat,
                                w.lon = way.lon,
                                w.geometry = way.geometry
                MERGE (w)-[:TAGS]->(t:Tag) 
                  ON CREATE SET t += way.tags
                WITH w, way.nodes AS nodes
                UNWIND nodes AS node
                MATCH (wn:OSMNode {osm_id: node})
                MERGE (wn)-[:PART_OF]->(w)
                MERGE (wn)-[:TAGS]->(t)',
                {batchSize: 1000, parallel: false}
            ) YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """)
        return result.values()

    def import_nodes_into_spatial_layer(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_nodes_into_spatial_layer)
            return result

    @staticmethod
    def _import_nodes_into_spatial_layer(tx):
        result = tx.run("""
            CALL apoc.periodic.iterate(
                'MATCH (n:OSMNode) RETURN n',
                'CALL spatial.addNode("spatial", n) YIELD node
                 RETURN node',
                {batchSize: 1000, parallel: false}
            ) YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """)
        return result.values()

    def set_location(self):
        """Insert the location in the POI, OSMNode, and RoadJunction nodes."""
        with self.driver.session() as session:
            result = session.execute_write(self._set_location)
            return result

    @staticmethod
    def _set_location(tx):
        result = tx.run("""
            CALL apoc.periodic.iterate(
                'MATCH (n) WHERE (n:OSMNode OR n:OSMWay) AND n.lat IS NOT NULL AND n.lon IS NOT NULL RETURN n',
                'SET n.location = point({latitude: tofloat(n.lat), longitude: tofloat(n.lon)})',
                {batchSize: 10000, parallel: true}
            ) YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """)
        return result.values()

    def set_index(self):
        with self.driver.session() as session:
            try:
                result = session.execute_write(self._set_index)
                return result
            except Exception as e:
                print(f"Index creation skipped or failed: {e}")

    @staticmethod
    def _set_index(tx):
        try:
            tx.run("CREATE INDEX IF NOT EXISTS FOR (n:OSMNode) ON (n.osm_id)")
            tx.run("CREATE INDEX IF NOT EXISTS FOR (n:PointOfInterest) ON (n.osm_id)")
            tx.run("CREATE INDEX IF NOT EXISTS FOR (n:OSMWay) ON (n.osm_id)")
            tx.run("CREATE INDEX IF NOT EXISTS FOR (n:RoadJunction) ON (n.driveable)")
            tx.run("CREATE POINT INDEX IF NOT EXISTS FOR (n:RoadJunction) ON (n.location)")
            tx.run("CREATE POINT INDEX IF NOT EXISTS FOR (n:OSMNode) ON (n.location)")
            print("Indexes created successfully")
        except Exception as e:
            print(f"Failed to create index: {e}")
        return []

    def mark_driveable_roadjunctions(self):
        with self.driver.session() as session:
            session.execute_write(self._mark_driveable_roadjunctions)

    @staticmethod
    def _mark_driveable_roadjunctions(tx):
        tx.run("""
               MATCH ()-[r:ROUTE]-()
               WHERE r.driveable = 'True' OR r.driveable = 'true'
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

    def connect_amenity_batch(self):
        """Connect the POI and OSMNode to the nearest RoadJunction."""
        with self.driver.session() as session:
            result = session.execute_write(self._connect_amenity_batch)

    @staticmethod
    def _connect_amenity_batch(tx):

        result = tx.run("""
            MATCH (osmn)
            WHERE (osmn:OSMNode OR osmn:OSMWay)
              AND NOT (osmn)-[:NEAR]->(:RoadJunction)
              AND osmn.lat IS NOT NULL AND osmn.lon IS NOT NULL
            WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS nodeLoc
            MATCH (rj:RoadJunction)
            WHERE rj.driveable = true AND rj.location IS NOT NULL AND point.distance(rj.location, nodeLoc) < 100
            WITH osmn, rj, point.distance(rj.location, nodeLoc) AS dist
            ORDER BY dist
            WITH osmn, collect({rj: rj, dist: dist})[0] AS nearest
            WHERE nearest IS NOT NULL
            WITH osmn, nearest.rj AS nearestRJ, nearest.dist AS nearestDist
            MERGE (osmn)-[r:NEAR]->(nearestRJ)
              ON CREATE SET r.distance = nearestDist, r.status = 'driveable_within_100m'
        """)

        result = tx.run("""
            // Connect remaining OSMNodes to nearest non-driveable RoadJunction
            MATCH (osmn)
            WHERE (osmn:OSMNode OR osmn:OSMWay)
              AND NOT (osmn)-[:NEAR]->(:RoadJunction)
              AND osmn.lat IS NOT NULL AND osmn.lon IS NOT NULL
            WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS osmLoc
            CALL {
              WITH osmn, osmLoc
              MATCH (rj:RoadJunction)
              WHERE rj.driveable = false AND rj.location IS NOT NULL
              RETURN rj, point.distance(osmLoc, rj.location) AS dist
              ORDER BY dist ASC
              LIMIT 1
            }
            MERGE (osmn)-[r:NEAR]->(rj)
              ON CREATE SET r.distance = dist, r.status = 'nearest_non_driveable'
        """)

        result = tx.run("""
            MATCH (osmn)-[:NEAR]->(nonRJ:RoadJunction)
            WHERE (osmn:OSMNode OR osmn:OSMWay)
              AND nonRJ.driveable = false 
              AND NOT (osmn)-[:NEAR]->(:RoadJunction {driveable: true})
              AND osmn.lat IS NOT NULL AND osmn.lon IS NOT NULL
            WITH DISTINCT osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS osmLoc
            CALL {
              WITH osmLoc
              MATCH (drvRJ:RoadJunction)
              WHERE drvRJ.driveable = true AND drvRJ.location IS NOT NULL
              RETURN drvRJ, point.distance(osmLoc, drvRJ.location) AS dist
              ORDER BY dist ASC
              LIMIT 1
            }
            MERGE (osmn)-[r:NEAR]->(drvRJ)
              ON CREATE SET r.distance = dist, r.status = 'nearest_driveable_from_non_driveable'
        """)

        return result.values()


def count_summary(driver):
    """Counts of the POI layer in the database, for the paper"""
    queries = [
        ("PointOfInterest nodes", "MATCH (n:PointOfInterest) RETURN count(n)"),
        ("OSMWay:PointOfInterest nodes", "MATCH (n:OSMWay) RETURN count(n)"),
        ("OSMNode nodes", "MATCH (n:OSMNode) RETURN count(n)"),
        ("Tag nodes", "MATCH (n:Tag) RETURN count(n)"),
        ("NEAR relationships", "MATCH ()-[r:NEAR]->() RETURN count(r)"),
        ("NEAR from OSMWay centroids", "MATCH (:OSMWay)-[r:NEAR]->() RETURN count(r)"),
        ("PART_OF relationships", "MATCH ()-[r:PART_OF]->() RETURN count(r)"),
        ("TAGS relationships", "MATCH ()-[r:TAGS]->() RETURN count(r)"),
        ("POI nodes with a location but without NEAR",
         "MATCH (n) WHERE (n:OSMNode OR n:OSMWay) AND n.lat IS NOT NULL AND NOT (n)-[:NEAR]->() RETURN count(n)"),
    ]
    lines = []
    with driver.session() as session:
        for label, q in queries:
            lines.append(f"{label}: {session.run(q).single()[0]}")
        for rec in session.run("MATCH ()-[r:NEAR]->() RETURN r.status AS status, count(r) AS n, "
                               "percentileCont(r.distance, 0.5) AS median, percentileCont(r.distance, 0.95) AS p95, "
                               "max(r.distance) AS max ORDER BY status"):
            lines.append(f"   NEAR {rec['status']}: {rec['n']} links, median {rec['median']:.1f} m, "
                         f"95th percentile {rec['p95']:.1f} m, max {rec['max']:.1f} m")
    return lines


class LargeAreaPOIGenerator:
    def __init__(self, greeter: App, overpass_url: str = None,
                 import_folder_path: str = '/var/lib/neo4j/import/'):
        self.greeter = greeter
        self.import_folder_path = import_folder_path
        self.overpass_api = overpy.Overpass(url=overpass_url) if overpass_url else overpy.Overpass()
        self.use_rate_limit = overpass_url is None
        self.rate_limit_lock = threading.Lock()
        self.last_api_call = 0
        self.min_api_interval = 1.0

    def get_import_folder_path(self):
        """Get the Neo4j import folder path"""
        if not self.import_folder_path:
            neo4j_home = self.greeter.get_path()[0][0]
            import_folder = self.greeter.get_import_folder_name()[0][0]
            self.import_folder_path = os.path.join(neo4j_home, import_folder)
        return self.import_folder_path

    @staticmethod
    def km_to_degrees(km_lat: float, km_lon: float, reference_lat: float) -> Tuple[float, float]:
        """Convert kilometers to degrees"""
        degrees_lat = km_lat / KM_PER_DEGREE
        degrees_lon = km_lon / (KM_PER_DEGREE * math.cos(math.radians(reference_lat)))
        return degrees_lat, degrees_lon

    def get_processed_sections(self):
        """Read or create a file to track processed section numbers"""
        processed_file = os.path.join(self.get_import_folder_path(), "processed_poi_sections.txt")
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
        processed_file = os.path.join(self.get_import_folder_path(), "processed_poi_sections.txt")
        with open(processed_file, 'a') as f:
            f.write(f"{section_num}\n")

    @staticmethod
    def study_area_bbox(center_lat, center_lon, total_distance_km):
        """Study area as in OSMnx bbox_from_point (Earth radius 6,371,009 m), the same square as the
        road network graph: (min_lat, min_lon, max_lat, max_lon)"""
        d_lat = math.degrees(total_distance_km * 1000 / 6371009)
        d_lon = d_lat / math.cos(math.radians(center_lat))
        return (center_lat - d_lat, center_lon - d_lon, center_lat + d_lat, center_lon + d_lon)

    def calculate_grid_points(self, center_lat: float, center_lon: float,
                              total_distance_km: float, area_size_km: float) -> List[Tuple[float, float]]:
        """Calculate all grid center points for the large area without overlap"""
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

    def rate_limited_overpass_query(self, query: str):
        """Execute Overpass query with rate limiting"""
        if not self.use_rate_limit:
            return self.overpass_api.query(query)
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_api_interval:
                time.sleep(self.min_api_interval - time_since_last)
            
            result = self.overpass_api.query(query)
            self.last_api_call = time.time()
            return result

    def query_section_data(self, lat: float, lon: float, area_size_km: float, section_num: int):
        """Query both ways and nodes for a section in a single optimized query"""
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

        try:
            result = self.rate_limited_overpass_query(f"""
                [out:json][timeout:90];
                (
                  node["amenity"]({min_lat},{min_lon},{max_lat},{max_lon});
                  way["amenity"]({min_lat},{min_lon},{max_lat},{max_lon});
                );
                (._;>;);
                out body;
            """)
            return result
        except Exception as e:
            print(f"Error querying section {section_num}: {e}")
            time.sleep(5)
            try:
                result = self.rate_limited_overpass_query(f"""
                    [out:json][timeout:90];
                    (
                      node["amenity"]({min_lat},{min_lon},{max_lat},{max_lon});
                      way["amenity"]({min_lat},{min_lon},{max_lat},{max_lon});
                    );
                    (._;>;);
                    out body;
                """)
                return result
            except Exception as e2:
                print(f"Retry failed for section {section_num}: {e2}")
                raise

    def process_single_section(self, lat: float, lon: float, section_num: int, area_size_km: float, spatial: bool):
        """Process a single grid section for POIs"""
        path = self.get_import_folder_path()
        
        result = self.query_section_data(lat, lon, area_size_km, section_num)
        
        list_node_way = []
        list_way = []
        way_node_ids = set()
        
        for way in result.ways:
            way_dict = {'type': 'way', 'id': way.id, 'tags': way.tags}
            node_ids = []
            
            for node in way.get_nodes(resolve_missing=False):
                node_dict = {
                    'type': 'node',
                    'id': node.id,
                    'id_way': way.id,
                    'lat': str(node.lat),
                    'lon': str(node.lon),
                    'geometry': f'POINT({node.lat} {node.lon})',
                    'tags': node.tags
                }
                list_node_way.append(node_dict)
                node_ids.append(node.id)
                way_node_ids.add(node.id)
            
            way_dict['nodes'] = node_ids
            centroid = way_centroid([(float(n.lat), float(n.lon)) for n in way.get_nodes(resolve_missing=False)])
            if centroid is not None:
                way_dict['lat'] = centroid[0]
                way_dict['lon'] = centroid[1]
                way_dict['geometry'] = f'POINT({centroid[0]} {centroid[1]})'
            list_way.append(way_dict)
        
        list_node = []
        for node in result.nodes:
            if node.id not in way_node_ids and node.tags.get('amenity'):
                node_dict = {
                    'type': 'node',
                    'id': node.id,
                    'lat': str(node.lat),
                    'lon': str(node.lon),
                    'geometry': f'POINT({node.lat} {node.lon})',
                    'tags': node.tags
                }
                list_node.append(node_dict)
        
        with open(os.path.join(path, 'nodeway.json'), "w") as f:
            json.dump({"elements": list_node_way}, f)
        print(f"Section {section_num}: Generated nodeway.json ({len(list_node_way)} nodes)")
        
        with open(os.path.join(path, "wayfile.json"), "w") as f:
            json.dump({"elements": list_way}, f)
        print(f"Section {section_num}: Generated wayfile.json ({len(list_way)} ways)")
        
        with open(os.path.join(path, 'nodefile.json'), "w") as f:
            json.dump({"elements": list_node}, f)
        print(f"Section {section_num}: Generated nodefile.json ({len(list_node)} nodes)")
        
        if list_node_way:
            self.greeter.import_node_way()
        if list_way:
            self.greeter.import_way()
        if list_node:
            self.greeter.import_node()
        
        print(f"Section {section_num}: Processed successfully")

    def generate_large_area_poi(self, center_lat: float, center_lon: float,
                                total_distance_km: float, area_size_km: float, spatial: bool):
        """Main method to generate POIs for a large area in grid-based batches"""
        print(f"Large area POI generation...")
        print(f"Center: ({center_lat}, {center_lon})")
        print(f"Total area: {total_distance_km}km")
        print(f"Section size: {area_size_km}km")
        self.study_bbox = self.study_area_bbox(center_lat, center_lon, total_distance_km)
        print(f"Study area: lat [{self.study_bbox[0]:.6f}, {self.study_bbox[2]:.6f}], "
              f"lon [{self.study_bbox[1]:.6f}, {self.study_bbox[3]:.6f}]")

        print("Creating database indexes...")
        self.greeter.set_index()

        grid_points = self.calculate_grid_points(center_lat, center_lon,
                                                 total_distance_km, area_size_km)

        processed_sections = self.get_processed_sections()
        print(f"Found {len(processed_sections)} previously processed sections")

        sections_to_process = [(i, lat, lon) for i, (lat, lon) in enumerate(grid_points, 1) 
                              if i not in processed_sections]
        
        print(f"Will process {len(sections_to_process)} new sections out of {len(grid_points)} total")

        try:
            for i, lat, lon in sections_to_process:
                print(f"\n--- Processing section {i}/{len(grid_points)} ---")
                self.process_single_section(lat, lon, i, area_size_km, spatial)
                self.save_processed_section(i)
                
                if i % 10 == 0 or i == len(grid_points):
                    print("\nRunning batch post-processing...")
                    self.greeter.set_location()
                    self.greeter.mark_driveable_roadjunctions()
                    self.greeter.connect_amenity_batch()
                    print("Batch post-processing complete")

            remaining = len(sections_to_process) % 10
            if remaining > 0:
                print("\nRunning final post-processing...")
                self.greeter.set_location()
                self.greeter.mark_driveable_roadjunctions()
                self.greeter.connect_amenity_batch()

            print("\n=== POI Import Completed Successfully ===")
            print(f"Processed {len(sections_to_process)} new sections")
            print(f"Total sections: {len(grid_points)}")

        except Exception as e:
            print(f"\n=== POI Import Failed ===")
            print(f"Error during POI generation: {e}")
            raise


def add_options():
    parser = argparse.ArgumentParser(description='Insertion of POI in the graph.')
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str, required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str, required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str, required=True)
    parser.add_argument('--latitude', '-x', dest='lat', type=float, required=True)
    parser.add_argument('--longitude', '-y', dest='lon', type=float, required=True)
    parser.add_argument('--distance', '-d', dest='dist', type=float, required=True)
    parser.add_argument('--spatial', '-s', dest='spatial', type=str, required=False, default='False')
    parser.add_argument('--area-size', '-a', dest='area_size', type=float, required=True, help='Size of each grid section in km')
    parser.add_argument('--overpass-url', dest='overpass_url', type=str, default=None,
                        help='Overpass API URL, e.g. a local instance (default: public server)')
    parser.add_argument('--resume', action='store_true',
                        help='continue an interrupted run (keep the progress file); without it the progress file '
                             'of an earlier run is removed first')
    parser.add_argument('--import-folder', dest='import_folder', type=str,
                        default='/var/lib/neo4j/import/',
                        help='Neo4j import folder (the JSON files are written here)')
    return parser


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)

    progress = os.path.join(options.import_folder, "processed_poi_sections.txt")
    if not options.resume and os.path.exists(progress):
        os.remove(progress)
        print(f"Removed old file {progress}")

    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)
    generator = LargeAreaPOIGenerator(greeter, overpass_url=options.overpass_url,
                                      import_folder_path=options.import_folder)

    try:
        generator.generate_large_area_poi(
            center_lat=options.lat,
            center_lon=options.lon,
            total_distance_km=options.dist,
            area_size_km=options.area_size,
            spatial=options.spatial.lower() == 'true'
        )
        total_time = time.time() - start_time
        side = 2 * options.dist
        bbox = generator.study_bbox
        n_cells = len(generator.calculate_grid_points(options.lat, options.lon, options.dist, options.area_size))
        lines = [f"POIs (Neo4j) - results for the paper ({time.strftime('%Y-%m-%d %H:%M')})",
                 f"Overpass: {options.overpass_url or 'public server'}",
                 f"Centre: {options.lat}, {options.lon}; D = {options.dist} km",
                 f"Study area: lat {bbox[0]:.6f} to {bbox[2]:.6f}, lon {bbox[1]:.6f} to {bbox[3]:.6f}; "
                 f"{side:g} x {side:g} km = {side * side:g} km2",
                 f"POI cells: {n_cells} (cell size {options.area_size} km)", ""]
        lines += count_summary(greeter.driver)
        lines += ["", f"Total execution time: {total_time:.2f} seconds"]
        path = os.path.join(options.import_folder, 'poi_paper_results_neo4j.txt')
        with open(path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"Results for the paper written to {path}")
    except Exception as e:
        print(f"Error during POI generation: {e}")
        raise
    finally:
        greeter.close()

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    return 0


if __name__ == "__main__":
    main()