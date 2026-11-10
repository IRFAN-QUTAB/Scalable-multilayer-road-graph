import osmnx as ox
import argparse
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
import os
import time
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
        if GraphDatabase is None:
            raise ImportError("The neo4j package is not installed (pip install neo4j). "
                              "Use --output-file to run without Neo4j.")
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
                        CALL apoc.import.graphml($file, {storeNodeIds: true, defaultRelationshipType: 'ROUTE'});
                    """, file=file)
        return result.values()

    def get_import_folder(self):
        with self.driver.session() as session:
            result = session.execute_read(self._get_import_folder)
            return result

    @staticmethod
    def _get_import_folder(tx):
        result = tx.run("""
                        SHOW SETTINGS 'server.directories.import' YIELD name, value RETURN value;
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
                           MATCH (n:RoadJunction) SET n.location = point({latitude: tofloat(n.y), longitude: tofloat(n.x)}),
                                            n.lat = tofloat(n.y), 
                                            n.lon = tofloat(n.x),
                                            n.geometry='POINT(' + n.y + ' ' + n.x +')';
                       """)
        return result.values()

    def set_distance(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_distance)
            return result

    @staticmethod
    def _set_distance(tx):
        result = tx.run("""
                           MATCH (n:RoadJunction)-[r:ROUTE]-() SET r.distance=tofloat(r.length), r.status='active'
                       """)
        return result.values()

    def rename_junction_id(self):
        with self.driver.session() as session:
            return session.execute_write(self._rename_junction_id)

    @staticmethod
    def _rename_junction_id(tx):
        return tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.id IS NOT NULL
            SET n.osmnode_id = n.id
            REMOVE n.id
        """).values()

    def rename_route_osmid(self):
        with self.driver.session() as session:
            return session.execute_write(self._rename_route_osmid)

    @staticmethod
    def _rename_route_osmid(tx):
        return tx.run("""
            MATCH ()-[r:ROUTE]->()
            WHERE r.osmid IS NOT NULL
            SET r.osmway_id = r.osmid
            REMOVE r.osmid
        """).values()

    def set_index(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_index)
            return result

    @staticmethod
    def _set_index(tx):
        return tx.run("""
            CREATE INDEX IF NOT EXISTS
            FOR (n:RoadJunction)
            ON (n.osmnode_id)
        """).values()

    def generate_spatial_layer(self):
        with self.driver.session() as session:
            result = session.execute_write(self._generate_spatial_layer)
            return result

    @staticmethod
    def _generate_spatial_layer(tx):
        result = tx.run("""
                call spatial.layers()
                """)
        if len(result.values()) == 0:
            result = tx.run("""
                call spatial.addWKTLayer('spatial', 'geometry')
                """)
        return result.values()

    def import_nodes_in_spatial_layer(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_nodes_in_spatial_layer)
            return result

    @staticmethod
    def _import_nodes_in_spatial_layer(tx):
        result = tx.run("""
        match (n:RoadJunction)
        CALL spatial.addNode('spatial', n) yield node return node;
        """)
        return result.values()

def add_options():
    parser = argparse.ArgumentParser(description='Creation of routing graph.')
    parser.add_argument('--latitude', '-x', dest='lat', type=float, required=True)
    parser.add_argument('--longitude', '-y', dest='lon', type=float, required=True)
    parser.add_argument('--distance', '-d', dest='dist', type=float, required=True)
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str, default=None)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str, default=None)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str, default=None)
    parser.add_argument('--nameFile', '-f', dest='file_name', type=str, default=None)
    parser.add_argument('--output-file', dest='output_file', type=str, default=None,
                        help='Save the GraphML file here and skip the Neo4j import (no Neo4j needed)')
    parser.add_argument('--overpass-url', dest='overpass_url', type=str, default=None,
                        help='Overpass API URL, e.g. a local instance (default: public server)')
    parser.add_argument('--overpass-timeout', dest='overpass_timeout', type=int, default=None,
                        help='Overpass query timeout in seconds (default: OSMnx default, 180)')
    parser.add_argument('--overpass-memory', dest='overpass_memory', type=int, default=None,
                        help='Overpass query memory limit in MB (default: server default)')
    return parser

def main(args=None):
    start_time = time.time()
    stop_mem, peak_mem = start_memory_monitor()
    argParser = add_options()
    options = argParser.parse_args(args=args)
    use_overpass_server(options.overpass_url, options.overpass_timeout, options.overpass_memory)

    if options.output_file:
        greeter = None
        path = options.output_file
    else:
        if not (options.neo4jURL and options.neo4juser and options.neo4jpwd and options.file_name):
            argParser.error("--neo4jURL, --neo4juser, --neo4jpwd and --nameFile are required "
                            "unless --output-file is given")
        greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)

        import_folder = greeter.get_import_folder()[0][0]
        neo4j_home = os.environ.get('NEO4J_HOME', '')
        path = os.path.join(neo4j_home, import_folder, options.file_name)

    G = ox.graph_from_point(
        (options.lat, options.lon),
        dist=int(options.dist),
        dist_type='bbox',
        simplify=False,
        custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service"]'
    )

    for u, v, k, data in G.edges(keys=True, data=True):
        highway = data.get('highway')
        access = data.get('access')

        data['driveable'] = True

        if access in {"no", "private"}:
            data['driveable'] = False
        elif highway == 'service':
            data['driveable'] = False
        elif highway == 'residential' and access == 'private':
            data['driveable'] = False

    ox.save_graphml(G, path)
    construction_time = time.time() - start_time
    stop_mem.set()
    print(f"Construction time (download + GraphML): {construction_time:.2f} seconds")
    print(f"Peak memory (all processes): {peak_mem[0] / 1024**3:.2f} GB")
    print(f"Peak memory (largest single process): {peak_mem[1] / 1024**3:.2f} GB")

    wcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    largest = G.subgraph(wcc[0]) if wcc else G
    scc_sizes = [len(c) for c in nx.strongly_connected_components(largest)]
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Weakly connected components: {len(wcc)}")
    print(f"Largest weakly connected component: {largest.number_of_nodes()} nodes, "
          f"{largest.number_of_edges()} edges")
    print(f"Strongly connected components (in the largest component): {len(scc_sizes)}")
    if scc_sizes:
        print(f"Largest strongly connected component: {max(scc_sizes)} nodes "
              f"({100.0 * max(scc_sizes) / largest.number_of_nodes():.2f}%)")

    if greeter is None:
        return 0

    greeter.generate_spatial_layer()
    greeter.creation_graph(options.file_name)
    greeter.set_label()
    greeter.rename_junction_id()
    greeter.set_location()
    greeter.set_distance()
    greeter.rename_route_osmid()
    greeter.set_index()
    greeter.close()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Execution time: {elapsed:.2f} seconds")
    return 0

main()