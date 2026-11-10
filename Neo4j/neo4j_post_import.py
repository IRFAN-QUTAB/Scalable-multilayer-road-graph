from neo4j import GraphDatabase
import argparse


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def set_labels(self):
        with self.driver.session() as session:
            session.execute_write(self._set_labels)

    @staticmethod
    def _set_labels(tx):
        tx.run("""
            MATCH (n)
            WHERE n.node_type IS NULL OR n.node_type = 'RoadJunction'
            SET n:RoadJunction
        """)
        print("? RoadJunction labels set")

        tx.run("""
            MATCH (n)
            WHERE n.node_type = 'OSMNode'
            SET n:OSMNode
        """)
        print("? OSMNode labels set")

        tx.run("""
            MATCH (n)
            WHERE n.node_type = 'OSMNode' AND n.poi_type = 'OSMNode'
            SET n:POI
        """)
        print("? OSMNode:POI labels set")

        tx.run("""
            MATCH (n)
            WHERE n.node_type = 'OSMWay'
            SET n:OSMWay:POI
        """)
        print("? OSMWay:POI labels set")

        tx.run("""
            MATCH (n)
            WHERE n.node_type = 'Tag'
            SET n:Tag
        """)
        print("? Tag labels set")

        tx.run("""
            MATCH (n)
            WHERE n.node_type = 'RoadOsm'
            SET n:RoadOsm
        """)
        print("? RoadOsm labels set")

    def set_location(self):
        with self.driver.session() as session:
            session.execute_write(self._set_location)

    @staticmethod
    def _set_location(tx):
        tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.y IS NOT NULL AND n.x IS NOT NULL
            SET n.location = point({latitude: toFloat(n.y), longitude: toFloat(n.x)}),
                n.lat = toFloat(n.y),
                n.lon = toFloat(n.x),
                n.geometry = 'POINT(' + n.y + ' ' + n.x + ')'
        """)
        print("? RoadJunction locations set")

        tx.run("""
            MATCH (n:OSMNode)
            WHERE n.lat IS NOT NULL AND n.lon IS NOT NULL
            SET n.location = point({latitude: toFloat(n.lat), longitude: toFloat(n.lon)})
        """)
        print("? OSMNode locations set")

    def set_distance(self):
        with self.driver.session() as session:
            session.execute_write(self._set_distance)

    @staticmethod
    def _set_distance(tx):
        tx.run("""
            MATCH (n:RoadJunction)-[r:ROUTE]-()
            SET r.distance = toFloat(r.length), r.status = 'active'
        """)
        print("? ROUTE distances set")

        tx.run("""
            MATCH ()-[r:NEAR]-()
            SET r.distance = toFloat(r.distance)
        """)
        print("? NEAR distances set")

    def mark_driveable(self):
        with self.driver.session() as session:
            session.execute_write(self._mark_driveable)

    @staticmethod
    def _mark_driveable(tx):
        tx.run("""
            MATCH ()-[r:ROUTE]-()
            WHERE r.driveable = 'True' OR r.driveable = 'true'
            SET r.driveable = true
        """)

        tx.run("""
            MATCH (n:RoadJunction)
            SET n.driveable = false
        """)

        tx.run("""
            MATCH (n:RoadJunction)-[r:ROUTE]-()
            WHERE r.driveable = true
            SET n.driveable = true
        """)
        print("? Driveable junctions marked")

    def set_indexes(self):
        with self.driver.session() as session:
            session.execute_write(self._set_indexes)

    @staticmethod
    def _set_indexes(tx):
        tx.run("CREATE INDEX IF NOT EXISTS FOR (n:RoadJunction) ON (n.osmnode_id)")
        print("? Index on RoadJunction.osmnode_id")

        tx.run("CREATE INDEX IF NOT EXISTS FOR (n:OSMNode) ON (n.osm_id)")
        print("? Index on OSMNode.osm_id")

        tx.run("CREATE INDEX IF NOT EXISTS FOR (n:POI) ON (n.osm_id)")
        print("? Index on POI.osm_id")

        tx.run("CREATE INDEX IF NOT EXISTS FOR (n:RoadOsm) ON (n.name)")
        print("? Index on RoadOsm.name")

    def generate_spatial_layer(self):
        with self.driver.session() as session:
            session.execute_write(self._generate_spatial_layer)

    @staticmethod
    def _generate_spatial_layer(tx):
        result = tx.run("""
            CALL spatial.layers()
        """)
        layers = result.values()
        if len(layers) == 0:
            tx.run("""
                CALL spatial.addWKTLayer('spatial', 'geometry')
            """)
            print("? Spatial layer created")
        else:
            print("? Spatial layer already exists")

    def import_nodes_in_spatial_layer(self):
        with self.driver.session() as session:
            session.execute_write(self._import_nodes_in_spatial_layer)

    @staticmethod
    def _import_nodes_in_spatial_layer(tx):
        result = tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.geometry IS NOT NULL
            CALL spatial.addNode('spatial', n) YIELD node
            RETURN count(node) as total
        """)
        count = result.single()['total']
        print(f"? Imported {count} RoadJunction nodes into spatial layer")

        result = tx.run("""
            MATCH (n:OSMNode)
            WHERE n.geometry IS NOT NULL
            CALL spatial.addNode('spatial', n) YIELD node
            RETURN count(node) as total
        """)
        count = result.single()['total']
        print(f"? Imported {count} OSMNode nodes into spatial layer")


def add_options():
    parser = argparse.ArgumentParser(description='Post-import setup for Neo4j graph database.')
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str,
                        default='bolt://localhost:7687',
                        help='Neo4j bolt URL (default: bolt://localhost:7687)')
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str,
                        default='neo4j',
                        help='Neo4j username (default: neo4j)')
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str,
                        required=True,
                        help='Neo4j password')
    return parser


def main(args=None):
    argParser = add_options()
    options = argParser.parse_args(args=args)

    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)
    try:
        print("\n=== Running post-import steps ===\n")

        print("Step 1: Setting labels...")
        greeter.set_labels()

        print("\nStep 2: Setting locations...")
        greeter.set_location()

        print("\nStep 3: Setting distances...")
        greeter.set_distance()

        print("\nStep 4: Marking driveable junctions...")
        greeter.mark_driveable()

        print("\nStep 5: Creating indexes...")
        greeter.set_indexes()

        print("\nStep 6: Setting up spatial layer...")
        greeter.generate_spatial_layer()
        greeter.import_nodes_in_spatial_layer()

        print("\n=== Post-import steps completed successfully! ===")
    except Exception as e:
        print(f"Error during post-import: {e}")
        raise
    finally:
        greeter.close()


if __name__ == "__main__":
    main()