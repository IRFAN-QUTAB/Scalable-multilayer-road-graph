# Scalable Multi-Layer Graph Framework for Road Network Analysis and Disruption Simulation

A Python-based framework that constructs multi-layer graph representations of urban road networks using OpenStreetMap data and Neo4j. It supports large-area graph generation with parallel processing, POI integration, dual graph construction, network analysis through centrality and community detection algorithms, resilience evaluation under road closure scenarios, and shortest-path routing with disruption simulation.

## Requirements

- [Neo4j Desktop](https://neo4j.com/docs/operations-manual/current/installation/) (DBMS version 4.4.1)
- [Graph Data Science](https://neo4j.com/docs/graph-data-science/current/installation/) plugin (version 1.6+)
- [APOC](https://neo4j.com/labs/apoc/4.1/installation/) plugin
- Python 3.8+

### Neo4j Configuration

Install the Neo4j Spatial plugin by placing the JAR file in the plugins folder of your Neo4j installation. 
Then edit your DBMS settings and add the following lines after the Other Neo4j system properties section:

```
apoc.import.file.enabled=true
dbms.security.procedures.unrestricted=jwt.security.*,apoc.*,gds.*,spatial.*
```

### Python Dependencies

```shell
pip install osmnx overpy neo4j pandas networkx scipy numpy matplotlib folium powerlaw
```

---

## Pipeline Overview

The framework follows a sequential pipeline. Each step produces CSV files or updates the Neo4j database for the next step.

```
1. Create Road Network Graph  →  fullgraph_nodes.csv, fullgraph_rels.csv
2. Add POIs to Graph          →  combined_nodes.csv, combined_edges.csv
3. Creation of Dual Graph      →  dual_nodes.csv, dual_edges.csv
4. Import CSVs into Neo4j     →  (neo4j-admin import)
5. Initialize Graph Properties →  labels, indexes, spatial layer
6. Network Analysis Algorithms →  metrics CSVs + plots
7. Resilience Analysis         →  scenario comparison CSVs + plots
8. Routing                     →  route comparison maps (HTML)
```

---

## 1. Create Road Network Graph

Generates a large-area road network graph from OSM by dividing the area into overlapping grid sections, downloading each in parallel, merging, deduplicating, filtering by boundary, and validating connectivity.

```shell
python "Create Road Network Graph" -x 44.645885 -y 10.9255707 -d 5 -o ./output
```

| Parameter | Description |
|-----------|-------------|
| `-x` | Latitude of the center point |
| `-y` | Longitude of the center point |
| `-d` | Total distance in km (radius of area of interest) |
| `-o` | Output folder for CSV files |

The script will interactively ask for section size (km) and overlap (km). It supports resumption if interrupted — already processed sections are skipped on restart.

**Output:** `fullgraph_nodes.csv` and `fullgraph_rels.csv` in the output folder.

---

## 2. Add POIs to Graph

Queries amenities (points of interest) from the Overpass API, links each POI to the nearest road junction using a KD-tree spatial index, and produces combined node and edge CSV files ready for Neo4j import.

```shell
python "Add POIs to Graph" -n fullgraph_nodes.csv -r fullgraph_rels.csv -o ./output -x 44.645885 -y 10.9255707 -d 5 -a 2
```

| Parameter | Description |
|-----------|-------------|
| `-n` | Path to fullgraph_nodes.csv |
| `-r` | Path to fullgraph_rels.csv |
| `-o` | Output folder |
| `-x`, `-y` | Center latitude and longitude |
| `-d` | Total distance in km |
| `-a` | Section size in km for Overpass queries |

**Output:** `combined_nodes.csv` and `combined_edges.csv` containing road junctions, OSM nodes, OSM ways, tags, and all relationship types (ROUTE, NEAR, PART_OF, TAGS).

---

## 3. Creation of Dual Graph

Transforms the primal junction graph into a dual graph where each named road becomes a node (RoadOsm) and two roads sharing a junction are connected by a CONNECTED edge. Handles connections through roundabouts and unnamed road segments.

```shell
python "Creation of Dual Graph" -n fullgraph_nodes.csv -r fullgraph_rels.csv -o ./output
```

| Parameter | Description |
|-----------|-------------|
| `-n` | Path to fullgraph_nodes.csv |
| `-r` | Path to fullgraph_rels.csv |
| `-o` | Output folder for dual graph CSVs |

**Output:** `dual_nodes.csv` and `dual_edges.csv`.

---


## 4. Import CSVs into Neo4j

Use `neo4j-admin import` to load the combined CSV files into your Neo4j database. The CSV files include neo4j-admin compatible headers.

---

## 5. Initialize Graph Properties

Configures the graph schema after importing it into Neo4j. Sets node labels (RoadJunction, OSMNode, POI, OSMWay, Tag, RoadOsm), geospatial locations, distance properties, driveable flags, indexes, and a spatial layer.

```shell
python "Initialize Graph Properties" -n bolt://localhost:7687 -u neo4j -p password
```

| Parameter | Description |
|-----------|-------------|
| `-n` | Neo4j bolt URL (default: bolt://localhost:7687) |
| `-u` | Neo4j username (default: neo4j) |
| `-p` | Neo4j password |

---


## 6. Network Analysis Algorithms

Runs a complete set of graph-theoretic metrics on the dual graph using Neo4j GDS projections, saves results to CSV, and generates publication-quality plots.

```shell
python "Network Analysis Algorithms" -n bolt://localhost:7687 -u neo4j -p password -o results
```

| Parameter | Description |
|-----------|-------------|
| `-n` | Neo4j bolt URL |
| `-u` | Neo4j username |
| `-p` | Neo4j password |
| `-o` | Output folder for results (default: results) |

**Computed metrics:** PageRank, Betweenness Centrality, Harmonic Closeness, Degree Distribution (with Clauset MLE power-law fit), Clustering Coefficient (standard and adapted), Connected Components (WCC + SCC), Louvain Community Detection, Average Path Length, Diameter, Assortativity (Newman's r).

**Output:** CSV files with node-level metrics, graph summary, degree distribution, clustering and assortativity spectra, conductance per community. Plots include degree distribution, log-log P(k), clustering spectrum, assortativity spectrum, correlation matrix, scatter plots, centrality histograms, and community size distribution.

---

## 7. Resilience Analysis

Evaluates network resilience by simulating progressive road closures based on PageRank ranking. Runs 13 scenarios: one non-closure baseline, six closing increasing percentages of top-ranked roads (2%, 5%, 10%, 20%, 30%, 50%), and six closing the same percentages of bottom-ranked roads.

```shell
python "Resilience Analysis" -n bolt://localhost:7687 -u neo4j -p password -o results_resilience
```

| Parameter | Description |
|-----------|-------------|
| `-n` | Neo4j bolt URL |
| `-u` | Neo4j username |
| `-p` | Neo4j password |
| `-o` | Output folder (default: results_resilience) |

**Metrics per scenario:** Diameter, Connected Components, Average Path Length, Clustering Coefficient (standard and adapted), Modularity, Communities, Average Harmonic Closeness, and Average Betweenness.

**Output:** Comparison CSV with all scenarios, PageRank distribution histograms per scenario, and bar charts comparing each metric across scenarios.

---

## 8. Routing

Computes shortest-distance routes between two POIs on the primal graph using Dijkstra's algorithm. Supports comparison between non-closure and closure scenarios. The closure percentage is based on PageRank ranking from the dual graph.

```shell
python "Routing" -s (xxxxx) -d (xxxxx) -n bolt://localhost:7687 -u neo4j -p password -o routing_results
```

| Parameter | Description |
|-----------|-------------|
| `-s` | OSM ID of the source POI |
| `-d` | OSM ID of the destination POI |
| `-n` | Neo4j bolt URL |
| `-u` | Neo4j username |
| `-p` | Neo4j password |
| `-o` | Output folder (default: routing_results) |

The script will ask for a closure percentage interactively. Enter 0 for non-closure only, or a value like 10 to close the top 10% of roads by PageRank.

**Output:** An interactive HTML map showing both routes (green for non-closure, red for closure) and closed road segments (orange), along with a distance and hop comparison in the terminal.

---

## Example: Modena and Rome

Modena (radius 5 km):
```shell
python "Create Road Network Graph" -x 44.645885 -y 10.9255707 -d 5 -o ./modena_output
```

Rome (radius 30 km):
```shell
python "Create Road Network Graph" -x 41.9028 -y 12.4964 -d 30 -o ./rome_output
```
