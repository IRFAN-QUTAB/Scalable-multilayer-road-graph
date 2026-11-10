# Scalable Multi-Layer Graph Framework for Road Network Analysis and Disruption Simulation

Code of the paper *A Scalable Multi-Layer Graph Framework for Road Network Analysis and Disruption Simulation*.

The framework builds, from OpenStreetMap (OSM) data, a multi-layer graph of a study area: the road network
(primal graph), the points of interest (POIs) linked to the road network, and the dual graph of the roads. The
road network and the POIs of a large area are built cell by cell in parallel worker processes on a single machine,
and all layers are written as CSV files in the input format of the Neo4j bulk importer (`neo4j-admin import`).
The dual graphs are analysed (centrality, clustering, assortativity, communities, power-law test) and used to
simulate road closures, with routing on the primal graph.

## Contents

| Folder | Script | What it does |
|---|---|---|
| `Road_Network` | `road_network_parallel.py` | Cell-based parallel construction of the road network graph (primal graph) |
| | `road_network_sequential.py` | Same construction, one cell after the other in a single process |
| | `road_network_single_request.py` | Road network graph of the same area with a single OSMnx request (reference graph); GraphML file or Neo4j |
| | `road_network_local_file.py` | Single OSMnx graph built from a local `.osm.pbf` file (osmium) |
| | `compare_with_single_request.py` | Compares the cell-based graph with the single-request graph (nodes, directed edges, attributes, shortest paths) |
| `POI Integration` | `poi_layer_parallel.py` | POIs downloaded cell by cell in parallel worker processes and linked to the road network (rules R1, R2, R3) |
| | `poi_layer_sequential.py` | Same POI layer, one cell after the other in a single process |
| | `poi_long_links.py` | Length of the POI links, links and POIs above given distances |
| | `poi_barrier_check.py` | Share of the POI links whose straight line crosses a barrier mapped in OSM |
| `Dual_Graph` | `dual_graph.py` | Dual graphs: name-based (main), name_split and stroke-based (COINS, momepy) |
| `Neo4j` | `neo4j_road_network.py` | Road network written cell by cell into Neo4j (transactional writes) |
| | `neo4j_poi_layer.py` | POI layer written cell by cell into Neo4j |
| | `neo4j_post_import.py` | Labels, locations, distances, driveable flag, indexes and spatial layer after `neo4j-admin import` |
| `Analysis` | `dual_metrics.py` | Structural metrics of a dual graph |
| | `dual_closures.py` | Road closure scenarios and routing on origin-destination pairs |
| | `compare_dual_graphs.py` | Comparison of the three dual graphs of a city (road length, centralities) |
| | `od_endpoint_check.py` | Splits the pairs without a path into pairs with an end point on a closed road and separated pairs |
| `Benchmarks` | `scaling_benchmark.py` | Construction time and memory with different numbers of worker processes |
| | `cell_overlap_benchmark.py` | Construction time, memory and duplicated work for different cell sizes and overlaps |
| | `compare_csv_rows.py` | Checks that two CSV files have the same columns and rows|

## Requirements

* Python 3.13 and the packages of `requirements.txt` (`pip install -r requirements.txt`).
  Versions used for the paper: OSMnx 2.0.2, NetworkX 3.7, pandas 3.1.0, NumPy 2.2.4, SciPy 1.15.3,
  python-igraph 1.0.0, powerlaw 2.0.0, momepy 1.0.0.
* An Overpass API server. The paper uses a local Overpass server with a snapshot of the OSM data of Italy, so that
  all runs use the same data; the public server (default of OSMnx) can also be used with `--overpass-url`.
* `osmium-tool`, only for `road_network_local_file.py`.
* Neo4j 5 with the APOC, Graph Data Science and Spatial plugins, only for the scripts of `4_neo4j` and for loading
  the CSV files into a database.

## Study areas of the paper

| Area | Centre (lat, lon) | D (km) | Road network cells: size / overlap (km) | POI cells (km) |
|---|---|---|---|---|
| Modena | 44.64588, 10.92557 | 5 | 2 / 1 | 5 |
| Rome | 41.9028, 12.4964 | 18.5 | 5 / 2 | 5 |
| Po Delta | 44.507836, 11.883350 | 80 | 5 / 2 | 5 |

The study area is the square of side 2D around the centre (bounding box, as in a single OSMnx request with
`dist_type='bbox'`). The paper uses 16 worker processes.

## How to run

Replace `URL` with the address of the Overpass interpreter (e.g. `http://localhost:12351/api/interpreter`).

### 1. Road network graph

```shell
cd 1_road_network
printf "5\n2\n" | python3 road_network_parallel.py --latitude 41.9028 --longitude 12.4964 --distance 18.5 \
    --output rome_cell --overpass-url URL --workers 16
```

The script asks for the cell size and the overlap in km (here 5 and 2, given through `printf`). A cell whose
request fails is retried (`--max-retries`); if it still fails, the run stops with an error. An interrupted run
can be restarted with the same command and continues from the completed cells.

Output: `fullgraph_nodes.csv` and `fullgraph_rels.csv` (road junctions and road segments), `diagnostics.txt`
(times of the phases, memory), `cell_timings.csv`.

Reference graph and comparison:

```shell
python3 road_network_single_request.py --latitude 41.9028 --longitude 12.4964 --distance 18500 \
    --output-file rome_single.graphml --overpass-url URL
python3 compare_with_single_request.py --cell-nodes rome_cell/fullgraph_nodes.csv --cell-rels rome_cell/fullgraph_rels.csv \
    --single-graphml rome_single.graphml --pairs 500 --report rome_comparison.txt
```

`road_network_single_request.py` takes the distance in metres.

### 2. POI Integration

```shell
cd 2_poi
python3 poi_layer_parallel.py --nodes-csv ../1_road_network/rome_cell/fullgraph_nodes.csv \
    --rels-csv ../1_road_network/rome_cell/fullgraph_rels.csv --output rome_poi \
    --latitude 41.9028 --longitude 12.4964 --distance 18.5 --area-size 5 --overpass-url URL --workers 16
python3 poi_long_links.py --area Rome=rome_poi --output poi_long_links
python3 poi_barrier_check.py --overpass-url URL --area Rome=rome_poi --output poi_barrier_check
```

Output: `combined_nodes.csv` and `combined_edges.csv` (road network, POIs, OSM ways, tags; relationships ROUTE,
NEAR, PART_OF, TAGS) and `poi_paper_results.txt`. Each POI is linked to the driveable road junctions within 100 m
(R1); a POI without such a junction is linked to the nearest non-driveable junction (R2) and then to the nearest
driveable junction (R3). Driveable junctions are the junctions of the largest strongly connected component of
the driveable network.

### 3. Dual graphs

```shell
cd 3_dual_graph
python3 dual_graph.py -n ../1_road_network/rome_cell/fullgraph_nodes.csv \
    -r ../1_road_network/rome_cell/fullgraph_rels.csv -o rome_dual --split-gap 2000
```

Parameters of the paper: roundabouts followed up to 50 hops, unnamed segments up to 500 hops (defaults), pieces of
the same name more than 2 km apart split in the name_split graph, COINS angle threshold 0 (default).

Output: `rome_dual/` (name-based), `rome_dual/name_split/`, `rome_dual/strokes/`, each with `dual_nodes.csv` and
`dual_edges.csv`; `edge_keys.csv` (road of every primal edge in the three graphs); `name_pieces.csv`;
`dual_report.txt`.

### 4. Neo4j

The CSV files have the headers of the Neo4j bulk importer (`:ID`, `:START_ID`, `:END_ID`, `:TYPE`). Load
`combined_nodes.csv` and `combined_edges.csv` (and, for the dual graph, `dual_nodes.csv` and `dual_edges.csv`)
into an empty database with `neo4j-admin database import full`, then run the post-import step:

```shell
python3 4_neo4j/neo4j_post_import.py --neo4jURL bolt://localhost:7687 --neo4juser neo4j --neo4jpwd PASSWORD
```

`neo4j_road_network.py` and `neo4j_poi_layer.py` write the same layers cell by cell into a running database (transactional
writes); they are used for the comparison of the construction strategies.

### 5. Analysis

```shell
cd 5_analysis
python3 dual_metrics.py --dual-dir ../3_dual_graph/rome_dual --label "Rome name" --output analysis/rome_name --bootstrap 2500
python3 dual_closures.py --nodes-csv ../1_road_network/rome_cell/fullgraph_nodes.csv \
    --rels-csv ../1_road_network/rome_cell/fullgraph_rels.csv --edge-keys ../3_dual_graph/rome_dual/edge_keys.csv \
    --dual-dir ../3_dual_graph/rome_dual --key name --label "Rome name" --output analysis/rome_name_closures --per-origin 50
python3 od_endpoint_check.py --nodes-csv ../1_road_network/rome_cell/fullgraph_nodes.csv \
    --rels-csv ../1_road_network/rome_cell/fullgraph_rels.csv --edge-keys ../3_dual_graph/rome_dual/edge_keys.csv \
    --key name --closures analysis/rome_name_closures
```

For the name_split and stroke-based graphs use `--dual-dir .../rome_dual/name_split --key name_split` and
`--dual-dir .../rome_dual/strokes --key stroke`. `compare_dual_graphs.py` compares the three graphs of a city
(`--metrics` with the output folders of `dual_metrics.py`).

`dual_closures.py` runs the closure scenarios of the paper: closures of the top and bottom 2-50% of the roads by
PageRank (RCS1, RCS2); closures with the same removed length (1, 2, 5, 10, 20% of the network length) by
PageRank, betweenness, degree, lowest PageRank and in random order (10 repetitions); closure of a 1 km section of
the five roads with the highest PageRank and of the whole road. Routing is done on the primal graph on 5,000
origin-destination pairs (100 random origins with 50 destinations each, seed 42), with shortest paths by length
and fastest paths by travel time (speed from the OSM `maxspeed` tag, or the mean known speed of the same road
type).

### 6. Benchmarks

```shell
cd 6_benchmarks
python3 scaling_benchmark.py --latitude 44.507836 --longitude 11.883350 --distance 80 --area-size 5 --overlap 2 \
    --overpass-url URL --output-base scaling --workers 1 2 4 8 16 32 --repeats 3
python3 cell_overlap_benchmark.py --latitude 41.9028 --longitude 12.4964 --distance 18.5 \
    --combinations 5:2 5:1 5:0 10:2 2:1 --workers 16 --repeats 3 --overpass-url URL --output-base cell_overlap
