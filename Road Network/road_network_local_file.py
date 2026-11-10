
import argparse
import datetime
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import networkx as nx
import osmnx as ox
import psutil
from osmnx import _osm_xml
from osmnx import _overpass

HIGHWAY_REGEX = "motorway|trunk|primary|secondary|tertiary|residential|unclassified|service"
CUSTOM_FILTER = f'["highway"~"{HIGHWAY_REGEX}"]'

DOCKER = {'container': None, 'host': None, 'inside': None}


def start_memory_monitor(interval=0.5):
    """Peak RAM of this script, of its child processes, and of the running osmium processes
    (osmium started with docker exec is not a child process, so it is found by its name):
    peak[0] = all these processes together, peak[1] = largest single process, peak[2] = osmium."""
    proc = psutil.Process(os.getpid())
    peak = [0, 0, 0]
    stop = threading.Event()

    def run():
        tick = 0
        osmium_procs = []
        while not stop.is_set():
            sizes = [proc.memory_info().rss]
            for c in proc.children(recursive=True):
                try:
                    if c.name() != 'osmium':
                        sizes.append(c.memory_info().rss)
                except psutil.Error:
                    pass
            if tick % 4 == 0:
                osmium_procs = [p for p in psutil.process_iter(['name']) if p.info.get('name') == 'osmium']
            osm = 0
            for p in osmium_procs:
                try:
                    osm += p.memory_info().rss
                except psutil.Error:
                    pass
            peak[0] = max(peak[0], sum(sizes) + osm)
            peak[1] = max(peak[1], max(sizes + [osm]))
            peak[2] = max(peak[2], osm)
            tick += 1
            stop.wait(interval)

    threading.Thread(target=run, daemon=True).start()
    return stop, peak


def local_network_reader(xml_path):
    """Replacement of the Overpass download: returns the ways of the local XML file that match
    the same highway filter, and the nodes of these ways, in the same format as Overpass."""
    pattern = re.compile(HIGHWAY_REGEX)

    def download(polygon, network_type, custom_filter):
        data = _osm_xml._overpass_json_from_xml(xml_path, "utf-8")
        ways = [e for e in data["elements"]
                if e.get("type") == "way" and pattern.search(str(e.get("tags", {}).get("highway", "")))]
        used = {n for w in ways for n in w.get("nodes", [])}
        nodes = [e for e in data["elements"] if e.get("type") == "node" and e["id"] in used]
        print(f"Local file: {len(ways)} ways and {len(nodes)} nodes after the highway filter")
        return [{"elements": nodes + ways}]

    return download


def to_container(path):
    """Path of a host file as seen inside the Docker container (bind mount host -> inside)"""
    path = os.path.abspath(path)
    host = DOCKER['host'].rstrip('/')
    if not (path == host or path.startswith(host + '/')):
        raise SystemExit(f"{path} is not inside the mounted folder {host}: put the .pbf file and "
                         f"--work-dir inside {host}")
    return DOCKER['inside'].rstrip('/') + path[len(host):]


def osmium_cmd(args, paths):
    """osmium command on the host, or inside the Docker container if --docker-container is given.
    paths: indexes of the arguments that are file paths (translated for the container)."""
    if DOCKER['container']:
        args = [to_container(a) if i in paths else a for i, a in enumerate(args)]
        return ['docker', 'exec', DOCKER['container'], 'osmium'] + args
    return ['osmium'] + args


def osmium(args, paths):
    cmd = osmium_cmd(args, paths)
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def osmium_output(args, paths):
    try:
        return subprocess.run(osmium_cmd(args, paths), capture_output=True, text=True, check=True).stdout
    except Exception as e:
        return f"(not available: {e})"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Single OSMnx graph from a local OSM file.')
    parser.add_argument('--pbf', required=True, help='local .osm.pbf file')
    parser.add_argument('--latitude', type=float, required=True)
    parser.add_argument('--longitude', type=float, required=True)
    parser.add_argument('--distance', type=float, required=True, help='D in meters (as in the single request)')
    parser.add_argument('--work-dir', required=True, help='folder for the extracted files and the results')
    parser.add_argument('--no-graphml', action='store_true', help='do not write the GraphML file')
    parser.add_argument('--docker-container', default=None,
                        help='run osmium inside this Docker container (e.g. overpass)')
    parser.add_argument('--docker-mount', default=None,
                        help='bind mount of the container as HOST:INSIDE, e.g. '
                             '/path/to/overpass_db:/db')
    parser.add_argument('--compare-cell-dir', default=None,
                        help='folder with fullgraph_nodes.csv and fullgraph_rels.csv of the cell-based run; '
                             'if given, compare_with_single_request.py is run at the end')
    parser.add_argument('--compare-script', default=os.path.join(here, 'compare_with_single_request.py'))
    parser.add_argument('--pairs', type=int, default=200, help='OD pairs for the shortest-path check')
    args = parser.parse_args()
    if args.docker_container:
        if not args.docker_mount or ':' not in args.docker_mount:
            parser.error('--docker-container needs --docker-mount HOST:INSIDE')
        DOCKER['container'] = args.docker_container
        DOCKER['host'], DOCKER['inside'] = args.docker_mount.split(':', 1)

    os.makedirs(args.work_dir, exist_ok=True)
    area_pbf = os.path.join(args.work_dir, 'area.osm.pbf')
    area_xml = os.path.join(args.work_dir, 'area_highway.osm')
    graphml = os.path.join(args.work_dir, 'local_file_graph.graphml')
    report = os.path.join(args.work_dir, 'local_file_results.txt')
    ox.settings.use_cache = False

    lines = []

    def out(text=''):
        print(text)
        lines.append(text)
        with open(report, 'w') as f:
            f.write("\n".join(lines) + "\n")

    out("Single OSMnx graph from a local OSM file")
    out(f"Date: {datetime.datetime.now().isoformat(timespec='seconds')}")
    out(f"CPU cores (logical): {os.cpu_count()} | RAM total {psutil.virtual_memory().total / 1024**3:.1f} GB, "
        f"available {psutil.virtual_memory().available / 1024**3:.1f} GB")
    out(f"Load average before the run (1, 5, 15 min): {os.getloadavg()}")
    out(f"Python {sys.version.split()[0]} | OSMnx {ox.__version__}")
    version = osmium_output(['--version'], set()).splitlines()
    out(f"osmium: {version[0] if version else 'n/a'}")
    out(f"Local file: {args.pbf} ({os.path.getsize(args.pbf) / 1024**3:.2f} GB)")
    for line in osmium_output(['fileinfo', args.pbf], {1}).splitlines():
        if 'timestamp' in line.lower():
            out(f"  {line.strip()}")

    left, bottom, right, top = ox.utils_geo.bbox_from_point((args.latitude, args.longitude), dist=args.distance)
    m_left, m_bottom, m_right, m_top = ox.utils_geo.bbox_from_point(
        (args.latitude, args.longitude), dist=args.distance + 2000)
    out(f"Study area: lat [{bottom:.6f}, {top:.6f}], lon [{left:.6f}, {right:.6f}] "
        f"(center {args.latitude}, {args.longitude}, D = {args.distance:.0f} m)")

    stop_mem, peak = start_memory_monitor()
    t0 = time.time()
    osmium(['extract', '-b', f'{m_left},{m_bottom},{m_right},{m_top}', '-s', 'complete_ways',
            args.pbf, '-o', area_pbf, '--overwrite'], paths={5, 7})
    t1 = time.time()
    osmium(['tags-filter', area_pbf, 'w/highway', '-o', area_xml, '--overwrite'], paths={1, 4})
    t2 = time.time()

    _overpass._download_overpass_network = local_network_reader(Path(area_xml))
    G = ox.graph_from_point((args.latitude, args.longitude), dist=int(args.distance), dist_type='bbox',
                            simplify=False, custom_filter=CUSTOM_FILTER)
    t3 = time.time()

    for u, v, k, data in G.edges(keys=True, data=True):
        highway, access = data.get('highway'), data.get('access')
        data['driveable'] = "True"
        if access in {"no", "private"} or highway == 'service':
            data['driveable'] = "False"
    if not args.no_graphml:
        ox.save_graphml(G, graphml)
    t4 = time.time()
    stop_mem.set()

    scc = [len(c) for c in nx.strongly_connected_components(G)]
    out(f"Phase time - osmium extract: {t1 - t0:.2f} seconds")
    out(f"Phase time - osmium tags-filter (OSM XML): {t2 - t1:.2f} seconds")
    out(f"Phase time - OSMnx graph from the local file: {t3 - t2:.2f} seconds")
    if not args.no_graphml:
        out(f"Phase time - writing the GraphML file: {t4 - t3:.2f} seconds")
    out(f"Construction time (extract + filter + graph): {t3 - t0:.2f} seconds")
    out(f"Construction time including the GraphML file (as reported for the single request): {t4 - t0:.2f} seconds")
    out(f"Peak memory (all processes, osmium included): {peak[0] / 1024**3:.2f} GB")
    out(f"Peak memory (largest single process): {peak[1] / 1024**3:.2f} GB")
    out(f"Peak memory of osmium: {peak[2] / 1024**3:.2f} GB"
        + (" (0 means that the osmium processes were not visible from this user)" if peak[2] == 0 else ""))
    out(f"Size of the extracted files: area.osm.pbf {os.path.getsize(area_pbf) / 1024**3:.2f} GB, "
        f"area_highway.osm {os.path.getsize(area_xml) / 1024**3:.2f} GB")
    out(f"Nodes: {G.number_of_nodes()}")
    out(f"Edges: {G.number_of_edges()}")
    out(f"Weakly connected components: {nx.number_weakly_connected_components(G)}")
    out(f"Strongly connected components: {len(scc)}")
    out(f"Largest strongly connected component: {max(scc)} nodes ({100.0 * max(scc) / G.number_of_nodes():.2f}%)")
    if not args.no_graphml:
        out(f"GraphML file: {graphml}")
    out("Not included in the times: download of the .osm.pbf file (as the build of the Overpass database "
        "is not included in the times of the other runs).")

    if args.compare_cell_dir and not args.no_graphml:
        del G
        cmp_report = os.path.join(args.work_dir, 'local_file_vs_cell_based.txt')
        cmd = [sys.executable, args.compare_script,
               '--cell-nodes', os.path.join(args.compare_cell_dir, 'fullgraph_nodes.csv'),
               '--cell-rels', os.path.join(args.compare_cell_dir, 'fullgraph_rels.csv'),
               '--single-graphml', graphml, '--pairs', str(args.pairs), '--report', cmp_report]
        out(f"\nComparison with the cell-based graph: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        out(f"Comparison exit code: {result.returncode} | report: {cmp_report}")
        if os.path.exists(cmp_report):
            out(open(cmp_report).read())
        else:
            out(result.stdout[-3000:] + result.stderr[-3000:])
    out(f"Finished {datetime.datetime.now().isoformat(timespec='seconds')}")


if __name__ == '__main__':
    main()
