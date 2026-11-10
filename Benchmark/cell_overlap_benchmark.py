
import argparse
import csv
import datetime
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import time

PATTERNS = {
    'cells': r"Will generate (\d+) sections",
    'download_s': r"Phase time - download and processing of the sections: ([\d.]+) seconds",
    'merge_s': r"Phase time - merge, deduplication and boundary filter: ([\d.]+) seconds",
    'lcc_s': r"Phase time - largest connected component and connectivity report: ([\d.]+) seconds",
    'construction_s': r"Construction time \(excluding user input\): ([\d.]+) seconds",
    'mem_all_gb': r"Peak memory \(all processes\): ([\d.]+) GB",
    'mem_max_gb': r"Peak memory \(largest single process\): ([\d.]+) GB",
    'nodes': r"^Nodes: (\d+)",
    'edges': r"^Edges: (\d+)",
    'retries': r"(Retry \d+/\d+)",
}


def parse(output):
    values = {}
    for key, pattern in PATTERNS.items():
        found = re.findall(pattern, output, flags=re.MULTILINE)
        if key == 'retries':
            values[key] = len(found)
        elif found:
            values[key] = found[-1]
    return values


def cell_work(out_dir):
    """sum of the nodes and edges built in all cells (cell_timings.csv of the run)"""
    path = os.path.join(out_dir, 'cell_timings.csv')
    if not os.path.exists(path):
        return None, None
    nodes = edges = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            nodes += int(float(row.get('nodes') or 0))
            edges += int(float(row.get('edges') or 0))
    return nodes, edges


def graph_fingerprint(out_dir):
    """SHA-256 of the sorted node ids and of the sorted edges (start node, end node, OSM way id)"""
    import pandas as pd
    n_path = os.path.join(out_dir, 'fullgraph_nodes.csv')
    r_path = os.path.join(out_dir, 'fullgraph_rels.csv')
    if not (os.path.exists(n_path) and os.path.exists(r_path)):
        return None

    def norm(s):
        s = s.astype(str).str.strip()
        return s.where(~s.str.endswith('.0'), s.str[:-2])

    nodes = norm(pd.read_csv(n_path, usecols=['osmnode_id'], dtype=str)['osmnode_id'])
    rels = pd.read_csv(r_path, usecols=['source_id', 'target_id', 'osmway_id'], dtype=str)
    edges = norm(rels['source_id']) + '|' + norm(rels['target_id']) + '|' + norm(rels['osmway_id'].fillna(''))
    h = hashlib.sha256()
    h.update('\n'.join(sorted(nodes)).encode())
    h.update(b'#')
    h.update('\n'.join(sorted(edges)).encode())
    return h.hexdigest()[:16]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description='Cell size and overlap benchmark.')
    p.add_argument('--latitude', type=float, required=True)
    p.add_argument('--longitude', type=float, required=True)
    p.add_argument('--distance', type=float, required=True, help='D in km')
    p.add_argument('--combinations', nargs='+', required=True,
                   help='cell size and overlap in km as cell:overlap, for example 5:2 5:1 10:2')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--overpass-url', type=str, default=None)
    p.add_argument('--output-base', type=str, required=True)
    p.add_argument('--script', type=str, default=os.path.join(here, '..', '1_road_network', 'road_network_parallel.py'))
    p.add_argument('--keep', action='store_true', help='keep the output folder of every run (default: deleted '
                                                        'after the results are written)')
    o = p.parse_args()

    combos = []
    for c in o.combinations:
        s, e = (float(x) for x in c.split(':'))
        if s <= 0 or e < 0 or e >= s:
            p.error(f"invalid combination {c}: the cell size must be positive and the overlap smaller than it")
        combos.append((s, e))

    os.makedirs(o.output_base, exist_ok=True)
    results_file = os.path.join(o.output_base, 'cell_overlap_results.txt')
    if os.path.exists(results_file):
        os.remove(results_file)

    def write(text=''):
        print(text)
        with open(results_file, 'a') as f:
            f.write(text + '\n')

    write("Cell size and overlap benchmark")
    write(f"Date: {datetime.datetime.now().isoformat(timespec='seconds')} | host {platform.node()} | "
          f"CPU cores {os.cpu_count()} | Python {sys.version.split()[0]}")
    write(f"Centre: {o.latitude}, {o.longitude} | D = {o.distance} km | workers {o.workers} | repeats {o.repeats}")
    write(f"Overpass: {o.overpass_url or 'public server'} | script: {o.script}")

    rows = []
    for rep in range(1, o.repeats + 1):
        for s, e in combos:
            name = f"cell_{s:g}_overlap_{e:g}_rep_{rep}"
            out_dir = os.path.join(o.output_base, name)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            cmd = [sys.executable, o.script, '--latitude', str(o.latitude), '--longitude', str(o.longitude),
                   '--distance', str(o.distance), '--output', out_dir, '--workers', str(o.workers)]
            if o.overpass_url:
                cmd += ['--overpass-url', o.overpass_url]
            write(f"\n--- cell {s:g} km, overlap {e:g} km, repeat {rep}, started "
                  f"{datetime.datetime.now().isoformat(timespec='seconds')} ---")
            t0 = time.time()
            proc = subprocess.run(cmd, input=f"{s}\n{e}\n", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            wall = time.time() - t0
            log_path = os.path.join(o.output_base, f"log_{name}.txt")
            with open(log_path, 'w') as f:
                f.write(proc.stdout)
            v = parse(proc.stdout)
            v.update({'cell': s, 'overlap': e, 'repeat': rep, 'ok': proc.returncode == 0, 'wall_s': wall})
            cn, ce = cell_work(out_dir)
            v['cell_nodes'], v['cell_edges'] = cn, ce
            v['fingerprint'] = graph_fingerprint(out_dir) if v['ok'] else None
            if cn and v.get('nodes'):
                v['dup_nodes'] = cn / float(v['nodes'])
            if ce and v.get('edges'):
                v['dup_edges'] = ce / float(v['edges'])
            v['expected_factor'] = (s / (s - e)) ** 2
            rows.append(v)
            write(f"exit code {proc.returncode} | wall {wall:.1f} s | log {log_path}")
            for k in ('cells', 'nodes', 'edges', 'download_s', 'merge_s', 'lcc_s', 'construction_s', 'mem_all_gb',
                      'mem_max_gb', 'retries', 'cell_nodes', 'cell_edges', 'dup_nodes', 'dup_edges',
                      'expected_factor', 'fingerprint'):
                val = v.get(k, 'n/a')
                write(f"  {k:<16} {val:.3f}" if isinstance(val, float) else f"  {k:<16} {val}")
            if not o.keep and os.path.exists(out_dir):
                shutil.rmtree(out_dir)

    import statistics

    def med(rs, key):
        vals = [float(r[key]) for r in rs if r.get(key) is not None]
        return statistics.median(vals) if vals else float('nan')

    write("\n" + "=" * 100)
    write("Summary (median of the repeats)")
    write(f"{'cell km':>7} {'overlap km':>10} {'cells':>7} {'download s':>10} {'merge s':>8} {'LCC s':>7} {'total s':>8} "
          f"{'mem all GB':>10} {'mem max GB':>10} {'dup nodes':>9} {'dup edges':>9} {'(s/(s-e))^2':>11}")
    for s, e in combos:
        rs = [r for r in rows if r['cell'] == s and r['overlap'] == e and r['ok']]
        if not rs:
            write(f"{s:>7g} {e:>10g}  all runs failed (see logs)")
            continue
        write(f"{s:>7g} {e:>10g} {med(rs, 'cells'):>7.0f} {med(rs, 'download_s'):>10.1f} {med(rs, 'merge_s'):>8.1f} "
              f"{med(rs, 'lcc_s'):>7.1f} {med(rs, 'construction_s'):>8.1f} {med(rs, 'mem_all_gb'):>10.2f} "
              f"{med(rs, 'mem_max_gb'):>10.2f} {med(rs, 'dup_nodes'):>9.2f} {med(rs, 'dup_edges'):>9.2f} "
              f"{(s / (s - e)) ** 2:>11.2f}")
    prints = {r['fingerprint'] for r in rows if r['ok']}
    counts = {(r.get('nodes'), r.get('edges')) for r in rows if r['ok']}
    write(f"\nSame final graph (node ids and edges) in all runs: {'yes' if len(prints) == 1 else 'NO'} "
          f"(fingerprints {sorted(str(x) for x in prints)}; nodes, edges {sorted(counts)})")
    if len(prints) > 1:
        for fp in sorted(str(x) for x in prints):
            write(f"   {fp}: " + ", ".join(f"{r['cell']:g}/{r['overlap']:g}" for r in rows if str(r['fingerprint']) == fp))
    failed = [(r['cell'], r['overlap'], r['repeat']) for r in rows if not r['ok']]
    write(f"Failed runs: {failed if failed else 'none'}")
    write("dup nodes / dup edges = nodes (edges) built in all cells / nodes (edges) of the final graph; "
          "(s/(s-e))^2 = overlap factor of the complexity formula (border cells not included)")
    write(f"Finished {datetime.datetime.now().isoformat(timespec='seconds')}")


if __name__ == '__main__':
    main()
