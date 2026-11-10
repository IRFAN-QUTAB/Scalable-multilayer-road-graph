
import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import time

PATTERNS = {
    'download_s': r"Phase time - download and processing of the sections: ([\d.]+) seconds",
    'merge_s': r"Phase time - merge, deduplication and boundary filter: ([\d.]+) seconds",
    'lcc_s': r"Phase time - largest connected component and connectivity report: ([\d.]+) seconds",
    'construction_s': r"Construction time \(excluding user input\): ([\d.]+) seconds",
    'mem_all_gb': r"Peak memory \(all processes\): ([\d.]+) GB",
    'mem_max_gb': r"Peak memory \(largest single process\): ([\d.]+) GB",
    'nodes': r"^Nodes: (\d+)",
    'edges': r"^Edges: (\d+)",
    'scc': r"^Strongly connected components: (\d+)",
    'cells': r"Will generate (\d+) sections",
    'retries': r"(Retry \d+/\d+)",
    'overpass_mean_s': r"waiting for the Overpass server \(request and response\): sum [\d.]+ s, mean ([\d.]+) s",
    'osmnx_mean_s': r"OSMnx graph construction in the worker: sum [\d.]+ s, mean ([\d.]+) s",
    'waiting_share': r"Share of the cell time spent waiting for the Overpass server: ([\d.]+)%",
    'workers_busy': r"workers busy ([\d.]+)% of the phase",
    'cpu_cores_used': r"download phase: [\d.]+ CPU-seconds = ([\d.]+) cores",
    'sys_cpu_download': r"Phase download: samples \d+, system CPU mean ([\d.]+)%",
    'load_download': r"Phase download: .*?load average mean ([\d.]+)",
    'overpass_procs_max': r"Phase download: .*?running Overpass server processes mean [\d.]+ / max (\d+)",
    'load_before': r"Load average before the run \(1, 5, 15 min\): \(([\d.]+)",
}
SUBSTEP = r"Sub-step time - (.+?): ([\d.]+) seconds"


def parse(output):
    values = {}
    values['substeps'] = re.findall(SUBSTEP, output)
    for key, pattern in PATTERNS.items():
        found = re.findall(pattern, output, flags=re.MULTILINE)
        if key == 'retries':
            values[key] = len(found)
        elif found:
            values[key] = found[-1]
    return values


def system_info():
    lines = [f"Date: {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"Host: {platform.node()}",
             f"CPU cores (os.cpu_count): {os.cpu_count()}",
             f"Python: {sys.version.split()[0]}"]
    try:
        import psutil
        lines.append(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    except ImportError:
        pass
    try:
        import osmnx
        lines.append(f"OSMnx: {osmnx.__version__}")
    except ImportError:
        pass
    return lines


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Scaling benchmark.')
    parser.add_argument('--latitude', type=float, required=True)
    parser.add_argument('--longitude', type=float, required=True)
    parser.add_argument('--distance', type=float, required=True, help='D in km')
    parser.add_argument('--area-size', type=float, required=True, help='cell side in km')
    parser.add_argument('--overlap', type=float, required=True, help='cell overlap in km')
    parser.add_argument('--overpass-url', type=str, default=None)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--output-base', type=str, required=True,
                        help='folder for the runs (one subfolder per run) and the results file')
    parser.add_argument('--script', type=str, default=os.path.join(here, '..', '1_road_network', 'road_network_parallel.py'),
                        help='path of road_network_parallel.py')
    parser.add_argument('--repeats', type=int, default=1, help='runs for every number of workers (default 1)')
    parser.add_argument('--cleanup', action='store_true',
                        help='delete the output folder of each run after its results are written')
    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)
    results_file = os.path.join(args.output_base, 'scaling_results.txt')

    def write(text=''):
        print(text)
        with open(results_file, 'a') as f:
            f.write(text + '\n')

    write("=" * 78)
    write("Scaling benchmark")
    for line in system_info():
        write(line)
    write(f"Script: {args.script}")
    write(f"Center: {args.latitude}, {args.longitude} | D = {args.distance} km | "
          f"cell = {args.area_size} km | overlap = {args.overlap} km")
    write(f"Overpass: {args.overpass_url or 'public server'}")
    write(f"Workers: {args.workers} | repeats: {args.repeats} (in rounds)")
    write("=" * 78)

    rows = []
    runs = [(r, w) for r in range(1, args.repeats + 1) for w in args.workers]
    for rep_no, w in runs:
        out_dir = os.path.join(args.output_base, f"run_workers_{w}_rep_{rep_no}")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        cmd = [sys.executable, args.script,
               '--latitude', str(args.latitude), '--longitude', str(args.longitude),
               '--distance', str(args.distance), '--output', out_dir, '--workers', str(w)]
        if args.overpass_url:
            cmd += ['--overpass-url', args.overpass_url]

        write(f"\n--- Run with {w} worker(s), repeat {rep_no}, started {datetime.datetime.now().isoformat(timespec='seconds')} ---")
        log_path = os.path.join(args.output_base, f"log_workers_{w}_rep_{rep_no}.txt")
        t0 = time.time()
        with open(log_path, 'w') as log:
            proc = subprocess.run(cmd, input=f"{args.area_size}\n{args.overlap}\n",
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log.write(proc.stdout)
        wall = time.time() - t0
        v = parse(proc.stdout)
        v['workers'] = w
        v['repeat'] = rep_no
        v['ok'] = proc.returncode == 0
        rows.append(v)

        write(f"Exit code: {proc.returncode} | wall time: {wall:.2f} s | full log: {log_path}")
        for key in ('cells', 'nodes', 'edges', 'scc', 'download_s', 'merge_s', 'lcc_s',
                    'construction_s', 'mem_all_gb', 'mem_max_gb', 'retries', 'load_before',
                    'overpass_mean_s', 'osmnx_mean_s', 'waiting_share', 'workers_busy',
                    'cpu_cores_used', 'sys_cpu_download', 'load_download', 'overpass_procs_max'):
            write(f"  {key:<19} {v.get(key, 'n/a')}")
        for name, value in v['substeps']:
            write(f"  sub-step: {name}: {value} s")
        if args.cleanup and os.path.exists(out_dir):
            shutil.rmtree(out_dir)

    import statistics

    def med(rs, key):
        vals = [float(r[key]) for r in rs if key in r]
        return statistics.median(vals) if vals else float('nan')

    write("\n" + "=" * 78)
    write("Summary: median of the repeats; speed-up and efficiency with respect to the first number of workers")
    ok = {w: [r for r in rows if r['workers'] == w and r['ok'] and 'construction_s' in r] for w in args.workers}
    base_w = next((w for w in args.workers if ok[w]), None)
    base_t = med(ok[base_w], 'construction_s') if base_w else float('nan')
    write(f"{'workers':>7} {'runs':>4} {'download':>9} {'merge':>7} {'LCC':>7} {'total':>8} {'min':>8} {'max':>8} "
          f"{'speed-up':>8} {'effic.':>7} {'overpass/cell':>13} {'osmnx/cell':>10} {'wait %':>7} "
          f"{'cores used':>10} {'sys CPU %':>9} {'load':>6} {'ovp procs':>9} {'mem all':>8} {'mem max':>8}")
    for w in args.workers:
        rs = ok[w]
        if not rs:
            write(f"{w:>7} all runs failed (see logs)")
            continue
        t = med(rs, 'construction_s')
        tmin = min(float(r['construction_s']) for r in rs)
        tmax = max(float(r['construction_s']) for r in rs)
        sp = base_t / t
        eff = sp * base_w / w
        write(f"{w:>7} {len(rs):>4} {med(rs, 'download_s'):>9.1f} {med(rs, 'merge_s'):>7.1f} {med(rs, 'lcc_s'):>7.1f} "
              f"{t:>8.1f} {tmin:>8.1f} {tmax:>8.1f} {sp:>8.2f} {eff:>7.1%} {med(rs, 'overpass_mean_s'):>13.3f} "
              f"{med(rs, 'osmnx_mean_s'):>10.3f} {med(rs, 'waiting_share'):>7.1f} {med(rs, 'cpu_cores_used'):>10.2f} "
              f"{med(rs, 'sys_cpu_download'):>9.1f} {med(rs, 'load_download'):>6.2f} {med(rs, 'overpass_procs_max'):>9.0f} "
              f"{med(rs, 'mem_all_gb'):>8.2f} {med(rs, 'mem_max_gb'):>8.2f}")
    write("Columns: download/merge/LCC/total in seconds (median); min/max = shortest and longest total; "
          "overpass/cell = mean waiting time for the Overpass server per cell (s); osmnx/cell = mean OSMnx "
          "construction time per cell (s); wait % = share of the cell time spent waiting for the Overpass server; "
          "cores used = CPU used by the run in the download phase; sys CPU % and load = whole server during the "
          "download phase (includes other users); ovp procs = largest number of running Overpass server processes.")
    graphs = {(r.get('nodes'), r.get('edges')) for r in rows if r['ok']}
    failed = [(r['workers'], r['repeat']) for r in rows if not r['ok']]
    write(f"\nSame number of nodes and edges in all runs: {'yes' if len(graphs) == 1 else 'NO'} {sorted(graphs)}")
    write(f"Failed runs (workers, repeat): {failed if failed else 'none'}")
    write(f"Finished {datetime.datetime.now().isoformat(timespec='seconds')}")


if __name__ == '__main__':
    main()
