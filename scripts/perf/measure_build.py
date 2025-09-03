#!/usr/bin/env python3
"""Measure C++ (or generic) build performance and emit structured metrics.

Intent (T1):
 - Run clean + warmed builds N times.
 - Collect wall time + (optionally) link stage duration.
 - Output perf/build-<git-sha>/summary.json

Environment / Arguments:
 --clean           Perform a clean before measuring (delete build dir)
 --repeat N        Number of measurement repetitions (default 3)
 --build-cmd CMD   Build command (default: cmake --build build -j)
 --configure-cmd CMD Configure step to run when doing a clean (optional)
 --touch FILE      A small translation unit to touch for incremental timing

Assumptions:
 - Git repository present.
 - If using CMake, build directory is ./build (override by passing custom build command).
 - Link time extraction optional: if an artifact build/link.log exists, we attempt naive parse.

Future Enhancements:
 - Parse -ftime-trace JSON for header cost aggregation (handled separately in trace_aggregate.py).
 - Capture resident set size (RSS) via /usr/bin/time -v (portable wrapper needed later).
"""
from __future__ import annotations
import argparse, subprocess, time, json, os, shutil, datetime, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT / 'perf'

def sh(cmd: str, cwd: Path | None = None, env=None):
    start = time.time()
    res = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dur = time.time() - start
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError(f"Command failed ({res.returncode}): {cmd}")
    return dur, res.stdout

def git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    except Exception:
        return 'unknown'

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--build-cmd', default='cmake --build build -j')
    ap.add_argument('--configure-cmd', default='')
    ap.add_argument('--touch', default='')
    return ap.parse_args()

def ensure_perf_dir(sha: str) -> Path:
    run_dir = PERF_DIR / f'build-{sha}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def clean_build_dir():
    build_dir = ROOT / 'build'
    if build_dir.exists():
        shutil.rmtree(build_dir)

def configure_if_needed(configure_cmd: str):
    if configure_cmd:
        print(f'[configure] {configure_cmd}')
        sh(configure_cmd, cwd=ROOT)

def measure_clean_build(build_cmd: str):
    print(f'[build:clean] {build_cmd}')
    dur, out = sh(build_cmd, cwd=ROOT)
    link_ms = extract_link_time(out)
    return dur * 1000.0, link_ms

def measure_incremental(build_cmd: str, touch_file: str):
    if not touch_file:
        return None
    f = ROOT / touch_file
    if not f.exists():
        print(f'[warn] touch file not found: {touch_file}', file=sys.stderr)
        return None
    now = time.time()
    os.utime(f, (now, now))
    print(f'[build:incremental] {build_cmd}')
    dur, out = sh(build_cmd, cwd=ROOT)
    return dur * 1000.0

def extract_link_time(output: str):
    # Placeholder heuristic: look for lines like "[100%] Linking CXX executable" and mark start
    lines = output.splitlines()
    link_start_idx = None
    for i,l in enumerate(lines):
        if 'Linking CXX executable' in l or 'Linking' in l:
            link_start_idx = i
    # Not very accurate; refine later by parsing -ftime-trace or ninja logs.
    return None

def main():
    args = parse_args()
    sha = git_sha()
    run_dir = ensure_perf_dir(sha)

    clean_times = []
    link_times = []
    incremental_times = []

    for i in range(args.repeat):
        if args.clean:
            clean_build_dir()
            configure_if_needed(args.configure_cmd)
        clean_ms, link_ms = measure_clean_build(args.build_cmd)
        clean_times.append(clean_ms)
        if link_ms is not None:
            link_times.append(link_ms)
        inc = measure_incremental(args.build_cmd, args.touch)
        if inc is not None:
            incremental_times.append(inc)

    def median_or_null(vals):
        return round(statistics.median(vals),2) if vals else None

    summary = {
        'git_sha': sha,
        'timestamp': datetime.datetime.utcnow().isoformat()+'Z',
        'clean_ms': median_or_null(clean_times),
        'incremental_ms': median_or_null(incremental_times),
        'link_ms': median_or_null(link_times),
        'cache_hit_ratio': None,
        'template_instantiations': None,
        'hdr_top': []
    }

    out_file = run_dir / 'summary.json'
    out_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Wrote {out_file}')

if __name__ == '__main__':
    main()
