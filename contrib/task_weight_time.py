#!/usr/bin/env python3
#
# Compute a per-task "weight_time" (1..10) from a buildstats directory.
#
# Buildstats layout (as produced by OE-Core's buildstats.bbclass):
#   <buildstats_dir>/<pn>-<pv>-<pr>/<taskname>
# Each task file contains a line like:
#   Elapsed time: 9.61 seconds
#
# The weight is computed on a log10 scale over the observed elapsed-time
# range in this build. Task times span ~5 orders of magnitude, so a linear
# scale collapses ~99% of tasks to weight 1; log10 spreads them evenly.
#
# Usage:
#   contrib/task_weight_time.py <buildstats_dir> [-o task_weights.json]
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ELAPSED_RE = re.compile(r'^Elapsed time:\s*([\d.]+)\s*seconds', re.MULTILINE)
# Recipe dir looks like "pn-pv-rN". pv can contain dots and (rarely) dashes.
# Strip the trailing "-<pv>-r<N>" to recover pn.
PN_RE = re.compile(r'^(?P<pn>.+)-[^-]+-r\d+$')

# task-depends.dot edges look like:  "pn.do_task" -> "pn.do_task"
# where "A -> B" means "A depends on B" (B must run before A).
DOT_EDGE_RE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')

# Clamp very-small elapsed values so log10 stays finite. 10ms is below the
# resolution we care about for scheduling decisions anyway.
MIN_ELAPSED = 0.01


def parse_elapsed(path: Path) -> float | None:
    try:
        text = path.read_text(errors='ignore')
    except OSError:
        return None
    m = ELAPSED_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def pn_from_recipe_dir(name: str) -> str:
    m = PN_RE.match(name)
    return m.group('pn') if m else name


def collect_timings(buildstats: Path) -> dict[tuple[str, str], float]:
    """Return {(pn, taskname): elapsed_seconds} for every task file found."""
    timings: dict[tuple[str, str], float] = {}
    for recipe_dir in sorted(buildstats.iterdir()):
        if not recipe_dir.is_dir():
            continue
        pn = pn_from_recipe_dir(recipe_dir.name)
        for task_file in recipe_dir.iterdir():
            if not task_file.is_file() or not task_file.name.startswith('do_'):
                continue
            elapsed = parse_elapsed(task_file)
            if elapsed is None:
                continue
            # If the same (pn, task) appears twice (multiconfig or rebuild),
            # keep the maximum -- worst case is what matters for scheduling.
            key = (pn, task_file.name)
            if elapsed > timings.get(key, -1.0):
                timings[key] = elapsed
    return timings


def build_weighter(elapsed_values: list[float]):
    """Return a function elapsed -> int weight in [1, 10] using log10 scaling."""
    clamped = [max(v, MIN_ELAPSED) for v in elapsed_values if v > 0]
    if not clamped:
        return lambda _e: 1, (0.0, 0.0)
    log_min = math.log10(min(clamped))
    log_max = math.log10(max(clamped))
    span = log_max - log_min

    def to_weight(elapsed: float) -> int:
        if elapsed <= 0 or span <= 0:
            return 1
        lg = math.log10(max(elapsed, MIN_ELAPSED))
        # Map [log_min, log_max] -> [1, 10].
        w = 1 + (lg - log_min) * 9.0 / span
        return int(max(1, min(10, round(w))))

    return to_weight, (log_min, log_max)


def parse_task_graph(dot_path: Path) -> dict[tuple[str, str], set[tuple[str, str]]]:
    """Parse a bitbake `task-depends.dot` file.

    Returns {(pn, taskname): set of direct revdep (pn, taskname) tuples},
    i.e. for each node t, the set of tasks that directly depend on t.

    The .dot file uses edges "A" -> "B" where A depends on B (A runs after B).
    So the reverse edge B -> A gives revdeps of B.
    """
    revdeps: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    all_nodes: set[tuple[str, str]] = set()

    def split_node(label: str) -> tuple[str, str] | None:
        # Labels look like "pn.do_taskname". pn may contain dots (rare) but
        # taskname always starts with "do_", so split on the last ".do_".
        idx = label.rfind('.do_')
        if idx < 0:
            return None
        return (label[:idx], label[idx + 1:])

    with dot_path.open() as fh:
        for line in fh:
            m = DOT_EDGE_RE.match(line)
            if not m:
                continue
            a = split_node(m.group(1))
            b = split_node(m.group(2))
            if not a or not b:
                continue
            all_nodes.add(a)
            all_nodes.add(b)
            # Edge A -> B means A depends on B, so A is a revdep of B.
            revdeps[b].add(a)

    # Ensure every seen node has an entry (leaves have empty revdeps).
    for n in all_nodes:
        revdeps.setdefault(n, set())
    return revdeps


def compute_downstream(
    revdeps: dict[tuple[str, str], set[tuple[str, str]]],
) -> dict[tuple[str, str], set[tuple[str, str]]]:
    """For each node t, return the transitive downstream cone
    (the set of tasks that (in)directly depend on t)."""
    downstream: dict[tuple[str, str], set[tuple[str, str]]] = {n: set() for n in revdeps}
    computed: set[tuple[str, str]] = set()
    sys.setrecursionlimit(50000)

    def visit(node: tuple[str, str], stack: set) -> set[tuple[str, str]]:
        if node in computed:
            return downstream[node]
        if node in stack:
            # Should not happen in a DAG, guard anyway.
            return set()
        stack.add(node)
        acc: set[tuple[str, str]] = set()
        for r in revdeps[node]:
            acc.add(r)
            acc |= visit(r, stack)
        stack.discard(node)
        downstream[node] = acc
        computed.add(node)
        return acc

    for n in revdeps:
        visit(n, set())
    return downstream


def build_unblock_weighter(values: dict[tuple[str, str], float]):
    """Return a function value -> int score in [1, 10] using log10(1 + value)."""
    if not values:
        return lambda _v: 1, 0.0
    max_v = max(values.values())
    if max_v <= 0:
        return lambda _v: 1, max_v
    log_max = math.log10(1 + max_v)

    def to_score(v: float) -> int:
        if v <= 0 or log_max <= 0:
            return 1
        w = 1 + math.log10(1 + v) / log_max * 9.0
        return int(max(1, min(10, round(w))))

    return to_score, max_v


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description='Compute per-task weight_time (1..10) from a buildstats dir, '
                    'and optionally combine with an unblock_score from a task '
                    'dependency graph to produce a real_weight (2..20).')
    ap.add_argument('buildstats', type=Path,
                    help='Path to a buildstats/<timestamp> directory')
    ap.add_argument('-o', '--output', type=Path, default=Path('task_weights.json'),
                    help='Output JSON path (default: ./task_weights.json)')
    ap.add_argument('-g', '--task-graph', type=Path, default=None,
                    help='Optional task-depends.dot (from `bitbake -g <target>`). '
                         'When supplied, unblock_score and real_weight are also computed.')
    args = ap.parse_args(argv)

    if not args.buildstats.is_dir():
        print(f"error: {args.buildstats} is not a directory", file=sys.stderr)
        return 2

    timings = collect_timings(args.buildstats)
    if not timings:
        print(f"error: no task timings found under {args.buildstats}", file=sys.stderr)
        return 1

    elapsed_values = list(timings.values())
    to_weight, (log_min, log_max) = build_weighter(elapsed_values)

    # Precompute weight_time per (pn, task) so we can weight downstream work.
    weight_time: dict[tuple[str, str], int] = {
        key: to_weight(v) for key, v in timings.items()
    }
    # Fallback for nodes present in the DAG but missing from buildstats
    # (setscene-only tasks, noexec sentinels, newly-added recipes). Median is
    # a safer neutral than 0 or 1 -- it neither erases nor over-inflates them.
    default_weight = _median(list(weight_time.values())) if weight_time else 1

    # Optional: parse dep graph and compute unblock scores.
    unblock_count: dict[tuple[str, str], int] = {}
    unblock_work: dict[tuple[str, str], int] = {}
    to_score = None
    max_unblock_work = 0
    missing_from_buildstats = 0
    if args.task_graph:
        if not args.task_graph.is_file():
            print(f"error: {args.task_graph} not found", file=sys.stderr)
            return 2
        revdeps = parse_task_graph(args.task_graph)
        downstream = compute_downstream(revdeps)
        for n, ds in downstream.items():
            unblock_count[n] = len(ds)
            unblock_work[n] = sum(weight_time.get(u, default_weight) for u in ds)
        # How many DAG nodes have no buildstats entry?
        missing_from_buildstats = sum(
            1 for n in revdeps if n not in weight_time
        )
        to_score, max_unblock_work = build_unblock_weighter(unblock_work)

    # Build nested {pn: {taskname: {...}}} structure.
    weights: dict[str, dict[str, dict]] = defaultdict(dict)
    wt_bucket: Counter = Counter()
    us_bucket: Counter = Counter()
    rw_bucket: Counter = Counter()
    for (pn, task), elapsed in sorted(timings.items()):
        wt = to_weight(elapsed)
        entry = {
            'elapsed': round(elapsed, 3),
            'weight_time': wt,
        }
        if to_score is not None:
            uc = unblock_count.get((pn, task), 0)
            uw = unblock_work.get((pn, task), 0)
            us = to_score(uw)
            entry['unblock_count'] = uc
            entry['unblock_work'] = uw
            entry['unblock_score'] = us
            entry['real_weight'] = wt + us
            us_bucket[us] += 1
            rw_bucket[wt + us] += 1
        weights[pn][task] = entry
        wt_bucket[wt] += 1

    positive = [v for v in elapsed_values if v > 0]
    output: dict = {
        'source': str(args.buildstats),
        'num_tasks': len(timings),
        'num_recipes': len(weights),
        'elapsed_range_seconds': [
            round(min(elapsed_values), 3),
            round(max(elapsed_values), 3),
        ],
        'log10_range': [round(log_min, 3), round(log_max, 3)],
        'scale_time': 'weight_time = round(1 + (log10(elapsed) - log10_min) * 9 / (log10_max - log10_min))',
        'weight_time_distribution': {str(k): wt_bucket[k] for k in sorted(wt_bucket)},
    }
    if to_score is not None:
        output.update({
            'task_graph': str(args.task_graph),
            'default_weight_for_missing_nodes': default_weight,
            'nodes_missing_from_buildstats': missing_from_buildstats,
            'max_unblock_work': max_unblock_work,
            'scale_unblock': 'unblock_score = round(1 + log10(1 + unblock_work) / log10(1 + max_unblock_work) * 9)',
            'scale_real': 'real_weight = weight_time + unblock_score  (range 2..20)',
            'unblock_score_distribution': {str(k): us_bucket[k] for k in sorted(us_bucket)},
            'real_weight_distribution': {str(k): rw_bucket[k] for k in sorted(rw_bucket)},
        })
    output['weights'] = weights

    args.output.write_text(json.dumps(output, indent=2, sort_keys=False))
    print(f"Wrote {args.output}")
    print(f"  tasks: {len(timings)} across {len(weights)} recipes")
    print(f"  elapsed: {min(positive):.3f}s .. {max(positive):.1f}s")
    print(f"  log10 range: [{log_min:.3f}, {log_max:.3f}]")
    print("  weight_time distribution:")
    _print_hist(wt_bucket)
    if to_score is not None:
        print(f"  DAG nodes missing from buildstats: {missing_from_buildstats}"
              f" (using default weight_time = {default_weight})")
        print(f"  unblock_work: 0 .. {max_unblock_work}")
        print("  unblock_score distribution:")
        _print_hist(us_bucket)
        print("  real_weight distribution:")
        _print_hist(rw_bucket)
    return 0


def _median(values: list[int]) -> int:
    if not values:
        return 1
    xs = sorted(values)
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) // 2


def _print_hist(bucket: Counter) -> None:
    if not bucket:
        return
    peak = max(bucket.values())
    for k in sorted(bucket):
        count = bucket[k]
        bar = '#' * max(1, int(50 * count / peak))
        print(f"    {k:2d}: {count:5d}  {bar}")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
