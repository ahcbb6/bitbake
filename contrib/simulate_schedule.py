#!/usr/bin/env python3
#
# Discrete-event simulator for BitBake scheduler A/B testing.
#
# Reads a task dependency graph (`task-depends.dot` from `bitbake -g`) and
# a buildstats directory (produced by inheriting `buildstats.bbclass`), then
# replays the build under a chosen scheduler at a chosen parallelism level
# and reports the simulated wall clock.
#
# All tetris weights are computed on the fly from raw elapsed times so
# parameters can be tuned per run without regenerating any cache file.
# Adaptive Tetris uses the live policy from bb.tetris at each dispatch.
# --no-tetris-dynamic restores the original static policy for comparison.
#
# Simulator caveats (each optimistic vs real bitbake):
#   * ignores pressure throttling (BB_PRESSURE_MAX_*, BB_LOADFACTOR_MAX)
#   * ignores per-taskname number_threads caps
#   * no setscene phase (all tasks run for their full elapsed time)
#   * no worker-fork or scheduling-tick overhead
# Task durations stay fixed even under CPU contention, so these results are
# diagnostic estimates, not predictions of relative real-build performance.
# By default all tasks share BB_NUMBER_THREADS worker slots, including fetches.
# A separate fetch pool is available only as an explicit hypothetical experiment.
#
# Usage:
#   contrib/simulate_schedule.py \
#       -g <task-depends.dot> \
#       -b <buildstats_dir> \
#       [-s speed|completion|tetris] \
#       [-j <threads>] \
#       [--compare] \
#       [--critical-path] \
#       [--sweep --no-tetris-dynamic]
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import heapq
import itertools
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from bb.tetris import TetrisPriority, estimate_task_durations

Node = tuple[str, str]  # (pn, taskname)

DOT_EDGE_RE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')
DOT_NODE_RE = re.compile(r'^\s*"([^"]+)"\s*\[')
ELAPSED_RE = re.compile(r'^Elapsed time:\s*([\d.]+)\s*seconds', re.MULTILINE)
# 'Child rusage ru_utime: N.NN' + '... ru_stime: N.NN'. Sum is real CPU-seconds
# consumed by the task's child processes (make/ninja/etc.).
CHILD_UTIME_RE = re.compile(r'^Child rusage ru_utime:\s*([\d.]+)', re.MULTILINE)
CHILD_STIME_RE = re.compile(r'^Child rusage ru_stime:\s*([\d.]+)', re.MULTILINE)
PN_RE = re.compile(r'^(?P<pn>.+)-[^-]+-r\d+$')

DEFAULT_DURATION = 1.0  # seconds; only used if we have no timing info at all
MIN_ELAPSED = 0.01      # clamp for log10


# --------------------------------------------------------------------- loading

def parse_task_graph(dot_path: Path) -> tuple[
    dict[Node, set[Node]],
    dict[Node, set[Node]],
]:
    """Parse task-depends.dot. Returns (depends, revdeps)."""
    depends: dict[Node, set[Node]] = defaultdict(set)
    revdeps: dict[Node, set[Node]] = defaultdict(set)

    def split(label: str) -> Node | None:
        idx = label.rfind('.do_')
        if idx < 0:
            return None
        return (label[:idx], label[idx + 1:])

    with dot_path.open() as fh:
        for line in fh:
            m = DOT_EDGE_RE.match(line)
            if not m:
                declaration = DOT_NODE_RE.match(line)
                n = split(declaration.group(1)) if declaration else None
                if n:
                    depends.setdefault(n, set())
                    revdeps.setdefault(n, set())
                continue
            a, b = split(m.group(1)), split(m.group(2))
            if not a or not b:
                continue
            # A -> B means A depends on B (B runs first).
            depends[a].add(b)
            revdeps[b].add(a)
            depends.setdefault(b, set())
            revdeps.setdefault(a, set())
    return depends, revdeps


def collect_timings_from_buildstats(
    buildstats: Path,
) -> dict[Node, tuple[float, float]]:
    """Walk a buildstats directory. Returns {(pn, taskname): (elapsed, cpu)}.

    Layout (from buildstats.bbclass):
        <buildstats>/<pn>-<pv>-<pr>/<taskname>
    Each task file has 'Elapsed time: N.NN seconds' plus
    'Child rusage ru_utime: N.NN' / 'ru_stime: N.NN' (child CPU seconds).

    If the buildstats path contains a single timestamped subdir (typical when
    pointing at TMPDIR/buildstats), descend into it automatically.
    """
    if not buildstats.is_dir():
        raise SystemExit(f"buildstats path is not a directory: {buildstats}")

    entries = sorted(p for p in buildstats.iterdir() if p.is_dir())
    if entries and all(re.match(r'^\d{10,}$', p.name) for p in entries):
        entries.sort(reverse=True)
        for cand in entries:
            if any(c.is_dir() for c in cand.iterdir()):
                buildstats = cand
                break

    timings: dict[Node, tuple[float, float]] = {}
    for recipe_dir in sorted(buildstats.iterdir()):
        if not recipe_dir.is_dir():
            continue
        m = PN_RE.match(recipe_dir.name)
        pn = m.group('pn') if m else recipe_dir.name
        for task_file in recipe_dir.iterdir():
            if not task_file.is_file() or not task_file.name.startswith('do_'):
                continue
            try:
                text = task_file.read_text(errors='ignore')
            except OSError:
                continue
            em = ELAPSED_RE.search(text)
            if not em:
                continue
            try:
                elapsed = float(em.group(1))
            except ValueError:
                continue
            um = CHILD_UTIME_RE.search(text)
            sm = CHILD_STIME_RE.search(text)
            cpu = 0.0
            try:
                if um:
                    cpu += float(um.group(1))
                if sm:
                    cpu += float(sm.group(1))
            except ValueError:
                pass
            key = (pn, task_file.name)
            # Keep the max if the same task appears twice (multiconfig).
            prev = timings.get(key)
            if prev is None or elapsed > prev[0]:
                timings[key] = (elapsed, cpu)
    return timings


def build_duration_map(
    timings: dict[Node, tuple[float, float]],
    all_nodes: set[Node],
    default: float | None = None,
) -> tuple[dict[Node, float], dict[Node, float], float, int]:
    """Return (elapsed_map, cpu_map, default_elapsed, filled_count).

    Missing nodes get the median elapsed and their cpu is assumed equal to it
    (single-core equivalent) so width defaults to 1.
    """
    elapsed = {k: v[0] for k, v in timings.items() if k in all_nodes}
    cpu = {k: v[1] for k, v in timings.items() if k in all_nodes}

    observed = [v for v in elapsed.values() if v > 0]
    if default is None:
        default = _median(observed) if observed else DEFAULT_DURATION

    filled = 0
    for n in all_nodes:
        if n not in elapsed or elapsed[n] <= 0:
            elapsed[n] = default
            cpu[n] = default
            filled += 1
        elif n not in cpu:
            cpu[n] = elapsed[n]
    return elapsed, cpu, default, filled


def compute_task_widths(
    elapsed: dict[Node, float],
    cpu: dict[Node, float],
) -> dict[Node, int]:
    """Approximate CPU demand from cpu/elapsed, independent of worker count."""
    widths: dict[Node, int] = {}
    for n, e in elapsed.items():
        c = cpu.get(n, 0.0)
        if e <= 0 or c <= 0:
            widths[n] = 1
            continue
        # This is reporting-only: one task still occupies one worker slot.
        w = max(1, int(round(c / e)))
        widths[n] = w
    return widths


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2.0


# ---------------------------------------------------------- tetris weight core
# All computed on the fly from raw elapsed times so parameters can be tuned
# without regenerating any cache file.

def log_normalize(
    values: dict[Node, float],
    out_lo: int = 1,
    out_hi: int = 10,
) -> dict[Node, int]:
    """Map values to integers in [out_lo, out_hi] via log10 scaling."""
    positive = [v for v in values.values() if v > 0]
    if not positive:
        return {k: out_lo for k in values}
    lg_min = math.log10(max(min(positive), MIN_ELAPSED))
    lg_max = math.log10(max(positive))
    span = lg_max - lg_min

    out: dict[Node, int] = {}
    span_out = out_hi - out_lo
    for k, v in values.items():
        if v <= 0 or span <= 0:
            out[k] = out_lo
            continue
        lg = math.log10(max(v, MIN_ELAPSED))
        w = out_lo + (lg - lg_min) * span_out / span
        out[k] = int(max(out_lo, min(out_hi, round(w))))
    return out


def compute_downstream_reduction(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    per_node_weight: dict[Node, int],
    agg: str = 'sum',
) -> dict[Node, float]:
    """Propagate downstream weights, matching the runtime scheduler.

    'sum' counts shared descendants once per dependency path, not once per
    unique node. 'max' finds the heaviest chain of normalized task weights.

    Iterative reverse-topo: no recursion, O(V+E).
    """
    # Reverse topo order (leaves first).
    remaining_rev = {n: len(revdeps[n]) for n in nodes}
    order: list[Node] = []
    frontier = [n for n in nodes if remaining_rev[n] == 0]
    while frontier:
        nxt = []
        for n in frontier:
            order.append(n)
            for d in depends[n]:
                remaining_rev[d] -= 1
                if remaining_rev[d] == 0:
                    nxt.append(d)
        frontier = nxt

    # Propagate upwards.
    out: dict[Node, float] = {n: 0.0 for n in nodes}
    if agg == 'sum':
        for n in order:
            s = 0.0
            for r in revdeps[n]:
                s += per_node_weight.get(r, 0) + out[r]
            out[n] = s
    elif agg == 'max':
        for n in order:
            best = 0.0
            for r in revdeps[n]:
                cand = per_node_weight.get(r, 0) + out[r]
                if cand > best:
                    best = cand
            out[n] = best
    else:
        raise ValueError(f"unknown agg: {agg}")
    return out


# ------------------------------------------------------------------ schedulers
#
# All schedulers share the signature
#     fn(nodes, depends, revdeps, duration, params) -> list[Node]
# and return a priority list (index 0 = highest priority). Only tetris uses
# `duration` and `params`; the others accept them for a uniform API.

def scheduler_basic(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    duration: dict[Node, float],
    params: dict,
) -> list[Node]:
    """Base ordering -- insertion order, matches RunQueueScheduler (name='basic')."""
    return list(nodes)


def scheduler_speed(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    duration: dict[Node, float],
    params: dict,
) -> list[Node]:
    """Port of BitBake's `RunQueueSchedulerSpeed`: sort by RunTaskEntry.weight,
    computed via RunQueueData.calculate_task_weights (weight=1, endpoints=10,
    reverse-propagated)."""
    weight = {n: 1 for n in nodes}
    endpoints = [n for n in nodes if not revdeps[n]]
    for n in endpoints:
        weight[n] = 10
    deps_left = {n: len(revdeps[n]) for n in nodes}
    frontier = list(endpoints)
    while frontier:
        nxt = []
        for tid in frontier:
            for rdep in depends[tid]:
                weight[rdep] = weight[rdep] + weight[tid]
                deps_left[rdep] -= 1
                if deps_left[rdep] == 0:
                    nxt.append(rdep)
        frontier = nxt

    # Runtime builds ascending weight buckets and reverses the complete list.
    return list(reversed(sorted(nodes, key=lambda n: weight[n])))


def scheduler_completion(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    duration: dict[Node, float],
    params: dict,
) -> list[Node]:
    """Port of `RunQueueSchedulerCompletion`: start from Speed order, then
    reorder so tasks of the same kind cluster together per recipe."""
    speed = scheduler_speed(nodes, depends, revdeps, duration, params)

    # Extract per-recipe task order from Speed's priority list.
    task_lists: dict[str, list[str]] = {}
    for pn, task in speed:
        task_lists.setdefault(pn, []).append(task)

    # Unify into a global task-kind order (see bb.runqueue.RunQueueSchedulerCompletion).
    all_tasks: list[str] = []
    for _pn, new_tasks in task_lists.items():
        index = 0
        old_task = all_tasks[index] if index < len(all_tasks) else None
        for new_task in new_tasks:
            if old_task == new_task:
                index += 1
                old_task = all_tasks[index] if index < len(all_tasks) else None
            else:
                if new_task in all_tasks:
                    index = all_tasks.index(new_task) + 1
                    old_task = all_tasks[index] if index < len(all_tasks) else None
                else:
                    all_tasks.insert(index, new_task)
                    index += 1
    all_tasks.reverse()

    # Reorder Speed by clustering tasks of each kind first (preserving relative
    # Speed order inside each cluster).
    prio_map = list(speed)
    task_index = 0
    for kind in all_tasks:
        for i in range(task_index, len(prio_map)):
            if prio_map[i][1] == kind:
                item = prio_map.pop(i)
                prio_map.insert(task_index, item)
                task_index += 1
    return prio_map


def scheduler_tetris(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    duration: dict[Node, float],
    params: dict,
) -> list[Node]:
    """Build the static ordering used as the adaptive policy's tie-break.

    With --no-tetris-dynamic this is the complete selection policy.

    Computes on the fly:
        weight_time[n]     = log10-normalized elapsed          in [1, resolution]
        downstream_raw[n]  = agg(weight_time over downstream)  agg in {sum, max}
        unblock_score[n]   = log10-normalized downstream_raw   in [1, resolution]
        real_weight[n]     = alpha * weight_time[n] + beta * unblock_score[n]

    Then computes PERT slack from raw `duration` (seconds) on the live DAG.
    Any do_fetch task whose slack exceeds `fetch_slack_pct` of the estimated
    makespan is *deferred*: sorted after every non-deferred task, so it only
    runs when the scheduler would otherwise leave a worker idle. Fetches on
    (or near) the critical path retain their normal priority.

    Ties preserve input node order, as in the runtime. DOT declaration order
    need not match runtime task insertion order, so tied tasks may still differ.

    Params (all optional):
        alpha (float, default 0.5)     weight_time coefficient
        beta  (float, default 3.0)     unblock_score coefficient
        agg   ('sum'|'max', default 'max')  downstream aggregation
        cluster (bool, default False)  apply completion-style tie-break clustering
        resolution (int, default 20)   log-quantization buckets for weight_time
                                       and unblock_score; higher = fewer ties
        defer_fetch (bool, default False)     enable non-critical fetch deferral
        fetch_slack_pct (float, default 1.0)  slack > pct*makespan => defer fetch
        timings (dict, default duration)     raw observed timings, before filling
                                             missing replay durations

    Defaults picked via grid search on core-image-minimal buildstats; see
    contrib/simulate_schedule.py --sweep --no-tetris-dynamic for the search procedure.
    """
    alpha = float(params.get('alpha', 0.5))
    beta = float(params.get('beta', 3.0))
    agg = params.get('agg', 'max')
    cluster = bool(params.get('cluster', False))
    resolution = int(params.get('resolution', 20))
    defer_fetch = bool(params.get('defer_fetch', False))
    fetch_slack_pct = float(params.get('fetch_slack_pct', 1.0))
    timings = params.get('timings', duration)

    # 1. weight_time from elapsed.
    observed_weights = (
        log_normalize(timings, out_lo=1, out_hi=resolution)
        if any(v > 0 for v in timings.values()) else {}
    )
    weight_time = {n: observed_weights.get(n, 4) for n in nodes}

    # 2. downstream aggregation (sum or max of downstream weight_times).
    downstream_raw = compute_downstream_reduction(
        nodes, depends, revdeps, weight_time, agg=agg,
    )

    # 3. normalize downstream to [1, resolution] via log10.
    # Every DAG has a sink with work=0, anchoring log10(1 + work) at zero.
    unblock_score = log_normalize(
        {n: 1 + w for n, w in downstream_raw.items()},
        out_lo=1, out_hi=resolution,
    )

    # 4. combine.
    real_weight = {
        n: alpha * weight_time[n] + beta * unblock_score[n]
        for n in nodes
    }

    # 5. identify non-critical fetches to defer.
    estimated = estimate_task_durations({n: n for n in nodes}, timings)
    deferred = tetris_deferred_tasks(
        nodes, depends, revdeps, estimated,
        defer_fetch and bool(timings), fetch_slack_pct,
    )

    # 6. sort. First key: not-deferred (deferred fetches trail). Then by
    # real_weight desc, then raw weight_time desc, preserving input ties.
    prio = sorted(
        nodes,
        key=lambda n: (n in deferred, -real_weight[n], -weight_time[n]),
    )

    if cluster:
        prio = _apply_kind_clustering(prio)
    return prio


def tetris_deferred_tasks(nodes, depends, revdeps, duration, enabled, slack_pct):
    if not enabled:
        return set()
    slack, makespan = compute_pert_slack(nodes, depends, revdeps, duration)
    threshold = makespan * slack_pct / 100.0
    return {n for n in nodes if n[1] == 'do_fetch' and slack[n] > threshold}


def make_tetris_policy(nodes, depends, revdeps, duration, priority, params):
    if not params.get('dynamic', True):
        return None
    timings = params.get('timings', duration)
    estimated = estimate_task_durations({n: n for n in nodes}, timings)
    deferred = tetris_deferred_tasks(
        nodes, depends, revdeps, estimated,
        params.get('defer_fetch', False) and bool(timings),
        params.get('fetch_slack_pct', 1.0),
    )
    return TetrisPriority(depends, revdeps, estimated, priority, deferred)


def _apply_kind_clustering(prio: list[Node]) -> list[Node]:
    """Reorder so tasks of the same kind cluster together (like completion),
    preserving the input priority order inside each cluster.

    Uses the task-kind order induced by the input list itself (kinds are
    encountered in priority order, and each kind's cluster is placed at the
    position of its first occurrence)."""
    # Discover kind order from the input (first-seen wins).
    kinds: list[str] = []
    seen: set[str] = set()
    for _pn, task in prio:
        if task not in seen:
            kinds.append(task)
            seen.add(task)

    # Bucket by kind, preserving in-cluster order.
    buckets: dict[str, list[Node]] = {k: [] for k in kinds}
    for n in prio:
        buckets[n[1]].append(n)

    out: list[Node] = []
    for k in kinds:
        out.extend(buckets[k])
    return out


SCHEDULERS: dict[str, Callable] = {
    'basic':      scheduler_basic,
    'speed':      scheduler_speed,
    'completion': scheduler_completion,
    'tetris':     scheduler_tetris,
}


# --------------------------------------------------------------- critical path

def compute_pert_slack(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    duration: dict[Node, float],
) -> tuple[dict[Node, float], float]:
    """PERT slack per node in seconds. Returns (slack, makespan_estimate).

    Forward pass: EST[n] = max(EFT[pred]).
    Backward pass: LFT[n] = min(LFT[rd] - duration[rd]); LFT[sink] = makespan.
    slack[n] = LFT[n] - duration[n] - EST[n]. slack == 0 means n is on the CP.
    """
    # Kahn's topological sort.
    remaining = {n: len(depends[n]) for n in nodes}
    frontier = [n for n in nodes if remaining[n] == 0]
    topo: list[Node] = []
    while frontier:
        nxt = []
        for n in frontier:
            topo.append(n)
            for rdep in revdeps[n]:
                remaining[rdep] -= 1
                if remaining[rdep] == 0:
                    nxt.append(rdep)
        frontier = nxt

    if len(topo) != len(nodes) or not nodes:
        return {n: 0.0 for n in nodes}, 0.0

    est: dict[Node, float] = {n: 0.0 for n in nodes}
    for n in topo:
        eft = est[n] + duration[n]
        for r in revdeps[n]:
            if eft > est[r]:
                est[r] = eft

    makespan = max(est[n] + duration[n] for n in nodes)

    lft: dict[Node, float] = {n: makespan for n in nodes}
    for n in reversed(topo):
        if revdeps[n]:
            best = float('inf')
            for r in revdeps[n]:
                cand = lft[r] - duration[r]
                if cand < best:
                    best = cand
            lft[n] = best

    slack = {}
    for n in nodes:
        s = lft[n] - duration[n] - est[n]
        slack[n] = s if s > 0.0 else 0.0
    return slack, makespan


def find_critical_path(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    duration: dict[Node, float],
) -> tuple[list[Node], float]:
    """Longest weighted path through the DAG.

    Returns (path, total_seconds) where path[0] is a source (no deps) and
    path[-1] is a sink (no revdeps) -- i.e. execution order along the chain.

    This is the makespan lower bound: no scheduler with unbounded parallelism
    can finish the build in less than this much wall time.
    """
    # Kahn's topological sort.
    remaining = {n: len(depends[n]) for n in nodes}
    frontier = [n for n in nodes if remaining[n] == 0]
    topo: list[Node] = []
    while frontier:
        nxt = []
        for n in frontier:
            topo.append(n)
            for rdep in revdeps[n]:
                remaining[rdep] -= 1
                if remaining[rdep] == 0:
                    nxt.append(rdep)
        frontier = nxt

    # Longest-path DP in topo order.
    #   ef[n] = earliest finish time = duration[n] + max(ef[pred] for pred in depends[n])
    #   parent[n] = pred that gave the max (for backtracking).
    ef: dict[Node, float] = {}
    parent: dict[Node, Node | None] = {}
    for n in topo:
        best_pred: Node | None = None
        best_val = 0.0
        for pred in depends[n]:
            v = ef.get(pred, 0.0)
            if v > best_val:
                best_val = v
                best_pred = pred
        ef[n] = best_val + duration[n]
        parent[n] = best_pred

    if not ef:
        return [], 0.0

    # Endpoint = node with max earliest-finish.
    end = max(ef, key=lambda n: ef[n])
    total = ef[end]

    # Walk parents backwards.
    path: list[Node] = []
    cur: Node | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, total


def makespan_lower_bound(
    duration: dict[Node, float],
    critical_path_len: float,
    threads: int,
    io_fetch_pool: int | None = None,
) -> tuple[float, str]:
    """Best possible finish time given the DAG and worker count.

    Returns (lower_bound_seconds, dominant_reason).

    Two independent lower bounds -- OPT is at least each of them:
        LB_cp   = critical path length         (dependency chain)
        LB_work = total task work / threads    (perfect load balancing)

    With a hypothetical separate fetch pool, bound each pool's work separately.
    These are lower bounds for the fixed-duration model, not real-build optima.
    """
    worker_work = sum(v for n, v in duration.items()
                      if io_fetch_pool is None or n[1] != 'do_fetch')
    bounds = [(critical_path_len, 'critical path'),
              (worker_work / threads, 'worker work / threads')]
    if io_fetch_pool is not None:
        fetch_work = sum(v for n, v in duration.items() if n[1] == 'do_fetch')
        bounds.append((fetch_work / io_fetch_pool, 'fetch work / fetch slots'))
    return max(bounds, key=lambda bound: bound[0])


def print_critical_path(
    path: list[Node],
    total: float,
    duration: dict[Node, float],
    max_rows: int = 30,
) -> None:
    print(f"Critical path: {len(path)} tasks, {fmt_hms(total)} ({total:.0f}s)")
    print("This is a fixed-duration lower bound; finishing other branches")
    print("can still take longer even if this chain runs without gaps.")
    print()

    # Sort tasks in the chain by duration (heaviest hitters first) to help
    # the user see where to focus recipe-level optimization.
    ranked = sorted(range(len(path)), key=lambda i: -duration[path[i]])
    show = ranked if len(ranked) <= max_rows else ranked[:max_rows]

    print(f"Heaviest {len(show)} of {len(path)} tasks on the critical path:")
    print(f"  {'pos':>4}  {'elapsed':>10}  {'%chain':>7}  task")
    for idx in show:
        pn, task = path[idx]
        d = duration[path[idx]]
        pct = (d / total) * 100 if total else 0
        print(f"  {idx + 1:>4}  {fmt_hms(d):>10}  {pct:>6.1f}%  {pn}:{task}")

    if len(path) > max_rows:
        remaining_secs = sum(duration[path[i]] for i in ranked[max_rows:])
        print(f"  ... {len(path) - max_rows} more, "
              f"cumulative {fmt_hms(remaining_secs)} "
              f"({remaining_secs/total*100:.1f}% of chain)")

    print()
    print("Full chain in execution order:")
    for i, n in enumerate(path):
        pn, task = n
        d = duration[n]
        print(f"  {i + 1:>4}. {fmt_hms(d):>9}  {pn}:{task}")


def find_top_chains(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    duration: dict[Node, float],
    k: int = 5,
) -> list[dict]:
    """Return the top-k heaviest chains ranked by *novel* work.

    Chain #1 is the true critical path. For each subsequent chain we pick the
    node whose parent-walk contributes the most seconds of work NOT covered by
    any earlier chain, then walk parent pointers backwards until we hit a node
    already covered ('merge point'). Only the novel prefix is reported, so
    trivial tail-fanout tasks that merely reuse the CP's tail don't dominate.

    Returns a list of dicts:
        {
          'endpoint':      terminal node of the chain,
          'ef_end':        earliest finish time at endpoint (seconds),
          'unique_path':   list of nodes new to this chain, in execution order,
          'merges_into':   node where this chain rejoins a prior one, or None
                           if the chain reaches all the way back to a source,
          'unique_secs':   sum of duration over unique_path,
        }
    """
    # Kahn's topological sort.
    remaining = {n: len(depends[n]) for n in nodes}
    frontier = [n for n in nodes if remaining[n] == 0]
    topo: list[Node] = []
    while frontier:
        nxt = []
        for n in frontier:
            topo.append(n)
            for rdep in revdeps[n]:
                remaining[rdep] -= 1
                if remaining[rdep] == 0:
                    nxt.append(rdep)
        frontier = nxt

    # Longest-path DP with parent pointers.
    ef: dict[Node, float] = {}
    parent: dict[Node, Node | None] = {}
    for n in topo:
        best_pred: Node | None = None
        best_val = 0.0
        for pred in depends[n]:
            v = ef.get(pred, 0.0)
            if v > best_val:
                best_val = v
                best_pred = pred
        ef[n] = best_val + duration[n]
        parent[n] = best_pred

    if not ef:
        return []

    covered: set[Node] = set()
    chains: list[dict] = []
    for _ in range(k):
        # For every node not yet covered, walk parents until hitting covered
        # (or DAG source) and sum novel duration. Track the node whose novel
        # path is heaviest.
        best_endpoint: Node | None = None
        best_novel_secs = -1.0
        best_rev_path: list[Node] = []
        best_merge: Node | None = None
        for n in ef:
            if n in covered:
                continue
            rev_path: list[Node] = []
            novel = 0.0
            cur: Node | None = n
            while cur is not None and cur not in covered:
                rev_path.append(cur)
                novel += duration[cur]
                cur = parent[cur]
            if novel > best_novel_secs:
                best_novel_secs = novel
                best_endpoint = n
                best_rev_path = rev_path
                best_merge = cur
        if best_endpoint is None or best_novel_secs <= 0.0:
            break
        best_rev_path.reverse()
        chains.append({
            'endpoint': best_endpoint,
            'ef_end': ef[best_endpoint],
            'unique_path': best_rev_path,
            'merges_into': best_merge,
            'unique_secs': best_novel_secs,
        })
        covered.update(best_rev_path)
    return chains


def print_top_chains(
    chains: list[dict],
    duration: dict[Node, float],
    max_rows_per_chain: int = 15,
) -> None:
    """Print the output of `find_top_chains` in a human-readable form."""
    if not chains:
        print("(no chains found)")
        return
    cp_len = chains[0]['ef_end']
    print(f"Top {len(chains)} heaviest chains (chain #1 is the critical path):")
    print()
    for i, c in enumerate(chains, 1):
        endpoint = c['endpoint']
        merge = c['merges_into']
        upath = c['unique_path']
        header = (
            f"=== Chain #{i}: ends at {endpoint[0]}:{endpoint[1]} "
            f"(EFT {fmt_hms(c['ef_end'])} = {c['ef_end']:.0f}s"
            f", {c['ef_end']/cp_len*100:.1f}% of CP)"
        )
        print(header)
        if merge is None:
            print(f"    ({len(upath)} tasks, reaches a DAG source)")
        else:
            print(f"    ({len(upath)} novel tasks, "
                  f"then merges into a prior chain at "
                  f"{merge[0]}:{merge[1]})")
        print(f"    novel path work: {fmt_hms(c['unique_secs'])} "
              f"({c['unique_secs']:.0f}s)")

        # Show the heaviest tasks in this novel segment.
        ranked = sorted(range(len(upath)), key=lambda j: -duration[upath[j]])
        show = ranked[:max_rows_per_chain]
        print(f"    Heaviest {len(show)} of {len(upath)} tasks in the novel segment:")
        print(f"      {'pos':>4}  {'elapsed':>10}  task")
        for idx in show:
            n = upath[idx]
            print(f"      {idx + 1:>4}  {fmt_hms(duration[n]):>10}  "
                  f"{n[0]}:{n[1]}")
        if len(upath) > max_rows_per_chain:
            tail = sum(duration[upath[j]] for j in ranked[max_rows_per_chain:])
            print(f"      ... {len(upath) - max_rows_per_chain} more, "
                  f"cumulative {fmt_hms(tail)}")
        print()


# -------------------------------------------------------------------- sim core

def simulate(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    duration: dict[Node, float],
    prio_map: list[Node],
    threads: int,
    width: dict[Node, int] | None = None,
    io_fetch_pool: int | None = None,
    policy: TetrisPriority | None = None,
) -> dict:
    """Discrete-event simulation. Returns stats dict.

    width[tid] = internal CPU demand of a task (models make -j / ninja -j
    inside do_compile). Matches real bitbake: each task takes exactly ONE
    worker slot to launch; width only informs utilization reporting so we
    can see when the pool is oversubscribed. Defaults to 1 everywhere,
    which reproduces the legacy behaviour.

    io_fetch_pool: an explicit hypothetical experiment, NOT real BitBake.
    If set, do_fetch tasks run in a separate pool rather than using workers.
    policy: the shared live Tetris policy, evaluated after every dispatch or
    completion. None uses the static priority list.
    """
    if threads < 1 or (io_fetch_pool is not None and io_fetch_pool < 1):
        raise ValueError("worker and fetch pool sizes must be positive")
    if width is None:
        width = {n: 1 for n in nodes}
    prio_rank = {n: i for i, n in enumerate(prio_map)}
    remaining = {n: len(depends[n]) for n in nodes}
    ready: list[tuple[int, Node]] = []
    completed: set[Node] = set()

    for n in nodes:
        if remaining[n] == 0:
            heapq.heappush(ready, (prio_rank[n], n))

    running: list[tuple[float, int, Node]] = []
    fetch_running: list[tuple[float, int, Node]] = []
    running_demand = 0  # Σ width[t] over currently-running CPU tasks
    fetch_slots = 0
    seq = 0
    now = 0.0

    total_task_time = 0.0
    busy_area = 0.0
    cpu_area = 0.0
    peak_parallelism = 0
    time_at_level = [0.0] * (threads + 1)

    def is_fetch(n: Node) -> bool:
        return io_fetch_pool is not None and n[1] == 'do_fetch'

    while ready or running or fetch_running:
        # One worker slot per task, exactly like real bitbake. Width only
        # affects the utilization metrics below.
        deferred: list[tuple[int, Node]] = []
        while ready:
            if policy is not None:
                eligible = [
                    n for _p, n in ready
                    if (fetch_slots < io_fetch_pool if is_fetch(n)
                        else len(running) < threads)
                ]
                if not eligible:
                    break
                tid = policy.select(eligible, completed, set(), threads, now)
                p = prio_rank[tid]
                ready.remove((p, tid))
                heapq.heapify(ready)
            else:
                p, tid = heapq.heappop(ready)
            if is_fetch(tid):
                if fetch_slots >= io_fetch_pool:
                    deferred.append((p, tid))
                    continue
                finish = now + duration[tid]
                heapq.heappush(fetch_running, (finish, seq, tid))
                fetch_slots += 1
                seq += 1
                total_task_time += duration[tid]
                continue
            if len(running) >= threads:
                deferred.append((p, tid))
                if io_fetch_pool is None:
                    break
                continue
            finish = now + duration[tid]
            heapq.heappush(running, (finish, seq, tid))
            running_demand += width[tid]
            seq += 1
            total_task_time += duration[tid]
            peak_parallelism = max(peak_parallelism, len(running))
        for item in deferred:
            heapq.heappush(ready, item)

        if not running and not fetch_running:
            break

        if running and fetch_running:
            next_finish = min(running[0][0], fetch_running[0][0])
        elif running:
            next_finish = running[0][0]
        else:
            next_finish = fetch_running[0][0]
        dt = next_finish - now
        busy_area += len(running) * dt
        cpu_area += running_demand * dt
        time_at_level[len(running)] += dt
        now = next_finish
        while running and running[0][0] == now:
            _f, _s, tid = heapq.heappop(running)
            completed.add(tid)
            running_demand -= width[tid]
            for rdep in revdeps[tid]:
                remaining[rdep] -= 1
                if remaining[rdep] == 0:
                    heapq.heappush(ready, (prio_rank[rdep], rdep))
        while fetch_running and fetch_running[0][0] == now:
            _f, _s, tid = heapq.heappop(fetch_running)
            completed.add(tid)
            fetch_slots -= 1
            for rdep in revdeps[tid]:
                remaining[rdep] -= 1
                if remaining[rdep] == 0:
                    heapq.heappush(ready, (prio_rank[rdep], rdep))

    unfinished = len(nodes) - len(completed)
    if unfinished:
        raise ValueError(f"{unfinished} tasks never became runnable (dependency cycle?)")

    sat = 0.0
    full_load = 0.0
    if now > 0 and threads > 0:
        for k in range(threads + 1):
            sat += ((k / threads) ** 2) * time_at_level[k] / now
        full_load = time_at_level[threads] / now

    return {
        'wall_clock_seconds': now,
        'total_task_seconds': total_task_time,
        'threads': threads,
        'nodes_total': len(nodes),
        'nodes_completed': len(completed),
        'nodes_unfinished': unfinished,
        'peak_parallelism': peak_parallelism,
        'avg_parallelism': (busy_area / now) if now > 0 else 0.0,
        'utilization': (busy_area / (now * threads)) if now > 0 else 0.0,
        'avg_cpu_demand': (cpu_area / now) if now > 0 else 0.0,
        'time_at_level': time_at_level,
        'saturation_score': sat,
        'full_load_fraction': full_load,
    }


# ------------------------------------------------------------------------ main

def fmt_hms(s: float) -> str:
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Simulate a bitbake build under a chosen scheduler.")
    ap.add_argument('-g', '--task-graph', type=Path, required=True,
                    help='task-depends.dot from `bitbake -g <target>`')
    ap.add_argument('-b', '--buildstats', type=Path, required=True,
                    help='buildstats dir (or TMPDIR/buildstats -- newest timestamp picked)')
    ap.add_argument('-s', '--scheduler', choices=sorted(SCHEDULERS),
                    default='tetris',
                    help='Scheduler to simulate (default: tetris)')
    ap.add_argument('-j', '--threads', type=int, default=8,
                    help='BB_NUMBER_THREADS (default: 8)')
    ap.add_argument('--compare', action='store_true',
                    help='Run every scheduler and print a comparison table')
    ap.add_argument('--critical-path', action='store_true',
                    help='Print the longest weighted path through the DAG '
                         '(makespan lower bound) and exit')
    ap.add_argument('--top-chains', type=int, metavar='K', default=None,
                    help='Print the K heaviest chains through the DAG (chain #1 '
                         'is the critical path; each subsequent chain shows '
                         'only its novel tail before merging back). Implies exit.')
    ap.add_argument('--sweep', action='store_true',
                    help='Grid-search static tetris params across j >= 8 '
                         '(requires --no-tetris-dynamic)')
    # Tetris tunables.
    ap.add_argument('--no-tetris-dynamic', dest='tetris_dynamic',
                    action='store_false', default=True,
                    help='Use the original static Tetris ordering instead of '
                         'adaptive critical-path/unlock selection (default: adaptive)')
    ap.add_argument('--tetris-alpha', type=float, default=0.5,
                    help='tetris: weight_time coefficient (default 0.5)')
    ap.add_argument('--tetris-beta', type=float, default=3.0,
                    help='tetris: unblock_score coefficient (default 3.0)')
    ap.add_argument('--tetris-agg', choices=('sum', 'max'), default='max',
                    help='tetris: downstream aggregation (default max)')
    ap.add_argument('--tetris-cluster', action='store_true',
                    help='tetris: apply task-kind clustering as tie-break')
    ap.add_argument('--tetris-resolution', type=int, default=20,
                    help='tetris: log-quantization buckets for weight_time and '
                         'unblock_score (default 20; higher = fewer ties, but '
                         'the sweep-optimum is 20 -- too fine actually hurts)')
    ap.add_argument('--tetris-defer-fetch', dest='tetris_defer_fetch',
                    action='store_true', default=False,
                    help='tetris: defer non-critical do_fetch tasks (default: off)')
    ap.add_argument('--no-tetris-defer-fetch', dest='tetris_defer_fetch',
                    action='store_false',
                    help='tetris: disable fetch deferral')
    ap.add_argument('--tetris-fetch-slack-pct', type=float, default=1.0,
                    help='tetris: defer a do_fetch when its PERT slack exceeds '
                         'this pct of estimated makespan (default 1.0)')
    ap.add_argument('--realistic', dest='realistic', action='store_true',
                    default=True,
                    help='Report estimated internal CPU demand from buildstats '
                         '(default: on). Does not change worker slots or '
                         'fixed task durations.')
    ap.add_argument('--no-realistic', dest='realistic', action='store_false',
                    help='Disable CPU-demand estimates; scheduling is unchanged')
    ap.add_argument('--io-fetch-concurrency', type=int, default=None,
                    help='Hypothetical separate fetch pool size; NOT BitBake '
                         'behavior. Default: fetches share worker slots.')
    args = ap.parse_args(argv)
    if args.threads < 1:
        ap.error('--threads must be >= 1')
    if args.io_fetch_concurrency is not None and args.io_fetch_concurrency < 1:
        ap.error('--io-fetch-concurrency must be >= 1')

    depends, revdeps = parse_task_graph(args.task_graph)
    all_nodes: set[Node] = set(depends) | set(revdeps)
    if not all_nodes:
        ap.error('task graph contains no tasks')
    timings = collect_timings_from_buildstats(args.buildstats)
    observed = {n: elapsed for n, (elapsed, _cpu) in timings.items()}
    duration, cpu, default_dur, filled = build_duration_map(timings, all_nodes)
    if args.realistic:
        width = compute_task_widths(duration, cpu)
    else:
        width = {n: 1 for n in all_nodes}
    io_pool = args.io_fetch_concurrency
    nodes = list(depends)

    print(f"Graph : {len(nodes)} tasks, "
          f"{sum(len(v) for v in depends.values())} edges")
    print(f"Timing: {len(nodes) - filled} from buildstats, "
          f"{filled} defaulted to {default_dur:.3f}s (median)")
    print(f"Total  task work: {fmt_hms(sum(duration.values()))} "
          f"({sum(duration.values()):.0f}s)")
    if args.realistic:
        wide = sum(1 for n in nodes if width[n] > 1)
        core_seconds = sum(cpu.get(n, duration[n]) for n in nodes)
        print(f"CPU-demand estimate: {wide} tasks use >1 core internally; "
              f"total CPU work {fmt_hms(core_seconds)} ({core_seconds:.0f}s)")
    if io_pool is not None:
        print(f"Hypothetical fetch pool: {io_pool} slots outside the worker "
              "pool (NOT real BitBake)")
    print(f"Threads: {args.threads}")
    print()

    if args.critical_path:
        path, total = find_critical_path(nodes, depends, revdeps, duration)
        print_critical_path(path, total, duration)
        return 0

    if args.top_chains is not None:
        if args.top_chains < 1:
            print(f"--top-chains must be >= 1", file=sys.stderr)
            return 2
        chains = find_top_chains(nodes, depends, revdeps, duration,
                                 k=args.top_chains)
        print_top_chains(chains, duration)
        return 0

    tetris_params = {
        'alpha': args.tetris_alpha,
        'beta': args.tetris_beta,
        'agg': args.tetris_agg,
        'cluster': args.tetris_cluster,
        'resolution': args.tetris_resolution,
        'defer_fetch': args.tetris_defer_fetch,
        'fetch_slack_pct': args.tetris_fetch_slack_pct,
        'timings': observed,
        'dynamic': args.tetris_dynamic,
    }

    def run(name: str, params: dict | None = None) -> dict:
        sched_fn = SCHEDULERS[name]
        p = params if params is not None else (tetris_params if name == 'tetris' else {})
        prio_map = sched_fn(nodes, depends, revdeps, duration, p)
        policy = (make_tetris_policy(nodes, depends, revdeps, duration, prio_map, p)
                  if name == 'tetris' else None)
        return simulate(nodes, depends, revdeps, duration, prio_map,
                        args.threads, width=width, io_fetch_pool=io_pool, policy=policy)

    if args.sweep:
        if args.tetris_dynamic:
            ap.error('--sweep tunes static coefficients; pass --no-tetris-dynamic')
        return _run_sweep(nodes, depends, revdeps, duration, width, io_pool, observed)

    if args.compare:
        rows = []
        for name in ('basic', 'speed', 'completion', 'tetris'):
            r = run(name)
            rows.append((name, r))
        best = min(r['wall_clock_seconds'] for _n, r in rows)
        _, cp_len = find_critical_path(nodes, depends, revdeps, duration)
        lb, lb_reason = makespan_lower_bound(duration, cp_len, args.threads, io_pool)
        print(f"{'scheduler':<12} {'wall':>10}  {'vs best':>8}  {'vs LB':>9}  "
              f"{'avg par':>8}  {'peak par':>8}  {'util':>6}  "
              f"{'sat':>6}  {'full%':>6}")
        for name, r in rows:
            wc = r['wall_clock_seconds']
            delta = (wc / best - 1) * 100 if best > 0 else 0.0
            lb_delta = (wc / lb - 1) * 100 if lb > 0 else 0.0
            print(f"{name:<12} {fmt_hms(wc):>10}  {delta:>+7.1f}%  "
                  f"{lb_delta:>+8.1f}%  "
                  f"{r['avg_parallelism']:>8.2f}  {r['peak_parallelism']:>8d}  "
                  f"{r['utilization']*100:>5.1f}%  "
                  f"{r['saturation_score']*100:>5.1f}%  "
                  f"{r['full_load_fraction']*100:>5.1f}%")
        print()
        print(f"  Model lower bound: {fmt_hms(lb)} ({lb:.0f}s), dominated by "
              f"{lb_reason}")
        print(f"    critical path : {fmt_hms(cp_len)} ({cp_len:.0f}s)")
        print("  Bounds and headroom apply only to this fixed-duration model.")
        print()
        print(f"  tetris params: alpha={tetris_params['alpha']} "
              f"beta={tetris_params['beta']} agg={tetris_params['agg']} "
              f"cluster={tetris_params['cluster']} "
              f"resolution={tetris_params['resolution']} "
              f"defer_fetch={tetris_params['defer_fetch']} "
              f"dynamic={tetris_params['dynamic']}")
        print("  util : Σ (k/N) × t[k] / wall   -- mean utilization (linear)")
        print("  sat  : Σ (k/N)² × t[k] / wall  -- saturation score (quadratic; punishes underuse)")
        print("  full%: t[N] / wall             -- fraction of wall time at max threads")
        print()
        _print_hist_compare(rows, args.threads)
    else:
        r = run(args.scheduler)
        _, cp_len = find_critical_path(nodes, depends, revdeps, duration)
        lb, lb_reason = makespan_lower_bound(duration, cp_len, args.threads, io_pool)
        lb_delta = (r['wall_clock_seconds'] / lb - 1) * 100 if lb > 0 else 0.0
        print(f"Scheduler: {args.scheduler}")
        if args.scheduler == 'tetris':
            print(f"  Params: alpha={tetris_params['alpha']} "
                  f"beta={tetris_params['beta']} agg={tetris_params['agg']} "
                  f"cluster={tetris_params['cluster']} "
                  f"resolution={tetris_params['resolution']} "
                  f"defer_fetch={tetris_params['defer_fetch']} "
                  f"dynamic={tetris_params['dynamic']}")
        print(f"  Simulated wall clock: {fmt_hms(r['wall_clock_seconds'])} "
              f"({r['wall_clock_seconds']:.0f}s)")
        print(f"  Model lower bound  : {fmt_hms(lb)} ({lb:.0f}s, {lb_reason})  "
              f"-> +{lb_delta:.1f}% above LB")
        print(f"  Peak parallelism   : {r['peak_parallelism']}")
        print(f"  Avg  parallelism   : {r['avg_parallelism']:.2f}")
        print(f"  Worker utilization : {r['utilization']*100:.1f}%  (linear Σ k/N)")
        if args.realistic:
            print(f"  Avg CPU demand     : {r['avg_cpu_demand']:.2f} cores "
                  "(estimate, no contention model)")
        print(f"  Saturation score   : {r['saturation_score']*100:.1f}%  (quadratic Σ (k/N)²)")
        print(f"  Full-load fraction : {r['full_load_fraction']*100:.1f}%  "
              f"(time at exactly {args.threads} workers)")
        print(f"  Tasks completed    : {r['nodes_completed']} / {r['nodes_total']}")
        if r['nodes_unfinished']:
            print(f"  WARNING: {r['nodes_unfinished']} tasks never became runnable"
                  f" (dependency cycle?)")
        print()
        _print_hist_single(r, args.threads)

    return 0


# --------------------------------------------------------------------- sweep
#
# Grid-search tetris params, evaluate each config across j in {8, 16, 32}, and
# rank by mean saturation score. (j <= 4 is intentionally excluded -- schedulers
# converge there since parallelism is capacity-bound, not scheduling-bound.)

SWEEP_THREADS = (8, 16, 32)
SWEEP_ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
SWEEP_BETAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
SWEEP_AGGS = ('sum', 'max')
SWEEP_CLUSTER = (False,)
SWEEP_RESOLUTION = (10, 20, 50, 100)
SWEEP_DEFER_FETCH = (False, True)


def _run_sweep(
    nodes: list[Node],
    depends: dict[Node, set[Node]],
    revdeps: dict[Node, set[Node]],
    duration: dict[Node, float],
    width: dict[Node, int],
    io_pool: int | None,
    timings: dict[Node, float],
) -> int:
    total_configs = (
        len(SWEEP_ALPHAS) * len(SWEEP_BETAS) * len(SWEEP_AGGS)
        * len(SWEEP_CLUSTER) * len(SWEEP_RESOLUTION) * len(SWEEP_DEFER_FETCH)
    )
    print(f"Sweep: α×β×agg×clust×res×defer = {len(SWEEP_ALPHAS)}×{len(SWEEP_BETAS)}"
          f"×{len(SWEEP_AGGS)}×{len(SWEEP_CLUSTER)}×{len(SWEEP_RESOLUTION)}"
          f"×{len(SWEEP_DEFER_FETCH)} "
          f"= {total_configs} configs × "
          f"{len(SWEEP_THREADS)} thread counts = "
          f"{total_configs * len(SWEEP_THREADS)} sims")
    print(f"Threads evaluated: {SWEEP_THREADS} (j <= 4 excluded)")
    print(f"Objective: mean saturation score across thread counts")
    print()

    # Baseline: what does speed score at each j?
    baselines: dict[int, dict] = {}
    for j in SWEEP_THREADS:
        prio = scheduler_speed(nodes, depends, revdeps, duration, {})
        baselines[j] = simulate(nodes, depends, revdeps, duration, prio, j,
                                width=width, io_fetch_pool=io_pool)

    print("Baselines (speed scheduler):")
    print(f"  {'j':>3}  {'wall':>10}  {'sat':>6}  {'full%':>6}")
    for j in SWEEP_THREADS:
        r = baselines[j]
        print(f"  {j:>3}  {fmt_hms(r['wall_clock_seconds']):>10}  "
              f"{r['saturation_score']*100:>5.1f}%  "
              f"{r['full_load_fraction']*100:>5.1f}%")
    print()

    # Sweep.
    results = []  # list of (mean_sat, mean_wall_delta, config, per_j_stats)
    total = 0
    for alpha, beta, agg, cluster, resolution, defer_fetch in itertools.product(
        SWEEP_ALPHAS, SWEEP_BETAS, SWEEP_AGGS, SWEEP_CLUSTER,
        SWEEP_RESOLUTION, SWEEP_DEFER_FETCH,
    ):
        # Skip the trivial all-zero config (equivalent to arbitrary ordering).
        if alpha == 0 and beta == 0:
            continue
        params = {
            'alpha': alpha, 'beta': beta, 'agg': agg, 'cluster': cluster,
            'resolution': resolution, 'defer_fetch': defer_fetch,
            'fetch_slack_pct': 1.0,
            'timings': timings,
        }
        prio = scheduler_tetris(nodes, depends, revdeps, duration, params)

        per_j = {}
        sats, wall_deltas = [], []
        for j in SWEEP_THREADS:
            r = simulate(nodes, depends, revdeps, duration, prio, j,
                         width=width, io_fetch_pool=io_pool)
            per_j[j] = r
            sats.append(r['saturation_score'])
            base_wall = baselines[j]['wall_clock_seconds']
            wall_deltas.append(r['wall_clock_seconds'] / base_wall - 1)
        mean_sat = sum(sats) / len(sats)
        mean_wall_delta = sum(wall_deltas) / len(wall_deltas)
        results.append((mean_sat, mean_wall_delta, params, per_j))
        total += len(SWEEP_THREADS)

    # Sort by mean sat desc.
    results.sort(key=lambda r: -r[0])

    # Print top 15 by sat.
    print(f"Top 15 configs by mean saturation score (of {len(results)} tested, "
          f"{total} sims):")
    print(f"  {'#':>2}  {'α':>4} {'β':>4} {'agg':>4} {'res':>4} {'def':>4}   "
          f"{'meanSat':>7} {'meanΔwall':>10}   "
          f"{'j=8 wall / sat':>18}  {'j=16 wall / sat':>18}  {'j=32 wall / sat':>18}")
    for i, (msat, mdel, params, per_j) in enumerate(results[:15], 1):
        cells = []
        for j in SWEEP_THREADS:
            r = per_j[j]
            cells.append(f"{fmt_hms(r['wall_clock_seconds']):>8} / {r['saturation_score']*100:>4.1f}%")
        print(f"  {i:>2}  {params['alpha']:>4.1f} {params['beta']:>4.1f} "
              f"{params['agg']:>4} {params['resolution']:>4} "
              f"{'Y' if params['defer_fetch'] else 'N':>4}   "
              f"{msat*100:>6.2f}% {mdel*100:>+9.2f}%   "
              + "  ".join(cells))

    # Also print bottom 5 (as sanity check).
    print()
    print(f"Bottom 5 configs by mean saturation score:")
    for i, (msat, mdel, params, per_j) in enumerate(results[-5:], 1):
        cells = []
        for j in SWEEP_THREADS:
            r = per_j[j]
            cells.append(f"{fmt_hms(r['wall_clock_seconds']):>8} / {r['saturation_score']*100:>4.1f}%")
        print(f"  {i:>2}  {params['alpha']:>4.1f} {params['beta']:>4.1f} "
              f"{params['agg']:>4} {params['resolution']:>4} "
              f"{'Y' if params['defer_fetch'] else 'N':>4}   "
              f"{msat*100:>6.2f}% {mdel*100:>+9.2f}%   "
              + "  ".join(cells))

    return 0


def _print_hist_single(r: dict, threads: int) -> None:
    tal = r['time_at_level']
    wall = r['wall_clock_seconds']
    print("Parallelism histogram (wall time at each concurrency level):")
    print(f"  {'workers':>7}  {'wall time':>11}  {'% of build':>10}  {'cum %':>8}   bar")
    cum = 0.0
    for k in range(threads, -1, -1):
        t = tal[k]
        if t <= 0:
            continue
        pct = (t / wall) * 100 if wall else 0
        cum += pct
        bar = '#' * int(round(pct / 2))  # each # = 2%
        print(f"  {k:>7}  {fmt_hms(t):>11}  {pct:>9.1f}%  {cum:>7.1f}%   {bar}")


def _print_hist_compare(rows: list[tuple[str, dict]], threads: int) -> None:
    print("Parallelism histogram (% of wall time at each level):")
    header = "  workers  " + "  ".join(f"{n:>10}" for n, _ in rows)
    print(header)
    walls = {n: r['wall_clock_seconds'] for n, r in rows}
    for k in range(threads, -1, -1):
        cells = []
        any_nonzero = False
        for n, r in rows:
            t = r['time_at_level'][k]
            w = walls[n]
            pct = (t / w) * 100 if w else 0
            if pct > 0:
                any_nonzero = True
            cells.append(f"{pct:>9.1f}%")
        if any_nonzero:
            print(f"  {k:>7}  " + "  ".join(cells))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
