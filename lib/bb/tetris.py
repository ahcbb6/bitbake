#
# Tetris runqueue scheduler for BitBake.
#
# Adaptive selection favors remaining critical-path seconds. When at most two
# workers' worth of tasks are eligible, a short final prerequisite can get an
# unlock-rate boost. At most one such dispatch precedes a path-first dispatch,
# and its estimated duration must fit within 5% of the leading remaining path.
# Eligible waiting time is added to path priority to prevent repeated bypasses.
# Completed work has zero remaining cost; setscene-covered work has zero own
# cost but still carries its downstream path until BitBake processes the skip.
#
# The static tie-break (or full policy with BB_TETRIS_DYNAMIC = "0") fuses:
#   * weight_time    -- log10-normalized elapsed time, computed on the fly
#                       from ${TMPDIR}/buildstats (written by buildstats.bbclass
#                       during previous builds in this TMPDIR)
#   * unblock_score  -- log10-normalized aggregate downstream weight_time,
#                       computed live from the task graph's revdeps
#
#   real_weight = alpha * weight_time + beta * unblock_score
#
# No sidecar file is required. On the first ever build in a fresh TMPDIR the
# buildstats dir is empty; the scheduler then degrades gracefully to using a
# neutral weight_time for every task, and the unblock_score alone still ranks
# tasks meaningfully (it comes from the live graph). Every subsequent build
# picks up the newest buildstats snapshot automatically.
#
# The default coefficients (alpha=0.5, beta=3.0) and aggregation ('max') were
# picked via grid search on core-image-minimal buildstats; see
# contrib/simulate_schedule.py --sweep --no-tetris-dynamic. The original intuition is:
#   * downstream criticality (unblock_score) matters much more than a task's
#     own duration
#   * 'max' aggregation ranks "unlocks the critical downstream child" above
#     "unlocks the fattest fan-out of tiny tasks"
#   * a small non-zero alpha tie-breaks heavy hubs ahead of light hubs
#
# Enable in local.conf with:
#
#   INHERIT           += "buildstats"           # so tetris has data next build
#   BB_SCHEDULERS     += "bb.tetris.RunQueueSchedulerTetris"
#   BB_SCHEDULER       = "tetris"
#
# Optional fetch deferral is disabled by default. Historical do_fetch times swing
# by 100x depending on cache hits vs. network, so their own duration is a poor
# priority signal. Instead, we compute a PERT critical-path slack per task
# from raw buildstats seconds, and any do_fetch whose slack exceeds a small
# percentage of the estimated makespan is *deferred*: sorted after every non-
# deferred task, so it only runs when a worker would otherwise be idle. Fetches
# on (or near) the critical path retain their normal priority.
#
# Optional tuning knobs (the static coefficients came from the original sweep):
#
#   BB_TETRIS_DYNAMIC          = "1"      # 0 restores static priority selection
#   BB_TETRIS_ALPHA            = "0.5"    # weight_time coefficient
#   BB_TETRIS_BETA             = "3.0"    # unblock_score coefficient
#   BB_TETRIS_AGG              = "max"    # 'max' or 'sum' downstream aggregation
#   BB_TETRIS_RESOLUTION       = "20"     # log-quantization buckets (sweep-optimum)
#   BB_TETRIS_BUILDSTATS       = "${TMPDIR}/buildstats"   # override source dir
#   BB_TETRIS_DEFER_FETCH      = "0"      # 1 enables experimental fetch deferral
#   BB_TETRIS_FETCH_SLACK_PCT  = "1.0"    # slack > pct*makespan => defer fetch
#
# SPDX-License-Identifier: MIT
#

import logging
import math
import os
import re
import time

from bb.runqueue import RunQueueScheduler, split_tid_mcfn

logger = logging.getLogger("BitBake.RunQueue.Tetris")

# Neutral fallback weight_time (roughly the midpoint of a resolution=20 scale).
# Used for tasks not present in the buildstats snapshot (new recipes, setscene-
# only tasks, or when no buildstats exist yet).
DEFAULT_WEIGHT_TIME = 4

# Static sweep-derived defaults; see module docstring above.
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 3.0
DEFAULT_AGG = "max"
# Log-quantization resolution for weight_time / unblock_score. Grid search on
# core-image-minimal buildstats shows 20 buckets is the sweet spot: coarse
# enough that the (alpha, beta) integer arithmetic still orders tasks the way
# the tuning intended, fine enough that most damaging ties are broken. Bumping
# to 100 actually regresses wall clock at j=8 by ~2 min because the finer
# ordering breaks a helpful accidental tie-break.
DEFAULT_RESOLUTION = 20

# Optional experimental deferral of non-critical fetches. Keep it disabled
# by default: delaying a fetch can postpone the work that would fill workers.
DEFAULT_DEFER_FETCH = False
DEFAULT_FETCH_SLACK_PCT = 1.0
DEFAULT_DYNAMIC = True

# Clamp for log10; 10ms is well below any scheduling-relevant granularity.
_MIN_ELAPSED = 0.01

# `Elapsed time: N.NN seconds` line written by buildstats.bbclass.
_ELAPSED_RE = re.compile(r'^Elapsed time:\s*([\d.]+)\s*seconds', re.MULTILINE)

# Recipe dir name looks like "pn-pv-rN"; strip trailing "-pv-rN" to recover pn.
_PN_RE = re.compile(r'^(?P<pn>.+)-[^-]+-r\d+$')

# A timestamped buildstats subdir looks like a run of >= 10 digits.
_TIMESTAMP_RE = re.compile(r'^\d{10,}$')


def estimate_task_durations(task_keys, timings):
    """Estimate seconds using observed times, task-kind medians, then global median."""
    if not timings:
        return {tid: 1.0 for tid in task_keys}
    by_task = {}
    for (_pn, task), value in timings.items():
        if value > 0:
            by_task.setdefault(task, []).append(value)
    medians = {
        task: sorted(values)[len(values) // 2]
        for task, values in by_task.items()
    }
    positive = sorted(value for value in timings.values() if value > 0)
    fallback = positive[len(positive) // 2] if positive else _MIN_ELAPSED
    return {
        tid: max(_MIN_ELAPSED, timings[key] if timings.get(key, 0) > 0
                 else medians.get(key[1], fallback))
        for tid, key in task_keys.items()
    }


class TetrisPriority:
    """Stateful priority policy shared by the runqueue and discrete-event simulator."""

    def __init__(self, depends, revdeps, duration, priority, deferred=()):
        self.depends = depends
        self.revdeps = revdeps
        self.duration = duration
        self.rank = {tid: rank for rank, tid in enumerate(priority)}
        self.deferred = set(deferred)
        self.remaining = {tid: len(deps) for tid, deps in depends.items()}
        indegree = self.remaining.copy()
        ready = [tid for tid, count in indegree.items() if not count]
        self.topo = []
        while ready:
            tid = ready.pop()
            self.topo.append(tid)
            for child in revdeps[tid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(self.topo) != len(depends):
            raise ValueError("Tetris: dependency graph contains a cycle")
        self.completed = set()
        self.covered = set()
        self.waiting = {}
        self.unlock_bypass = False
        self.last_reason = None
        self.path = {}
        self._update_paths()

    def _update_paths(self):
        for tid in reversed(self.topo):
            if tid in self.completed:
                self.path[tid] = 0.0
            else:
                own = 0.0 if tid in self.covered else self.duration[tid]
                self.path[tid] = own + max(
                    (self.path[child] for child in self.revdeps[tid]), default=0.0
                )

    def select(self, eligible, completed, covered, workers, now):
        if len(completed) != len(self.completed):
            for tid in completed.difference(self.completed):
                for child in self.revdeps[tid]:
                    self.remaining[child] -= 1
                # All predecessors of a completed task are also complete.
                # No pending ancestor path needs recalculation here.
                self.path[tid] = 0.0
            self.completed = set(completed)
        if covered != self.covered:
            self.covered = set(covered)
            self._update_paths()

        eligible = set(eligible)
        self.waiting = {tid: self.waiting.get(tid, now) for tid in eligible}
        skipped = eligible.intersection(covered)
        if skipped:
            self.last_reason = "covered"
            return min(skipped, key=self.rank.__getitem__)

        candidates = eligible.difference(self.deferred) or eligible
        critical = max(
            candidates,
            key=lambda tid: (self.path[tid] + now - self.waiting[tid],
                             self.path[tid], -self.rank[tid]),
        )
        selected = critical
        self.last_reason = "critical-path"
        if len(eligible) <= 2 * workers and not self.unlock_bypass:
            budget = self.path[critical] * 0.05
            best = None
            for tid in candidates:
                if self.duration[tid] > budget:
                    continue
                gain = sum(
                    self.duration[child] for child in self.revdeps[tid]
                    if self.remaining[child] == 1
                    and child not in self.completed and child not in covered
                )
                if gain <= 0:
                    continue
                score = (gain / self.duration[tid], self.path[tid], -self.rank[tid])
                if best is None or score > best[0]:
                    best = (score, tid)
            if best is not None:
                selected = best[1]
                if selected != critical:
                    self.last_reason = "unlock"
        self.unlock_bypass = selected != critical
        self.waiting.pop(selected)
        return selected


class RunQueueSchedulerTetris(RunQueueScheduler):
    """Adaptive path/unlock scheduling, with the original static score as tie-break."""

    name = "tetris"

    def __init__(self, runqueue, rqdata):
        RunQueueScheduler.__init__(self, runqueue, rqdata)

        alpha, beta, agg, resolution, defer_fetch, fetch_slack_pct, dynamic = (
            self._resolve_tunables()
        )

        # Load raw elapsed times from ${TMPDIR}/buildstats (or override) and
        # log-normalize them into weight_time in [1, resolution]. No JSON,
        # no manual regeneration step -- every rebuild reflects the latest
        # available timings.
        bs_path = self._resolve_buildstats_path()
        timings, source = self._load_timings_from_buildstats(bs_path)
        weight_time_lookup = self._log_normalize_timings(timings, resolution)

        # Assign weight_time per tid.
        wt_by_tid = {}
        wt_hits = 0
        task_of = {}
        for tid in self.rqdata.runtaskentries:
            pn, taskname = self._pn_task_for_tid(tid)
            wt = weight_time_lookup.get((pn, taskname), DEFAULT_WEIGHT_TIME)
            wt_by_tid[tid] = wt
            task_of[tid] = taskname
            if (pn, taskname) in weight_time_lookup:
                wt_hits += 1

        # Topological order (deps before revdeps). Reused for both downstream
        # aggregation and PERT slack.
        topo = self._topo_order()

        # Compute per-tid downstream aggregate of weight_time (sum or max)
        # via reverse-topological propagation. Avoids materializing the full
        # downstream set per tid, so it stays O(V + E) in both time and space.
        unblock_work = self._reduce_downstream(wt_by_tid, agg, topo=topo)

        # Log10-normalize unblock_work into unblock_score in [1, resolution].
        max_work = max(unblock_work.values()) if unblock_work else 0
        log_max = math.log10(1 + max_work) if max_work > 0 else 0.0
        span_out = resolution - 1

        def to_score(w):
            if w <= 0 or log_max <= 0:
                return 1
            score = 1 + math.log10(1 + w) / log_max * span_out
            return int(max(1, min(resolution, round(score))))

        real_weight = {
            tid: alpha * wt_by_tid[tid] + beta * to_score(unblock_work[tid])
            for tid in self.rqdata.runtaskentries
        }

        # Identify non-critical do_fetch tasks to defer. Requires raw timings
        # (log-normalized weights are too compressed for meaningful CP slack).
        # Falls back to no-defer if we have no timings at all.
        deferred = set()
        makespan_est = 0.0
        slack_threshold = 0.0
        if defer_fetch and timings:
            duration = self._estimate_durations(timings)
            slack, makespan_est = self._compute_slack(duration, topo)
            slack_threshold = makespan_est * fetch_slack_pct / 100.0
            for tid in self.rqdata.runtaskentries:
                if (task_of[tid] == 'do_fetch'
                        and slack.get(tid, 0.0) > slack_threshold):
                    deferred.add(tid)

        # Sort: non-deferred first, then by real_weight desc, then by raw
        # weight_time desc. Deferred fetches trail behind everything else so
        # they only run when the scheduler would otherwise be idle.
        self.prio_map = sorted(
            self.rqdata.runtaskentries.keys(),
            key=lambda t: (t not in deferred, real_weight[t], wt_by_tid[t]),
            reverse=True,
        )
        self.rev_prio_map = {tid: rank for rank, tid in enumerate(self.prio_map)}
        self.priority = None
        if dynamic:
            self.priority = TetrisPriority(
                {tid: entry.depends for tid, entry in self.rqdata.runtaskentries.items()},
                {tid: entry.revdeps for tid, entry in self.rqdata.runtaskentries.items()},
                self._estimate_durations(timings), self.prio_map, deferred,
            )

        # Summary logging so users can sanity-check what the scheduler picked up.
        n_fetch = sum(1 for t in task_of.values() if t == 'do_fetch')
        logger.debug(
            "Tetris: alpha=%s beta=%s agg=%s resolution=%s defer_fetch=%s "
            "fetch_slack_pct=%s; loaded %d timings from %s (%d/%d tids matched); "
            "makespan_est=%.1fs slack_threshold=%.1fs; "
            "deferred %d/%d do_fetch tasks",
            alpha, beta, agg, resolution, defer_fetch, fetch_slack_pct,
            len(weight_time_lookup), source,
            wt_hits, len(self.rqdata.runtaskentries),
            makespan_est, slack_threshold,
            len(deferred), n_fetch,
        )
        top = ", ".join(
            "%s(rw=%.1f)" % (t, real_weight[t])
            for t in self.prio_map[:5]
        )
        logger.debug("Tetris: top 5 by real_weight: %s", top)

    def select_buildable_task(self, buildable):
        if self.priority is None:
            return super().select_buildable_task(buildable)
        tid = self.priority.select(
            buildable, self.rq.runq_complete, self.rq.tasks_covered,
            self.rq.number_tasks, time.monotonic(),
        )
        logger.debug("Tetris: eligible=%d selected=%s reason=%s remaining_path=%.2fs",
                     len(buildable), tid, self.priority.last_reason, self.priority.path[tid])
        return tid

    # ------------------------------------------------------------------ helpers

    def _resolve_tunables(self):
        """Read BB_TETRIS_* from the datastore, with fallbacks."""
        cfg = self.rq.cfgData
        def _f(name, default):
            v = cfg.getVar(name)
            if v is None or v == "":
                return default
            try:
                return float(v)
            except ValueError:
                logger.warning("Tetris: %s=%r not a float; using default %s", name, v, default)
                return default
        def _i(name, default):
            v = cfg.getVar(name)
            if v is None or v == "":
                return default
            try:
                iv = int(v)
                if iv < 2:
                    raise ValueError
                return iv
            except ValueError:
                logger.warning("Tetris: %s=%r not an int >= 2; using default %s", name, v, default)
                return default
        def _b(name, default):
            v = cfg.getVar(name)
            if v is None or v == "":
                return default
            return str(v).strip().lower() in ("1", "true", "yes", "on")
        alpha = _f("BB_TETRIS_ALPHA", DEFAULT_ALPHA)
        beta = _f("BB_TETRIS_BETA", DEFAULT_BETA)
        agg = (cfg.getVar("BB_TETRIS_AGG") or DEFAULT_AGG).strip().lower()
        if agg not in ("sum", "max"):
            logger.warning("Tetris: BB_TETRIS_AGG=%r invalid; using default %s", agg, DEFAULT_AGG)
            agg = DEFAULT_AGG
        resolution = _i("BB_TETRIS_RESOLUTION", DEFAULT_RESOLUTION)
        defer_fetch = _b("BB_TETRIS_DEFER_FETCH", DEFAULT_DEFER_FETCH)
        fetch_slack_pct = _f("BB_TETRIS_FETCH_SLACK_PCT", DEFAULT_FETCH_SLACK_PCT)
        dynamic = _b("BB_TETRIS_DYNAMIC", DEFAULT_DYNAMIC)
        return alpha, beta, agg, resolution, defer_fetch, fetch_slack_pct, dynamic

    def _resolve_buildstats_path(self):
        """Return the buildstats base directory to scan, or None.

        Preference order:
          1. BB_TETRIS_BUILDSTATS (explicit override)
          2. BUILDSTATS_BASE (set by buildstats.bbclass)
          3. ${TMPDIR}/buildstats (default location)
        Returns None if nothing exists on disk.
        """
        cfg = self.rq.cfgData
        for var in ("BB_TETRIS_BUILDSTATS", "BUILDSTATS_BASE"):
            v = cfg.getVar(var)
            if v:
                return v
        tmpdir = cfg.getVar("TMPDIR")
        if tmpdir:
            return os.path.join(tmpdir, "buildstats")
        return None

    def _load_timings_from_buildstats(self, base):
        """Walk `base` and return ({(pn, taskname): elapsed_seconds}, source_str).

        Layout (from buildstats.bbclass):
            <base>/<timestamp>/<pn>-<pv>-<pr>/<taskname>
        Each task file has an 'Elapsed time: N.NN seconds' line.

        If `base` itself contains one or more timestamp subdirs, the newest
        one that actually holds recipe subdirs is used -- that's the snapshot
        of the previous build. On any failure (missing dir, empty dir, no
        elapsed lines) returns an empty dict; the scheduler then degrades to
        pure downstream-fanout scoring.
        """
        if not base or not os.path.isdir(base):
            logger.warning(
                "Tetris: buildstats dir %s not found; every task will use "
                "default weight_time=%d. Inherit 'buildstats' in local.conf "
                "so the next build has timing data.",
                base, DEFAULT_WEIGHT_TIME,
            )
            return {}, "<missing>"

        scan_dir = base
        try:
            entries = sorted(
                os.path.join(base, n) for n in os.listdir(base)
                if os.path.isdir(os.path.join(base, n))
            )
        except OSError as exc:
            logger.warning("Tetris: cannot list %s (%s); using defaults", base, exc)
            return {}, "<unreadable>"

        # If every subdir is a timestamp, descend into the newest one that
        # actually contains recipe dirs.
        if entries and all(_TIMESTAMP_RE.match(os.path.basename(p)) for p in entries):
            entries.sort(reverse=True)
            for cand in entries:
                try:
                    if any(
                        os.path.isdir(os.path.join(cand, n))
                        for n in os.listdir(cand)
                    ):
                        scan_dir = cand
                        break
                except OSError:
                    continue

        timings = {}
        try:
            recipe_dirs = sorted(os.listdir(scan_dir))
        except OSError as exc:
            logger.warning("Tetris: cannot list %s (%s); using defaults", scan_dir, exc)
            return {}, "<unreadable>"

        for recipe_name in recipe_dirs:
            recipe_path = os.path.join(scan_dir, recipe_name)
            if not os.path.isdir(recipe_path):
                continue
            m = _PN_RE.match(recipe_name)
            pn = m.group('pn') if m else recipe_name
            try:
                task_names = os.listdir(recipe_path)
            except OSError:
                continue
            for task_name in task_names:
                if not task_name.startswith('do_'):
                    continue
                task_path = os.path.join(recipe_path, task_name)
                try:
                    with open(task_path, errors='ignore') as fh:
                        text = fh.read()
                except OSError:
                    continue
                em = _ELAPSED_RE.search(text)
                if not em:
                    continue
                try:
                    elapsed = float(em.group(1))
                except ValueError:
                    continue
                key = (pn, task_name)
                # Keep max if the same task appears twice (multiconfig).
                if elapsed > timings.get(key, -1.0):
                    timings[key] = elapsed

        if not timings:
            logger.warning(
                "Tetris: no Elapsed timings found under %s; using defaults",
                scan_dir,
            )
        return timings, scan_dir

    def _log_normalize_timings(self, timings, resolution):
        """{(pn, task): elapsed_seconds} -> {(pn, task): int in [1, resolution]}.

        Log10 spread over the observed min..max range. Task times span ~5
        orders of magnitude in a typical image build, so a linear scale would
        collapse 99% of tasks to weight 1.
        """
        positive = [v for v in timings.values() if v > 0]
        if not positive:
            return {}
        lg_min = math.log10(max(min(positive), _MIN_ELAPSED))
        lg_max = math.log10(max(positive))
        span = lg_max - lg_min
        span_out = resolution - 1

        out = {}
        for key, v in timings.items():
            if v <= 0 or span <= 0:
                out[key] = 1
                continue
            lg = math.log10(max(v, _MIN_ELAPSED))
            w = 1 + (lg - lg_min) * span_out / span
            out[key] = int(max(1, min(resolution, round(w))))
        return out

    def _pn_task_for_tid(self, tid):
        try:
            (mc, _fn, taskname, taskfn) = split_tid_mcfn(tid)
            pn = self.rqdata.dataCaches[mc].pkg_fn[taskfn]
        except (KeyError, ValueError):
            return None, None
        return pn, taskname

    def _topo_order(self):
        """Return tids in topological order (all depends of a tid precede it).

        Kahn's algorithm on the live dep-DAG, O(V+E). If the graph has a
        cycle (shouldn't happen -- RunQueueData.prepare validates), the
        returned list will be shorter than runtaskentries; callers should
        check length before relying on it.
        """
        entries = self.rqdata.runtaskentries
        in_deg = {tid: len(entries[tid].depends) for tid in entries}
        ready = [tid for tid, d in in_deg.items() if d == 0]
        topo = []
        while ready:
            n = ready.pop()
            topo.append(n)
            for r in entries[n].revdeps:
                in_deg[r] -= 1
                if in_deg[r] == 0:
                    ready.append(r)
        return topo

    def _reduce_downstream(self, wt_by_tid, agg, topo=None):
        """Return {tid: aggregate_of_downstream_weight_times}.

        `agg` is 'sum' (fat fan-out wins) or 'max' (heaviest downstream chain
        wins). Uses reverse-topological propagation on the dep-DAG so we
        never materialize the full downstream set per tid -- O(V + E) total.
        """
        entries = self.rqdata.runtaskentries
        if topo is None:
            topo = self._topo_order()

        if len(topo) != len(entries):
            # A cycle would have been caught by RunQueueData.prepare() already,
            # but guard anyway so we never wedge the build.
            logger.warning(
                "Tetris: dep graph not fully sorted (%d/%d); results may be partial",
                len(topo),
                len(entries),
            )

        # Reverse-propagate: leaves first, aggregate downstream child + own.
        down = {tid: 0 for tid in entries}
        if agg == "sum":
            for n in reversed(topo):
                total = 0
                for r in entries[n].revdeps:
                    total += wt_by_tid.get(r, DEFAULT_WEIGHT_TIME) + down[r]
                down[n] = total
        else:  # 'max'
            for n in reversed(topo):
                best = 0
                for r in entries[n].revdeps:
                    cand = wt_by_tid.get(r, DEFAULT_WEIGHT_TIME) + down[r]
                    if cand > best:
                        best = cand
                down[n] = best
        return down

    def _estimate_durations(self, timings):
        """Return {tid: seconds} for every tid.

        Preference: observed (pn, task) time from buildstats; failing that,
        the median observed time for that task-name (e.g. do_configure across
        recipes); failing that, the global median. Missing/zero timings clamp
        to _MIN_ELAPSED so slack math stays well-defined.
        """
        return estimate_task_durations(
            {tid: self._pn_task_for_tid(tid) for tid in self.rqdata.runtaskentries},
            timings,
        )

    def _compute_slack(self, duration, topo):
        """PERT slack per tid, in seconds. Returns (slack, makespan_estimate).

        Forward pass computes earliest-start; backward pass computes
        latest-finish given the sink deadline == estimated makespan; slack
        is LFT - duration - EST. A slack of 0 means the task lies on the
        critical path.

        If the topo order is incomplete (cycle), returns zero slack for all
        tids so the deferral logic no-ops.
        """
        entries = self.rqdata.runtaskentries
        if len(topo) != len(entries) or not entries:
            return {tid: 0.0 for tid in entries}, 0.0

        # Forward pass: EST[tid] = max(EFT[dep]) over predecessors.
        est = {tid: 0.0 for tid in entries}
        for tid in topo:
            eft = est[tid] + duration[tid]
            for r in entries[tid].revdeps:
                if eft > est[r]:
                    est[r] = eft

        makespan = 0.0
        for tid in entries:
            eft = est[tid] + duration[tid]
            if eft > makespan:
                makespan = eft

        # Backward pass: LFT[sink]=makespan; LFT[tid]=min(LFT[rd]-duration[rd]).
        lft = {tid: makespan for tid in entries}
        for tid in reversed(topo):
            rds = entries[tid].revdeps
            if rds:
                best = float('inf')
                for r in rds:
                    cand = lft[r] - duration[r]
                    if cand < best:
                        best = cand
                lft[tid] = best

        slack = {}
        for tid in entries:
            s = lft[tid] - duration[tid] - est[tid]
            slack[tid] = s if s > 0.0 else 0.0
        return slack, makespan
