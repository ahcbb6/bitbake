#
# SPDX-License-Identifier: GPL-2.0-only
#

import importlib.util
import io
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bb.runqueue import RunQueueScheduler
from bb.tetris import RunQueueSchedulerTetris, TetrisPriority, estimate_task_durations


spec = importlib.util.spec_from_file_location(
    "simulate_schedule",
    Path(__file__).resolve().parents[3] / "contrib" / "simulate_schedule.py",
)
simulator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simulator)


class TetrisHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="tetris-history-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.stats = self.root / "tmp" / "buildstats"
        self.config = {
            "TOPDIR": str(self.root), "MACHINE": "qemux86-64",
            "BB_NUMBER_THREADS": "8", "PARALLEL_MAKE": "-j 8",
        }

    def load(self, base=None, version=None):
        cfg = SimpleNamespace(getVar=self.config.get)
        scheduler = RunQueueSchedulerTetris.__new__(RunQueueSchedulerTetris)
        scheduler.rq = SimpleNamespace(
            cfgData=cfg,
            cooker=SimpleNamespace(databuilder=SimpleNamespace(mcdata={"": cfg})),
        )
        scheduler.rqdata = SimpleNamespace(runtaskentries={}, dataCaches={})
        if version:
            scheduler.rqdata.runtaskentries = {"/llvm.bb:do_compile": None}
            scheduler.rqdata.dataCaches = {"": SimpleNamespace(
                pkg_fn={"/llvm.bb": "llvm-native"},
                pkg_pepvpr={"/llvm.bb": ("0", version, "r0")},
            )}
        return scheduler._load_timings_from_buildstats(str(base or self.stats))[0]

    def record(self, snapshot, recipe, task, elapsed, ended, status="PASSED"):
        path = self.stats / snapshot / recipe / task
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(
                f"Event: TaskStarted\nStarted: {ended - elapsed}\n"
                f"Elapsed time: {elapsed} seconds\nStatus: {status}\nEnded: {ended}\n"
            )
        return path

    def test_compile_restore_rebuild_retains_estimate_outside_tmp(self):
        self.record("20260101000000", "llvm-native-22.1.3-r0",
                    "do_compile", 7500, 10000)
        self.assertEqual(self.load()[("llvm-native", "do_compile")], 7500)
        shutil.rmtree(self.root / "tmp")
        self.record("20260102000000", "llvm-native-22.1.3-r0",
                    "do_populate_sysroot_setscene", 0.5, 20000)
        self.record("20260102000000", "small-1.0-r0", "do_compile", 13, 20001)
        timings = self.load()
        self.assertEqual(timings[("llvm-native", "do_populate_sysroot_setscene")], 0.5)
        self.assertEqual(
            estimate_task_durations({"llvm": ("llvm-native", "do_compile")}, timings),
            {"llvm": 7500},
        )
        shutil.rmtree(self.root / "tmp")
        with self.assertLogs("BitBake.RunQueue.Tetris", level="WARNING"):
            self.assertEqual(self.load()[("llvm-native", "do_compile")], 7500)

    def test_successful_samples_across_snapshots_retries_and_versions(self):
        self.record("20260101000000", "llvm-native-22.1.3-r0",
                    "do_compile", 7500, 10000)
        self.record("20260102000000", "llvm-native-22.1.8-r0",
                    "do_compile", 6000, 20000)
        path = self.record("20260102000000", "llvm-native-22.1.8-r0",
                           "do_compile", 50, 21000, status="FAILED")
        with path.open("a") as stream:
            stream.write("Event: TaskStarted\nStarted: 22000\n")
        (self.stats / "20260103000000").mkdir()
        self.assertEqual(self.load()[("llvm-native", "do_compile")], 6000)
        # Reimporting an older archive must not replace a newer persisted value.
        self.assertEqual(
            self.load(self.stats / "20260101000000")[("llvm-native", "do_compile")], 6000
        )
        self.record("20260104000000", "llvm-native-22.1.8-r0",
                    "do_compile", 5500, 30000)
        self.assertEqual(self.load()[("llvm-native", "do_compile")], 5750)
        self.assertEqual(self.load(version="22.1.3")[("llvm-native", "do_compile")], 7500)
        self.assertEqual(self.load(version="22.1.8")[("llvm-native", "do_compile")], 5750)
        self.assertEqual(self.load(version="23.0.0")[("llvm-native", "do_compile")], 5750)

    def test_rolling_median_evicts_old_samples_and_deduplicates_imports(self):
        key = ("llvm-native", "do_compile")
        for i, elapsed in enumerate((10, 20, 30, 40, 50, 1000, 60)):
            self.record(f"2026010{i + 1}000000", "llvm-native-22.1.8-r0",
                        "do_compile", elapsed, 10000 + i * 2000)
        self.assertEqual(self.load()[key], 50)
        self.assertEqual(self.load()[key], 50)
        # A copied archive with the same completed execution is not a new sample.
        self.record("20260108000000", "llvm-native-22.1.8-r0",
                    "do_compile", 60, 22000)
        self.assertEqual(self.load()[key], 50)
        with sqlite3.connect(self.root / "cache" / "tetris.sqlite3") as db:
            rows = db.execute("SELECT ended, elapsed FROM timings_v2 ORDER BY ended").fetchall()
        self.assertEqual(rows, [(14000, 30), (16000, 40), (18000, 50),
                                (20000, 1000), (22000, 60)])
        self.config["BB_TETRIS_HISTORY"] = ""
        self.assertEqual(self.load()[key], 50)
        del self.config["BB_TETRIS_HISTORY"]
        shutil.rmtree(self.root / "tmp")
        with self.assertLogs("BitBake.RunQueue.Tetris", level="WARNING"):
            self.assertEqual(self.load()[key], 50)

    def test_legacy_history_migrates_as_version_unknown_fallback(self):
        key = ("llvm-native", "do_compile")
        self.record("20260101000000", "llvm-native-22.1.3-r0",
                    "do_compile", 7500, 10000)
        self.load()
        with sqlite3.connect(self.root / "cache" / "tetris.sqlite3") as db:
            db.execute("CREATE TABLE timings_v1 AS "
                       "SELECT scope, pn, task, ended, elapsed FROM timings_v2")
            db.execute("DROP TABLE timings_v2")
        shutil.rmtree(self.root / "tmp")
        with self.assertLogs("BitBake.RunQueue.Tetris", level="WARNING"):
            self.assertEqual(self.load(version="22.1.8")[key], 7500)
        self.record("20260102000000", "llvm-native-22.1.8-r0",
                    "do_compile", 6000, 20000)
        self.assertEqual(self.load(version="22.1.8")[key], 6000)

    def test_configuration_scopes_do_not_share_retained_timings(self):
        self.record("20260101000000", "llvm-native-22.1.3-r0",
                    "do_compile", 7500, 10000)
        self.load()
        shutil.rmtree(self.root / "tmp")
        for variable, value in (("MACHINE", "qemuarm64"), ("DISTRO", "other"),
                                ("BB_NUMBER_THREADS", "16"), ("PARALLEL_MAKE", "-j 16")):
            with self.subTest(variable=variable), \
                 patch.dict(self.config, {variable: value}), \
                 self.assertLogs("BitBake.RunQueue.Tetris", level="WARNING"):
                self.assertEqual(self.load(), {})
        with self.assertLogs("BitBake.RunQueue.Tetris", level="WARNING"):
            self.assertEqual(self.load()[("llvm-native", "do_compile")], 7500)

    def test_disabled_or_broken_persistence_still_merges_snapshots(self):
        self.record("20260101000000", "llvm-native-22.1.3-r0",
                    "do_compile", 7500, 10000)
        self.record("20260102000000", "small-1.0-r0", "do_compile", 13, 20000)
        self.config["BB_TETRIS_HISTORY"] = ""
        self.assertEqual(self.load()[("llvm-native", "do_compile")], 7500)
        self.assertFalse((self.root / "cache").exists())
        invalid = self.root / "invalid.sqlite3"
        invalid.write_text("not a database")
        self.config["BB_TETRIS_HISTORY"] = str(invalid)
        with self.assertLogs("BitBake.RunQueue.Tetris", level="WARNING") as logs:
            self.assertEqual(self.load()[("llvm-native", "do_compile")], 7500)
        self.assertTrue(any("cannot update timing history" in line for line in logs.output))


class TetrisSimulationTests(unittest.TestCase):
    def test_priority_matches_runtime(self):
        nodes = [
            ("z", "do_fetch"), ("a", "do_fetch"),
            ("z", "do_compile"), ("a", "do_compile"),
            ("image", "do_rootfs"), ("other", "do_build"),
            ("new", "do_compile"),
        ]
        depends = {n: set() for n in nodes}
        revdeps = {n: set() for n in nodes}
        for parent, child in [(0, 2), (1, 3), (2, 4), (3, 4), (0, 6)]:
            depends[nodes[child]].add(nodes[parent])
            revdeps[nodes[parent]].add(nodes[child])
        entries = {
            n: SimpleNamespace(depends=depends[n], revdeps=revdeps[n])
            for n in nodes
        }
        full = dict(zip(nodes, [0.02, 1, 20, 1000, 100, 1, 1]))
        partial = {n: full[n] for n in nodes[:4]}
        partial[("outside-graph", "do_compile")] = 50000
        snapshots = [full, partial, {}, dict.fromkeys(nodes, 0.0)]

        def init_base(self, rq, rqdata):
            self.rq = rq
            self.rqdata = rqdata

        for timings in snapshots:
            for agg in ("max", "sum"):
                for defer in (None, False, True):
                    with self.subTest(timings=timings, agg=agg, defer=defer):
                        config = {"BB_TETRIS_AGG": agg}
                        params = {"agg": agg, "timings": timings}
                        if defer is not None:
                            config["BB_TETRIS_DEFER_FETCH"] = str(int(defer))
                            params["defer_fetch"] = defer
                        rq = SimpleNamespace(cfgData=SimpleNamespace(getVar=config.get))
                        rqdata = SimpleNamespace(runtaskentries=entries)
                        with patch.object(RunQueueScheduler, "__init__", init_base), \
                             patch.object(RunQueueSchedulerTetris, "_pn_task_for_tid",
                                          lambda self, tid: tid), \
                             patch.object(RunQueueSchedulerTetris, "_load_timings_from_buildstats",
                                          return_value=(timings, "<test>")):
                            runtime = RunQueueSchedulerTetris(rq, rqdata)
                        duration, _, _, _ = simulator.build_duration_map(
                            {n: (v, v) for n, v in timings.items()}, set(nodes)
                        )
                        actual = simulator.scheduler_tetris(
                            nodes, depends, revdeps, duration,
                            params,
                        )
                        self.assertEqual(actual, runtime.prio_map)

    def test_graph_includes_isolated_tasks(self):
        dot = (
            'digraph depends {\n'
            '"isolated.do_build" [label="isolated"]\n'
            '"a.do_compile" -> "a.do_fetch"\n'
            '}\n'
        )
        with patch.object(Path, "open", return_value=io.StringIO(dot)):
            depends, revdeps = simulator.parse_task_graph(Path("test.dot"))
        self.assertEqual(set(depends), {
            ("isolated", "do_build"), ("a", "do_compile"), ("a", "do_fetch"),
        })
        self.assertEqual(depends[("isolated", "do_build")], set())
        self.assertEqual(revdeps[("isolated", "do_build")], set())
        self.assertEqual(depends[("a", "do_compile")], {("a", "do_fetch")})

    def test_duration_map_excludes_unrelated_work(self):
        a, b = ("a", "do_compile"), ("b", "do_compile")
        elapsed, cpu, default, filled = simulator.build_duration_map(
            {a: (10, 80), ("outside", "do_compile"): (10000, 20000)}, {a, b}
        )
        self.assertEqual(elapsed, {a: 10, b: 10})
        self.assertEqual(cpu, {a: 80, b: 10})
        self.assertEqual((default, filled), (10, 1))
        self.assertEqual(simulator.makespan_lower_bound(elapsed, 10, 1)[0], 20)

    def test_cpu_demand_does_not_change_worker_metrics(self):
        a, f = ("a", "do_compile"), ("f", "do_fetch")
        nodes = [a, f]
        graph = {n: set() for n in nodes}
        elapsed = {n: 10 for n in nodes}
        widths = simulator.compute_task_widths(elapsed, {a: 80, f: 10})
        baseline = simulator.simulate(nodes, graph, graph, elapsed, nodes, 1)
        measured = simulator.simulate(nodes, graph, graph, elapsed, nodes, 1, widths)
        for key in ("wall_clock_seconds", "avg_parallelism", "utilization",
                    "saturation_score", "full_load_fraction", "time_at_level"):
            self.assertEqual(measured[key], baseline[key])
        self.assertEqual(measured["wall_clock_seconds"], 20)
        self.assertEqual(measured["avg_cpu_demand"], 4.5)
        self.assertEqual(measured["utilization"], 1)

    def test_separate_fetch_pool_has_no_head_of_line_blocking(self):
        a, b, f = ("a", "do_compile"), ("b", "do_compile"), ("f", "do_fetch")
        nodes = [a, b, f]
        graph = {n: set() for n in nodes}
        duration = {a: 10, b: 10, f: 100}
        for priority in ([a, b, f], [a, f, b]):
            with self.subTest(priority=priority):
                result = simulator.simulate(
                    nodes, graph, graph, duration, priority, 1, io_fetch_pool=1
                )
                self.assertEqual(result["wall_clock_seconds"], 100)
                self.assertEqual(result["utilization"], 0.2)
                self.assertEqual(result["avg_parallelism"], 0.2)
                bound, _ = simulator.makespan_lower_bound(duration, 100, 1, 1)
                self.assertEqual(bound, result["wall_clock_seconds"])
                self.assertEqual(sum(result["time_at_level"]), 100)

    def test_invalid_capacity_and_cycles_fail_explicitly(self):
        node = ("a", "do_compile")
        for threads, fetch_slots in [(0, None), (1, 0), (-1, 2)]:
            with self.subTest(threads=threads, fetch_slots=fetch_slots):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    simulator.simulate(
                        [node], {node: set()}, {node: set()}, {node: 1},
                        [node], threads, io_fetch_pool=fetch_slots,
                    )
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            simulator.simulate(
                [node], {node: {node}}, {node: {node}}, {node: 1}, [node], 1
            )


class AdaptiveTetrisTests(unittest.TestCase):
    def policy(self, duration, edges=(), deferred=()):
        depends = {tid: set() for tid in duration}
        revdeps = {tid: set() for tid in duration}
        for parent, child in edges:
            depends[child].add(parent)
            revdeps[parent].add(child)
        return TetrisPriority(depends, revdeps, duration, list(duration), deferred)

    def test_short_unlock_is_bounded_and_only_for_shallow_queue(self):
        duration = {"critical": 100, "gate": 1, "gate2": 1,
                    "child": 20, "child2": 20, "filler": 10}
        edges = [("gate", "child"), ("gate2", "child2")]
        policy = self.policy(duration, edges)
        self.assertEqual(policy.select({"critical", "gate"}, set(), set(), 1, 0), "gate")
        self.assertEqual(policy.last_reason, "unlock")
        # A second attractive gate cannot repeatedly postpone the long path.
        self.assertEqual(
            policy.select({"critical", "gate2"}, {"gate"}, set(), 1, 1), "critical"
        )
        policy = self.policy(duration, edges)
        self.assertEqual(
            policy.select({"critical", "gate", "filler"}, set(), set(), 1, 0),
            "critical",
        )
        duration["gate"] = 6
        policy = self.policy(duration, edges)
        self.assertEqual(policy.select({"critical", "gate"}, set(), set(), 1, 0), "critical")

    def test_unlock_gain_updates_when_last_other_prerequisite_finishes(self):
        policy = self.policy(
            {"critical": 100, "critical2": 100, "gate": 1, "other": 1, "child": 20},
            [("gate", "child"), ("other", "child")],
        )
        self.assertEqual(policy.select({"critical", "gate"}, set(), set(), 1, 0), "critical")
        self.assertEqual(
            policy.select({"critical2", "gate"}, {"other"}, set(), 1, 1), "gate"
        )
        self.assertEqual(policy.last_reason, "unlock")
        self.assertEqual(policy.remaining["child"], 1)

    def test_covered_work_has_zero_cost_and_coverage_can_be_revoked(self):
        policy = self.policy(
            {"root": 1, "cached": 1000, "tail": 50, "other": 60},
            [("root", "cached"), ("cached", "tail")],
        )
        self.assertEqual(
            policy.select({"root", "other"}, set(), {"cached"}, 1, 0), "other"
        )
        self.assertEqual(policy.path["root"], 51)
        policy.select({"root"}, set(), set(), 1, 1)
        self.assertEqual(policy.path["root"], 1051)
        self.assertEqual(
            policy.select({"cached", "other"}, {"root"}, {"cached"}, 1, 2), "cached"
        )
        policy.select({"tail", "other"}, {"root", "cached"}, {"cached"}, 1, 3)
        self.assertEqual(policy.path["cached"], 0)
        self.assertEqual(policy.remaining["tail"], 0)

    def test_aging_and_explicit_fetch_deferral(self):
        policy = self.policy({"first": 100, "later": 100, "waiting": 1})
        self.assertEqual(
            policy.select({"first", "waiting"}, set(), set(), 1, 0), "first"
        )
        self.assertEqual(
            policy.select({"later", "waiting"}, {"first"}, set(), 1, 150), "waiting"
        )
        policy = self.policy({"fetch": 1000, "compile": 10}, deferred={"fetch"})
        self.assertEqual(policy.select({"fetch", "compile"}, set(), set(), 1, 0), "compile")
        self.assertEqual(policy.select({"fetch"}, {"compile"}, set(), 1, 10), "fetch")

    def test_adaptive_simulation_reduces_dependency_tail(self):
        a, b, gate, child = [(n, "do_compile") for n in ("a", "b", "gate", "child")]
        duration = {a: 60, b: 60, gate: 1, child: 100}
        policy = self.policy(duration, [(gate, child)])
        nodes = list(duration)
        static = simulator.simulate(
            nodes, policy.depends, policy.revdeps, duration, nodes, 2
        )
        adaptive = simulator.simulate(
            nodes, policy.depends, policy.revdeps, duration, nodes, 2, policy=policy
        )
        self.assertEqual(static["wall_clock_seconds"], 161)
        self.assertEqual(adaptive["wall_clock_seconds"], 120)
        self.assertEqual(adaptive["nodes_completed"], len(nodes))
        self.assertGreater(adaptive["utilization"], static["utilization"])

    def test_runtime_keeps_all_eligibility_checks_and_static_opt_out(self):
        running = "/running.bb:do_compile"
        capped = "/capped.bb:do_compile"
        stamped = "/stamped.bb:do_install"
        held = "/held.bb:do_install"
        unknown = "/unknown.bb:do_install"
        allowed = "/allowed.bb:do_install"
        nodes = [capped, stamped, held, unknown, allowed]
        scheduler = RunQueueSchedulerTetris.__new__(RunQueueSchedulerTetris)
        scheduler.rq = SimpleNamespace(
            runq_running={running}, runq_complete=set(), holdoff_tasks={held},
            tasks_covered=set(), tasks_notcovered=set(nodes) - {unknown},
            stats=SimpleNamespace(active=1), number_tasks=2,
            cfgData=SimpleNamespace(getVarFlag=lambda task, flag: "1" if task == "do_compile" else None),
            build_stamps={running: "busy-stamp"}, can_start_task=lambda: True,
        )
        scheduler.buildable = set(nodes)
        scheduler.stamps = dict(zip(nodes, nodes))
        scheduler.stamps[stamped] = "busy-stamp"
        scheduler.skip_maxthread = {}
        scheduler.prio_map = nodes
        scheduler.rev_prio_map = {tid: i for i, tid in enumerate(nodes)}
        scheduler.priority = self.policy(dict.fromkeys(nodes, 1))
        with patch.object(scheduler, "exceeds_max_pressure", return_value=True):
            self.assertIsNone(scheduler.next())
        with patch.object(scheduler, "exceeds_max_pressure", return_value=False):
            self.assertEqual(scheduler.next(), allowed)
            scheduler.priority = None
            self.assertEqual(scheduler.next(), allowed)
            scheduler.rq.can_start_task = lambda: False
            self.assertIsNone(scheduler.next())
