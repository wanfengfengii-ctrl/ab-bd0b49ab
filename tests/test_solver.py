"""Unit tests for the exact delay-plan solver."""

from __future__ import annotations

import random
import sys

import pytest

from app.solver import (
    Anchor,
    Instance,
    _ramps_of,
    compile_plan,
)

sys.setrecursionlimit(100_000)


def make_instance(targets, min_delay, max_delay, max_step, max_ramps, anchors):
    return Instance(
        targets=tuple(targets),
        min_delay=min_delay,
        max_delay=max_delay,
        max_step=max_step,
        max_ramps=max_ramps,
        anchors=tuple(Anchor(element=e, delay=d) for e, d in anchors),
    )


def brute_force_best(inst: Instance):
    """Exhaustive reference: best (maxErr, totalErr, ramps, sequence)."""
    n = inst.n
    anchor_map = {a.element: a.delay for a in inst.anchors}
    best = None

    def rec(i, prev_v, prev_k, ramps, seq, max_e, tot_e):
        nonlocal best
        if best is not None:
            if max_e > best[0]:
                return
            if max_e == best[0] and tot_e > best[1]:
                return
            if max_e == best[0] and tot_e == best[1]:
                if ramps > best[2]:
                    return
                if ramps == best[2] and tuple(seq) > best[3][: len(seq)]:
                    return
        if i == n:
            key = (max_e, tot_e, ramps, tuple(seq))
            if best is None or key < best:
                best = key
            return
        pinned = anchor_map.get(i)
        if pinned is not None:
            candidates = (pinned,)
        else:
            # try values closest to the target first so the incumbent
            # becomes strong early and prunes hard
            candidates = sorted(
                range(inst.min_delay, inst.max_delay + 1),
                key=lambda v: (abs(v - inst.targets[i]), v),
            )
        for v in candidates:
            if i == 0:
                k, r = None, 0
            else:
                k = v - prev_v
                if abs(k) > inst.max_step:
                    continue
                if i == 1:
                    r = 1
                elif k == prev_k:
                    r = ramps
                else:
                    r = ramps + 1
                if r > inst.max_ramps:
                    continue
            e = abs(v - inst.targets[i])
            rec(i + 1, v, k, r, seq + [v], max(max_e, e), tot_e + e)

    rec(0, None, None, 0, [], 0, 0)
    return best


def result_key(result):
    return (
        result["maxAbsError"],
        result["totalAbsError"],
        result["rampCount"],
        tuple(result["delays"]),
    )


def check_feasible_shape(inst: Instance, result: dict):
    """Independently re-verify every contract of a feasible response."""
    n = inst.n
    delays = result["delays"]
    assert len(delays) == n
    assert len(result["errors"]) == n
    assert result["errors"] == [d - t for d, t in zip(delays, inst.targets)]
    anchor_map = {a.element: a.delay for a in inst.anchors}
    for i, v in enumerate(delays):
        assert inst.min_delay <= v <= inst.max_delay
        if i in anchor_map:
            assert v == anchor_map[i]
        if i:
            assert abs(v - delays[i - 1]) <= inst.max_step
    diffs = [delays[i + 1] - delays[i] for i in range(n - 1)]
    runs = 1 + sum(1 for i in range(1, len(diffs)) if diffs[i] != diffs[i - 1])
    assert runs == result["rampCount"] <= inst.max_ramps
    abs_errors = [abs(e) for e in result["errors"]]
    assert result["maxAbsError"] == max(abs_errors)
    assert result["totalAbsError"] == sum(abs_errors)
    # ramp boundaries must be contiguous, cover everything and match delays
    ramps = result["ramps"]
    assert len(ramps) == runs
    assert ramps[0]["startElement"] == 0
    assert ramps[-1]["endElement"] == n - 1
    for left, right in zip(ramps, ramps[1:]):
        assert left["endElement"] == right["startElement"]
    for ramp in ramps:
        s, e, k = ramp["startElement"], ramp["endElement"], ramp["slope"]
        assert e > s
        assert ramp["startValue"] == delays[s]
        assert ramp["endValue"] == delays[e]
        assert delays[e] - delays[s] == k * (e - s)
        for i in range(s, e):
            assert delays[i + 1] - delays[i] == k


# ---------------------------------------------------------------------------
# hand-crafted cases
# ---------------------------------------------------------------------------

def test_exact_arithmetic_progression_single_ramp():
    inst = make_instance(
        targets=[2 * i for i in range(12)],
        min_delay=0, max_delay=30, max_step=5, max_ramps=1,
        anchors=[(0, 0), (11, 22)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is True
    assert res["delays"] == [2 * i for i in range(12)]
    assert res["errors"] == [0] * 12
    assert res["maxAbsError"] == 0
    assert res["totalAbsError"] == 0
    assert res["rampCount"] == 1
    assert res["ramps"] == [{
        "startElement": 0, "endElement": 11, "slope": 2,
        "startValue": 0, "endValue": 22,
    }]
    check_feasible_shape(inst, res)


def test_anchors_are_hit_exactly():
    inst = make_instance(
        targets=[5] * 16,
        min_delay=0, max_delay=20, max_step=3, max_ramps=4,
        anchors=[(0, 2), (7, 11), (15, 4)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is True
    assert res["delays"][0] == 2
    assert res["delays"][7] == 11
    assert res["delays"][15] == 4
    check_feasible_shape(inst, res)


def test_max_error_is_minimised_before_total_error():
    # step target 0 -> 10 with maxStep 1: the error must spread, and the
    # minimal possible max error is 5 (a plateau of 5 in the middle).
    inst = make_instance(
        targets=[0] * 6 + [10] * 6,
        min_delay=0, max_delay=10, max_step=1, max_ramps=4,
        anchors=[(0, 0), (11, 10)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is True
    assert res["maxAbsError"] == 5
    check_feasible_shape(inst, res)


def test_lexicographic_tie_break():
    # Two co-optimal plans with (maxErr, totalErr, ramps) = (1, 11, 2):
    #   A = [0,1,2,2,...,2]  and  B = [0,...,0,1,2]
    # B is lexicographically smaller and must be selected.
    inst = make_instance(
        targets=[1] * 12,
        min_delay=0, max_delay=3, max_step=1, max_ramps=2,
        anchors=[(0, 0), (11, 2)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is True
    assert res["maxAbsError"] == 1
    assert res["totalAbsError"] == 11
    assert res["rampCount"] == 2
    assert res["delays"] == [0] * 10 + [1, 2]
    check_feasible_shape(inst, res)


def test_ramp_count_is_minimised_after_errors():
    # reachable with zero error using exactly 3 ramps; a 4th ramp must not
    # appear even though it is allowed
    inst = make_instance(
        targets=[0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        min_delay=0, max_delay=10, max_step=2, max_ramps=6,
        anchors=[(0, 0), (2, 2), (4, 0), (6, 2), (11, 7)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is True
    assert res["maxAbsError"] == 0
    assert res["rampCount"] == 3
    check_feasible_shape(inst, res)


def test_anchor_step_conflict_reports_interval_and_no_delays():
    inst = make_instance(
        targets=[0] * 12,
        min_delay=0, max_delay=40, max_step=2, max_ramps=4,
        anchors=[(1, 0), (5, 20)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is False
    assert res["reason"] == "anchor-step-conflict"
    assert res["delays"] is None
    assert res["errors"] is None
    assert res["ramps"] is None
    conflict = res["conflict"]
    assert conflict["startElement"] == 1
    assert conflict["endElement"] == 5
    assert conflict["requiredChange"] == 20
    assert conflict["allowedChange"] == 8


def test_anchor_step_conflict_is_deterministic():
    inst = make_instance(
        targets=[3] * 14,
        min_delay=0, max_delay=50, max_step=1, max_ramps=5,
        anchors=[(2, 0), (4, 9), (10, 40), (13, 41)],
    )
    first = compile_plan(inst)
    second = compile_plan(inst)
    assert first == second
    assert first["feasible"] is False
    assert first["reason"] == "anchor-step-conflict"
    # smallest (start, end) conflicting pair is reported
    assert first["conflict"]["startElement"] == 2
    assert first["conflict"]["endElement"] == 4


def test_anchor_out_of_bounds():
    inst = make_instance(
        targets=[0] * 12,
        min_delay=0, max_delay=10, max_step=2, max_ramps=3,
        anchors=[(0, 0), (6, 11)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is False
    assert res["reason"] == "anchor-out-of-bounds"
    assert res["delays"] is None


def test_ramp_limit_exceeded_reports_minimum_required():
    inst = make_instance(
        targets=[0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        min_delay=0, max_delay=10, max_step=2, max_ramps=2,
        anchors=[(0, 0), (2, 2), (4, 0), (6, 2), (11, 7)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is False
    assert res["reason"] == "ramp-limit-exceeded"
    assert res["minRampsRequired"] == 3
    assert res["delays"] is None


def test_targets_outside_interval_are_approximated():
    # targets sit far outside [0, 10]: the plan pins to the bounds and the
    # unavoidable errors (5 at the low end, 15 at the high end) define E*
    inst = make_instance(
        targets=[-5] * 6 + [25] * 6,
        min_delay=0, max_delay=10, max_step=3, max_ramps=2,
        anchors=[(0, 0), (11, 10)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is True
    assert res["maxAbsError"] == 15
    check_feasible_shape(inst, res)


def test_zero_max_step_forces_constant_plan():
    inst = make_instance(
        targets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        min_delay=0, max_delay=20, max_step=0, max_ramps=1,
        anchors=[(0, 6), (11, 6)],
    )
    res = compile_plan(inst)
    assert res["feasible"] is True
    assert res["delays"] == [6] * 12
    assert res["rampCount"] == 1
    check_feasible_shape(inst, res)


def test_result_is_deterministic():
    rng = random.Random(7)
    targets = [rng.randint(0, 30) for _ in range(24)]
    inst = make_instance(
        targets=targets,
        min_delay=0, max_delay=30, max_step=4, max_ramps=5,
        anchors=[(0, 10), (12, 22), (23, 8)],
    )
    assert compile_plan(inst) == compile_plan(inst)


# ---------------------------------------------------------------------------
# exhaustive cross-check against brute force
# ---------------------------------------------------------------------------

def random_small_instance(rng: random.Random) -> Instance:
    n = 12
    lo = rng.randint(0, 2)
    hi = lo + rng.randint(2, 4)
    max_step = rng.randint(1, 2)
    max_ramps = rng.randint(1, 3)
    targets = [rng.randint(lo - 1, hi + 1) for _ in range(n)]
    elements = rng.sample(range(n), rng.randint(2, 3))
    anchors = [(e, rng.randint(lo, hi)) for e in sorted(elements)]
    return make_instance(targets, lo, hi, max_step, max_ramps, anchors)


@pytest.mark.parametrize("seed", range(60))
def test_matches_brute_force(seed):
    rng = random.Random(10_000 + seed)
    inst = random_small_instance(rng)
    res = compile_plan(inst)
    best = brute_force_best(inst)
    if best is None:
        assert res["feasible"] is False
        assert res["delays"] is None
    else:
        assert res["feasible"] is True
        check_feasible_shape(inst, res)
        assert result_key(res) == best
