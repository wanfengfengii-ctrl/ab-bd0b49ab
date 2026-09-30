"""Exact integer solver for ultrasound probe delay-plan compilation.

A *delay plan* assigns one integer delay to every element.  A plan is
feasible when it

* stays inside the global ``[min_delay, max_delay]`` interval,
* hits every anchor exactly,
* keeps ``|x[i+1] - x[i]| <= max_step`` for adjacent elements, and
* uses at most ``max_ramps`` *ramps*, where a ramp is a maximal run of
  equal adjacent differences (i.e. one constant-slope segment).

Feasible plans are ranked lexicographically by

1. maximum absolute error against the target delays,
2. total absolute error,
3. actual ramp count,
4. lexicographic order of the delay sequence itself.

All decisions use integer arithmetic only.  The solver is a dynamic
program over ``(position, value, incoming slope, ramps used)``:

* pass 1 (forward, bottleneck cost) finds the minimal max-abs-error ``E*``;
* pass 2 (backward suffix DP, full layers) minimises the combined cost
  ``total_error * (ramp_cap + 1) + ramps`` under the tightened domains
  ``|x[i] - target[i]| <= E*`` — because ramps ``<= ramp_cap < multiplier``
  this is exactly the lexicographic pair ``(total_error, ramps)``;
* pass 3 greedily rebuilds the lexicographically smallest optimal
  sequence using the suffix layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sentinel for unreachable states.  Must satisfy
#   INF + max_real_cost < 2**31   (layers are int32)
# With the validated input envelope (|values| <= 100_000, ramps <= 47)
# the real combined cost stays below ~5e8, so 1e9 is safe.
INF = 1_000_000_000

# Resource envelope: total suffix-DP cells (and peak forward-layer cells)
# are rejected above this budget with a 422-style error.  60M int32 cells
# correspond to roughly 240 MB of suffix layers.
MAX_STATE_CELLS = 60_000_000


class SolverResourceError(RuntimeError):
    """Raised when an instance exceeds the solver's resource envelope."""


@dataclass(frozen=True)
class Anchor:
    element: int
    delay: int


@dataclass(frozen=True)
class Instance:
    targets: tuple
    min_delay: int
    max_delay: int
    max_step: int
    max_ramps: int
    anchors: tuple

    @property
    def n(self) -> int:
        return len(self.targets)


# ---------------------------------------------------------------------------
# domains
# ---------------------------------------------------------------------------

def _position_domains(inst: Instance, e_bound: int | None) -> list[tuple[int, int]]:
    """Inclusive [lo, hi] value interval allowed at each element.

    Intersects the global bounds, the reachability envelope implied by the
    anchors (``|x[i] - a.delay| <= max_step * |i - a.element|``), the
    max-error bound ``e_bound`` (when given) and the anchor pins.
    """
    anchor_map = {a.element: a.delay for a in inst.anchors}
    domains = []
    for i in range(inst.n):
        lo = inst.min_delay
        hi = inst.max_delay
        for a in inst.anchors:
            dist = i - a.element
            if dist < 0:
                dist = -dist
            reach = inst.max_step * dist
            if a.delay - reach > lo:
                lo = a.delay - reach
            if a.delay + reach < hi:
                hi = a.delay + reach
        if e_bound is not None:
            t = inst.targets[i]
            if t - e_bound > lo:
                lo = t - e_bound
            if t + e_bound < hi:
                hi = t + e_bound
        v = anchor_map.get(i)
        if v is not None:
            if v > lo:
                lo = v
            if v < hi:
                hi = v
        domains.append((lo, hi))
    return domains


def _slope_ranges(domains: list[tuple[int, int]], max_step: int) -> list[tuple[int, int] | None]:
    """Inclusive slope range into each position ``i >= 1``."""
    ranges: list[tuple[int, int] | None] = [None]
    for i in range(1, len(domains)):
        lo_prev, hi_prev = domains[i - 1]
        lo_cur, hi_cur = domains[i]
        k_lo = max(-max_step, lo_cur - hi_prev)
        k_hi = min(max_step, hi_cur - lo_prev)
        ranges.append((k_lo, k_hi))
    return ranges


def _domains_valid(domains, slopes) -> bool:
    if any(lo > hi for lo, hi in domains):
        return False
    return all(r is None or r[0] <= r[1] for r in slopes[1:])


def _error_arrays(domains, targets) -> list[np.ndarray]:
    out = []
    for (lo, hi), t in zip(domains, targets):
        vals = np.arange(lo, hi + 1, dtype=np.int64)
        out.append(np.abs(vals - t).astype(np.int32))
    return out


def _layer_cells(domains, slopes, ramp_cap) -> tuple[int, int]:
    """(total, peak) DP cells for the given domains and ramp cap."""
    total = 0
    peak = 0
    for i, (lo, hi) in enumerate(domains):
        w = max(1, hi - lo + 1)
        if i == 0:
            kw = 1
        else:
            kw = max(1, slopes[i][1] - slopes[i][0] + 1)
        cells = ramp_cap * w * kw
        total += cells
        peak = max(peak, cells)
    return total, peak


def _guard_budget(domains, slopes, ramp_cap) -> None:
    total, peak = _layer_cells(domains, slopes, ramp_cap)
    if total > MAX_STATE_CELLS or peak > MAX_STATE_CELLS:
        raise SolverResourceError(
            f"instance requires about {total} DP states, "
            f"which exceeds the solver limit of {MAX_STATE_CELLS}"
        )


# ---------------------------------------------------------------------------
# forward DP (bottleneck for pass 1, summed cost for the min-ramps probe)
# ---------------------------------------------------------------------------

def _forward_step(g_prev, lo_prev, k_lo_prev, lo_cur, k_lo_cur, k_hi_cur,
                  err_cur, mode, multiplier):
    ramp_cap = g_prev.shape[0]
    w_prev = g_prev.shape[1]
    kw_prev = g_prev.shape[2]
    k_hi_prev = k_lo_prev + kw_prev - 1
    w_cur = err_cur.shape[0]
    kw_cur = k_hi_cur - k_lo_cur + 1
    values_cur = lo_cur + np.arange(w_cur)

    # continue the current ramp with the same slope
    same = np.full((ramp_cap, w_cur, kw_cur), INF, dtype=np.int32)
    ks_lo = max(k_lo_cur, k_lo_prev)
    ks_hi = min(k_hi_cur, k_hi_prev)
    for k in range(ks_lo, ks_hi + 1):
        src = values_cur - (k + lo_prev)
        valid = (src >= 0) & (src < w_prev)
        if valid.any():
            same[:, valid, k - k_lo_cur] = g_prev[:, src[valid], k - k_lo_prev]

    # open a new ramp with a different slope:
    # change[r', v', k'] = min over k_prev != k' of g_prev[r'-1, v'-k', k_prev]
    # The predecessor value v'-k' is fixed per target slope k', so first
    # precompute the two smallest values along the previous slope axis.
    change = np.full((ramp_cap, w_cur, kw_cur), INF, dtype=np.int32)
    if ramp_cap > 1:
        head = g_prev[: ramp_cap - 1]                # (R-1, w_prev, kw_prev)
        arg1 = head.argmin(axis=2)
        if kw_prev > 1:
            part = np.partition(head, 1, axis=2)
            min1 = part[..., 0]
            min2 = part[..., 1]
        else:
            min1 = head[..., 0]
            min2 = np.full_like(min1, INF)
        for k in range(k_lo_cur, k_hi_cur + 1):
            src = values_cur - (k + lo_prev)
            valid = (src >= 0) & (src < w_prev)
            if not valid.any():
                continue
            k_prev_idx = k - k_lo_prev
            if 0 <= k_prev_idx < kw_prev:
                picked = np.where(arg1[:, src[valid]] == k_prev_idx,
                                  min2[:, src[valid]], min1[:, src[valid]])
            else:
                picked = min1[:, src[valid]]
            if mode == "sum":
                picked = picked + 1  # a slope change opens one more ramp
            change[1:, valid, k - k_lo_cur] = picked

    best = np.minimum(same, change)
    if mode == "bottleneck":
        return np.maximum(best, err_cur[None, :, None])
    err = (err_cur.astype(np.int64) * multiplier).astype(np.int32)
    return best + err[None, :, None]


def _forward_dp(domains, targets, max_step, ramp_cap, mode, multiplier=1,
                error_weight=1):
    """Rolling forward DP; returns the layer at the last position."""
    n = len(domains)
    slopes = _slope_ranges(domains, max_step)
    errs = _error_arrays(domains, targets)
    if error_weight == 0:
        errs = [np.zeros_like(e) for e in errs]

    lo0, hi0 = domains[0]
    lo1, hi1 = domains[1]
    k_lo1, k_hi1 = slopes[1]
    w1 = hi1 - lo1 + 1
    g = np.full((ramp_cap, w1, k_hi1 - k_lo1 + 1), INF, dtype=np.int32)
    values1 = lo1 + np.arange(w1)
    e0_all = errs[0].astype(np.int64)
    e1_all = errs[1].astype(np.int64)
    for k in range(k_lo1, k_hi1 + 1):
        src = values1 - (k + lo0)
        valid = (src >= 0) & (src <= hi0 - lo0)
        if not valid.any():
            continue
        e0 = e0_all[src[valid]]
        e1 = e1_all[valid]
        if mode == "bottleneck":
            cost = np.maximum(e0, e1)
        else:
            cost = (e0 + e1) * multiplier + 1
        g[0, valid, k - k_lo1] = cost.astype(np.int32)

    for i in range(2, n):
        g = _forward_step(g, domains[i - 1][0], slopes[i - 1][0],
                          domains[i][0], slopes[i][0], slopes[i][1],
                          errs[i], mode, multiplier)
    return g


# ---------------------------------------------------------------------------
# backward suffix DP (summed combined cost, all layers kept for greedy)
# ---------------------------------------------------------------------------

def _suffix_step(s_next, lo_next, k_lo_next, lo_cur, k_lo_cur, k_hi_cur,
                 err_cur, multiplier):
    ramp_cap = s_next.shape[0]
    w_next = s_next.shape[1]
    kw_next = s_next.shape[2]
    k_hi_next = k_lo_next + kw_next - 1
    w_cur = err_cur.shape[0]
    kw_cur = k_hi_cur - k_lo_cur + 1
    values_cur = lo_cur + np.arange(w_cur)

    # continue the current ramp with the same slope
    same = np.full((ramp_cap, w_cur, kw_cur), INF, dtype=np.int32)
    ks_lo = max(k_lo_cur, k_lo_next)
    ks_hi = min(k_hi_cur, k_hi_next)
    for k in range(ks_lo, ks_hi + 1):
        dst = values_cur + k - lo_next
        valid = (dst >= 0) & (dst < w_next)
        if valid.any():
            same[:, valid, k - k_lo_cur] = s_next[:, dst[valid], k - k_lo_next]

    # open a new ramp with a different slope (costs one extra ramp)
    change = np.full((ramp_cap, w_cur, kw_cur), INF, dtype=np.int32)
    if ramp_cap > 1:
        stacked = np.full((ramp_cap - 1, w_cur, kw_next), INF, dtype=np.int32)
        for kk in range(kw_next):
            dst = values_cur + (k_lo_next + kk) - lo_next
            valid = (dst >= 0) & (dst < w_next)
            if valid.any():
                stacked[:, valid, kk] = s_next[1:][:, dst[valid], kk]
        best1 = stacked.min(axis=2)
        arg1 = stacked.argmin(axis=2)
        jj, vv = np.indices(best1.shape)
        stacked[jj, vv, arg1] = INF
        best2 = stacked.min(axis=2)
        kk_idx = np.arange(k_lo_cur, k_hi_cur + 1) - k_lo_next
        in_next = (kk_idx >= 0) & (kk_idx < kw_next)
        is_arg = in_next[None, None, :] & (kk_idx[None, None, :] == arg1[:, :, None])
        change[: ramp_cap - 1] = np.where(is_arg, best2[:, :, None],
                                          best1[:, :, None]) + 1

    out = np.minimum(same, change)
    err = (err_cur.astype(np.int64) * multiplier).astype(np.int32)
    return out + err[None, :, None]


def _suffix_dp(domains, targets, max_step, ramp_cap, multiplier):
    n = len(domains)
    slopes = _slope_ranges(domains, max_step)
    errs = _error_arrays(domains, targets)
    layers: list[np.ndarray | None] = [None] * n

    lo, hi = domains[-1]
    k_lo, k_hi = slopes[-1]
    base = np.empty((ramp_cap, hi - lo + 1, k_hi - k_lo + 1), dtype=np.int32)
    base[:] = (errs[-1].astype(np.int64) * multiplier).astype(np.int32)[:, None]
    layers[-1] = base

    for i in range(n - 2, 0, -1):
        layers[i] = _suffix_step(layers[i + 1], domains[i + 1][0],
                                 slopes[i + 1][0], domains[i][0],
                                 slopes[i][0], slopes[i][1],
                                 errs[i], multiplier)
    return layers, slopes, errs


# ---------------------------------------------------------------------------
# lexicographic reconstruction
# ---------------------------------------------------------------------------

def _first_position_costs(layers, slopes, errs, domains, multiplier):
    """Best total combined cost for each candidate value at element 0."""
    s1 = layers[1]
    lo0, hi0 = domains[0]
    lo1 = domains[1][0]
    k_lo1, k_hi1 = slopes[1]
    w1 = s1.shape[1]
    values0 = np.arange(lo0, hi0 + 1)
    best = np.full(hi0 - lo0 + 1, INF, dtype=np.int64)
    e0 = errs[0].astype(np.int64) * multiplier
    for k in range(k_lo1, k_hi1 + 1):
        dst = values0 + k - lo1
        valid = (dst >= 0) & (dst < w1)
        if valid.any():
            # the first difference always opens the first ramp (+1)
            cand = e0[valid] + 1 + s1[0, dst[valid], k - k_lo1].astype(np.int64)
            best[valid] = np.minimum(best[valid], cand)
    return best


def _greedy_sequence(layers, slopes, errs, domains, multiplier):
    """Rebuild the lexicographically smallest optimal delay sequence."""
    n = len(domains)
    ramp_cap = layers[1].shape[0]
    per_v0 = _first_position_costs(layers, slopes, errs, domains, multiplier)
    c_star = int(per_v0.min())
    lo0 = domains[0][0]
    x0 = lo0 + int(np.argmin(per_v0))  # argmin returns the first minimum
    seq = [x0]

    # element 1: smallest value that keeps the total cost at c_star
    residual = c_star - int(errs[0][x0 - lo0]) * multiplier - 1
    lo1, hi1 = domains[1]
    k_lo1, k_hi1 = slopes[1]
    s1 = layers[1]
    chosen = None
    for v1 in range(lo1, hi1 + 1):
        k = v1 - x0
        if k < k_lo1 or k > k_hi1:
            continue
        if int(s1[0, v1 - lo1, k - k_lo1]) == residual:
            chosen = (v1, k)
            break
    if chosen is None:
        raise RuntimeError("greedy reconstruction failed at element 1")
    seq.append(chosen[0])
    k_cur = chosen[1]
    r_cur = 0  # ramp index 0 == one ramp used so far

    for i in range(2, n):
        lo_prev, _ = domains[i - 1]
        k_lo_prev, _ = slopes[i - 1]
        s_prev = layers[i - 1]
        residual = (int(s_prev[r_cur, seq[-1] - lo_prev, k_cur - k_lo_prev])
                    - int(errs[i - 1][seq[-1] - lo_prev]) * multiplier)
        lo_i, hi_i = domains[i]
        k_lo_i, k_hi_i = slopes[i]
        s_i = layers[i]
        chosen = None
        for v in range(lo_i, hi_i + 1):
            k = v - seq[-1]
            if k < k_lo_i or k > k_hi_i:
                continue
            opens_new_ramp = k != k_cur
            r_new = r_cur + 1 if opens_new_ramp else r_cur
            if r_new >= ramp_cap:
                continue
            # a slope change is charged one extra ramp at this transition
            cand = int(s_i[r_new, v - lo_i, k - k_lo_i]) + (1 if opens_new_ramp else 0)
            if cand == residual:
                chosen = (v, k, r_new)
                break
        if chosen is None:
            raise RuntimeError(f"greedy reconstruction failed at element {i}")
        seq.append(chosen[0])
        k_cur = chosen[1]
        r_cur = chosen[2]
    return seq, c_star


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ramps_of(seq):
    """Maximal runs of equal adjacent differences as (start, end, slope).

    ``start``/``end`` are inclusive element indices; the ramp covers
    elements ``start..end`` with per-step change ``slope``.
    """
    diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    ramps = []
    start = 0
    for i in range(1, len(diffs)):
        if diffs[i] != diffs[i - 1]:
            ramps.append((start, i, diffs[i - 1]))
            start = i
    ramps.append((start, len(diffs), diffs[-1]))
    return ramps


def _infeasible(reason: str, detail: dict) -> dict:
    out = {
        "feasible": False,
        "reason": reason,
        # never emit a partial delay table for infeasible requests
        "delays": None,
        "errors": None,
        "ramps": None,
        "maxAbsError": None,
        "totalAbsError": None,
        "rampCount": None,
    }
    out.update(detail)
    return out


def _min_ramps(inst: Instance, domains, slopes) -> int | None:
    """Smallest ramp count of any constraint-satisfying sequence."""
    ramp_cap = inst.n - 1
    try:
        _guard_budget(domains, slopes, ramp_cap)
    except SolverResourceError:
        return None
    g = _forward_dp(domains, inst.targets, inst.max_step, ramp_cap,
                    mode="sum", multiplier=1, error_weight=0)
    best = int(g.min())
    return best if best < INF else None


def _assert_solution(inst: Instance, seq, ramp_cap) -> None:
    anchor_map = {a.element: a.delay for a in inst.anchors}
    assert len(seq) == inst.n
    for i, v in enumerate(seq):
        assert inst.min_delay <= v <= inst.max_delay
        if i in anchor_map:
            assert v == anchor_map[i]
        if i:
            assert abs(v - seq[i - 1]) <= inst.max_step
    assert len(_ramps_of(seq)) <= ramp_cap


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def compile_plan(inst: Instance) -> dict:
    """Compile one delay plan; returns the API response payload."""
    n = inst.n
    ramp_cap = min(inst.max_ramps, n - 1)

    # anchors must sit inside the global delay interval
    for a in inst.anchors:
        if not (inst.min_delay <= a.delay <= inst.max_delay):
            return _infeasible("anchor-out-of-bounds", {
                "element": a.element,
                "delay": a.delay,
                "minDelay": inst.min_delay,
                "maxDelay": inst.max_delay,
            })

    # anchors must be mutually reachable under the max-step limit;
    # report the lexicographically smallest conflicting pair (stable output)
    ordered = sorted(inst.anchors, key=lambda a: a.element)
    conflict = None
    for x in range(len(ordered)):
        for y in range(x + 1, len(ordered)):
            a, b = ordered[x], ordered[y]
            if abs(b.delay - a.delay) > inst.max_step * (b.element - a.element):
                conflict = (a, b)
                break
        if conflict:
            break
    if conflict:
        a, b = conflict
        return _infeasible("anchor-step-conflict", {
            "conflict": {
                "startElement": a.element,
                "endElement": b.element,
                "startDelay": a.delay,
                "endDelay": b.delay,
                "requiredChange": abs(b.delay - a.delay),
                "allowedChange": inst.max_step * (b.element - a.element),
                "maxStep": inst.max_step,
            }
        })

    # pass 1: minimal achievable max-abs-error under the ramp cap
    domains1 = _position_domains(inst, None)
    slopes1 = _slope_ranges(domains1, inst.max_step)
    if not _domains_valid(domains1, slopes1):
        return _infeasible("constraints-unsatisfiable", {})
    _guard_budget(domains1, slopes1, ramp_cap)
    g1 = _forward_dp(domains1, inst.targets, inst.max_step, ramp_cap,
                     mode="bottleneck")
    e_star = int(g1.min())
    if e_star >= INF:
        min_ramps = _min_ramps(inst, domains1, slopes1)
        detail = {"maxRamps": inst.max_ramps}
        if min_ramps is not None:
            detail["minRampsRequired"] = min_ramps
        return _infeasible("ramp-limit-exceeded", detail)

    # pass 2+3: minimise (total error, ramps), then lexicographic order
    domains2 = _position_domains(inst, e_star)
    slopes2 = _slope_ranges(domains2, inst.max_step)
    if not _domains_valid(domains2, slopes2):
        raise RuntimeError("tightened domains unexpectedly infeasible")
    _guard_budget(domains2, slopes2, ramp_cap)
    multiplier = ramp_cap + 1
    layers, slopes2, errs = _suffix_dp(domains2, inst.targets, inst.max_step,
                                       ramp_cap, multiplier)
    seq, c_star = _greedy_sequence(layers, slopes2, errs, domains2, multiplier)

    errors = [d - t for d, t in zip(seq, inst.targets)]
    abs_errors = [abs(e) for e in errors]
    ramps = _ramps_of(seq)

    # cheap internal consistency checks (n <= 48)
    _assert_solution(inst, seq, ramp_cap)
    assert max(abs_errors) == e_star
    assert sum(abs_errors) * multiplier + len(ramps) == c_star

    return {
        "feasible": True,
        "delays": seq,
        "errors": errors,
        "maxAbsError": max(abs_errors),
        "totalAbsError": sum(abs_errors),
        "rampCount": len(ramps),
        "ramps": [
            {
                "startElement": s,
                "endElement": e,
                "slope": k,
                "startValue": seq[s],
                "endValue": seq[e],
            }
            for s, e, k in ramps
        ],
    }
