"""Exact stability surveys over families of polygons: parallel, cached on disk.

Each polygon gets the exact boundary-normalized crease minimum of `crease_exact.crease_minimum`
(negative = relatively K-unstable) and its critical line. Results are stored in a JSON file keyed by a
caller-chosen string, so a survey can be extended or rerun without recomputing.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import mpmath

from .crease_exact import crease_minimum
from .polygon import Polygon


def margin_record(P: Polygon) -> dict:
    """The crease minimum of P and its critical line n·x = s, as floats (n = None if no crease beats 1)."""
    r = crease_minimum(P)
    rec = {"value": float(r.value), "n": None, "s": None, "theta": None, "cell": None, "kind": None}
    if r.crease is not None:
        c = r.crease
        rec.update(n=[float(v) for v in c.normal], s=float(c.s), theta=c.theta, cell=list(c.cell), kind=c.kind,
                   value_str=mpmath.nstr(r.value, 30))
    return rec


def _cached_map(fn, items: dict, cache, processes, verbose) -> dict:
    """{key: fn(item)} in parallel, reusing and extending the JSON cache file."""
    path = Path(cache) if cache else None
    done = json.loads(path.read_text()) if path and path.exists() else {}
    todo = [k for k in items if k not in done]
    if verbose:
        print(f"{len(items) - len(todo)} cached, {len(todo)} to compute")
    if todo:
        with ProcessPoolExecutor(processes) as pool:
            for key, rec in zip(todo, pool.map(fn, [items[k] for k in todo], chunksize=1)):
                done[key] = rec
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(done, indent=0))
    return {k: done[k] for k in items}


def survey(polygons: dict[str, Polygon], cache: str | Path | None = None, processes: int | None = None,
           verbose: bool = True) -> dict[str, dict]:
    """margin_record for every polygon, reusing and extending the cache file (keys are the dict keys)."""
    return _cached_map(margin_record, polygons, cache, processes, verbose)


# ---------------------------------------------------------------- threshold labels


def label_threshold(job, lo: float = 4.0, hi: float = 2000.0, tol: float = 1e-10) -> dict:
    """The label m* at which family(*params, 1/m) becomes unstable, for job = (family, params).

    `family` must be a module-level function (it is sent to worker processes) building the polygon with
    one label 1/m; equivalently, label 1 on that edge and m on the others. Stable for m < m*, assumed
    monotone in m. Illinois iteration on log m in [lo, hi] for a zero of the exact crease minimum, with m
    rounded to denominators ≤ 10⁶, so m* is accurate to about 1e-6 relative. Also returns the balanced
    crease at m* (the line f = (c0, c1, c2), crease max(0, c0 + c1 x + c2 y)). m_star None: no sign
    change in [lo, hi] (ends gives the two margins).
    """
    family, params = job

    def margin(x):
        m = Fraction(math.exp(x)).limit_denominator(10**6)
        rec = margin_record(family(*params, 1 / m))
        return float(m), rec

    a, b = math.log(lo), math.log(hi)
    _, ra = margin(a)
    _, rb = margin(b)
    fa, fb = ra["value"], rb["value"]
    if not (fa > 0 > fb):
        return {"m_star": None, "ends": [fa, fb]}
    side, m, rec = 0, None, None
    for _ in range(60):
        x = (a * fb - b * fa) / (fb - fa)
        m, rec = margin(x)
        f, x = rec["value"], math.log(m)
        if abs(f) < tol or b - a < 1e-12:
            break
        if f > 0:
            a, fa = x, f
            if side == 1:
                fb /= 2
            side = 1
        else:
            b, fb = x, f
            if side == -1:
                fa /= 2
            side = -1
    line = None if rec["n"] is None else [-rec["s"], *rec["n"]]
    return {"m_star": m, "value": rec["value"], "line": line, "kind": rec["kind"]}


def label_threshold_acg(job, lo: float = 4.0, hi: float = 3000.0, steps: int = 45) -> dict:
    """As label_threshold, by bisection on the sign of ACG's criterion (quadrilaterals only): an
    independent check. ACG samples its conic, so its m* is slightly high (relative 1e-5)."""
    from .ambitoric import acg_margin

    family, params = job

    def unstable(x):
        m = Fraction(math.exp(x)).limit_denominator(10**6)
        return acg_margin(family(*params, 1 / m))["value"] < -1e-12

    a, b = math.log(lo), math.log(hi)
    if unstable(a) or not unstable(b):
        return {"m_star": None}
    for _ in range(steps):
        mid = (a + b) / 2
        a, b = (a, mid) if unstable(mid) else (mid, b)
    return {"m_star": math.exp((a + b) / 2)}


def thresholds(jobs: dict[str, tuple], cache: str | Path | None = None, processes: int | None = None,
               verbose: bool = True, method: str = "exact") -> dict[str, dict]:
    """label_threshold (or, method="acg", label_threshold_acg) for every (family, params), cached like `survey`."""
    fn = {"exact": label_threshold, "acg": label_threshold_acg}[method]
    return _cached_map(fn, jobs, cache, processes, verbose)


def edges_crossed(P: Polygon, line, tol: float = 1e-9) -> tuple[int, ...]:
    """Indices of the edges of P met by the line c0 + c1 x + c2 y = 0 (an edge through which it passes a
    vertex counts on both sides)."""
    c0, c1, c2 = (float(v) for v in line)
    vals = [c0 + c1 * float(x) + c2 * float(y) for x, y in P.vertices]
    n = len(vals)
    return tuple(i for i in range(n)
                 if vals[i] * vals[(i + 1) % n] < 0 or abs(vals[i]) < tol or abs(vals[(i + 1) % n]) < tol)
