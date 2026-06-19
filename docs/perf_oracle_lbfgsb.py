#!/usr/bin/env python3
"""Head-to-head SciPy oracle for fsci-opt L-BFGS-B Criterion rows.

Mirrors `crates/fsci-opt/benches/optimize_bench.rs::bench_lbfgsb` so the
reported ratios compare the Rust port against the original SciPy public path
on the same deterministic objective shapes.
"""

import json
import statistics
import time

import numpy as np
from scipy.optimize import minimize


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def quadratic(x):
    x = np.asarray(x, dtype=float)
    return float(np.dot(x, x))


def percentile(sorted_values, q):
    if not sorted_values:
        raise ValueError("empty sample")
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def run_case(case_id, func, x0, reps=11):
    samples = []
    outcomes = []
    for _ in range(reps):
        started = time.perf_counter()
        res = minimize(
            func,
            np.asarray(x0, dtype=float),
            method="L-BFGS-B",
            tol=1.0e-8,
            options={"maxiter": 2000},
        )
        samples.append((time.perf_counter() - started) * 1.0e6)
        outcomes.append(
            {
                "success": bool(res.success),
                "fun": float(res.fun),
                "nit": int(res.nit),
                "nfev": int(res.nfev),
                "njev": int(res.njev),
            }
        )
    ordered = sorted(samples)
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples)
    return {
        "case": case_id,
        "reps": reps,
        "p50_us": percentile(ordered, 0.50),
        "p95_us": percentile(ordered, 0.95),
        "p99_us": percentile(ordered, 0.99),
        "mean_us": mean,
        "cv_pct": 0.0 if mean == 0.0 else stdev / mean * 100.0,
        "outcome": outcomes[-1],
    }


def main():
    cases = [
        ("lbfgsb/rosenbrock_unconstrained_fd/2", rosenbrock, [-1.2, 1.0]),
        (
            "lbfgsb/rosenbrock_unconstrained_fd/10",
            rosenbrock,
            [-1.2 if i % 2 == 0 else 1.0 for i in range(10)],
        ),
        (
            "lbfgsb/quadratic_unconstrained_fd/32",
            quadratic,
            [float(i % 7) - 3.0 for i in range(32)],
        ),
    ]
    payload = {
        "oracle": "scipy.optimize.minimize(method='L-BFGS-B')",
        "cases": [run_case(*case) for case in cases],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
