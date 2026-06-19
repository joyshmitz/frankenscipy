#!/usr/bin/env python3
"""SciPy oracle timing for fsci-fft cross_spectral_density gauntlet rows."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from typing import Callable

import numpy as np
import scipy
import scipy.fft


FS = 48_000.0
SIZES = (4096, 65_536)


def make_real_pair(n: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n, dtype=np.float64)
    t = i / float(n)
    x = np.sin(2.0 * np.pi * t) + 0.25 * np.cos(13.0 * np.pi * t)
    y = np.cos(3.0 * np.pi * t) - 0.5 * np.sin(17.0 * np.pi * t)
    return x, y


def scipy_rfft_formula(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx = scipy.fft.rfft(x)
    fy = scipy.fft.rfft(y)
    csd = fx * np.conj(fy) / (len(x) * FS)
    freqs = scipy.fft.rfftfreq(len(x), 1.0 / FS)
    return freqs, csd


def scipy_full_fft_formula(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx = scipy.fft.fft(x.astype(np.complex128))
    fy = scipy.fft.fft(y.astype(np.complex128))
    n_freq = len(x) // 2 + 1
    csd = fx[:n_freq] * np.conj(fy[:n_freq]) / (len(x) * FS)
    freqs = np.arange(n_freq, dtype=np.float64) * FS / len(x)
    return freqs, csd


def time_us(
    label: str,
    n: int,
    fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    reps: int,
    warmups: int,
) -> dict[str, object]:
    x, y = make_real_pair(n)
    for _ in range(warmups):
        fn(x, y)

    samples: list[float] = []
    out: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(reps):
        started = time.perf_counter_ns()
        out = fn(x, y)
        samples.append((time.perf_counter_ns() - started) / 1000.0)

    samples.sort()
    assert out is not None
    p95_index = int(0.95 * (len(samples) - 1))
    return {
        "label": label,
        "n": n,
        "reps": reps,
        "warmups": warmups,
        "p50_us": statistics.median(samples),
        "p95_us": samples[p95_index],
        "min_us": samples[0],
        "max_us": samples[-1],
        "nfreq": int(len(out[0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=120)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()

    results = []
    for n in SIZES:
        results.append(time_us("scipy_rfft_formula", n, scipy_rfft_formula, args.reps, args.warmups))
        results.append(time_us("scipy_full_fft_formula", n, scipy_full_fft_formula, args.reps, args.warmups))

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "fs": FS,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
