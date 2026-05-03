# Benchmark Baselines — 2026-05-03

Published snapshot of `docs/baseline_*.json` against SPEC §17 budgets. This file is the
deliverable for bead `frankenscipy-d64uj` (reality-check D.11) and the human-readable artifact
that lives alongside the machine-readable JSONs the `benchmark_gate` binary consumes.

The five baseline files were captured 2026-04-13 via `cargo bench` on the rch remote worker.
Future regenerations should bump `baseline_version`, update `generated_at`, and append a row
to §3 (history) below.

## 1. SPEC §17 budgets

The benchmark gate enforces these p95 latency budgets (see
`crates/fsci-conformance/src/bin/benchmark_gate.rs:22-28`):

| Family    | Class                            | Budget |
|-----------|----------------------------------|--------|
| linalg    | dense solve, 4k–8k               | 650 ms |
| sparse    | sparse matvec                    | 220 ms |
| opt       | optimizer iteration              | 180 ms |
| integrate | IVP solve step                   | 320 ms |
| fft       | FFT transform                    | 210 ms |

Headroom column in §2 below uses the largest measured p95 (`upper_*` of the slowest case in
each family) divided into the budget.

## 2. Current baselines

All five families are within budget for sizes captured. The single watch-item is `linalg
solve` extrapolated to 4000×4000 — the table-stakes class for SPEC §17 — which is estimated at
~22000 ms (34× over budget). That's tracked separately as a BLAS/LAPACK acceleration item.

### 2.1 linalg — `docs/baseline_linalg.json`

| Benchmark                         | Median       | Upper (p95) | Status |
|-----------------------------------|-------------:|------------:|--------|
| `baseline_solve / 100x100`        |   300.71 µs  |   305.73 µs | ✓      |
| `baseline_solve / 500x500`        |    30.98 ms  |    31.41 ms | ✓      |
| `baseline_solve / 1000x1000`      |   344.93 ms  |   390.25 ms | ✓ (high variance — 20% outliers) |

Extrapolation: 4000×4000 ≈ 22 000 ms (O(n³) from 1000×1000) → **34× over the 650 ms budget**.
Action item: BLAS/LAPACK acceleration or algorithm change required before any 4k+ solve can
ship under the §17 envelope.

### 2.2 fft — `docs/baseline_fft.json`

| Benchmark                | Median   | Upper (p95) | Status |
|--------------------------|---------:|------------:|--------|
| `baseline_fft / 1024`    |  20.57 µs|    21.15 µs | ✓      |
| `baseline_fft / 65536`   |   3.75 ms|     4.12 ms | ✓      |

Headroom: 210 ms / ~15 ms estimate at 262144 = **~14× headroom**.

### 2.3 opt — `docs/baseline_opt.json`

BFGS on Rosenbrock and quadratic objectives across dimensions 2/5/10:

| Benchmark                       | Median       | Upper (p95) | Status |
|---------------------------------|-------------:|------------:|--------|
| `bfgs / rosenbrock_dim2`        |    2.90 µs   |    3.00 µs  | ✓      |
| `bfgs / rosenbrock_dim5`        |   20.62 µs   |   20.91 µs  | ✓      |
| `bfgs / rosenbrock_dim10`       |   76.57 µs   |   77.68 µs  | ✓      |
| `bfgs / quadratic_dim2`         |  597.49 ns   |  671.75 ns  | ✓      |
| `bfgs / quadratic_dim5`         |  649.22 ns   |  656.10 ns  | ✓      |
| `bfgs / quadratic_dim10`        |  938.20 ns   |  946.46 ns  | ✓      |

Headroom: 180 ms / 0.077 ms = **~2 337× headroom** at the largest captured size.

### 2.4 sparse — `docs/baseline_sparse.json`

| Benchmark                              | Median       | Upper (p95) | Status |
|----------------------------------------|-------------:|------------:|--------|
| `sparse_spmv / 100x100_d5_nnz500`      |  656.06 ns   |  664.75 ns  | ✓      |
| `sparse_spmv / 1000x1000_d1_nnz10126`  |   13.23 µs   |   13.43 µs  | ✓      |
| `sparse_spmv / 10000x10000_d0_nnz99930`|  153.32 µs   |  154.50 µs  | ✓      |

Headroom: 220 ms / 0.155 ms = **~1 428× headroom**.

### 2.5 integrate — `docs/baseline_integrate.json`

| Benchmark                          | Median     | Upper (p95) | Status |
|------------------------------------|-----------:|------------:|--------|
| `solve_ivp / exponential_rk45`     |  18.33 µs  |  18.51 µs   | ✓      |
| `solve_ivp / lorenz_rk45`          |  25.41 µs  |  25.77 µs   | ✓      |
| `validate_tol / scalar`            |  12.48 ns  |  12.67 ns   | ✓      |
| `validate_tol / vector_100`        |  66.30 ns  |  67.09 ns   | ✓      |

Headroom: 320 ms / 0.026 ms = **~12 308× headroom**.

## 3. Regression gate

The CI gate is `crates/fsci-conformance/src/bin/benchmark_gate.rs`. It does **three**
things:

1. **SPEC budget check** (`--check-spec`). For every benchmark in every baseline, takes
   `upper_ms` (p95 if available, else median) and asserts it is ≤ the family's SPEC budget
   from §1. Exit code 1 on any violation.
2. **Regression check** (`--compare <dir> [--regression-tolerance F]`). Loads
   `baseline_*.json` files from `<dir>` and compares the `family/group/name` slot of each
   benchmark against the same slot in `--baselines-dir`. Reports a per-benchmark relative
   delta on both `median_ms` and `upper_ms` (p95). Exit code 1 if any delta exceeds the
   tolerance (default 5%, override with `--regression-tolerance`).
3. **Baseline integrity check** (default invocation). Loads all `baseline_*.json` files from
   `--baselines-dir` (default `docs/`) and parses them; failure to parse is exit code 2.

Workflow after a fresh benchmark capture:

```bash
# 1. Capture criterion output (per-family)
cd /data/projects/frankenscipy
rch exec -- cargo bench -p fsci-linalg --bench baseline_solve

# 2. Roll the criterion output up into a baseline_*.json under e.g. /tmp/new-baselines/
#    (the existing ad-hoc capture script in the repo's bench README does this; the same
#    schema BaselineFile defined in benchmark_gate.rs governs the file shape).

# 3. Run the regression gate
cargo run --release -p fsci-conformance --bin benchmark_gate -- \
    --baselines-dir docs --compare /tmp/new-baselines \
    --regression-tolerance 0.05
# exit code 0  → all benchmarks within 5% of baseline (and SPEC budget when --check-spec)
# exit code 1  → at least one benchmark regressed
# exit code 2  → IO/JSON error
```

The SPEC-only invocation:

```bash
cargo run --release -p fsci-conformance --bin benchmark_gate -- \
    --baselines-dir docs --check-spec
```

## 4. RaptorQ sidecar

SPEC §6 calls for RaptorQ-everywhere durability on long-lived artifacts. Each
`baseline_*.json` is a candidate. As of today none of the five baseline files have a
`.raptorq.json` sidecar — they sit naked in `docs/`. The follow-on for this bead is to plumb
the existing RaptorQ pipeline (used today only for `crates/fsci-conformance/fixtures/artifacts/
FSCI-P2C-*/parity_report.raptorq.json`) so each `baseline_*.json` regeneration writes a
sidecar. Tracked as the third remaining sub-item on `frankenscipy-d64uj`.

## 5. History

| Date       | baseline_version | Change                                                         |
|------------|------------------|----------------------------------------------------------------|
| 2026-04-13 | 1.0.0            | Initial capture (all five families). Linalg 1000×1000 high variance noted. |
| 2026-05-03 | (no rebench)     | Published this artifact + documented gate workflow + implemented `--compare` regression mode (5% default tolerance) + flagged RaptorQ-sidecar as the remaining follow-on under the same bead. |

## 6. Honesty notes

- The CPU field on every baseline reads `unknown (via rch remote)`. That is true today but
  ought to be filled in by the rch worker on capture; otherwise headroom claims are
  not portable across hardware. Tracked as a minor follow-on.
- The 1000×1000 linalg solve has 20% outliers per the baseline's own note. Deciding whether
  to widen the criterion sample size or accept the variance is part of the regression-gate
  follow-on (a >5% threshold against a noisy baseline is meaningless).
- Two SPEC §17 families — special and stats — are NOT represented in the baseline set at
  all. Adding them is a prerequisite to claiming the gate covers SPEC §17 in full; both are
  blocked behind `frankenscipy-b6z3m` and `frankenscipy-nb5gj` respectively (bench targets
  must exist before they can be baselined).
