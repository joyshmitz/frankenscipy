# frankenscipy-96n2y - jnjnp_zeros tighter frontier evidence

Date: 2026-06-19
Agent: cod-b / MistyBirch
Worker: `hz1` for Rust build/test/bench via `rch`
Local SciPy oracle host: `thinkstation1`

## Lever

`jnjnp_zeros(nt)` previously seeded the adaptive output frontier with
`per = sqrt(nt) + 4` serial roots and `n_max = 2*sqrt(nt) + 6` orders, then
doubled both dimensions together when the cutoff proof failed. The kept outputs
for the gauntlet sizes are much smaller (`nt=128` only reaches max serial `m=7`
and max order `n=19`), so this lever starts from a tighter envelope and expands
the serial and order dimensions independently using the existing exact
frontier proof.

## Commands

Baseline and candidate bench:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot
```

Focused correctness:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo test -p fsci-special jnjnp -- --nocapture
```

Build gate:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo check -p fsci-special --all-targets
```

SciPy oracle:

```bash
python3 - <<'PY'
import platform, sys, time, statistics
import numpy as np
import scipy
import scipy.special as sc
print(platform.node(), sys.version.split()[0], np.__version__, scipy.__version__)
for nt in (64, 128):
    sc.jnjnp_zeros(nt)
    reps = []
    for _ in range(9):
        start = time.perf_counter()
        for _ in range(1000):
            sc.jnjnp_zeros(nt)
        reps.append((time.perf_counter() - start) / 1000)
    print(nt, statistics.median(reps), statistics.mean(reps), min(reps), max(reps))
PY
```

## Same-worker Rust A/B

| Workload | Baseline current mean | Candidate current mean | Candidate/baseline | Verdict |
| --- | ---: | ---: | ---: | --- |
| `jnjnp_zeros(nt=64)` | 4.7127 ms | 2.2230 ms | 0.472x time, 2.12x faster | keep |
| `jnjnp_zeros(nt=128)` | 8.5181 ms | 6.1605 ms | 0.723x time, 1.38x faster | keep |

Criterion reported statistically significant improvements for `rust_current`:
`nt=64` changed by -52.933% and `nt=128` changed by -28.210% on `hz1`.

## SciPy Head-to-head

Local oracle versions: Python 3.13.7, NumPy 2.4.3, SciPy 1.17.1.

| Workload | Candidate Rust mean | SciPy median | Candidate/SciPy | Verdict |
| --- | ---: | ---: | ---: | --- |
| `jnjnp_zeros(nt=64)` | 2.2230 ms | 424.10 us | 5.24x slower | residual loss |
| `jnjnp_zeros(nt=128)` | 6.1605 ms | 799.97 us | 7.70x slower | residual loss |

SciPy win/loss/neutral: `0/2/0`.
Same-worker internal keep/loss/neutral: `2/0/0`.

## Correctness

- PASS: `jnyn_and_jnjnp_zeros_match_scipy`.
- PASS: `jnjnp_adaptive_envelope_matches_oversized_reference`.
- PASS: `jnjnp_frontier_matches_scipy_bench_cutoffs`.
- PASS: `cargo check -p fsci-special --all-targets` via `rch` on `hz1`.

`cargo fmt --check` is blocked by broad pre-existing rustfmt drift across the
workspace, and `rustfmt --check crates/fsci-special/src/bessel.rs` is blocked by
pre-existing formatting drift outside the edited block.

## Decision

KEEP. The tighter, dimension-specific frontier is a real same-worker internal
win and preserves the SciPy output contract at the gauntlet sizes. It does not
catch SciPy; do not repeat envelope retuning without a new profile. Route the
remaining loss to lower-constant Bessel root generation, cached cross-order
recurrences, or a true heap-streamed global zero enumerator.
