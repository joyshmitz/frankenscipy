# fsci-spatial pdist Chebyshev Wide SIMD Evidence

- Date: 2026-06-20
- Agent: cod-a / BlackThrush
- Bead: `frankenscipy-4tkgx`
- Decision: KEEP
- Lever: replace the generic Chebyshev helper's scalar iterator/fold with an
  8-lane `std::simd` absolute-difference max plus an explicit NaN mask. This
  keeps the scalar NaN-propagating max-fold contract and accelerates every
  non-dim4 Chebyshev batch route that calls the helper.

## Commands

- Same-worker baseline:
  `AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-spatial --bin perf_pdist_sweep`
  (`vmi1227854`, log `baseline_pdist_sweep_rch.txt`)
- Same-worker candidate:
  `AGENT_NAME=BlackThrush RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-spatial --bin perf_pdist_sweep`
  (`vmi1227854`, log `candidate_pdist_sweep_rch.txt`)
- SciPy oracle:
  local SciPy 1.17.1 / NumPy 2.4.3, log `scipy_oracle_pdist_sweep.txt`
- Criterion:
  `AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-spatial --bench spatial_bench -- pdist_highdim/chebyshev/n1000_d64 --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
  (`vmi1264463`, log `criterion_pdist_highdim_chebyshev_n1000_d64_rch.txt`)

## Same-Worker Target Delta

| Workload | Baseline Rust | Candidate Rust | Self delta | SciPy oracle | Candidate vs SciPy |
| --- | ---: | ---: | ---: | ---: | --- |
| `pdist/chebyshev/n512/d16` | 1.735 ms | 0.576 ms | 3.01x faster | 0.560 ms | Rust 1.03x slower |
| `pdist/chebyshev/n512/d64` | 8.195 ms | 0.931 ms | 8.80x faster | 2.172 ms | Rust 2.33x faster |
| `pdist/chebyshev/n2048/d64` | 78.381 ms | 10.575 ms | 7.41x faster | 40.949 ms | Rust 3.87x faster |

The d16 row is now a tiny residual SciPy loss instead of the prior 3.36x loss.
The d64 rows flip to clear SciPy wins.

## Full Candidate Sweep vs SciPy

| Workload | Candidate Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | --- |
| `pdist/euclidean/n512/d4` | 0.162 ms | 0.307 ms | Rust 1.90x faster |
| `pdist/cityblock/n512/d4` | 0.106 ms | 0.196 ms | Rust 1.85x faster |
| `pdist/sqeuclidean/n512/d4` | 0.110 ms | 0.177 ms | Rust 1.61x faster |
| `pdist/chebyshev/n512/d4` | 0.154 ms | 0.177 ms | Rust 1.15x faster |
| `pdist/euclidean/n512/d16` | 0.476 ms | 0.761 ms | Rust 1.60x faster |
| `pdist/cityblock/n512/d16` | 0.424 ms | 0.623 ms | Rust 1.47x faster |
| `pdist/sqeuclidean/n512/d16` | 0.431 ms | 0.547 ms | Rust 1.27x faster |
| `pdist/chebyshev/n512/d16` | 0.576 ms | 0.560 ms | Rust 1.03x slower |
| `pdist/euclidean/n512/d64` | 0.633 ms | 2.210 ms | Rust 3.49x faster |
| `pdist/cityblock/n512/d64` | 0.556 ms | 2.748 ms | Rust 4.94x faster |
| `pdist/sqeuclidean/n512/d64` | 0.709 ms | 2.039 ms | Rust 2.88x faster |
| `pdist/chebyshev/n512/d64` | 0.931 ms | 2.172 ms | Rust 2.33x faster |
| `pdist/euclidean/n4096/d4` | 11.336 ms | 50.667 ms | Rust 4.47x faster |
| `pdist/cosine/n4096/d4` | 11.345 ms | 48.631 ms | Rust 4.29x faster |
| `pdist/chebyshev/n2048/d64` | 10.575 ms | 40.949 ms | Rust 3.87x faster |
| `pdist/cityblock/n2048/d64` | 5.872 ms | 48.429 ms | Rust 8.25x faster |

Strict candidate sweep score versus the SciPy oracle: `15/1/0`.

## Correctness and Gates

- PASS: focused NaN-fold bit identity:
  `cargo test -p fsci-spatial pdist_wide_chebyshev_matches_scalar_nan_fold --lib -- --nocapture`
  via rch on `vmi1227854`.
- PASS: `cargo check -p fsci-spatial --all-targets` via rch on `vmi1153651`.
- PASS: `cargo clippy -p fsci-spatial --all-targets --no-deps -- -D warnings`
  via rch on `vmi1149989`.
- PASS: `cargo fmt --check -p fsci-spatial`.
- PASS: local live SciPy conformance:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-conformance --test diff_spatial_pdist_cdist -- --nocapture`.
- BLOCKED/INFRA: the same conformance test failed on rch worker `hz2` before
  comparison because that worker image lacks the Python `scipy` module.
- PASS: focused Criterion bench runs and includes
  `pdist_highdim/chebyshev/n1000_d64` with time interval
  `[28.945 ms 36.717 ms 42.736 ms]`.
- PASS: `git diff --check`.
- BLOCKED/EXISTING: changed-file `ubs` exits 1 on the existing broad
  `fsci-spatial` inventory: one old test `panic!` plus older unwrap, assert,
  direct-indexing, and allocation warnings. It reports no unsafe blocks, no
  formatter issue, and no clippy/check/test build failure for this patch.

## Negative Evidence

Do not retry the scalar iterator/fold Chebyshev helper for d16/d64. The
remaining d16 row is only 1.03x slower than SciPy after the vector helper and
needs a deeper layout or across-pairs kernel if it is worth chasing. The d64
losses from `frankenscipy-i0ghz` are closed by this lever.
