# frankenscipy-8l8r1.129 Gaussian Filter Cache-Planned 2D Reflect

Date: 2026-06-20
Agent: cod-b / MistyBirch
Verdict: KEEP as a measured same-worker Rust speedup; residual SciPy loss remains.

## Lever

Specialize `gaussian_filter` for the common 2-D `BoundaryMode::Reflect`,
order-0, all-axes path. The fast path precomputes reflect source-index plans for
rows and columns once, then runs separable vertical and horizontal passes over
row chunks. Other modes, dimensions, orders, and axis selections stay on the
existing generic path.

This is deliberately not the previously rejected scalar row-contiguous
interior-tap peel. It targets the first-axis generic N-D filter overhead for the
2-D separable Gaussian case.

## SciPy Oracle

Command:

```bash
python3 docs/perf_oracle_ndimage.py
```

Result:

```text
scipy ndimage correlate 5x5 256x256: 910.53 us
scipy ndimage gaussian sigma2 256x256: 1465.23 us
```

SciPy Gaussian reference: `1.46523 ms`.

## Rust Baselines And Candidate Runs

Current baseline before edit, RCH worker `vmi1227854`:

```bash
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- \
  correlate_gaussian/gaussian_sigma2/256 --noplot
```

Result: `[2.7741 ms 2.8418 ms 2.9136 ms]`, ratio vs SciPy `1.94x slower`.

Candidate, RCH worker `vmi1152480`, first sample:

```text
[2.0409 ms 2.1316 ms 2.2291 ms]
```

Ratio vs SciPy: `1.45x slower`.

Candidate, RCH worker `vmi1152480`, repeat sample:

```text
[1.8608 ms 1.9680 ms 2.0797 ms]
```

Ratio vs SciPy: `1.34x slower`.

Clean `ae454655` baseline copied to the same `vmi1152480` worker and run with
an isolated target dir:

```bash
ssh -i ~/.ssh/contabo_vps_ed25519 -o BatchMode=yes root@109.205.181.92 \
  'cd /data/projects/.scratch/frankenscipy-cod-b-gaussian-baseline-vmi115-20260620 && \
   AGENT_NAME=MistyBirch \
   CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b-vmi115-baseline \
   cargo bench -p fsci-ndimage --bench ndimage_bench -- \
   correlate_gaussian/gaussian_sigma2/256 --noplot'
```

Result: `[3.1976 ms 3.2989 ms 3.4050 ms]`, ratio vs SciPy `2.25x slower`.

Same-worker Rust improvement on `vmi1152480`: `3.2989 / 1.9680 = 1.68x`.

Additional routing-only clean baseline on RCH worker `vmi1149989`:

```text
[5.5490 ms 5.8852 ms 6.2679 ms]
```

Ratio vs SciPy: `4.02x slower`. Not used as the keep/reject proof because the
candidate sample was not on this worker.

## Correctness And Build Gates

Focused crate check via RCH:

```bash
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo check -p fsci-ndimage --all-targets
```

Result: PASS on `hz1`. Pre-existing warnings remained in `fsci-interpolate` and
`crates/fsci-ndimage/src/bin/diff_geom.rs`.

Focused Gaussian unit tests via RCH:

```bash
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo test -p fsci-ndimage gaussian_filter --lib -- --nocapture
```

Result: PASS on `hz1`, `13 passed / 0 failed / 230 filtered`. This includes
`gaussian_filter_reflect_2d_fast_path_matches_generic_sequential_path`, which
checks the new 2-D Reflect specialization against the existing generic
sequential Gaussian helper.

Live SciPy Gaussian conformance, local because RCH worker `vmi1152480` does not
have SciPy installed:

```bash
AGENT_NAME=MistyBirch FSCI_REQUIRE_SCIPY_ORACLE=1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b-local \
  cargo test -p fsci-conformance --test diff_ndimage_gaussian_filter -- --nocapture
```

Result: PASS, `1 passed / 0 failed`.

RCH conformance attempt:

```text
FAILED before executing comparisons: ModuleNotFoundError: No module named 'scipy'
```

Final hygiene gates:

- PASS: `git diff --check`.
- PASS: `ubs crates/fsci-ndimage/src/lib.rs docs/NEGATIVE_EVIDENCE.md docs/GAUNTLET_RELEASE_SCORECARD.md docs/progress/perf-negative-results.md tests/artifacts/perf/frankenscipy-8l8r1.128-gaussian-cache-planned/EVIDENCE.md .beads/issues.jsonl`; exit 0, no criticals, broad pre-existing warning inventory in `lib.rs`.
- BLOCKED: `cargo fmt --check -p fsci-ndimage` still reports pre-existing formatting drift in `crates/fsci-ndimage/benches/ndimage_bench.rs` and `crates/fsci-ndimage/src/bin/diff_fourier.rs`. The touched `src/lib.rs` was formatted directly with `rustfmt --edition 2024`.
- BLOCKED: `cargo clippy -p fsci-ndimage --all-targets -- -D warnings` remains blocked before ndimage by pre-existing `fsci-linalg` dependency lints.

## Decision

KEEP. The fast path is a real same-worker improvement for the tracked
`gaussian_sigma2/256` workload: `1.68x` faster than the old path on
`vmi1152480`. It remains `1.34x` slower than SciPy, so the residual route is
deeper constants: vectorized row/column dot kernels, cache blocking between
passes, or a transposed scratch layout that improves the vertical pass without
reintroducing the rejected scalar border/interior split.
