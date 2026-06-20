# LSAP First-Scan Initialization Gauntlet

- Date: 2026-06-20
- Agent: cod-b / BlackThrush
- Bead: `frankenscipy-8l8r1.137`
- Crate: `fsci-opt`
- Target: `linear_sum_assignment/dense`
- Decision: KEEP the first-scan initialization specialization; REJECT the
  compact selected-row/column lists and remaining-template copy subvariants.

## Lever

The shortest augmenting path search previously reset the whole `path` vector to
`UNASSIGNED` and the whole `shortest_path_costs` vector to `INFINITY` before
immediately writing every remaining column during the first relaxation step.
The kept lever peels that first row scan: it initializes `path[col]` and
`shortest_path_costs[col]` directly from the start row, applies the existing
SciPy tie-break, and only enters the later relaxation loop if the first chosen
column is already assigned.

This is a cache/branch lever, not a numerical change. The primal-dual update,
augmenting path reconstruction, and `row4col[col] == UNASSIGNED` tie preference
are unchanged.

## Commands

- Invalid requested form, recorded as tooling negative evidence:
  `AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench --release`
- Valid per-crate Criterion baseline/final form:
  `AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-opt --bench optimize_bench -- linear_sum_assignment/dense --sample-size 10 --measurement-time 1 --warm-up-time 1`
- Local SciPy oracle:
  `AGENT_NAME=BlackThrush python3 ... scipy.optimize.linear_sum_assignment ...`

## Benchmark Evidence

Same-worker current-vs-final gate on `vmi1227854`:

| Workload / route | Median | Interval / value | Ratio / verdict |
| --- | ---: | ---: | --- |
| Restored current Rust `dense/500` | 20.320 ms | [19.471, 21.702] ms | baseline |
| Final first-scan Rust `dense/500` | 21.009 ms | [19.964, 22.147] ms | neutral: 0.97x baseline speed, intervals overlap |
| SciPy 1.17.1 `dense/500` | 18.906268 ms | local p50 | Rust remains 1.11x slower |
| Restored current Rust `dense/1000` | 176.03 ms | [164.60, 189.98] ms | baseline |
| Final first-scan Rust `dense/1000` | 124.20 ms | [114.81, 135.33] ms | KEEP: 1.42x faster than current |
| SciPy 1.17.1 `dense/1000` | 125.511679 ms | local p50 | Rust is 1.01x faster, parity/slight win |

Strict SciPy score for final source: `1/1/0`.

Same-worker internal current-vs-final score: `1/0/1`.

## Rejected Subvariants

| Subvariant | Worker | n=500 median | n=1000 median | Verdict |
| --- | --- | ---: | ---: | --- |
| Compact selected-row/selected-column lists | `vmi1152480` | 31.409 ms vs 26.604 ms baseline | 267.86 ms vs 243.82 ms baseline | REJECT: n=500 regressed 1.18x; n=1000 also worse/no significant win |
| Remaining-template `copy_from_slice` | `vmi1227854` | 22.854 ms vs 20.320 ms baseline | 161.26 ms vs 176.03 ms baseline | REJECT: n=500 significant regression; n=1000 no statistically reliable change |

Rejected subvariant score: selected lists `0/2/0`; remaining template `0/1/1`.

## Correctness And Gates

- PASS: rch focused assignment tests after warning fix:
  `cargo test -p fsci-opt linear_sum_assignment --lib -- --nocapture` =
  9 passed / 0 failed.
- PASS: rch per-crate all-targets check:
  `cargo check -p fsci-opt --all-targets`.
- PASS: rch no-deps clippy:
  `cargo clippy -p fsci-opt --all-targets --no-deps -- -D warnings`.
- PASS: rch release build:
  `cargo build --release -p fsci-opt`.
- PASS: local live SciPy conformance:
  `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_opt_linear_sum_assignment -- --nocapture`
  = 1 passed / 0 failed.
- PASS: touched-file formatting:
  `rustfmt --edition 2024 --config skip_children=true --check crates/fsci-opt/src/lib.rs`.
- PASS: `git diff --check`.
- BLOCKED/EXISTING: changed-file UBS exits nonzero on the existing broad
  `crates/fsci-opt/src/lib.rs` inventory: test-only panic callbacks, unwrap and
  assert inventory, direct indexing inventory, and allocation/cast warnings.
  No finding points at the changed first-scan SAP block.
- NOTE: `rustfmt --edition 2024 --check crates/fsci-opt/src/lib.rs` without
  `skip_children=true` follows pre-existing child module formatting drift in
  `crates/fsci-opt/src/linesearch.rs`, outside this patch.
- NOTE: local conformance compiled unrelated workspace crates and emitted
  pre-existing warnings in `fsci-special` and `fsci-interpolate`.

## Negative Evidence

- Do not rerun compact selected row/column lists in this SAP kernel without a
  new proof that the extra list maintenance beats the existing boolean scans.
- Do not rerun remaining-template copy initialization for this workload unless
  the n=500 regression is eliminated and n=1000 is significant on the same
  worker.
- Do not use `cargo bench --release`; Cargo rejects it. Criterion benches
  already use the optimized bench profile through `cargo bench`.
- The remaining n=500 loss is a lower-level constant-factor issue. Credible
  next routes are dense matrix storage/API specialization, row-slice layout
  removal without copying, or a more invasive LAPJV-style kernel.
