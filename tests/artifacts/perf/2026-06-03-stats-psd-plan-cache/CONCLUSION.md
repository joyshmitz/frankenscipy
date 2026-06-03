# fsci-stats psd_welch 128-point plan cache

Bead: `frankenscipy-jznkb`
Agent: `OliveSnow`
Date: 2026-06-02 EDT

## Profile-backed target

Source profile: `tests/artifacts/perf/2026-06-03-stats-qmc-discrepancy/reprofile_after_qmc_broad_rch.txt`.

After the QMC discrepancy pass, the shifted top stats row was:

- `time_series/psd_welch/4096_w128_o64`: 1.1153 ms median on RCH worker `vmi1153651`

The prior segment-buffer reuse lever was rejected in `tests/artifacts/perf/2026-06-03-stats-psd-segment-reuse/` because it regressed 908.64 us to 995.07 us.

## One lever kept

`psd_welch` now reuses an immutable safe-Rust `OnceLock` plan for the profiled 128-point Welch case. The plan contains the same Hann window and `(cos, sin)` twiddle table that the function previously rebuilt on every call. Other window sizes still construct the same per-call plan.

No external BLAS/LAPACK/XLA linkage and no unsafe code.

## Isomorphism proof

- Input validation order is unchanged: empty data, zero window, invalid `fs`, overlap clamp, hop, and frequency count keep the same order.
- Public API, output length, output ordering, and empty/error behavior are unchanged.
- For the 128-point path, Hann coefficients are computed by the same expression and iteration order as before, then stored immutably.
- Twiddles are computed by the same `(k, n)` nested order and the same `angle.cos()` / `angle.sin()` expressions as before, then stored immutably.
- Segment order, frequency order, and sample order inside each frequency are unchanged.
- Floating-point accumulation order is unchanged: `d * w` is rounded into the segment value, then `re += s * cos` and `im -= s * sin` proceed in the same order.
- No RNG, tie-breaking, or global mutable numerical state is introduced; the `OnceLock` value is deterministic and immutable after initialization.

## Golden proof

RCH `perf_stats golden` before/after output was byte-identical.

- Before sha256: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`
- After sha256: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`
- Comparison: `golden_cmp=identical`

## Benchmark gate

Focused RCH Criterion baseline:

- `psd_welch/4096_w128_o64`: 985.45 us median on `vmi1156319`

Focused RCH Criterion after:

- `psd_welch/4096_w128_o64`: 764.89 us median on `vmi1153651`

Result: 1.29x faster.

Score: `5.2` = Impact 1.29 x Confidence 0.8 / Effort 0.2. The lever passes the Score>=2.0 keep gate.

## Validation

- `rch exec -- cargo check -p fsci-stats --lib --locked`: pass.
- `rch exec -- cargo clippy -p fsci-stats --lib --bin perf_stats --bench stats_bench --locked -- -D warnings`: pass.
- `cargo fmt -p fsci-stats --check`: pass.
- `ubs crates/fsci-stats/src/lib.rs`: exit 0, 0 critical, existing broad warning inventory remains.
- `rch exec -- cargo test -p fsci-stats --lib --locked -- --nocapture`: attempted, then stopped after extended runtime in existing `tests::at_risk_continuous_pdfs_integrate_to_one`; captured output shows no failures before stop.

## Verdict

Kept. The cached 128-point plan removes repeated deterministic setup work from the profiled Welch case while preserving bit-identical output.
