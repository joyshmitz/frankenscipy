# fsci-stats psd_welch stack segment negative result

Bead: `frankenscipy-3zkix`
Agent: `OliveSnow`
Date: 2026-06-02 EDT

## Profile-backed target

Source profile: `tests/artifacts/perf/2026-06-03-stats-psd-plan-cache/reprofile_after_psd_plan_cache_broad_rch.txt`.

After the 128-point plan cache pass, the top stats row remained:

- `time_series/psd_welch/4096_w128_o64`: 751.08 us median on RCH worker `vmi1153651`

## One lever tested

For the cached 128-point Welch path, the tested source used a fixed `[f64; 128]` stack segment instead of allocating a `Vec<f64>` segment per Welch segment.

No external BLAS/LAPACK/XLA linkage and no unsafe code.

## Isomorphism proof

- Input validation order was unchanged.
- Public API, output length, output ordering, and empty/error behavior were unchanged.
- Segment order, frequency order, and sample order inside each frequency were unchanged.
- Floating-point arithmetic order was unchanged: each segment sample still computed `d * w`, then `re += s * cos` and `im -= s * sin` proceeded in the same order.
- No RNG, tie-breaking, or global-state surface exists in this routine.

## Golden proof

RCH `perf_stats golden` before/after output was byte-identical.

- Before sha256: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`
- After sha256: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`
- Comparison: `golden_cmp=identical`

## Benchmark gate

Focused RCH Criterion baseline:

- `psd_welch/4096_w128_o64`: 772.65 us median on `vmi1153651`

Focused RCH Criterion after:

- `psd_welch/4096_w128_o64`: 770.95 us median on `vmi1153651`

Result: 1.00x, indistinguishable from noise.

Score: `0.0` because there is no real measured win. The lever fails the Score>=2.0 keep gate.

## Verdict

Abandoned. Production `psd_welch` was restored to the pre-lever plan-cache source.
