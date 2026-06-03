# fsci-stats psd_welch segment-buffer reuse negative result

Bead: `frankenscipy-8l8r1.3`
Duplicate bead closed with same result: `frankenscipy-8l8r1.4`
Agent: `OliveSnow`
Date: 2026-06-02 EDT

## Profile-backed target

Source profile: `tests/artifacts/perf/2026-06-03-stats-qmc-discrepancy/reprofile_after_qmc_broad_rch.txt`.

After the QMC discrepancy pass, the shifted top stats row was:

- `time_series/psd_welch/4096_w128_o64`: 1.1153 ms median on RCH worker `vmi1153651`

## One lever tested

`psd_welch` allocated a fresh windowed `Vec<f64>` for each overlapping Welch segment. The tested lever allocated that segment buffer once and refilled it per segment, preserving the existing behavior that `d * w` is computed once per segment sample before every frequency uses it.

No external BLAS/LAPACK/XLA linkage and no unsafe code.

## Isomorphism proof

- Validation order was unchanged: empty input, zero window, invalid `fs`, overlap clamp, and hop computation stayed before the segment loop.
- Public API, error behavior, output length, and output ordering were unchanged.
- Segment order was unchanged.
- Frequency order was unchanged.
- Sample order inside each frequency was unchanged.
- Floating-point arithmetic for each segment sample was unchanged: compute `d * w`, then use the resulting `s` in `re += s * cos` and `im -= s * sin`.
- No RNG, tie-breaking, or global-state surface exists in this routine.

## Golden proof

RCH `perf_stats golden` before/after output was byte-identical.

- Before sha256: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`
- After sha256: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`
- Comparison: `golden_cmp=identical`

## Benchmark gate

Focused RCH Criterion baseline:

- `psd_welch/4096_w128_o64`: 908.64 us median on `vmi1149989`

Focused RCH Criterion after:

- `psd_welch/4096_w128_o64`: 995.07 us median on `vmi1156319`

Result: 0.91x, a regression.

Score: `0.0` because the impact is negative. The lever fails the Score>=2.0 keep gate.

## Verdict

Abandoned. Production `psd_welch` was restored to the pre-lever source.

Do not retry simple segment-buffer reuse for the current twiddle-table implementation. The DFT inner loop dominates; this allocation reduction does not buy enough locality or allocator relief and measured slower in the focused RCH run.
