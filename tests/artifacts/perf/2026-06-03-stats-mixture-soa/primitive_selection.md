# QMC 2D Discrepancy SoA Primitive Selection

Bead: `frankenscipy-rh43r`

Target: `qmc_discrepancy/mixture/512x2`

Profile evidence:
- Post-PSD stats reprofile kept QMC 2D discrepancy rows prominent.
- Fresh focused RCH baseline for this bead: `[276.30 us, 282.08 us, 287.92 us]` in `baseline_mixture_512x2_rch.txt`.

Rejected prior levers:
- The broad 2D invariant cache was already shipped.
- Direct `delta.powi(2)` to `delta * delta` was rejected twice as a regression and must not be repeated.

Candidate primitive:
- Split cached 2D discrepancy point data from array-of-structs into separate row-major arrays for `x0`, `x1`, `centered0`, `centered1`, `abs0`, and `abs1`.
- The alien-graveyard match is SoA/columnar storage for cache-friendly numeric scans and explicit sequential streams.

Behavior contract:
- Public validation order and error surfaces remain unchanged.
- Row order remains ascending input row order.
- Pair loop order remains `i` outer, `j` inner, both ascending.
- Coordinate order remains coordinate 0 then coordinate 1.
- Formula term order, including `delta.powi(2)`, remains unchanged.
- Golden QMC output must remain byte-identical: `1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf`.
- No RNG, tie-breaking, or global state exists in these discrepancy routines.

Score target:
- Impact: 2.0. The target is a visible hot row, but this is a layout-only lever after prior large QMC optimization.
- Confidence: 3.0. The change preserves arithmetic order, but allocation and cache effects may be noisy.
- Effort: 2.0. One local data layout change plus proof.
- Score: `3.0 = 2.0 * 3.0 / 2.0`.

Decision:
- Trial the lever only if the diff stays layout-only.
- Reject and restore source if golden changes, validation fails, or focused RCH timing lacks a real win.
