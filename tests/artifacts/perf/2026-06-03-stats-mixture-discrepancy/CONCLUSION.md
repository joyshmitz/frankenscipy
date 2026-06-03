# Mixture Discrepancy Duplicate Negative Result

Bead: `frankenscipy-8l8r1.6`

Verdict: abandoned as a duplicate of `frankenscipy-8l8r1.5`.

Target: `qmc_discrepancy/mixture/512x2`.

Fresh focused RCH baseline:

- `baseline_mixture_512x2_rch.txt`
- Worker: `vmi1293453`
- Criterion time: `[270.19 us, 280.60 us, 287.66 us]`

Duplicate candidate:

- Replace `delta.powi(2)` with `delta * delta` in the 2D mixture pair loop.

Behavior proof:

- Golden before: `golden_qmc_before.txt`
- Golden after: `golden_qmc_after.txt`
- SHA256 stayed `1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf`.
- Byte comparison was identical.

Performance result:

- `after_mixture_512x2_rch.txt`
- Worker: `vmi1264463`
- Criterion time: `[680.56 us, 826.10 us, 949.47 us]`

The candidate regressed and was already rejected by `frankenscipy-8l8r1.5`; no source change was kept. Score: `0.0`.
