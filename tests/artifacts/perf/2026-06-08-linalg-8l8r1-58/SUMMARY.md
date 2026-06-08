# frankenscipy-8l8r1.58 closeout

Bead: `frankenscipy-8l8r1.58`
Target: packed/two-stage communication-avoiding bidiagonalization after fused-step keep.
Date: 2026-06-08

## Baseline

Fresh crate-scoped RCH baselines captured for the current fused-step reducer:

```text
worker=vmi1293453 elapsed_ms=192.095384 digest=0x90cdd3f8f71ed2c1
worker=vmi1167313 elapsed_ms=434.090993 digest=0x90cdd3f8f71ed2c1
public_route_worker=vmi1149989 routed_lstsq_ms=62.114397 routed_pinv_ms=66.772115
```

The invalid Criterion invocation in this directory is not benchmark evidence.

## Attempted Lever

Structural-zero write elision in the fused Golub-Kahan reducer.

Proof passed:

```text
test tests::bidiag_fused_step_matches_workspace_reference_bits ... ok
```

Same-worker benchmark on `vmi1167313` regressed:

```text
baseline elapsed_ms=434.090993
after    elapsed_ms=571.395627
ratio    0.7597x
digest   0x90cdd3f8f71ed2c1
```

The source was manually restored to the pre-lever reducer.

## Prior Packed-Panel Evidence

The true packed-panel Stage 1 lane was already rejected in
`tests/artifacts/perf/2026-06-08-linalg-two-stage-packed-panel/reject_packed_panel_stage1_verified_delta.md`:

```text
baseline_current_golub_kahan_ms=431.652279
packed_panel_stage1_ms=10628.935808
speedup=0.040611
```

That attempt preserved the fixed-input route but encoded a full far-rectangle delta as a wide replay, making it the wrong primitive.

## Decision

Close `.58` as rejected for this packed/two-stage family and route immediately to the next profile-backed primitive. This is not a ceiling: `frankenscipy-8l8r1.59` is opened and claimed for a guarded normal-equation thin-SVD route that bypasses the reducer only when public reconstruction, rank, tie, and sign gates accept; otherwise it falls back to the existing bidiagonal path.
