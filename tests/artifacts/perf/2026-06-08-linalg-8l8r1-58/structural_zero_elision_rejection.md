# Structural-zero elision rejection

Bead: `frankenscipy-8l8r1.58`
Date: 2026-06-08

## Lever

Skip writing structural zeros in the fused Golub-Kahan reducer after each left and right reflector application.

## Proof

`proof_structural_zero_elision_bits_rch.txt` passed:

```text
test tests::bidiag_fused_step_matches_workspace_reference_bits ... ok
```

The digest stayed unchanged in the after benchmark:

```text
digest=0x90cdd3f8f71ed2c1
```

## Benchmark

Same-worker comparison on `vmi1167313`:

```text
baseline elapsed_ms=434.090993
after    elapsed_ms=571.395627
ratio    0.7597x
```

## Verdict

Rejected. The lever preserved behavior but regressed the reduction probe and scores below the keep threshold. Do not repeat this fused-step cleanup family; continue with the packed/two-stage communication-avoiding primitive.
