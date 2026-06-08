# Reject: verified-delta packed-panel Stage 1

Bead: `frankenscipy-8l8r1.52`

## Lever Tried

Private opt-in Stage 1 packed-panel bidiagonalization prototype:

- fixed panel width: `16`
- deterministic Householder generation reused from the current Golub-Kahan route
- far trailing region snapshotted before and after each panel
- verified safe-Rust far update replayed the exact delta through the existing
  rank-k update loop

This preserved the current route ordering by keeping the public SVD/lstsq/pinv
entry points unchanged.

## Behavior Proof

RCH worker: `vmi1153651`

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1153651 rch exec -- cargo test -p fsci-linalg --release --lib --locked packed_panel_bidiag_stage1 -- --nocapture
```

Artifact:
`tests/artifacts/perf/2026-06-08-linalg-two-stage-packed-panel/proof_packed_panel_stage1_rch.txt`

Result:

- `packed_panel_bidiag_stage1_is_deterministic_for_fixed_input`: passed
- `packed_panel_bidiag_stage1_matches_current_route_on_128x64`: passed
- fixed-input reflectors and band matrix matched the current route bitwise
- reconstruction, Q orthogonality, and V orthogonality checks passed

Public behavior remained unchanged because no public route was kept. Ordering,
tie-breaking, rank thresholding, floating-point public operation order, and RNG
state are therefore preserved by restoration. The original public golden SHA
anchor remains:

```text
1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225
```

## Benchmark

Same-worker RCH comparison on `vmi1153651`:

Baseline artifact:
`tests/artifacts/perf/2026-06-08-linalg-two-stage-packed-panel/baseline_bidiag_large_reduction_perf_probe_rch.txt`

After artifact:
`tests/artifacts/perf/2026-06-08-linalg-two-stage-packed-panel/after_packed_panel_stage1_perf_probe_rch.txt`

```text
baseline_current_golub_kahan_ms=431.652279
baseline_digest=0x90cdd3f8f71ed2c1
packed_panel_stage1_ms=10628.935808
packed_panel_band_digest=0x22123c9126a1ef63
packed_panel_stage1_digest=0x9cb3c203a9202074
speedup=0.040611
```

## Decision

Reject. The verified-delta implementation proves the ordering contract, but its
far update is the wrong primitive: it encodes a full far rectangle delta as a
wide identity-ranked replay, making the Stage 1 path about `24.6x` slower than
the current reducer on the same worker.

Source was restored by patch. No linalg code from this rejected lever is kept.

## Next Primitive

Attack the actual algorithmic primitive next: compact WY-style Householder panel
accumulation that forms true narrow `Y/T` panel state while the panel is being
generated, then applies one cache-blocked `A22 -= V T (V^T A22)`-class safe-Rust
update. Target ratio: at least `1.35x` for the 1024x512 reduction probe before
public routing is considered.
