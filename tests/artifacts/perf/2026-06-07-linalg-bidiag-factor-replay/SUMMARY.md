# Bidiag Factor Replay Workspace Rejection

Bead: `frankenscipy-8l8r1.44`

## Target

Profile-backed follow-up after acceptance-certificate and diagonal-output
micro-levers failed.

Fresh evidence:

- Public route baseline on `ts1`: `lstsq=74.970714 ms`, `pinv=77.441517 ms`
- Stage probe on `ts1`: `thin_bidiag_factor_replay` `replay_ms=250.248466`
- Focused factor replay baseline on `vmi1149989`: `replay_ms=300.625236`
- Public golden payload baseline passed on `ts1`

## Lever Tried

Reuse one `right_dot_workspace` while replaying right Householder reflectors in
`deterministic_thin_svd_from_reduction_parts`, replacing the per-reflector
allocation inside `apply_householder_right`.

This preserved reflector order, row/column traversal, floating-point operations,
singular-value ordering, public behavior, and RNG absence. It only changed
workspace allocation lifetime.

## Proof

RCH focused replay proof passed:

```text
command: RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-linalg --release --lib --locked thin_bidiag_reflector_replay_matches_dense_product_reference -- --nocapture
worker: ts1
result: ok
```

After-run digest stayed identical:

```text
reduction_digest=0x90cdd3f8f71ed2c1
reference_digest=0x22223a463752097f
replay_digest=0x8f521a39638fb520
```

## Rebench

Same-worker comparison against the prior `ts1` stage probe:

```text
baseline replay_ms=250.248466
after    replay_ms=251.184854
```

The lever did not improve the target and slightly regressed the focused replay
probe.

## Decision

Rejected. Source restored; `git diff -- crates/fsci-linalg/src/lib.rs` is empty.

Score: `0.0`.

Next primitive should move past replay allocation lifetime and attack the actual
reflector application work: cache-oblivious/block reflector replay or two-stage
communication-avoiding bidiagonalization with explicit public golden proof.
