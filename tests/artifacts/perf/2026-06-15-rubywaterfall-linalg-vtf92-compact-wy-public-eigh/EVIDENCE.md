# `frankenscipy-vtf92` Baseline Evidence

## Target

Integrate compact-WY full-to-band reduction into the public native symmetric `eigh` route without re-entering the scalar dense Householder path. The complete route still needs band-to-tridiagonal eigenvectors plus the correct back-transform, so this bead remains open.

## Baseline

Source: `baseline_public_eigh_native_route_rch.txt`

- Worker: RCH `ovh-a`
- Command: `cargo test -j 1 -p fsci-linalg --lib public_eigh_native_route_perf_probe --release --locked -- --ignored --nocapture --test-threads=1`
- 400x400 routed public `eigh`: `99.425739 ms`; direct nalgebra `81.920655 ms`; speedup `0.823938x`; digest `0x0dbbde75b75c8612`
- 800x800 routed public `eigh`: `496.097807 ms`; direct nalgebra `560.444540 ms`; speedup `1.129706x`; digest `0xad8a7e5fa1980bfb`
- 1200x1200 routed public `eigh`: `1685.043645 ms`; direct nalgebra `2253.501546 ms`; speedup `1.337355x`; digest `0x181b3486089d0e4a`

## Routing Decision

The existing compact-WY helper surface can reduce a full symmetric matrix to band form, and `frankenscipy-ql7n6` already routed large `eig_banded(..., eigvals_only=false)` calls through native symmetric `eigh`. That is not enough to wire public `eigh` through full-to-band: without a complete band-to-tridiagonal/eigenvector transform, the route would add band reduction work and then perform a second dense Householder reduction.

`frankenscipy-r5sjf` is therefore a narrow child guard for the profile-backed 400x400 regression found by this baseline. `frankenscipy-vtf92` stays open for the deeper algorithmic primitive.
