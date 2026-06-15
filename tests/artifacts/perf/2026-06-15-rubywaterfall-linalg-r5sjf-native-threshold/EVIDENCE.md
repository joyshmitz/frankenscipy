# `frankenscipy-r5sjf` Evidence

## Lever

Raise `PUBLIC_NATIVE_EIGH_MIN_DIM` from `256` to `512` so the measured 400x400 public `eigh` case uses the existing nalgebra route while 800x800 and 1200x1200 remain on the native safe-Rust route.

This is intentionally a threshold guard, not the compact-WY/two-stage implementation. `frankenscipy-vtf92` remains open for the deeper full-to-band plus band-to-tridiagonal/back-transform route.

## Baseline

Source: `../2026-06-15-rubywaterfall-linalg-vtf92-compact-wy-public-eigh/baseline_public_eigh_native_route_rch.txt`

- Worker: RCH `ovh-a`
- 400x400 routed public `eigh`: `99.425739 ms`
- 400x400 direct nalgebra reference: `81.920655 ms`
- Speedup before guard: `0.823938x`
- 800x800 routed digest: `0xad8a7e5fa1980bfb`
- 1200x1200 routed digest: `0x181b3486089d0e4a`

## After

Source: `after_public_eigh_native_threshold_rch.txt`

- Worker: RCH `ovh-a`
- 400x400 routed public `eigh`: `81.959196 ms`
- 400x400 direct nalgebra reference: `74.295887 ms`
- Guarded-route max absolute drift: `0.0`
- 800x800 routed digest: `0xad8a7e5fa1980bfb`
- 1200x1200 routed digest: `0x181b3486089d0e4a`

At the guarded 400x400 size, the public routed time improved from `99.425739 ms` to `81.959196 ms` (`1.21x`) by avoiding the unprofitable native path.

## Proof

- Ordering/tie policy: unchanged. The only route change is the size threshold.
- Floating-point and RNG: no RNG added; existing deterministic route behavior preserved.
- Golden digest: `eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a`.
- Large native digests unchanged for 800x800 and 1200x1200.
- Score: `Impact 2.0 * Confidence 4.0 / Effort 1.0 = 8.0`.

## Gates

- `rch exec -- cargo check -j 1 -p fsci-linalg --lib --locked`
- `rch exec -- cargo clippy -j 1 -p fsci-linalg --lib --no-deps --locked -- -D warnings`
- `cargo fmt -p fsci-linalg -- --check`
- `ubs crates/fsci-linalg/src/lib.rs crates/fsci-linalg/src/bin/diff_expm_frechet.rs .skill-loop-progress.md .beads/issues.jsonl`

The first clippy transcript records the local `type_complexity` failure in `expm_frechet`; the follow-up transcript records the passing run after introducing the `DenseMatrix` alias.
