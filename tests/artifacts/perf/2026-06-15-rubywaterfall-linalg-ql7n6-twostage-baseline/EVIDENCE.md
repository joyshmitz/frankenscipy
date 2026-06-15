# frankenscipy-ql7n6 Native Banded Eigenvector Evidence

## Verdict

Accepted a focused `ql7n6` production lever: for `eig_banded(..., eigvals_only=false)` at `n >= 128`, expand the lower-band matrix once and route eigenvectors through the already accepted safe-Rust `symmetric_eigh_native` path instead of the older scalar dense Householder branch.

This is not the full compact-WY public `eigh` integration. The remaining full-to-band production path is tracked by successor bead `frankenscipy-vtf92`.

## Timing

Same probe and shape family before/after:

| shape | bandwidth | before | after | speedup |
|---:|---:|---:|---:|---:|
| 128x128 | 32 | 16.803222 ms | 2.898347 ms | 5.7975x |
| 256x256 | 32 | 113.211171 ms | 16.648166 ms | 6.8002x |

Score: `Impact 4.0 * Confidence 4.0 / Effort 1.5 = 10.67`, keep.

## Isomorphism

The public contract is preserved for this route:

- `eig_banded` input validation and lower-band storage contract are unchanged.
- `eigvals_only=true` remains on the existing values-only branch.
- `n < 128` remains on the previous scalar branch.
- The new `n >= 128` eigenvector branch uses deterministic `symmetric_eigh_native`; no RNG or tolerance weakening was introduced.
- Dense reference eigenvalue digests were preserved for the perf probe shapes.

Observed after-route proof:

| shape | max value drift | eigenvector residual | values digest | dense values digest |
|---:|---:|---:|---:|---:|
| 128x128 | 7.560e-12 | 1.648e-12 | 0xd6dbb9200f65bd92 | 0x7bdbe4513b62a730 |
| 256x256 | 0.0 | 7.731e-12 | 0x09ed4d367faab431 | 0x09ed4d367faab431 |

The 128x128 candidate and dense digests differ because eigenvalue algorithms take different valid paths under the tolerance contract; the measured drift remains inside the existing tolerance. The 256x256 values digest matches the dense reference exactly for the probe fixture.

## Gates

RCH gates run crate-scoped:

- `cargo test -j 1 -p fsci-linalg --lib eig_banded --release --locked -- --nocapture --test-threads=1`
- `cargo test -j 1 -p fsci-linalg --lib symmetric_eigh_native_matches_nalgebra_and_timing --release --locked -- --nocapture --test-threads=1`
- `cargo test -j 1 -p fsci-linalg --lib svd_reroutes_rank_deficient_to_jacobi_and_reconstructs --release --locked -- --nocapture --test-threads=1`
- `cargo test -j 1 -p fsci-linalg --lib srht_transform_is_a_norm_preserving_embedding --release --locked -- --nocapture --test-threads=1`
- `cargo check -j 1 -p fsci-linalg --lib --locked`
- `cargo clippy -j 1 -p fsci-linalg --lib --no-deps --locked -- -D warnings`
- `cargo fmt -p fsci-linalg -- --check`

The clippy pass required local cleanup of pre-existing `needless_range_loop` findings in the linalg file so the changed crate could be gated cleanly.
