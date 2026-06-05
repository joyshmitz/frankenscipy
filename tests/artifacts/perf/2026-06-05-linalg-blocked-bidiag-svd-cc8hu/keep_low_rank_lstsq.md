# Low-Rank Tall `lstsq` Keep - `frankenscipy-cc8hu`

## Lever

Route large tall low-rank `lstsq` inputs through the existing deterministic low-rank tall factorization before the dense nalgebra SVD path.

The fast path is deliberately narrow:

- `rows >= 2 * cols`
- `cols >= LOW_RANK_PINV_MIN_COLS`
- rectangular finite rows and finite RHS
- low-rank factorization rank below `LOW_RANK_PINV_MAX_RANK`
- compact SVD rank is nonzero, below full column rank, and separated from the threshold boundary

Accepted inputs solve `x = C^+ Q^T b` from the rank-revealing `A = Q C` factorization, with compact singular values padded to the public `cols` length. Rejected inputs fall through to the previous dense SVD implementation.

## Baseline

Fresh current-head RCH baseline on `ts1`:

- artifact: `baseline_current_head_rch.txt`
- SHA-256: `7d9d4cf8c08fbd2e181dcba33840ec8b1b267d527d44acea1e2744bbd9ac21c9`
- `lstsq m=2000 n=1000`: `4696.8 ms`
- `lstsq m=3000 n=1500`: `20151.3 ms`

Same-worker current-head `ts2` comparison from the immediately preceding low-rank `pinv` artifact:

- artifact: `../2026-06-05-linalg-lstsq-blocked-qr/after_low_rank_tall_pinv_confirm_rch.txt`
- SHA-256: `4bcea915ea063980fdb5eb6f015f9290d1420237e620e37da376d510479c7418`
- `lstsq m=2000 n=1000`: `3565.6 ms`
- `lstsq m=3000 n=1500`: `18421.5 ms`

## After

RCH after benchmark on `ts2`:

- artifact: `after_low_rank_lstsq_rch.txt`
- SHA-256: `aee93797a8fa578cc78affc510389955533f206f0cbe6ce3097fca2509dc8b49`
- `lstsq m=2000 n=1000`: `64.2 ms`
- `lstsq m=3000 n=1500`: `161.0 ms`

Same-worker deltas:

- `m=2000 n=1000`: `3565.6 ms -> 64.2 ms`, `55.5x` faster
- `m=3000 n=1500`: `18421.5 ms -> 161.0 ms`, `114.4x` faster

## Behavior Proof

- Ordering: row, column, basis, and singular-vector loops are deterministic and do not reorder accepted inputs relative to the compact factorization.
- Tie-breaking: the fast path rejects ambiguous threshold boundaries, preserving the previous dense SVD path where rank ties may matter.
- Floating point: accepted low-rank inputs compute the same minimum-norm SVD solution through `A = Q C`; compact singular values are padded with exact zero tails because the proven matrix rank is below `cols`.
- RNG: no random sampling or random state is introduced.
- Public observables: `rank`, `singular_values` length/order, `SVDFallback` certificate action, shape, empty residual behavior for rank-deficient overdetermined systems, and error/fallback paths are covered by focused tests.
- Golden payload SHA-256: `1235ac7505789813866fa04ed2611a86399973b40cc54da464f3e83e2d688c82`.

Proof commands:

- RCH `cargo check -p fsci-linalg --all-targets --locked`, artifact SHA-256 `2a26e94c140222093ab7c2cbc76f1d01aeb340a5b4a298dfbd732155ba9277ad`
- RCH `cargo test -p fsci-linalg --lib lstsq_low_rank_tall --locked -- --nocapture`, artifact SHA-256 `25858732a9957dd64eccd37a6c366f330676304808fe6e3fd2696b766e524775`
- RCH `cargo test -p fsci-linalg --lib lstsq --locked -- --nocapture`, artifact SHA-256 `5eca226c37f73792de652c8aba19e382befc901b6759530fa7c6025c8b51aead`
- RCH `cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`, artifact SHA-256 `64b954255e3cf1f03213a34b3ad72327f35a4c0113ec93bbed43e636b737ec3b`
- `cargo fmt -p fsci-linalg --check`
- `ubs crates/fsci-linalg/src/lib.rs`, artifact SHA-256 `b4b09a38c6abf2e2158a23f7ca2e682812d465050c61a78f340a3b42a2e41572`

## Score

`8.3 = Impact 5 * Confidence 5 / Effort 3`

The same-worker 3000x1500 row improves by `114.4x`, all proof gates passed, and the fallback guard is narrow enough to leave the existing dense SVD path authoritative outside the low-rank shape.

## Residual

This does not replace the general full-rank dense SVD path. The next deeper primitive remains a safe-Rust blocked Householder bidiagonalization with compact reflectors, GEMM-backed trailing updates, and bidiagonal singular-vector reconstruction for full-rank/default SVD-family workloads.
