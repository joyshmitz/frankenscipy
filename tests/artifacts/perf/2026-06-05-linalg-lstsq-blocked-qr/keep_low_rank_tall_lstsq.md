# fsci-linalg low-rank tall lstsq keep

## Lever

Ship a deterministic safe-Rust compact-SVD least-squares path for large
low-rank tall matrices. For inputs that pass the existing low-rank tall
factorization gate, write `A = Q C`, compute the SVD of compact `C`, solve
`x = V Sigma^+ U^T Q^T b`, and publish the compact singular values padded with
zeros to the current public length `cols`.

The route is active only for tall large finite inputs (`rows >= 2 * cols`,
`cols >= 512`) with rank <= 16, a full reconstruction residual proof, and a
clear rank-boundary gap. Full-rank, ambiguous, non-finite, ragged, custom
negative/NaN-cond, and high-residual cases fall back to the existing full SVD.

## Baseline

Artifact: `baseline_cc8hu_current_rch.txt`

Worker: `ts2`

SHA-256:

```text
fddcfe8667678f7812f70ddb1db7dbb6f8ae1a14676efbe210fa2357abb12c41
```

Rows:

| operation | baseline |
| --- | ---: |
| `lstsq m=2000 n=1000` | `3563.8 ms` |
| `pinv  m=2000 n=1000` | `76.3 ms` |
| `lstsq m=3000 n=1500` | `20620.4 ms` |
| `pinv  m=3000 n=1500` | `223.1 ms` |
| `spd-solve n=1024` | `94.1 ms` |
| `spd-solve n=2048` | `353.5 ms` |

## After

Artifact: `after_low_rank_tall_lstsq_rch.txt`

Worker: `ts2`

SHA-256:

```text
a0c07d3921931b3098f90ce8df6d4133f72fcc5a63c43b216d4608ca2928693b
```

Rows:

| operation | baseline | after | ratio |
| --- | ---: | ---: | ---: |
| `lstsq m=2000 n=1000` | `3563.8 ms` | `66.3 ms` | `53.75x` |
| `pinv  m=2000 n=1000` | `76.3 ms` | `80.1 ms` | control |
| `lstsq m=3000 n=1500` | `20620.4 ms` | `162.7 ms` | `126.74x` |
| `pinv  m=3000 n=1500` | `223.1 ms` | `200.6 ms` | control |
| `spd-solve n=1024` | `94.1 ms` | `98.0 ms` | control |
| `spd-solve n=2048` | `353.5 ms` | `369.5 ms` | control |

## Behavior proof

Golden payload: `golden_lstsq_low_rank_tall_payload.txt`

Golden SHA-256:

```text
1235ac7505789813866fa04ed2611a86399973b40cc54da464f3e83e2d688c82
```

The golden test checks:

- small low-rank tall input against the existing full SVD reference:
  rank `3`, max `x` absolute difference `1.51014774019131437e-8`, max singular
  value absolute difference `1.65848885038321390e-15`.
- the only displayed singular-value difference is the full SVD numerical tail
  `1.65848885038321390e-15` becoming exact zero after the reconstruction-
  certified low-rank route; this is below the existing `1e-7` test envelope and
  does not change rank.
- production-gated `1024x512` public route returns rank `3`, singular-value
  vector length `512`, `SVDFallback` certificate action, reciprocal condition
  estimate `0.0`, and deterministic selected `x`/singular entries.
- ordering is fixed: input columns are scanned in order, modified Gram-Schmidt
  uses fixed two-pass reorthogonalization, compact singular values retain the
  SVD order, trailing singular values are appended zeros, and no RNG is used.

RCH proof commands:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-linalg --lib lstsq --locked -- --nocapture
RCH_FORCE_REMOTE=1 rch exec -- cargo check -p fsci-linalg --all-targets --locked
RCH_FORCE_REMOTE=1 rch exec -- cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings
cargo fmt -p fsci-linalg --check
ubs crates/fsci-linalg/src/lib.rs crates/fsci-linalg/src/bin/perf_lstsq_probe.rs
```

Artifacts:

- `cargo_test_lstsq_low_rank_tall_full_rch.txt`: `10 passed; 0 failed`
- `cargo_check_lstsq_low_rank_tall_rch.txt`: exit `0`
- `cargo_clippy_lstsq_low_rank_tall_rch.txt`: exit `0`
- `ubs_lstsq_low_rank_tall.txt`: exit `0`

## Score

Score: `8.3 = Impact 5 * Confidence 5 / Effort 3`

Decision: KEEP. The shifted profile target's `lstsq 3000x1500` row moves from
`20.620s` to `0.163s` on the same worker with golden-output SHA and focused
singular-value proof. General full-rank rectangular SVD still needs the blocked
Householder bidiagonalization primitive as a follow-up.
