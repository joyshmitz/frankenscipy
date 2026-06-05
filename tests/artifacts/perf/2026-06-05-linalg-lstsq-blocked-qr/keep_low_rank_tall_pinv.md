# fsci-linalg low-rank tall pinv keep

## Lever

Ship a deterministic safe-Rust rank-revealing tall factorization for `pinv`
when the input is tall (`rows >= 2 * cols`), large (`cols >= 512`), finite,
numerically low-rank (`rank <= 16`), and passes a full reconstruction residual
gate. The fast path reduces `A = Q C`, computes the SVD of compact `C`, and
forms `C^+ Q^T`. All other inputs fall back to the existing full SVD path.

Default `lstsq` is intentionally not routed through this lever because its
public result includes singular values and rank threshold behavior that require
the deeper blocked-bidiagonalization primitive.

## Baseline

Artifact: `baseline_bidiag_current_head_rch.txt`

Worker: `ts2`

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo run --release -p fsci-linalg --bin perf_lstsq_probe --locked
```

SHA-256:

```text
849ab458850d19b7a02d2fd880a91b584e56e0cefcadeeb112b24497c9075fac
```

Rows:

| operation | baseline |
| --- | ---: |
| `lstsq m=2000 n=1000` | `3598.4 ms` |
| `pinv  m=2000 n=1000` | `3793.3 ms` |
| `lstsq m=3000 n=1500` | `21041.5 ms` |
| `pinv  m=3000 n=1500` | `20887.2 ms` |
| `spd-solve n=1024` | `87.2 ms` |
| `spd-solve n=2048` | `388.9 ms` |

## After

Artifact: `after_low_rank_tall_pinv_rch.txt`

Worker: `ts2`

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo run --release -p fsci-linalg --bin perf_lstsq_probe --locked
```

SHA-256:

```text
53722ffacc9461be74eade49f595779452f0b8f89cceb3c456c4789282f0b6ba
```

Rows:

| operation | baseline | after | ratio |
| --- | ---: | ---: | ---: |
| `lstsq m=2000 n=1000` | `3598.4 ms` | `3598.1 ms` | `1.00x` |
| `pinv  m=2000 n=1000` | `3793.3 ms` | `79.1 ms` | `47.96x` |
| `lstsq m=3000 n=1500` | `21041.5 ms` | `20834.3 ms` | `1.01x` |
| `pinv  m=3000 n=1500` | `20887.2 ms` | `218.0 ms` | `95.81x` |
| `spd-solve n=1024` | `87.2 ms` | `85.8 ms` | control |
| `spd-solve n=2048` | `388.9 ms` | `365.6 ms` | control |

Secondary confirmation artifact: `after_low_rank_tall_pinv_confirm_rch.txt`
on `ts2`, SHA-256
`4bcea915ea063980fdb5eb6f015f9290d1420237e620e37da376d510479c7418`,
confirmed `pinv m=3000 n=1500` at `219.4 ms`.

## Behavior proof

Golden payload: `golden_pinv_low_rank_tall_payload.txt`

Golden SHA-256:

```text
3f67147c7d93c71f778f47d57205c75c28cb0062c30130be28797b17360dde97
```

The golden test checks:

- small low-rank tall input: fast compact-SVD result against the existing full
  SVD `pinv` reference, same rank `3`, max absolute difference
  `9.23919560591457412e-6` under the existing relative tolerance envelope.
- production-gated `1024x512` low-rank tall input: public `pinv` route returns
  rank `3`, `SVDFallback` certificate action, shape `(1024,512)`, reciprocal
  condition estimate `0.0` for the rank-deficient spectrum, and
  `fallback_active=false`.
- deterministic ordering: columns are scanned in input order; basis vectors use
  modified Gram-Schmidt with fixed two-pass reorthogonalization; compact SVD
  singular modes retain nalgebra ordering; no RNG is introduced.
- fallback parity: ragged, non-finite, full-rank, ambiguous rank-boundary, large
  residual, and high-rank inputs return `None` from the fast probe and execute
  the previous full SVD path.

RCH proof commands:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-linalg --lib pinv --locked -- --nocapture
RCH_FORCE_REMOTE=1 rch exec -- cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings
cargo fmt -p fsci-linalg --check
ubs crates/fsci-linalg/src/lib.rs crates/fsci-linalg/src/bin/perf_lstsq_probe.rs
```

Artifacts:

- `cargo_test_pinv_low_rank_tall_full_rch.txt`: `13 passed; 0 failed`
- `cargo_clippy_low_rank_tall_pinv_rch.txt`: exit `0`
- `ubs_low_rank_tall_pinv.txt`: exit `0`

## Score

Score: `8.3 = Impact 5 * Confidence 5 / Effort 3`

Decision: KEEP. The profile target's `pinv` row moves from `20.887s` to
`0.218s` on the same worker while preserving the tested public rank,
Moore-Penrose result tolerance, certificate shape/action, ordering, and fallback
contract. The remaining `lstsq` row still requires the blocked Householder
bidiagonalization primitive.
