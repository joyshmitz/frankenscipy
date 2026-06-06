# Stage 4a Keep: Column-Major Right Householder Workspace

Bead: `frankenscipy-ox9ly`

## Target

Continue the profile-backed full-rank rectangular SVD chain by reducing private
Golub-Kahan bidiagonalization cost before the full DLABRD/compact-WY blocked
panel implementation.

This lever optimizes the current right-reflector trailing updates. Public
`svd`, `svdvals`, `lstsq`, and `pinv` remain unwired and continue to use the
existing `safe_svd` route.

## Coordination

Agent Mail reservation attempts for the linalg file and artifact directory
returned `Resource is temporarily busy` even after the prior stale lease passed
`2026-06-06T01:20:06Z`. Work proceeded under the active Beads claim for
`frankenscipy-ox9ly`, with no edits to unrelated peer-owned files.

## Lever

`apply_householder_right` previously computed one row at a time. For a
column-major `DMatrix`, that strided across memory for each dot/update.

The kept change:

- adds `apply_householder_right_with_workspace`,
- accumulates all row dots by streaming down each reflector column,
- applies the same per-row scale back by streaming down each column,
- reuses one dot workspace across all right reflectors in the bidiagonal
  reduction, and
- preserves each row's dot-product summation order and every scalar update.

No unsafe code and no C BLAS/LAPACK linkage.

## Baseline And After

RCH same-worker private A/B probe on `ts1`:

```text
cargo test -p fsci-linalg --release --lib bidiag_right_workspace_perf_probe --locked -- --ignored --nocapture
```

Output:

```text
BIDIAG_RIGHT_WORKSPACE_PERF_BEGIN
shape=1024x512
rowwise_ms=1369.417561
workspace_ms=272.149569
rowwise_digest=0x90cdd3f8f71ed2c1
workspace_digest=0x90cdd3f8f71ed2c1
speedup=5.031856
BIDIAG_RIGHT_WORKSPACE_PERF_END
```

The earlier workspace-only private probe on `ts1` printed
`elapsed_ms=198.897163` with the same digest `0x90cdd3f8f71ed2c1`; the A/B
probe above is the acceptance evidence because it compares rowwise and workspace
implementations in the same release test binary.

Public Criterion guard on `ts1` after this private lever:

```text
cargo bench -p fsci-linalg --bench linalg_bench --locked -- 'lstsq/512x256|pinv/512x256' --sample-size 10 --warm-up-time 1 --measurement-time 3

lstsq/512x256  [85.649 ms 86.398 ms 87.104 ms]
pinv/512x256   [90.158 ms 91.000 ms 91.880 ms]
```

This is in the existing Stage 2/3 public guard band. No public speedup is claimed
because the private bidiagonal route remains unwired.

## Proof

Focused bit identity:

```text
cargo test -p fsci-linalg --lib bidiag_right_workspace_matches_rowwise_reference_bits --locked -- --nocapture
```

Result on `ts1`: `1 passed; 0 failed`.

Full focused bidiagonal suite:

```text
cargo test -p fsci-linalg --lib bidiag --locked -- --nocapture
```

Result on `ts1`: `16 passed; 0 failed; 2 ignored`.

Golden payloads remained unchanged:

- Stage 1 bidiagonal payload SHA-256:
  `c9aa3bec7c677f420ba88d2676e675423702e20ecf4239186c9059732e2c4ad4`
- Stage 3 thin-SVD payload SHA-256:
  `086c0c88cc52d431b9a497f7da60d64a25f2acde49ab0b387f6c03f44547fc73`
- Public SVD/lstsq/pinv payload SHA-256:
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`

Public golden guard:

```text
cargo test -p fsci-linalg --lib public_svd_lstsq_pinv_golden_payload --locked -- --nocapture
```

Result on `ts1`: `1 passed; 0 failed`.

## Isomorphism

- Ordering: public singular-value ordering remains on `safe_svd`; private
  bidiagonal factor ordering is bit-identical to the rowwise reference.
- Tie-breaking: unchanged; reflector generation order and scalar operations per
  row are preserved.
- Floating point: rowwise and workspace reductions have identical reflector,
  diagonal, superdiagonal, and bidiagonal bits in the focused proof; public
  golden SHA is unchanged.
- RNG: none.
- Errors and certificates: public routes, rank thresholds, fallback
  certificates, and error classes remain unchanged because this private
  primitive is still unwired.

## Validation

- `cargo fmt -p fsci-linalg --check`: passed
- RCH `cargo check -p fsci-linalg --all-targets --locked`: passed on `ts1`
- RCH `cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`:
  passed on `ts1`
- `ubs crates/fsci-linalg/src/lib.rs`: exit `0`, zero critical issues; existing
  warning inventory remains in the large file.

## Verdict

Keep.

Score: `10.0 = Impact 4 * Confidence 5 / Effort 2`.

Next deeper primitive under `frankenscipy-ox9ly`: DLABRD-style paneling with
compact left/right reflector accumulators and GEMM-backed far trailing updates.
