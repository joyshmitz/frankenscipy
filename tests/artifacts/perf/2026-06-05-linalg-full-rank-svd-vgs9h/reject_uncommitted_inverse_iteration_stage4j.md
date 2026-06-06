# Stage 4j Rejection: Uncommitted Tridiagonal Inverse-Iteration Backend Trial

Bead: `frankenscipy-z65tz`
Date: 2026-06-06T18:53:42-04:00

## Baseline

Fresh RCH baseline on `ts1`:

```text
cargo test -p fsci-linalg --release --lib thin_bidiag_factor_replay_perf_probe --locked -- --ignored --nocapture
test wall time: 4.47s
reduction_digest=0x90cdd3f8f71ed2c1
reference_ms=539.194886
replay_ms=272.890680
speedup=1.975864
u_max_abs_diff=5.10702591327572009e-15
vt_max_abs_diff=2.33146835171282873e-15
```

The factor assembly path is still much smaller than the full ignored-test wall time, so the profile-backed target remains the private bidiagonal SVD/Jacobi backend.

## Trial

Attempted one private backend lever:

- form the existing bidiagonal Gram tridiagonal diagonals directly,
- compute tridiagonal eigenvalues,
- recover deterministic vectors via shifted inverse iteration with Gram-Schmidt against prior vectors,
- retain the dense Jacobi backend as the proof reference/fallback.

## Proof Result

The proof was invalidated before a meaningful correctness or speed decision:

```text
RCH ts1 proof attempt 1:
cargo test -p fsci-linalg --release --lib bidiag_svd_tridiagonal_inverse_matches_jacobi_reference --locked -- --nocapture
result: compile failed from a source mix missing the helper constants/functions.

RCH ts1 proof attempt 2:
same proof command
result: exit 0 but ran 0 tests; the worker saw constants without the routing/test use sites.
```

After RCH artifact retrieval, `git diff -- crates/fsci-linalg/src/lib.rs` was empty. No candidate source was kept.

## Behavior Proof

Clean-source public golden guard passed after the aborted trial:

```text
RCH ts1:
cargo test -p fsci-linalg --release --lib public_svd_lstsq_pinv_golden_payload --locked -- --nocapture
result: passed
public golden SHA remains 1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225
```

Isomorphism:

- Ordering/tie-breaking: unchanged because no linalg source remains changed.
- Floating-point behavior: unchanged for public routes; the clean public golden payload passed.
- RNG: unchanged; SVD path has no RNG.
- Golden output: public SVD/lstsq/pinv SHA unchanged.

## Decision

Reject this uncommitted inverse-iteration trial as an invalid proof/bench path.

Score: `0.0 = Impact unknown * Confidence 0 / Effort high`

Next primitive: run the next backend experiment from a source-coherent, commit-backed scratch branch/worktree so RCH cannot test a mixed source snapshot. The algorithmic target remains a real bidiagonal-specialized SVD backend: implicit-shift bidiagonal QR with vector accumulation, dqds singular values plus inverse iteration/MRRR-style vector recovery, or divide-and-conquer bidiagonal SVD. Target remains at least `2.5x` on the private `1024x512` SVD stage with reconstruction, orthogonality, ordering/tie-break, and public-golden proof.
