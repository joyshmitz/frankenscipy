# Optimization Result — eliminate redundant input clone in dense solve

**Bead:** `frankenscipy-perf-linalg-matrix-copies-mh8ch` (perf)
**Skill:** extreme-software-optimization · **Lever:** 1 (one change)
**File:** `crates/fsci-linalg/src/lib.rs` — `solve_with_portfolio_internal`

## Change

`effective_a` was always a deep copy of the input matrix:

```rust
let effective_a = if options.transposed { transpose(a) } else { a.to_vec() };
```

`a.to_vec()` deep-clones the `Vec<Vec<f64>>` (~8 MB at n=1000) on the common
non-transposed path, even though `effective_a` is **read-only** downstream
(condition diagnostics, policy check, solver dispatch all borrow it). Replaced
with a `Cow` that borrows on the non-transposed path and only owns when a
transpose is genuinely required:

```rust
let effective_a: Cow<[Vec<f64>]> = if options.transposed {
    Cow::Owned(transpose(a))
} else {
    Cow::Borrowed(a)
};
```

This removes one of the three full-matrix copies per solve identified in
`hotspot_table.md` (rank 2). The remaining two (`dmatrix_from_rows`,
`matrix.clone()` before `.lu()`) are tracked in the sibling beads.

## Isomorphism Proof

- **Ordering preserved:** yes — identical data, identical traversal; only the
  storage (borrowed vs owned) changes.
- **Tie-breaking unchanged:** yes — no comparisons altered.
- **Floating-point:** identical — the `DMatrix` built from `effective_a` is
  byte-for-byte the same; LU, rcond, solve, and backward-error are unchanged.
- **RNG seeds:** N/A.
- **Golden outputs:** bit-exact f64 patterns of every solution element across
  6 sizes × 3 seeds × {non-transposed, transposed} (36 cases), incl.
  `backward_error`:
  - `golden/golden_before.txt` sha256 `5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd`
  - `golden/golden_after.txt`  sha256 `5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd`
  - **IDENTICAL ✓**
- Both the borrowed (non-transposed) and owned (transposed) branches are
  exercised by the golden set.

## Measured Before/After (same-session A/B, identical host load)

Scenario: `solve(A,b)`, n=1000 diagonally-dominant, cold single solve.
`hyperfine --warmup 3 --runs 30`, both binaries built `release-perf`.
Checksums identical (`-9.844828e-1`).

| metric | old (a.to_vec) | new (Cow borrow) | delta |
|--------|----------------|------------------|-------|
| p50 | 128.2 ms | 123.2 ms | **−3.9%** |
| p95 | 140.4 ms | 132.8 ms | **−5.4%** |
| mean ± σ | 127.6 ± 8.1 | 123.3 ± 5.5 | −3.4% (new better at every pct) |
| **peak RSS** | 55.4 MB | **47.5 MB** | **−14.3% (deterministic, = one 8 MB buffer)** |
| **minor page faults (cold)** | 15,765 | **13,802** | **−12.5%** |
| cold system time | ~31–45 ms | ~29 ms | lower |

Artifacts: `ab_old_vs_new.json`, `golden/`.

## Opportunity Score

Impact 2 (deterministic −14% RSS / −12.5% faults; −5.4% p95) ×
Confidence 5 (bit-identical golden + structural allocation removal + A/B) /
Effort 1 (4-line change) = **10 ≥ 2.0 ✓**

The wall-clock win is bounded because the O(n³) LU (~75% of time, sibling bead
`perf-linalg-lu-scalar`) and the two remaining copies still dominate. The
memory/allocation win is deterministic and unambiguous.
