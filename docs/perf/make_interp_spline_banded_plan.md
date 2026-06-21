# Plan: compact-banded `make_interp_spline` (close the remaining ~120x scipy loss)

Status: DESIGN (authored under a disk-low / no-build window — implement + verify
when benches resume). Owner: cc / MistyBirch.

## Why
`make_interp_spline(x, y, k)` (crates/fsci-interpolate/src/lib.rs, ~line 1766) is
**29-175x slower than scipy** and the gap grows O(n²) (measured k=3: n=1000 6.81ms
vs scipy 0.23ms; n=3000 84.26ms vs 0.48ms). The byte-identical `solve_banded`
switch (commit 318898bb) took it to 1.45x (n=3000 → 58ms); the **build** is now the
O(n²) bottleneck. scipy is O(n·k) (banded throughout). Three O(n²) sources remain:

1. `let mut a_mat = vec![vec![0.0; n]; n];` — dense n×n alloc/zero (~72MB at n=3000).
2. `eval_basis_all(&t, x[i], k, n)` per row does a **degree-0 interval LINEAR SCAN**
   `for i in 0..n { if t[i] <= x < t[i+1] ... }` (O(n)) AND allocates a length-n
   `Vec` (O(n)); called n times → O(n²).
3. `a_mat[i][..n].copy_from_slice(&basis[..n])` — O(n) per row.

## Target
Match scipy's O(n·k): no n×n storage, no per-row O(n) scan/alloc.

## Implementation

### 1. Byte-exact binary-search interval finder
Replace the degree-0 linear scan with a binary search that returns the SAME index
`μ` the scan finds (the half-open interval `t[μ] <= x < t[μ+1]`, with the existing
right-endpoint special case `x == t[μ+1] && μ+1 == t.len()-k-1`). Use
`t.partition_point(|&ti| ti <= x) - 1` then walk back over any repeated knots so the
chosen μ is the FIRST index satisfying the scan's predicate (the scan sets the
lowest such i as `lo`; de-Boor uses `lo`). VERIFY byte-exactly with a test that, for
random sorted x and knot vectors (incl. repeated interior knots + both endpoints),
the binary-search μ equals the linear-scan `lo` for every site. This alone removes
source (2)'s O(n) scan (byte-exact) but not the allocs.

### 2. Compact basis eval (k+1 values + offset)
Factor the de-Boor recursion in `eval_basis_all` into `eval_basis_compact(t, x, k, μ)
-> [f64; K1]` (K1 = k+1) operating on a small stack/`Vec` of length k+1 (indices
μ-k..μ), returning the k+1 nonzero B-spline values + the column offset `μ-k`. The
arithmetic/op-order must match the current in-place ascending sweep so values are
bit-identical. No length-n alloc.

### 3. Compact banded storage + LU solve (the real win)
Collocation row i has its k+1 nonzeros at columns `[off_i, off_i+k]` (off_i from the
interval finder). Build LAPACK-style band storage `ab` of shape `(2*kl+ku+1) × n`
with `kl = ku = k` (partial pivoting fills the upper band to `2k`, so allocate
`ku_eff = 2k`; total bands `kl + ku_eff + 1 = 3k+1`). Store `A[i][j]` at
`ab[kl + ku_eff + i - j][j]` (column-major band, the dgbsv convention) or a simpler
row-band `band[i][j-off_i]` + per-row offset — pick whichever makes the solver
indexing cleanest.

Port a banded LU with partial pivoting (dgbtrf/dgbtrs-style) operating ONLY within
the band. It must reproduce `solve_banded`'s elimination/pivot ORDER so the result
is byte-identical to today's (already scipy-parity) output:
- pivot search over rows `[col, col+kl]`,
- row swap within the band window,
- elimination of rows `[col+1, col+kl]`, updating columns `[col, col+ku_eff]`,
- back-substitution over the band.
`solve_banded` (crates/fsci-interpolate/src/lib.rs) is the byte-identical reference
to diff against on dense-expanded small systems.

## Verification (when disk recovers)
- New unit test: `make_interp_spline` compact-band output == the current dense path
  (`solve_banded` on the dense build) to_bits / ≤1e-12, for k∈{1,2,3,5}, n∈{8,50,200},
  random + clustered sites, repeated interior knots.
- Existing fsci-interpolate suite must stay 172/0 (scipy-parity tests included).
- Bench `make_interp_spline/k3` (already added, commit d049502d): expect n=3000
  58ms → ~1-2ms (O(n·k)), i.e. parity-ish with scipy 0.48ms (closes the loss).

## Gotchas
- The flat-dense lever (used for RBF, commit 17e29927) does NOT apply here: dropping
  the zero-skip on a banded matrix is O(n³). Banded storage is mandatory.
- Keep `solve_banded` / `solve_dense_system` for the other callers (smoothing/lsq
  splines use solve_banded on dense storage; RBF uses solve_dense_system_flat).
- Pivoting can be skipped IF the collocation is provably totally-positive
  (Schoenberg-Whitney), but scipy uses general gbsv — keep pivoting to stay
  byte-identical to the current (parity-verified) result.

---

## Reference implementation (paste + `cargo check` + verify when disk recovers)

Authored in a no-build window — TREAT AS UNVERIFIED until `cargo test -p
fsci-interpolate` passes (esp. the byte-diff test below). Edge cases flagged inline
must be checked against `eval_basis_all`'s linear scan before trusting.

```rust
/// Byte-exact binary-search replacement for eval_basis_all's degree-0 linear scan:
/// returns (lo, hi), the inclusive span of indices i in 0..n where the degree-0
/// indicator is 1 — i.e. (t[i] <= x < t[i+1]) OR the right-endpoint special case
/// (x == t[i+1] && i+1 == t.len()-k-1). Returns None if x is outside the knot span.
/// VERIFY: for random sorted t (incl. repeated interior knots) + x at interior,
/// both endpoints, and exactly on knots, (lo,hi) must equal the scan's (lo,hi).
fn bspline_knot_span(t: &[f64], x: f64, k: usize, n: usize) -> Option<(usize, usize)> {
    if n == 0 { return None; }
    // largest i with t[i] <= x  (i in 0..t.len())
    let pp = t.partition_point(|&ti| ti <= x); // count of t[.] <= x
    if pp == 0 { return None; }                 // x < t[0]
    let mut hi = pp - 1;                         // candidate: t[hi] <= x
    // walk back to the first index whose half-open interval is non-empty & holds x.
    // For the normal case t[i] <= x < t[i+1] with distinct knots, lo == hi == that i.
    // Repeated knots: skip empty intervals t[i]==t[i+1]; the scan's `lo` is the FIRST
    // matching index, `hi` the LAST. EDGE: x exactly on an interior knot t[m] makes
    // pp include it → hi=m, but the scan puts x in interval [t[m], t[m+1]); confirm
    // hi indexes that interval (clamp hi to n-1 and to t.len()-k-2 for the endpoint).
    // EDGE: x == right boundary → scan's endpoint case (i+1==t.len()-k-1); map hi to it.
    if hi >= n { hi = n - 1; }                   // x beyond last data interval
    // lo: walk back over indices i<hi that ALSO satisfy the predicate (repeated knots
    // give an empty [t[i],t[i+1]) so they DON'T match — lo stays = hi for distinct x).
    let mut lo = hi;
    while lo > 0 && t.get(lo).is_some() && t.get(lo - 1).map_or(false, |&p| p == t[lo]) {
        // t[lo-1]==t[lo] => interval [t[lo-1],t[lo]) empty => not a match; stop.
        break; // (placeholder — replicate scan semantics exactly; verify before trust)
    }
    Some((lo, hi))
}

/// Compact de-Boor: the k+1 nonzero B-spline values at columns [lo-k, hi] plus the
/// column offset (lo-k). Mirrors eval_basis_all's in-place ascending sweep EXACTLY
/// (same denom>0 guards, same read of basis[i] and basis[i+1], with basis[hi+1]
/// implicitly 0), but in a local buffer of length (hi - (lo-k) + 2) so the trailing
/// basis[hi+1]=0 read stays in-bounds. Returns (offset = lo-k, values).
fn eval_basis_compact(t: &[f64], x: f64, k: usize, n: usize) -> Option<(usize, Vec<f64>)> {
    let (lo, hi) = bspline_knot_span(t, x, k, n)?;
    let off = lo.saturating_sub(k);
    let len = hi + 1 - off + 1; // [off .. hi] inclusive + one trailing slot for basis[hi+1]
    let mut b = vec![0.0f64; len];
    // degree-0: global index i in [lo, hi] -> local i-off
    for i in lo..=hi { b[i - off] = 1.0; }
    for p in 1..=k {
        let start = lo.saturating_sub(p);
        for i in start..=hi {
            let li = i - off;
            let mut val = 0.0;
            if i + p < t.len() {
                let dl = t[i + p] - t[i];
                if dl > 0.0 { val += (x - t[i]) / dl * b[li]; }
            }
            if i + p + 1 < t.len() && i + 1 < n {
                let dr = t[i + p + 1] - t[i + 1];
                if dr > 0.0 { val += (t[i + p + 1] - x) / dr * b[li + 1]; }
            }
            b[li] = val;
        }
    }
    b.truncate(hi + 1 - off); // drop the trailing 0 slot -> exactly the k+1 (or hi-off+1) values
    Some((off, b))
}
```

Banded LU on compact storage: port `solve_banded`'s pivot/eliminate/back-sub ORDER
(it is the byte-identical reference) onto a flat `band: Vec<f64>` of shape
`(3k+1) * n` with the dgbsv index map `A[i][j] -> band[(2k + i - j) * n + j]`
(valid for `j-k <= i <= j+2k`). Use `split_at_mut` on the flat buffer per pivot
column (mirror solve_banded's pivot-row borrow). Keep partial pivoting to stay
byte-identical to today's parity-verified output.

### Byte-diff test (add before flipping make_interp_spline)
```rust
// new compact-band make_interp_spline coeffs == current solve_banded(dense) coeffs
// to_bits, for k in {1,2,3,5}, n in {8,50,200}, random + clustered + repeated-knot x.
```

---

## Plan 2: factor-once GCV trace loop (the real O(n³)→O(n²), cargo-needed)

Owner-note (cc): per the 2026-06-20 correction, gcv_optimal_lambda's trace loop is
STILL O(n³·iters) — each of the n columns does `let lhs = vec![vec![0.0;n];n]`
(O(n²) alloc+zero) + a fresh `solve_banded` (re-factors the SAME lhs). lhs is
loop-invariant (= XᵀWX + λ XᵀWE), so factor it ONCE per λ and substitute n RHS.

### Implementation (crates/fsci-interpolate/src/lib.rs)
Split `solve_banded` (line 2741, the Vec<Vec> variant) into two phases, keeping the
EXACT same FP order so the result is byte-identical to today's n separate calls:
- `fn factor_banded(a: &mut [Vec<f64>], bw) -> Result<Vec<usize>, _>` — the
  pivot/eliminate loop (2741's first half), recording the row-swap permutation
  `perm[col] = max_row`; leaves a holding L (below diag) + U (band) in place.
- `fn subst_banded(a: &[Vec<f64>], perm: &[usize], b: &mut [f64], bw)` — apply the
  same row swaps to b (in col order), forward-substitute with L, back-substitute with
  U (2741's `b[row] -= factor*b[col]` + the final back-sub block). Returns x.

Trace loop becomes:
```
let mut lhs = build_band(xtwx, xte, lam, n);     // O(n·bw) band fill, ONCE per λ
let perm = factor_banded(&mut lhs, 4)?;          // O(n·bw²), ONCE per λ
let mut tr = 0.0;
let mut b = vec![0.0; n];                          // reused
for col in 0..n {
    for i in 0..n { b[i] = xtwx[i][col]; }        // O(n) (or band-restrict to |i-col|≤4)
    let z = subst_banded(&lhs, &perm, &mut b, 4); // O(n·bw)
    tr += z[col];
}
```
→ per-λ trace cost O(n·bw² + n·n·bw) = O(n²·bw); whole gcv O(n²·iters). The m-solve
(5826) keeps its single solve_banded.

### Verify (cargo recovery)
- Byte-identity: factor_banded+subst_banded on a random banded system must equal
  `solve_banded` to_bits (add a unit test). Then make_smoothing_spline must stay
  byte-identical → existing scipy-parity + suite 172/0.
- Bench: make_smoothing_spline at large n (the GCV path) should drop ~n× on the trace.
- GOTCHA: partial-pivoting row swaps — `subst_banded` must replay `perm` in the SAME
  order factor_banded applied them (col ascending), and the back-sub window is
  [i, i+2·bw] (U upper bandwidth grows to 2·bw under pivoting). Keep the zero-skips so
  it stays bit-identical to solve_banded.

### Plan 2 — paste-ready reference code (factor-once GCV trace, O(n³)→O(n²))

UNVERIFIED (no-cargo authoring) — paste, `cargo check`, add the byte-diff test, then
run make_smoothing_spline scipy-parity + suite 172/0. Split of `solve_banded`
(line 2741) into factor + substitute; byte-identical to n separate solve_banded calls
(same pivot/elim/FP order; stored L multiplier replaces the eliminated a[row][col],
which back-sub never reads since it reads only j>i).

```rust
fn factor_banded(a: &mut [Vec<f64>], bw: usize) -> Result<Vec<usize>, InterpError> {
    let n = a.len();
    let mut perm = vec![0usize; n];
    for col in 0..n {
        let row_hi = (col + bw + 1).min(n);
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (off, a_row) in a[(col + 1)..row_hi].iter().enumerate() {
            let v = a_row[col].abs();
            if v > max_val { max_val = v; max_row = col + 1 + off; }
        }
        if max_val < 1e-14 {
            return Err(InterpError::InvalidArgument { detail: "singular matrix".to_string() });
        }
        if max_row != col { a.swap(col, max_row); }
        perm[col] = max_row;
        let col_hi = (col + 2 * bw + 1).min(n);
        let pivot_diag = a[col][col];
        for row in (col + 1)..row_hi {
            if a[row][col] == 0.0 { continue; }
            let factor = a[row][col] / pivot_diag;
            let (head, tail) = a.split_at_mut(row);
            let pivot = &head[col];
            let target = &mut tail[0];
            for j in (col + 1)..col_hi {          // U part only (j>col); a[row][col] reused for L
                let pval = pivot[j];
                if pval != 0.0 { target[j] -= factor * pval; }
            }
            target[col] = factor;                  // store L multiplier
        }
    }
    Ok(perm)
}

fn subst_banded(a: &[Vec<f64>], perm: &[usize], b: &mut [f64], bw: usize) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {                              // forward: same swap+L order as solve_banded
        if perm[col] != col { b.swap(col, perm[col]); }
        let row_hi = (col + bw + 1).min(n);
        for row in (col + 1)..row_hi {
            let factor = a[row][col];
            if factor != 0.0 { b[row] -= factor * b[col]; }
        }
    }
    let mut x = vec![0.0; n];                      // back: U
    for i in (0..n).rev() {
        let mut s = b[i];
        let j_hi = (i + 2 * bw + 1).min(n);
        for j in (i + 1)..j_hi {
            let aij = a[i][j];
            if aij != 0.0 { s -= aij * x[j]; }
        }
        x[i] = s / a[i][i];
    }
    x
}
```
Trace loop (replaces per-column build+solve): build `lhs` once (band-restricted),
`let perm = factor_banded(&mut lhs, 4)?;` then `for col { for i {b[i]=xtwx[i][col];}
let z = subst_banded(&lhs,&perm,&mut b,4); tr += z[col]; }`. → O(n·bw² + n²·bw) per λ.
BYTE-DIFF TEST: factor_banded+subst_banded == solve_banded to_bits on random banded
systems, before flipping the trace loop.

### Plan 2 — CORRECTION (2026-06-20, cc): factor-once is TOLERANCE-parity, NOT byte-identical
On re-review of the paste-ready code above: the "byte-identical to n solve_banded calls"
claim is WRONG. With partial pivoting, a column col's stored L multiplier (a[row][col])
gets MOVED by a LATER column col'>col row-swap (a.swap(col', max_row') swaps whole rows,
including the already-stored L of column col). So a factor-once solve that stores L and
substitutes (LAPACK getrf/getrs style) must either apply the permutation to b FIRST then
forward-subst (perm-first), or track the moved L — both do the FP ops in a DIFFERENT
ORDER than solve_banded's INTERLEAVED swap-then-eliminate-b. Same math, but
TOLERANCE-parity (≤~1e-12), NOT byte-for-byte.
CONSEQUENCES for the recovery impl:
- Do NOT gate on a to_bits byte-diff vs solve_banded — it WILL differ. Gate on a
  TOLERANCE diff (≤1e-10) vs solve_banded AND the make_smoothing_spline scipy-parity test
  (which is already tolerance-based — so flipping to factor-once is fine if it stays within
  that tolerance).
- Safest impl: standard banded getrf (store L + ipiv) + getrs (apply ipiv to b, forward
  with L, back with U). The subst code above is close but its interleaved swap+L-read is
  the fragile part — prefer perm-first getrs.
- This means the factor-once trace win is a (tolerance-parity) numerical change to
  make_smoothing_spline's GCV-selected lambda — verify the chosen lambda + coefficients
  stay within scipy tolerance, not that they're unchanged bit-for-bit.

### Plan 2 — SHIPPED via banded Cholesky (2026-06-20, cc)
The LU factor_banded/subst_banded above is NOT used: for a physical-row-swap Vec<Vec>,
later-column swaps scatter the stored L multipliers, so a perm-first/interleaved getrs is
wrong or non-banded. The GCV `lhs = XᵀWX + λXᵀE` is SPD, so it ships as banded CHOLESKY
(chol_banded + chol_subst, no pivoting → no scatter), factored once per λ. Verified
interpolate 173/0 (tolerance-parity; scipy-parity tests pass).
