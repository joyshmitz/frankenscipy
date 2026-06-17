//! Building blocks for the cosine-sine (CS) decomposition (`scipy.linalg.cossin`).
//!
//! `cossin` matches LAPACK's `dorcsd`, which chains `dorbdb` (simultaneous
//! bidiagonalization of the 2×2 partitioned orthogonal matrix) and `dbbcsd`
//! (bidiagonal CS values via an implicit QR sweep). Achieving scipy-identical
//! factor signs requires faithful ports of those routines — a multi-stage
//! effort (see bead frankenscipy-5tmu1).
//!
//! This module starts that port with the foundational Householder primitives
//! `dlarfgp` and `dlarf`, transcribed from the LAPACK reference with the exact
//! conventions `dorbdb` relies on. They are independently verifiable (the
//! Householder reflection property) and are `pub(crate)` until `dorbdb`/`dbbcsd`
//! consume them — mirroring how `fitpack_cyclic` shipped its solvers ahead of
//! the spline driver.
#![allow(dead_code)]

/// Generate an elementary Householder reflector with a **non-negative** beta,
/// faithfully matching LAPACK `DLARFGP`.
///
/// On input the reflector acts on the vector `[alpha; tail]` (length `n =
/// tail.len() + 1`). On output `tail` is overwritten with `v[1..]` (the implicit
/// `v[0] = 1`), and `(beta, tau)` are returned such that
/// `H = I − tau·v·vᵀ` satisfies `H·[alpha; tail] = [beta; 0]` with `beta ≥ 0`.
///
/// The extreme-underflow rescaling loop of the reference (`KNT`) is omitted: it
/// only triggers for subnormal `beta`, which never arises for the orthogonal
/// inputs of the CS decomposition.
pub(crate) fn dlarfgp(alpha_in: f64, tail: &mut [f64]) -> (f64, f64) {
    // n == 1 (empty tail): H is trivial; pick the sign so beta = |alpha| ≥ 0.
    if tail.is_empty() {
        return (alpha_in.abs(), if alpha_in >= 0.0 { 0.0 } else { 2.0 });
    }

    let eps = f64::EPSILON;
    let xnorm = tail.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut alpha = alpha_in;

    if xnorm <= eps * alpha.abs() {
        // H = [±1, 0; 0, I], sign chosen so beta = |alpha| ≥ 0.
        if alpha >= 0.0 {
            return (alpha, 0.0);
        }
        for v in tail.iter_mut() {
            *v = 0.0;
        }
        return (-alpha, 2.0);
    }

    // General case. beta = SIGN(hypot(alpha, xnorm), alpha) (Fortran SIGN: +0 → +).
    let mut beta = if alpha >= 0.0 {
        alpha.hypot(xnorm)
    } else {
        -alpha.hypot(xnorm)
    };
    let savealpha = alpha;
    alpha += beta;
    let tau;
    if beta < 0.0 {
        beta = -beta;
        tau = -alpha / beta;
    } else {
        alpha = xnorm * (xnorm / alpha);
        tau = alpha / beta;
        alpha = -alpha;
    }

    let smlnum = f64::MIN_POSITIVE / f64::EPSILON;
    if tau.abs() <= smlnum {
        // Denormal tau loses accuracy; flush per the reference.
        if savealpha >= 0.0 {
            return (beta, 0.0);
        }
        for v in tail.iter_mut() {
            *v = 0.0;
        }
        return (-savealpha, 2.0);
    }

    // Scale the reflector tail v[1..] = x[1..] / alpha.
    for v in tail.iter_mut() {
        *v /= alpha;
    }
    (beta, tau)
}

/// Apply a Householder reflector `H = I − tau·v·vᵀ` (with `v[0] = 1`, the rest in
/// `v_tail`) to the columns of a row-major `rows × cols` matrix `c` from the
/// **left** (`H·C`), matching the `'L'` side of LAPACK `DLARF`/`DLARF1F`.
///
/// `v` has length `rows`: `v[0] = 1` implicitly, `v_tail = v[1..]`. `tau == 0`
/// is a no-op.
pub(crate) fn dlarf_left(c: &mut [Vec<f64>], v_tail: &[f64], tau: f64) {
    if tau == 0.0 {
        return;
    }
    let rows = c.len();
    if rows == 0 {
        return;
    }
    let cols = c[0].len();
    // v[0] = 1, v[1..] = v_tail.
    let v = |i: usize| if i == 0 { 1.0 } else { v_tail[i - 1] };
    // For each column j: w = vᵀ·C[:,j]; C[:,j] -= tau·w·v.
    for j in 0..cols {
        let mut w = 0.0;
        for i in 0..rows {
            w += v(i) * c[i][j];
        }
        let tw = tau * w;
        for i in 0..rows {
            c[i][j] -= tw * v(i);
        }
    }
}

/// Apply a Householder reflector to the rows of a row-major `rows × cols` matrix
/// `c` from the **right** (`C·H`), matching the `'R'` side of LAPACK `DLARF`.
///
/// `v` has length `cols`: `v[0] = 1` implicitly, `v_tail = v[1..]`.
pub(crate) fn dlarf_right(c: &mut [Vec<f64>], v_tail: &[f64], tau: f64) {
    if tau == 0.0 {
        return;
    }
    let rows = c.len();
    if rows == 0 {
        return;
    }
    let cols = c[0].len();
    let v = |j: usize| if j == 0 { 1.0 } else { v_tail[j - 1] };
    // For each row i: w = C[i,:]·v; C[i,:] -= tau·w·v.
    for row in c.iter_mut().take(rows) {
        let mut w = 0.0;
        for j in 0..cols {
            w += row[j] * v(j);
        }
        let tw = tau * w;
        for j in 0..cols {
            row[j] -= tw * v(j);
        }
    }
}

/// Compute the CS-decomposition angle vector `theta` from the upper-left block
/// `x11` (`p × q`) of an `m × m` orthogonal matrix, matching the `theta` returned
/// by `scipy.linalg.cossin(..., separate=True)`.
///
/// The cosines of the angles are the `r = min(p, q, m-p, m-q)` smallest singular
/// values of `x11` (the larger `min(p,q) − r` singular values are ≈ 1 and form
/// the identity blocks). With `n11 = min(p,q) − r` and the singular values `σ`
/// in descending order, `theta[i] = arccos(σ[n11 + i])`, ascending.
pub(crate) fn cs_angles(
    x11: &[Vec<f64>],
    p: usize,
    q: usize,
    m: usize,
) -> Result<Vec<f64>, crate::LinalgError> {
    let r = p.min(q).min(m - p).min(m - q);
    let n11 = p.min(q) - r;
    let res = crate::svd(x11, crate::DecompOptions::default())?;
    let theta = (0..r)
        .map(|i| res.s[n11 + i].clamp(-1.0, 1.0).acos())
        .collect();
    Ok(theta)
}

/// Cosine-sine decomposition angles and middle factor of an `m × m` orthogonal
/// matrix `x` partitioned with a `p × q` upper-left block, matching the
/// `compute_u=False, compute_vh=False` mode of `scipy.linalg.cossin`.
///
/// Returns `(theta, cs)`: `theta` is the vector of CS angles (radians), and `cs`
/// is the `m × m` cosine-sine middle factor (default `swap_sign=False`
/// convention). The orthogonal factors `u`/`vh` are not yet produced (their
/// LAPACK-identical signs require the `dorbdb`/`dbbcsd` port — see bead
/// frankenscipy-5tmu1); this function provides the angle/structure side, which
/// is independent of that sign convention.
///
/// `x` must be `m × m` orthogonal with `0 < p < m`, `0 < q < m`.
pub fn cossin_angles(
    x: &[Vec<f64>],
    p: usize,
    q: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), crate::LinalgError> {
    let m = x.len();
    let x11: Vec<Vec<f64>> = x[..p].iter().map(|row| row[..q].to_vec()).collect();
    let theta = cs_angles(&x11, p, q, m)?;
    let cs = build_cs_matrix(&theta, p, q, m);
    Ok((theta, cs))
}

/// Assemble the `m × m` cosine-sine middle factor `CS` from the angle vector
/// `theta`, matching the construction in scipy's `_cossin` (default
/// `swap_sign=False`: the `-S`/`-I` blocks sit in the upper-right).
///
/// `p`, `q` are the row/column counts of the upper-left block `X11`; `r =
/// min(p, q, m-p, m-q)` is the number of nontrivial angles (`theta.len() == r`).
/// The identity-block ranks are `n11 = min(p,q)-r`, `n12 = min(p,m-q)-r`,
/// `n21 = min(m-p,q)-r`, `n22 = min(m-p,m-q)-r`.
pub(crate) fn build_cs_matrix(theta: &[f64], p: usize, q: usize, m: usize) -> Vec<Vec<f64>> {
    let r = p.min(q).min(m - p).min(m - q);
    let n11 = p.min(q) - r;
    let n12 = p.min(m - q) - r;
    let n21 = (m - p).min(q) - r;
    let n22 = (m - p).min(m - q) - r;
    let mut cs = vec![vec![0.0_f64; m]; m];

    // Leading identity block.
    for i in 0..n11 {
        cs[i][i] = 1.0;
    }
    // Upper-right -I block.
    {
        let xs = n11 + r;
        let ys = n11 + n21 + n22 + 2 * r;
        for i in 0..n12 {
            cs[xs + i][ys + i] = -1.0;
        }
    }
    // Lower-left I block.
    {
        let xs = p + n22 + r;
        let ys = n11 + r;
        for i in 0..n21 {
            cs[xs + i][ys + i] = 1.0;
        }
    }
    // Middle I block.
    for i in 0..n22 {
        cs[p + i][q + i] = 1.0;
    }
    // Cosine diagonals (upper-left and lower-right).
    for (i, &t) in theta.iter().enumerate().take(r) {
        cs[n11 + i][n11 + i] = t.cos();
        cs[p + n22 + i][n11 + r + n21 + n22 + i] = t.cos();
    }
    // Sine blocks: -S (upper-right) and +S (lower-left).
    for (i, &t) in theta.iter().enumerate().take(r) {
        cs[n11 + i][n11 + n21 + n22 + r + i] = -t.sin();
        cs[p + n22 + i][n11 + i] = t.sin();
    }
    cs
}

// Apply H = I − tau·v·vᵀ (v[r0]=1, `vtail` at rows r0+1..) to columns `c0..c1`
// of the row-major matrix `c`, from the left.
fn apply_left_block(c: &mut [Vec<f64>], r0: usize, c0: usize, c1: usize, vtail: &[f64], tau: f64) {
    if tau == 0.0 {
        return;
    }
    for col in c0..c1 {
        let mut w = c[r0][col];
        for (k, &vt) in vtail.iter().enumerate() {
            w += vt * c[r0 + 1 + k][col];
        }
        let tw = tau * w;
        c[r0][col] -= tw;
        for (k, &vt) in vtail.iter().enumerate() {
            c[r0 + 1 + k][col] -= tw * vt;
        }
    }
}

// Apply C·H (H = I − tau·w·wᵀ, w[c0]=1, `wtail` at cols c0+1..) to rows `r0..r1`
// of the row-major matrix `c`, from the right.
fn apply_right_block(c: &mut [Vec<f64>], r0: usize, r1: usize, c0: usize, wtail: &[f64], tau: f64) {
    if tau == 0.0 {
        return;
    }
    for row in c.iter_mut().take(r1).skip(r0) {
        let mut w = row[c0];
        for (k, &wt) in wtail.iter().enumerate() {
            w += wt * row[c0 + 1 + k];
        }
        let tw = tau * w;
        row[c0] -= tw;
        for (k, &wt) in wtail.iter().enumerate() {
            row[c0 + 1 + k] -= tw * wt;
        }
    }
}

/// Reflector scalars and angles from [`dorbdb_balanced`].
pub(crate) struct OrbdbResult {
    pub theta: Vec<f64>,
    pub phi: Vec<f64>,
    pub taup1: Vec<f64>,
    pub taup2: Vec<f64>,
    pub tauq1: Vec<f64>,
    pub tauq2: Vec<f64>,
}

/// Simultaneous bidiagonalization (LAPACK `DORBDB`, `TRANS='N'`, default signs)
/// of the four `q×q` blocks of a `2q×2q` partitioned orthogonal matrix (the
/// balanced square case `p = q = m/2`). On return the blocks hold the packed
/// Householder reflector tails; the angles and reflector scalars are returned.
pub(crate) fn dorbdb_balanced(
    x11: &mut [Vec<f64>],
    x12: &mut [Vec<f64>],
    x21: &mut [Vec<f64>],
    x22: &mut [Vec<f64>],
    q: usize,
) -> OrbdbResult {
    let (p, mp) = (q, q); // P = M-P = q
    let mut theta = vec![0.0f64; q];
    let mut phi = vec![0.0f64; q.saturating_sub(1)];
    let mut taup1 = vec![0.0; q];
    let mut taup2 = vec![0.0; q];
    let mut tauq1 = vec![0.0; q.saturating_sub(1)];
    let mut tauq2 = vec![0.0; q];

    for i in 0..q {
        // Step 1: column update mixing X11/X12 (and X21/X22) by phi[i-1].
        if i > 0 {
            let (c, s) = (phi[i - 1].cos(), phi[i - 1].sin());
            for r in i..p {
                x11[r][i] = c * x11[r][i] - s * x12[r][i - 1];
            }
            for r in i..mp {
                x21[r][i] = c * x21[r][i] - s * x22[r][i - 1];
            }
        }
        // Step 2: theta[i] = atan2(||X21(i:,i)||, ||X11(i:,i)||).
        let n21 = (i..mp).map(|r| x21[r][i] * x21[r][i]).sum::<f64>().sqrt();
        let n11 = (i..p).map(|r| x11[r][i] * x11[r][i]).sum::<f64>().sqrt();
        theta[i] = n21.atan2(n11);
        // Step 3: column reflectors P1, P2.
        {
            let mut tail: Vec<f64> = (i + 1..p).map(|r| x11[r][i]).collect();
            let (beta, tau) = dlarfgp(x11[i][i], &mut tail);
            x11[i][i] = beta;
            for (k, r) in (i + 1..p).enumerate() {
                x11[r][i] = tail[k];
            }
            taup1[i] = tau;
        }
        {
            let mut tail: Vec<f64> = (i + 1..mp).map(|r| x21[r][i]).collect();
            let (beta, tau) = dlarfgp(x21[i][i], &mut tail);
            x21[i][i] = beta;
            for (k, r) in (i + 1..mp).enumerate() {
                x21[r][i] = tail[k];
            }
            taup2[i] = tau;
        }
        // Step 4: apply P1/P2 from the left.
        let v1: Vec<f64> = (i + 1..p).map(|r| x11[r][i]).collect();
        if i + 1 < q {
            apply_left_block(x11, i, i + 1, q, &v1, taup1[i]);
        }
        apply_left_block(x12, i, i, q, &v1, taup1[i]);
        let v2: Vec<f64> = (i + 1..mp).map(|r| x21[r][i]).collect();
        if i + 1 < q {
            apply_left_block(x21, i, i + 1, q, &v2, taup2[i]);
        }
        apply_left_block(x22, i, i, q, &v2, taup2[i]);
        // Step 5: row update mixing X11/X21 (and X12/X22) by theta[i].
        let (st, ct) = (theta[i].sin(), theta[i].cos());
        if i + 1 < q {
            for col in i + 1..q {
                x11[i][col] = -st * x11[i][col] + ct * x21[i][col];
            }
        }
        for col in i..q {
            x12[i][col] = -st * x12[i][col] + ct * x22[i][col];
        }
        // Step 6: phi[i] = atan2(||X11(i,i+1:)||, ||X12(i,i:)||).
        if i + 1 < q {
            let n11r = (i + 1..q).map(|c| x11[i][c] * x11[i][c]).sum::<f64>().sqrt();
            let n12r = (i..q).map(|c| x12[i][c] * x12[i][c]).sum::<f64>().sqrt();
            phi[i] = n11r.atan2(n12r);
        }
        // Step 7: row reflectors Q1, Q2.
        if i + 1 < q {
            let mut tail: Vec<f64> = (i + 2..q).map(|c| x11[i][c]).collect();
            let (beta, tau) = dlarfgp(x11[i][i + 1], &mut tail);
            x11[i][i + 1] = beta;
            for (k, c) in (i + 2..q).enumerate() {
                x11[i][c] = tail[k];
            }
            tauq1[i] = tau;
        }
        {
            let mut tail: Vec<f64> = (i + 1..q).map(|c| x12[i][c]).collect();
            let (beta, tau) = dlarfgp(x12[i][i], &mut tail);
            x12[i][i] = beta;
            for (k, c) in (i + 1..q).enumerate() {
                x12[i][c] = tail[k];
            }
            tauq2[i] = tau;
        }
        // Step 8: apply Q1/Q2 from the right to the trailing rows.
        if i + 1 < q {
            let w1: Vec<f64> = (i + 2..q).map(|c| x11[i][c]).collect();
            apply_right_block(x11, i + 1, p, i + 1, &w1, tauq1[i]);
            apply_right_block(x21, i + 1, mp, i + 1, &w1, tauq1[i]);
        }
        let w2: Vec<f64> = (i + 1..q).map(|c| x12[i][c]).collect();
        if i + 1 < p {
            apply_right_block(x12, i + 1, p, i, &w2, tauq2[i]);
        }
        if i + 1 < mp {
            apply_right_block(x22, i + 1, mp, i, &w2, tauq2[i]);
        }
    }
    OrbdbResult {
        theta,
        phi,
        taup1,
        taup2,
        tauq1,
        tauq2,
    }
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    a
}

// Accumulate the orthogonal factor from column Householder reflectors stored in
// `src` (reflector i: v[i]=1, tail = src[i+1..n][i]).
pub(crate) fn build_from_col_reflectors(src: &[Vec<f64>], tau: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut a = identity(n);
    for i in (0..tau.len()).rev() {
        let tail: Vec<f64> = (i + 1..n).map(|r| src[r][i]).collect();
        apply_left_block(&mut a, i, 0, n, &tail, tau[i]);
    }
    a
}

// Accumulate the orthogonal factor from row Householder reflectors stored in
// `src` (reflector i: w[i+col_off]=1, tail = src[i][i+col_off+1..n]).
pub(crate) fn build_from_row_reflectors(
    src: &[Vec<f64>],
    tau: &[f64],
    n: usize,
    col_off: usize,
) -> Vec<Vec<f64>> {
    let mut a = identity(n);
    for i in (0..tau.len()).rev() {
        let cs = i + col_off;
        let tail: Vec<f64> = (cs + 1..n).map(|c| src[i][c]).collect();
        apply_left_block(&mut a, cs, 0, n, &tail, tau[i]);
    }
    a
}

/// Build the `2q×2q` bidiagonal-block matrix `BigB` from the CS angles, matching
/// LAPACK's `DBBCSD` input layout (balanced square, `M-P-Q = 0`).
pub(crate) fn build_bigb(theta: &[f64], phi: &[f64], q: usize) -> Vec<Vec<f64>> {
    let m = 2 * q;
    let mut b = vec![vec![0.0; m]; m];
    let c: Vec<f64> = theta.iter().map(|t| t.cos()).collect();
    let s: Vec<f64> = theta.iter().map(|t| t.sin()).collect();
    // This matches the bidiagonalization produced by [`dorbdb_balanced`]:
    // B11 upper-bidiagonal, B22 lower-bidiagonal (LAPACK DBBCSD convention),
    // B12 lower-bidiagonal but sign-negated, and B21 upper-bidiagonal positive.
    for i in 0..q {
        let cp_prev = if i == 0 { 1.0 } else { phi[i - 1].cos() };
        let cp_cur = if i + 1 < q { phi[i].cos() } else { 1.0 };
        b[i][i] = c[i] * cp_prev; // B11 D
        b[q + i][i] = s[i] * cp_prev; // B21 D (positive)
        b[i][q + i] = -s[i] * cp_cur; // B12 D (negated)
        b[q + i][q + i] = c[i] * cp_cur; // B22 D
    }
    for i in 0..q.saturating_sub(1) {
        let sp = phi[i].sin();
        b[i][i + 1] = -s[i] * sp; // B11 superdiagonal
        b[i + 1][q + i] = -c[i + 1] * sp; // B12 subdiagonal (negated)
        b[q + i][i + 1] = c[i] * sp; // B21 superdiagonal (positive, upper)
        b[q + i + 1][q + i] = -s[i + 1] * sp; // B22 subdiagonal
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (n, k, m) = (a.len(), b.len(), b[0].len());
        let mut c = vec![vec![0.0; m]; n];
        for i in 0..n {
            for l in 0..k {
                let ail = a[i][l];
                for j in 0..m {
                    c[i][j] += ail * b[l][j];
                }
            }
        }
        c
    }
    fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (n, m) = (a.len(), a[0].len());
        let mut t = vec![vec![0.0; n]; m];
        for i in 0..n {
            for j in 0..m {
                t[j][i] = a[i][j];
            }
        }
        t
    }

    // Deterministic 2q×2q orthogonal matrix (Q factor of a pseudo-random matrix).
    fn orthogonal(m: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut a = vec![vec![0.0; m]; m];
        for row in a.iter_mut() {
            for v in row.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *v = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            }
        }
        crate::qr(&a, crate::DecompOptions::default()).unwrap().q
    }

    #[test]
    fn dorbdb_reconstructs_balanced() {
        for &q in &[2usize, 3, 4, 5, 6] {
            for seed in 0..4u64 {
                let m = 2 * q;
                let x = orthogonal(m, seed + q as u64 * 100);
                let mut x11: Vec<Vec<f64>> = x[..q].iter().map(|r| r[..q].to_vec()).collect();
                let mut x12: Vec<Vec<f64>> = x[..q].iter().map(|r| r[q..].to_vec()).collect();
                let mut x21: Vec<Vec<f64>> = x[q..].iter().map(|r| r[..q].to_vec()).collect();
                let mut x22: Vec<Vec<f64>> = x[q..].iter().map(|r| r[q..].to_vec()).collect();

                let res = dorbdb_balanced(&mut x11, &mut x12, &mut x21, &mut x22, q);
                let p1 = build_from_col_reflectors(&x11, &res.taup1, q);
                let p2 = build_from_col_reflectors(&x21, &res.taup2, q);
                let q1 = build_from_row_reflectors(&x11, &res.tauq1, q, 1);
                let q2 = build_from_row_reflectors(&x12, &res.tauq2, q, 0);
                let bigb = build_bigb(&res.theta, &res.phi, q);

                // diag(P1,P2) · BigB · diag(Q1,Q2)ᵀ
                let mut big_p = vec![vec![0.0; m]; m];
                let mut big_q = vec![vec![0.0; m]; m];
                for i in 0..q {
                    for j in 0..q {
                        big_p[i][j] = p1[i][j];
                        big_p[q + i][q + j] = p2[i][j];
                        big_q[i][j] = q1[i][j];
                        big_q[q + i][q + j] = q2[i][j];
                    }
                }
                let bqt = transpose(&big_q);
                let recon = matmul(&matmul(&big_p, &bigb), &bqt);
                let mut maxerr = 0.0f64;
                for i in 0..m {
                    for j in 0..m {
                        maxerr = maxerr.max((recon[i][j] - x[i][j]).abs());
                    }
                }
                assert!(
                    maxerr < 1e-10,
                    "q={q} seed={seed}: reconstruction error {maxerr:.3e}"
                );
            }
        }
    }

    // Reconstruct H = I - tau v vᵀ and apply to x; check H·x = beta·e1, beta≥0.
    fn check_reflector(x: &[f64]) {
        let n = x.len();
        let alpha = x[0];
        let mut tail: Vec<f64> = x[1..].to_vec();
        let (beta, tau) = dlarfgp(alpha, &mut tail);
        assert!(beta >= 0.0, "beta must be non-negative, got {beta}");

        // v = [1, tail...]; H·x via the application routine.
        let mut col: Vec<Vec<f64>> = x.iter().map(|&xi| vec![xi]).collect();
        dlarf_left(&mut col, &tail, tau);
        let hx: Vec<f64> = col.iter().map(|r| r[0]).collect();
        assert!((hx[0] - beta).abs() < 1e-12, "H·x[0] = {} vs beta {beta}", hx[0]);
        for (i, &v) in hx.iter().enumerate().skip(1) {
            assert!(v.abs() < 1e-12, "H·x[{i}] = {v}, expected 0");
        }
        // beta = ±‖x‖.
        let nrm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((beta - nrm).abs() < 1e-12, "beta {beta} vs ‖x‖ {nrm}");
        let _ = n;
    }

    #[test]
    fn dlarfgp_reflects_to_positive_beta() {
        check_reflector(&[3.0, 4.0]);
        check_reflector(&[-3.0, 4.0]);
        check_reflector(&[3.0, -4.0, 12.0]);
        check_reflector(&[-1.0, -2.0, -2.0]);
        check_reflector(&[0.0, 3.0, 4.0]);
        check_reflector(&[5.0]); // n=1
        check_reflector(&[-5.0]); // n=1, negative
        check_reflector(&[2.0, 1e-20]); // tail negligible vs alpha
    }

    fn assert_cs(theta: &[f64], p: usize, q: usize, m: usize, want_flat: &[f64]) {
        let cs = build_cs_matrix(theta, p, q, m);
        for i in 0..m {
            for j in 0..m {
                let w = want_flat[i * m + j];
                assert!(
                    (cs[i][j] - w).abs() < 1e-6,
                    "CS[{i}][{j}] = {} vs {w}",
                    cs[i][j]
                );
            }
        }
    }

    #[test]
    fn cossin_angles_matches_scipy_full() {
        // Full 6x6 orthogonal X, p=q=3; scipy.linalg.cossin oracle.
        let x = vec![
            vec![-0.87391881375, 0.167562492975, -0.235103506178, 0.350213063322, -0.006202049134, 0.173860300728],
            vec![0.000460285786, -0.628191216821, -0.001453845601, 0.54997211923, -0.208189382876, -0.509471607163],
            vec![-0.261214975137, 0.006389522704, -0.246757897299, -0.471708053971, 0.418748888032, -0.687733474157],
            vec![-0.141882309423, -0.492064853719, 0.399089339788, 0.00027656113, 0.687774023242, 0.324709262791],
            vec![0.023462349781, -0.528172950388, -0.67836107675, -0.326551827636, -0.150226127071, 0.362084405606],
            vec![0.383863726197, 0.236987962287, -0.514187432999, 0.495718486188, 0.534478332503, 0.026319645199],
        ];
        let (theta, cs) = cossin_angles(&x, 3, 3).unwrap();
        let want_theta = [0.186263718828, 0.906992013555, 1.410204824056];
        for (g, w) in theta.iter().zip(want_theta.iter()) {
            assert!((g - w).abs() < 1e-9, "theta {g} vs {w}");
        }
        let want_cs = [
            0.982703009127, 0.0, 0.0, -0.185188541364, 0.0, 0.0, 0.0, 0.616117785876, 0.0, 0.0,
            -0.787654031874, 0.0, 0.0, 0.0, 0.159902126352, 0.0, 0.0, -0.987132873522,
            0.185188541364, 0.0, 0.0, 0.982703009127, 0.0, 0.0, 0.0, 0.787654031874, 0.0, 0.0,
            0.616117785876, 0.0, 0.0, 0.0, 0.987132873522, 0.0, 0.0, 0.159902126352,
        ];
        for r in 0..6 {
            for c in 0..6 {
                assert!(
                    (cs[r][c] - want_cs[r * 6 + c]).abs() < 1e-9,
                    "cs[{r}][{c}] = {} vs {}",
                    cs[r][c],
                    want_cs[r * 6 + c]
                );
            }
        }
    }

    #[test]
    fn cs_angles_match_scipy() {
        // Case A: m=6, p=q=3, r=3, n11=0.
        let a_x11 = vec![
            vec![-0.5299718695, -0.2604647036, -0.4936740867],
            vec![-0.105673699, -0.0378100654, 0.1484074059],
            vec![0.3840964564, -0.4523877513, -0.6696005085],
        ];
        let a_theta = cs_angles(&a_x11, 3, 3, 6).unwrap();
        let a_want = [0.159776054, 0.8461927778, 1.45411065];
        for (g, w) in a_theta.iter().zip(a_want.iter()) {
            assert!((g - w).abs() < 1e-8, "A theta {g} vs {w}");
        }

        // Case B: m=8, p=q=5, r=3, n11=2 (identity block present).
        let b_x11 = vec![
            vec![-0.597715282, -0.2284842244, 6.461e-06, -0.6729100837, 0.2199323067],
            vec![-0.1438553634, -0.4268619363, -0.5315047497, -0.0442671386, -0.2683036699],
            vec![-0.2517464206, 0.5361883302, -0.0607574542, -0.0670250601, -0.7047191734],
            vec![-0.2152718386, -0.016432015, 0.7485130952, -0.1263420574, -0.2090926243],
            vec![0.0634716335, -0.1290424626, 0.0822154112, -0.0422565627, -0.1356398819],
        ];
        let b_theta = cs_angles(&b_x11, 5, 5, 8).unwrap();
        let b_want = [0.4701951603, 1.0653266212, 1.4673215];
        for (g, w) in b_theta.iter().zip(b_want.iter()) {
            assert!((g - w).abs() < 1e-7, "B theta {g} vs {w}");
        }
    }

    #[test]
    fn build_cs_matrix_matches_scipy() {
        // scipy.linalg.cossin CS oracles (separate=False, swap_sign=False).
        assert_cs(
            &[0.27701, 0.340936, 1.327589, 1.483862],
            4,
            4,
            8,
            &[
                0.961877, 0.0, 0.0, 0.0, -0.273481, 0.0, 0.0, 0.0, 0.0, 0.942442, 0.0, 0.0, 0.0,
                -0.334369, 0.0, 0.0, 0.0, 0.0, 0.240817, 0.0, 0.0, 0.0, -0.970571, 0.0, 0.0, 0.0,
                0.0, 0.086825, 0.0, 0.0, 0.0, -0.996224, 0.273481, 0.0, 0.0, 0.0, 0.961877, 0.0,
                0.0, 0.0, 0.0, 0.334369, 0.0, 0.0, 0.0, 0.942442, 0.0, 0.0, 0.0, 0.0, 0.970571, 0.0,
                0.0, 0.0, 0.240817, 0.0, 0.0, 0.0, 0.0, 0.996224, 0.0, 0.0, 0.0, 0.086825,
            ],
        );
        // Unbalanced p<q.
        assert_cs(
            &[0.042939, 0.54894, 0.872127],
            3,
            5,
            8,
            &[
                0.999078, 0.0, 0.0, 0.0, 0.0, -0.042926, 0.0, 0.0, 0.0, 0.853078, 0.0, 0.0, 0.0,
                0.0, -0.521784, 0.0, 0.0, 0.0, 0.643199, 0.0, 0.0, 0.0, 0.0, -0.765699, 0.042926,
                0.0, 0.0, 0.0, 0.0, 0.999078, 0.0, 0.0, 0.0, 0.521784, 0.0, 0.0, 0.0, 0.0, 0.853078,
                0.0, 0.0, 0.0, 0.765699, 0.0, 0.0, 0.0, 0.0, 0.643199, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
        );
        // Smaller unbalanced p=2,q=3,m=6.
        assert_cs(
            &[0.193469, 0.956576],
            2,
            3,
            6,
            &[
                0.981343, 0.0, 0.0, 0.0, -0.192264, 0.0, 0.0, 0.576321, 0.0, 0.0, 0.0, -0.817223,
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.192264, 0.0, 0.0, 0.0, 0.981343, 0.0, 0.0, 0.817223,
                0.0, 0.0, 0.0, 0.576321, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
        );
    }

    #[test]
    fn dlarf_left_right_match_explicit_h() {
        // Build H explicitly from (v, tau) and compare to dlarf application.
        let x = [2.0, -1.0, 3.0];
        let mut tail = x[1..].to_vec();
        let (_, tau) = dlarfgp(x[0], &mut tail);
        let v = [1.0, tail[0], tail[1]];
        // Explicit H = I - tau v vᵀ.
        let mut h = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = (if i == j { 1.0 } else { 0.0 }) - tau * v[i] * v[j];
            }
        }
        // Random matrix C.
        let c0 = vec![
            vec![1.0, 2.0, -1.0],
            vec![0.5, -3.0, 2.0],
            vec![4.0, 1.0, 0.0],
        ];
        // Left: H·C.
        let mut left = c0.clone();
        dlarf_left(&mut left, &tail, tau);
        for i in 0..3 {
            for j in 0..3 {
                let expect: f64 = (0..3).map(|k| h[i][k] * c0[k][j]).sum();
                assert!((left[i][j] - expect).abs() < 1e-12, "left[{i}][{j}]");
            }
        }
        // Right: C·H.
        let mut right = c0.clone();
        dlarf_right(&mut right, &tail, tau);
        for i in 0..3 {
            for j in 0..3 {
                let expect: f64 = (0..3).map(|k| c0[i][k] * h[k][j]).sum();
                assert!((right[i][j] - expect).abs() < 1e-12, "right[{i}][{j}]");
            }
        }
    }
}
