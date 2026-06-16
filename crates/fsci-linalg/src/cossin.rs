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

#[cfg(test)]
mod tests {
    use super::*;

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
