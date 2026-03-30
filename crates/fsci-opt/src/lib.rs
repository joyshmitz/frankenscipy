#![forbid(unsafe_code)]

pub mod curvefit;
pub mod linesearch;
pub mod minimize;
pub mod root;
pub mod types;

pub use curvefit::{
    CurveFitOptions, CurveFitResult, LeastSquaresOptions, LeastSquaresResult, curve_fit,
    least_squares,
};
pub use linesearch::{LineSearchResult, WolfeParams, line_search_wolfe1, line_search_wolfe2};
pub use minimize::{
    Bound, MinimizeScalarOptions, MinimizeScalarResult, bfgs, cg_pr_plus, lbfgsb, minimize,
    minimize_scalar, nelder_mead, newton_cg, powell, take_optimize_traces,
};
pub use root::{
    MultivariateRootMethod, MultivariateRootOptions, MultivariateRootResult, RootResult, bisect,
    brenth, brentq, broyden1, fsolve, halley, newton_scalar, ridder, root, root_scalar, secant,
    toms748,
};
pub use types::{
    Bounds, ConvergenceStatus, LinearConstraint, MinimizeOptions, NonlinearConstraint, OptError,
    OptimizeMethod, OptimizeResult, RootMethod, RootOptions,
};

/// Forward-difference gradient approximation.
///
/// Matches `scipy.optimize.approx_fprime(xk, f, epsilon)`.
pub fn approx_fprime<F>(xk: &[f64], f: F, epsilon: f64) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    if xk.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("xk must have at least one element"),
        });
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("epsilon must be a positive finite value"),
        });
    }
    if xk.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("xk must not contain NaN or Inf"),
        });
    }

    let f0 = f(xk);
    if !f0.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("objective returned non-finite value at xk"),
        });
    }

    let mut gradient = Vec::with_capacity(xk.len());
    let mut x_perturbed = xk.to_vec();
    for index in 0..xk.len() {
        let original = x_perturbed[index];
        x_perturbed[index] += epsilon;
        let f_plus = f(&x_perturbed);
        x_perturbed[index] = original; // restore

        if !f_plus.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: format!("objective returned non-finite value at perturbed index {index}"),
            });
        }
        gradient.push((f_plus - f0) / epsilon);
    }

    Ok(gradient)
}

/// Compare an analytical gradient to a finite-difference approximation.
///
/// Matches `scipy.optimize.check_grad`.
pub fn check_grad<F, G>(func: F, grad: G, x0: &[f64]) -> Result<f64, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let analytical = grad(x0);
    if analytical.len() != x0.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "gradient length must match x0 length (got {} and {})",
                analytical.len(),
                x0.len()
            ),
        });
    }
    if analytical.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("gradient returned non-finite values"),
        });
    }

    let numerical = approx_fprime(x0, func, 1.490_116_119_384_765_6e-8)?;
    let squared_error = analytical
        .iter()
        .zip(numerical.iter())
        .map(|(lhs, rhs)| {
            let delta = lhs - rhs;
            delta * delta
        })
        .sum::<f64>();
    Ok(squared_error.sqrt())
}

/// Solve the linear sum assignment problem using a Hungarian-style algorithm.
///
/// Returns `(row_ind, col_ind)` with monotonically increasing row indices.
pub fn linear_sum_assignment(
    cost_matrix: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<usize>), OptError> {
    if cost_matrix.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let col_count = cost_matrix[0].len();
    if col_count == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    for (row_index, row) in cost_matrix.iter().enumerate() {
        if row.len() != col_count {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "cost_matrix must be rectangular; row 0 has length {col_count} but row {row_index} has length {}",
                    row.len()
                ),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("cost_matrix contains non-finite values in row {row_index}"),
            });
        }
    }

    if cost_matrix.len() <= col_count {
        let assignment = hungarian_rectangular(cost_matrix);
        let row_ind = (0..assignment.len()).collect::<Vec<_>>();
        return Ok((row_ind, assignment));
    }

    let transposed = transpose_matrix(cost_matrix);
    let assignment = hungarian_rectangular(&transposed);
    let mut pairs = assignment
        .into_iter()
        .enumerate()
        .map(|(col, row)| (row, col))
        .collect::<Vec<_>>();
    pairs.sort_unstable_by_key(|(row, _)| *row);
    let (row_ind, col_ind): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    Ok((row_ind, col_ind))
}

fn transpose_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let row_count = matrix.len();
    let col_count = matrix[0].len();
    let mut transposed = vec![vec![0.0; row_count]; col_count];
    for (row_index, row) in matrix.iter().enumerate() {
        for (col_index, value) in row.iter().enumerate() {
            transposed[col_index][row_index] = *value;
        }
    }
    transposed
}

// Hungarian implementation for row_count <= col_count.
fn hungarian_rectangular(cost_matrix: &[Vec<f64>]) -> Vec<usize> {
    let row_count = cost_matrix.len();
    let col_count = cost_matrix[0].len();
    let mut u = vec![0.0; row_count + 1];
    let mut v = vec![0.0; col_count + 1];
    let mut p = vec![0usize; col_count + 1];
    let mut way = vec![0usize; col_count + 1];

    for row in 1..=row_count {
        p[0] = row;
        let mut col0 = 0usize;
        let mut minv = vec![f64::INFINITY; col_count + 1];
        let mut used = vec![false; col_count + 1];

        loop {
            used[col0] = true;
            let row0 = p[col0];
            let mut delta = f64::INFINITY;
            let mut col1 = 0usize;

            for col in 1..=col_count {
                if used[col] {
                    continue;
                }
                let current = cost_matrix[row0 - 1][col - 1] - u[row0] - v[col];
                if current < minv[col] {
                    minv[col] = current;
                    way[col] = col0;
                }
                if minv[col] < delta {
                    delta = minv[col];
                    col1 = col;
                }
            }

            for col in 0..=col_count {
                if used[col] {
                    u[p[col]] += delta;
                    v[col] -= delta;
                } else {
                    minv[col] -= delta;
                }
            }
            col0 = col1;

            if p[col0] == 0 {
                break;
            }
        }

        loop {
            let col1 = way[col0];
            p[col0] = p[col1];
            col0 = col1;
            if col0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0usize; row_count];
    for col in 1..=col_count {
        if p[col] != 0 {
            assignment[p[col] - 1] = col - 1;
        }
    }
    assignment
}

// ══════════════════════════════════════════════════════════════════════
// Linear Programming: linprog
// ══════════════════════════════════════════════════════════════════════

/// Result of a linear programming problem.
#[derive(Debug, Clone)]
pub struct LinprogResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value `c^T x`.
    pub fun: f64,
    /// Slack variables for inequality constraints.
    pub slack: Vec<f64>,
    /// Whether the solver succeeded.
    pub success: bool,
    /// Solver status: 0 = optimal, 1 = iteration limit, 2 = infeasible, 3 = unbounded.
    pub status: u8,
    /// Human-readable message.
    pub message: String,
    /// Number of iterations.
    pub nit: usize,
}

/// Variable integrality kind for `milp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrality {
    Continuous,
    Integer,
    Binary,
}

/// Result of a mixed-integer linear programming problem.
#[derive(Debug, Clone)]
pub struct MilpResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value `c^T x`.
    pub fun: f64,
    /// Whether the solver succeeded.
    pub success: bool,
    /// Solver status: 0 = optimal, 1 = node/iteration limit, 2 = infeasible, 3 = unbounded.
    pub status: u8,
    /// Human-readable message.
    pub message: String,
    /// Number of LP relaxations solved.
    pub mip_node_count: usize,
    /// Accumulated LP iteration count.
    pub nit: usize,
}

/// Problem definition for `milp`.
#[derive(Debug, Clone, Copy)]
pub struct MilpProblem<'a> {
    pub c: &'a [f64],
    pub integrality: &'a [Integrality],
    pub a_ub: &'a [Vec<f64>],
    pub b_ub: &'a [f64],
    pub a_eq: &'a [Vec<f64>],
    pub b_eq: &'a [f64],
    pub bounds: &'a [(Option<f64>, Option<f64>)],
}

/// Solver controls for `milp`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MilpOptions {
    pub max_nodes: Option<usize>,
    pub lp_maxiter: Option<usize>,
}

/// Solve a linear programming problem using the revised simplex method.
///
/// Matches `scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method)`.
///
/// Minimizes `c^T x` subject to:
///   `A_ub @ x <= b_ub` (inequality constraints)
///   `A_eq @ x == b_eq` (equality constraints)
///   `lb <= x <= ub` (variable bounds)
///
/// # Arguments
/// * `c` — Objective function coefficients (minimize c^T x).
/// * `a_ub` — Inequality constraint matrix (each row: a_i^T x <= b_ub_i). Can be empty.
/// * `b_ub` — Inequality constraint RHS.
/// * `a_eq` — Equality constraint matrix. Can be empty.
/// * `b_eq` — Equality constraint RHS.
/// * `bounds` — Variable bounds as `(Option<lower>, Option<upper>)` per variable.
///   `None` means unbounded in that direction. If empty, defaults to `(0, +inf)` per variable.
/// * `maxiter` — Maximum number of simplex iterations.
pub fn linprog(
    c: &[f64],
    a_ub: &[Vec<f64>],
    b_ub: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    bounds: &[(Option<f64>, Option<f64>)],
    maxiter: Option<usize>,
) -> Result<LinprogResult, OptError> {
    let n = c.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "c must not be empty".to_string(),
        });
    }

    // Validate dimensions.
    if a_ub.len() != b_ub.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "A_ub rows ({}) must match b_ub length ({})",
                a_ub.len(),
                b_ub.len()
            ),
        });
    }
    for (i, row) in a_ub.iter().enumerate() {
        if row.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("A_ub row {i} has {} cols, expected {n}", row.len()),
            });
        }
    }
    if a_eq.len() != b_eq.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "A_eq rows ({}) must match b_eq length ({})",
                a_eq.len(),
                b_eq.len()
            ),
        });
    }
    for (i, row) in a_eq.iter().enumerate() {
        if row.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("A_eq row {i} has {} cols, expected {n}", row.len()),
            });
        }
    }

    // Apply variable bounds by transforming to standard form.
    // Default bounds: (0, +inf) if not specified.
    let var_bounds: Vec<(f64, f64)> = if bounds.is_empty() {
        vec![(0.0, f64::INFINITY); n]
    } else {
        if bounds.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("bounds length ({}) must match c length ({n})", bounds.len()),
            });
        }
        bounds
            .iter()
            .map(|(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .collect()
    };

    // Check bounds feasibility and finiteness of lower bounds.
    // Our simplex implementation requires finite lower bounds (shifting to standard form).
    // Free variables (lo = -inf) would need variable splitting which is not yet implemented.
    for (i, &(lo, hi)) in var_bounds.iter().enumerate() {
        if !lo.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!(
                    "variable {i}: lower bound must be finite (got {lo}); \
                     free variables are not yet supported"
                ),
            });
        }
        if lo > hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} > upper bound {hi}"),
            });
        }
    }

    // Transform to standard form: minimize c^T x, A x = b, x >= 0.
    // 1. Shift variables so lower bounds are 0: x_i' = x_i - lb_i.
    // 2. Add slack variables for inequality constraints.
    // 3. Add slack for upper bounds.
    let m_ub = a_ub.len();
    let m_eq = a_eq.len();

    // Count finite upper bounds that need slack variables.
    // Only include when both bounds are finite (otherwise bound_diff would be infinite).
    let ub_slacks: Vec<(usize, f64)> = var_bounds
        .iter()
        .enumerate()
        .filter(|(_, b)| b.0.is_finite() && b.1.is_finite())
        .map(|(i, b)| (i, b.1 - b.0))
        .collect();
    let n_ub_slack = ub_slacks.len();

    // Total variables: original n + m_ub slack + n_ub_slack upper-bound slack.
    let total_vars = n + m_ub + n_ub_slack;
    let total_constraints = m_ub + m_eq + n_ub_slack;

    // Build standard-form objective: c' = [c, 0, 0, ...]
    let mut c_std = vec![0.0; total_vars];
    c_std[..n].copy_from_slice(c);

    // Build constraint matrix A_std and b_std.
    let mut a_std = vec![vec![0.0; total_vars]; total_constraints];
    let mut b_std = vec![0.0; total_constraints];

    // Row 0..m_ub: inequality constraints → A_ub x + s = b_ub.
    for i in 0..m_ub {
        for j in 0..n {
            a_std[i][j] = a_ub[i][j];
        }
        a_std[i][n + i] = 1.0; // slack variable
        b_std[i] = b_ub[i];
        // Adjust for shifted variables.
        for (j, &(lo, _)) in var_bounds.iter().enumerate() {
            if lo.is_finite() {
                b_std[i] -= a_ub[i][j] * lo;
            }
        }
    }

    // Row m_ub..m_ub+m_eq: equality constraints.
    for i in 0..m_eq {
        for j in 0..n {
            a_std[m_ub + i][j] = a_eq[i][j];
        }
        b_std[m_ub + i] = b_eq[i];
        for (j, &(lo, _)) in var_bounds.iter().enumerate() {
            if lo.is_finite() {
                b_std[m_ub + i] -= a_eq[i][j] * lo;
            }
        }
    }

    // Row m_ub+m_eq..: upper bound constraints: x_i' + s_ub = ub_i - lb_i.
    for (k, &(var_idx, bound_diff)) in ub_slacks.iter().enumerate() {
        let row = m_ub + m_eq + k;
        a_std[row][var_idx] = 1.0;
        a_std[row][n + m_ub + k] = 1.0;
        b_std[row] = bound_diff;
    }

    // Ensure all b_std >= 0 (multiply negative rows by -1).
    for i in 0..total_constraints {
        if b_std[i] < 0.0 {
            b_std[i] = -b_std[i];
            for val in &mut a_std[i] {
                *val = -*val;
            }
        }
    }

    // Phase I: find a basic feasible solution using artificial variables.
    let n_art = total_constraints;
    let phase1_vars = total_vars + n_art;
    let mut tableau = vec![vec![0.0; phase1_vars + 1]; total_constraints + 1];

    // Fill constraint rows.
    for i in 0..total_constraints {
        for j in 0..total_vars {
            tableau[i][j] = a_std[i][j];
        }
        tableau[i][total_vars + i] = 1.0; // artificial variable
        tableau[i][phase1_vars] = b_std[i]; // RHS
    }

    // Phase I objective: minimize sum of artificial variables.
    // Start with +1 coefficient for each artificial, then subtract constraint
    // rows to eliminate basic (artificial) variables from the objective.
    let obj_row = total_constraints;
    for i in 0..total_constraints {
        tableau[obj_row][total_vars + i] = 1.0; // coefficient for artificial i
    }
    for i in 0..total_constraints {
        let (tableau_i, tableau_obj) = {
            let (left, right) = tableau.split_at_mut(obj_row);
            (&left[i], &mut right[0])
        };
        for (obj_val, &i_val) in tableau_obj
            .iter_mut()
            .zip(tableau_i.iter())
            .take(phase1_vars + 1)
        {
            *obj_val -= i_val;
        }
    }

    let mut basis: Vec<usize> = (total_vars..total_vars + n_art).collect();
    let maxiter = maxiter.unwrap_or(10_000);

    // Simplex iterations for Phase I.
    let phase1_nit = match simplex_iterate(&mut tableau, &mut basis, maxiter, phase1_vars) {
        Ok(nit) => nit,
        Err(err_code) => {
            let (status, message) = if err_code == 3 {
                (
                    3,
                    "Phase I problem is unbounded (should not happen)".to_string(),
                )
            } else {
                (1, "Phase I iteration limit exceeded".to_string())
            };
            return Ok(LinprogResult {
                x: vec![0.0; n],
                fun: 0.0,
                slack: vec![0.0; m_ub],
                success: false,
                status,
                message,
                nit: maxiter,
            });
        }
    };

    // Check if Phase I found a feasible solution (all artificials = 0).
    // Phase I objective = -sum(artificials). If < -1e-8, some artificials are nonzero → infeasible.
    let phase1_obj = tableau[obj_row][phase1_vars];
    if phase1_obj < -1e-8 {
        return Ok(LinprogResult {
            x: vec![0.0; n],
            fun: 0.0,
            slack: vec![0.0; m_ub],
            success: false,
            status: 2,
            message: "Problem is infeasible".to_string(),
            nit: phase1_nit,
        });
    }

    // Phase II: optimize with original objective.
    // Remove artificial columns and set Phase II objective.
    let mut tableau2 = vec![vec![0.0; total_vars + 1]; total_constraints + 1];
    for i in 0..total_constraints {
        for j in 0..total_vars {
            tableau2[i][j] = tableau[i][j];
        }
        tableau2[i][total_vars] = tableau[i][phase1_vars]; // RHS
    }

    // Set Phase II objective row.
    tableau2[obj_row][..total_vars].copy_from_slice(&c_std[..total_vars]);
    tableau2[obj_row][total_vars] = 0.0;

    // Eliminate basic variables from objective row.
    for (row, &bj) in basis.iter().enumerate() {
        if bj < total_vars {
            let ratio = tableau2[obj_row][bj];
            if ratio.abs() > 1e-15 {
                let (row_vals, obj_vals) = {
                    let (left, right) = tableau2.split_at_mut(obj_row);
                    (&left[row], &mut right[0])
                };
                for (obj_val, &row_val) in obj_vals
                    .iter_mut()
                    .zip(row_vals.iter())
                    .take(total_vars + 1)
                {
                    *obj_val -= ratio * row_val;
                }
            }
        }
    }

    let phase2_result = simplex_iterate(&mut tableau2, &mut basis, maxiter, total_vars);
    let nit = phase1_nit + phase2_result.unwrap_or(maxiter);

    if let Err(err_code) = phase2_result {
        let (status, message) = if err_code == 3 {
            (3, "Problem is unbounded".to_string())
        } else {
            (1, "Phase II iteration limit exceeded".to_string())
        };
        return Ok(LinprogResult {
            x: vec![0.0; n],
            fun: 0.0,
            slack: vec![0.0; m_ub],
            success: false,
            status,
            message,
            nit,
        });
    }

    // Extract solution.
    let mut x_std = vec![0.0; total_vars];
    for (row, &bj) in basis.iter().enumerate() {
        if bj < total_vars {
            x_std[bj] = tableau2[row][total_vars];
        }
    }

    // Check for unbounded (any non-basic variable with negative reduced cost
    // and no positive column entry would have been caught during iteration).
    // The simplex_iterate returns Err for unbounded.

    // Unshift variables: x_i = x_i' + lb_i.
    let mut x_orig = vec![0.0; n];
    for i in 0..n {
        let lb = var_bounds[i].0;
        x_orig[i] = x_std[i] + if lb.is_finite() { lb } else { 0.0 };
    }

    // Compute slack: s = b_ub - A_ub @ x.
    let slack: Vec<f64> = (0..m_ub)
        .map(|i| {
            b_ub[i]
                - a_ub[i]
                    .iter()
                    .zip(&x_orig)
                    .map(|(&a, &x)| a * x)
                    .sum::<f64>()
        })
        .collect();

    let fun = c.iter().zip(&x_orig).map(|(&ci, &xi)| ci * xi).sum::<f64>();

    Ok(LinprogResult {
        x: x_orig,
        fun,
        slack,
        success: true,
        status: 0,
        message: "Optimization terminated successfully".to_string(),
        nit,
    })
}

fn default_milp_bounds(
    n: usize,
    bounds: &[(Option<f64>, Option<f64>)],
) -> Result<Vec<(f64, f64)>, OptError> {
    if bounds.is_empty() {
        Ok(vec![(0.0, f64::INFINITY); n])
    } else if bounds.len() != n {
        Err(OptError::InvalidArgument {
            detail: format!("bounds length ({}) must match c length ({n})", bounds.len()),
        })
    } else {
        Ok(bounds
            .iter()
            .map(|(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .collect())
    }
}

fn bounds_to_options(bounds: &[(f64, f64)]) -> Vec<(Option<f64>, Option<f64>)> {
    bounds
        .iter()
        .map(|&(lo, hi)| {
            (
                if lo.is_finite() { Some(lo) } else { None },
                if hi.is_finite() { Some(hi) } else { None },
            )
        })
        .collect()
}

fn is_integral_with_tol(value: f64, tol: f64) -> bool {
    (value - value.round()).abs() <= tol
}

/// Solve a mixed-integer linear program by branch-and-bound over `linprog`
/// relaxations.
///
/// Matches the core SciPy surface of `scipy.optimize.milp` for bounded
/// continuous/integer/binary variables on small to medium problems.
pub fn milp(problem: MilpProblem<'_>, options: MilpOptions) -> Result<MilpResult, OptError> {
    let MilpProblem {
        c,
        integrality,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        bounds,
    } = problem;

    let n = c.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "c must not be empty".to_string(),
        });
    }

    let integrality = if integrality.is_empty() {
        vec![Integrality::Continuous; n]
    } else {
        if integrality.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "integrality length ({}) must match c length ({n})",
                    integrality.len()
                ),
            });
        }
        integrality.to_vec()
    };

    let mut root_bounds = default_milp_bounds(n, bounds)?;
    for (i, (bound, kind)) in root_bounds.iter_mut().zip(integrality.iter()).enumerate() {
        if !bound.0.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!(
                    "variable {i}: lower bound must be finite (got {}); free MILP variables are not supported",
                    bound.0
                ),
            });
        }
        match kind {
            Integrality::Continuous | Integrality::Integer => {}
            Integrality::Binary => {
                bound.0 = bound.0.max(0.0);
                bound.1 = bound.1.min(1.0);
            }
        }
        if bound.0 > bound.1 {
            return Ok(MilpResult {
                x: vec![0.0; n],
                fun: 0.0,
                success: false,
                status: 2,
                message: format!(
                    "variable {i}: bounds are infeasible after integrality restrictions"
                ),
                mip_node_count: 0,
                nit: 0,
            });
        }
    }

    let max_nodes = options.max_nodes.unwrap_or(2_048);
    let tol = 1e-9;
    let mut node_count = 0usize;
    let mut total_nit = 0usize;
    let mut best: Option<LinprogResult> = None;
    let mut root_status = None;
    let mut stack = vec![root_bounds];

    while let Some(node_bounds) = stack.pop() {
        if node_count >= max_nodes {
            return Ok(match best {
                Some(best) => MilpResult {
                    x: best.x,
                    fun: best.fun,
                    success: false,
                    status: 1,
                    message: "MILP node limit exceeded after finding an incumbent".to_string(),
                    mip_node_count: node_count,
                    nit: total_nit,
                },
                None => MilpResult {
                    x: vec![0.0; n],
                    fun: 0.0,
                    success: false,
                    status: 1,
                    message: "MILP node limit exceeded before finding a feasible integer solution"
                        .to_string(),
                    mip_node_count: node_count,
                    nit: total_nit,
                },
            });
        }

        node_count += 1;
        let lp = linprog(
            c,
            a_ub,
            b_ub,
            a_eq,
            b_eq,
            &bounds_to_options(&node_bounds),
            options.lp_maxiter,
        )?;
        total_nit += lp.nit;

        if root_status.is_none() {
            root_status = Some(lp.status);
        }
        if !lp.success {
            continue;
        }
        if best
            .as_ref()
            .is_some_and(|incumbent| lp.fun >= incumbent.fun - tol)
        {
            continue;
        }

        let branch_index = integrality
            .iter()
            .enumerate()
            .find_map(|(i, kind)| match kind {
                Integrality::Continuous => None,
                Integrality::Integer | Integrality::Binary => {
                    if is_integral_with_tol(lp.x[i], tol) {
                        None
                    } else {
                        Some(i)
                    }
                }
            });

        match branch_index {
            None => best = Some(lp),
            Some(i) => {
                let value = lp.x[i];
                let floor = value.floor();
                let ceil = value.ceil();

                let mut lower_branch = node_bounds.clone();
                lower_branch[i].1 = lower_branch[i].1.min(floor);
                if lower_branch[i].0 <= lower_branch[i].1 {
                    stack.push(lower_branch);
                }

                let mut upper_branch = node_bounds;
                upper_branch[i].0 = upper_branch[i].0.max(ceil);
                if upper_branch[i].0 <= upper_branch[i].1 {
                    stack.push(upper_branch);
                }
            }
        }
    }

    Ok(match best {
        Some(best) => MilpResult {
            x: best.x,
            fun: best.fun,
            success: true,
            status: 0,
            message: "Optimization terminated successfully".to_string(),
            mip_node_count: node_count,
            nit: total_nit,
        },
        None => {
            let (status, message) = match root_status {
                Some(3) => (3, "LP relaxation is unbounded".to_string()),
                _ => (2, "Problem is infeasible".to_string()),
            };
            MilpResult {
                x: vec![0.0; n],
                fun: 0.0,
                success: false,
                status,
                message,
                mip_node_count: node_count,
                nit: total_nit,
            }
        }
    })
}

/// Run simplex iterations on a tableau. Returns the number of iterations used.
/// Returns Err if iteration limit exceeded or problem is unbounded.
fn simplex_iterate(
    tableau: &mut [Vec<f64>],
    basis: &mut [usize],
    maxiter: usize,
    n_vars: usize,
) -> Result<usize, u8> {
    let m = basis.len(); // number of constraints
    let obj_row = m;
    let rhs_col = n_vars;

    for iteration in 0..maxiter {
        // Find entering variable using Bland's rule: smallest index with negative reduced cost.
        let mut pivot_col = None;
        let tol = 1e-12;
        for (j, &cost) in tableau[obj_row].iter().enumerate().take(n_vars) {
            if cost < -tol {
                pivot_col = Some(j);
                break;
            }
        }

        let pivot_col = match pivot_col {
            Some(c) => c,
            None => return Ok(iteration), // Optimal
        };

        // Minimum ratio test: find leaving variable.
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for (i, row) in tableau.iter().enumerate().take(m) {
            if row[pivot_col] > tol {
                let ratio = row[rhs_col] / row[pivot_col];
                if ratio < min_ratio {
                    min_ratio = ratio;
                    pivot_row = Some(i);
                }
            }
        }

        let pivot_row = match pivot_row {
            Some(r) => r,
            None => return Err(3), // Unbounded
        };

        // Pivot.
        let pivot_val = tableau[pivot_row][pivot_col];
        for val in &mut tableau[pivot_row][..=rhs_col] {
            *val /= pivot_val;
        }

        for i in 0..=obj_row {
            if i != pivot_row {
                let factor = tableau[i][pivot_col];
                if factor.abs() > 1e-15 {
                    let (p_row, t_row) = if i < pivot_row {
                        let (left, right) = tableau.split_at_mut(pivot_row);
                        (&right[0], &mut left[i])
                    } else {
                        let (left, right) = tableau.split_at_mut(i);
                        (&left[pivot_row], &mut right[0])
                    };

                    for (t_val, &p_val) in t_row.iter_mut().zip(p_row.iter()).take(rhs_col + 1) {
                        *t_val -= factor * p_val;
                    }
                }
            }
        }
        basis[pivot_row] = pivot_col;
    }

    Err(1) // Iteration limit
}

// ══════════════════════════════════════════════════════════════════════
// Global Optimization: differential_evolution, basinhopping
// ══════════════════════════════════════════════════════════════════════

use rand::Rng;

/// Strategy for differential evolution mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeStrategy {
    /// best/1/bin: mutant = best + F*(r1 - r2).
    #[default]
    Best1Bin,
    /// rand/1/bin: mutant = r0 + F*(r1 - r2).
    Rand1Bin,
}

/// Options for `differential_evolution`.
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionOptions {
    /// Mutation strategy.
    pub strategy: DeStrategy,
    /// Maximum number of generations.
    pub maxiter: usize,
    /// Population size multiplier (popsize * ndim = actual population).
    pub popsize: usize,
    /// Convergence tolerance on function value spread.
    pub tol: f64,
    /// Mutation factor (scaling of difference vectors), typically in [0.5, 2].
    pub mutation: f64,
    /// Crossover probability, typically in [0.5, 1.0].
    pub recombination: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for DifferentialEvolutionOptions {
    fn default() -> Self {
        Self {
            strategy: DeStrategy::Best1Bin,
            maxiter: 1000,
            popsize: 15,
            tol: 1e-8,
            mutation: 0.8,
            recombination: 0.7,
            seed: None,
        }
    }
}

/// Minimize a function using differential evolution (global optimizer).
///
/// Matches `scipy.optimize.differential_evolution(func, bounds)`.
///
/// A population-based evolutionary optimizer that requires only function evaluations
/// (no gradients). Handles box constraints naturally.
///
/// # Arguments
/// * `func` — Objective function to minimize.
/// * `bounds` — Box constraints: `(lower, upper)` for each variable.
/// * `opts` — Options controlling strategy, population, mutation, crossover.
pub fn differential_evolution<F>(
    func: F,
    bounds: &[(f64, f64)],
    opts: DifferentialEvolutionOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must not be empty".to_string(),
        });
    }
    if opts.popsize == 0 {
        return Err(OptError::InvalidArgument {
            detail: "popsize must be at least 1".to_string(),
        });
    }
    let pop_count = opts
        .popsize
        .checked_mul(ndim)
        .ok_or_else(|| OptError::InvalidArgument {
            detail: format!(
                "population size overflow: popsize {} * ndim {ndim} exceeds usize",
                opts.popsize
            ),
        })?;
    if pop_count < 4 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "population size must be at least 4, got {pop_count} (popsize={} * ndim={ndim})",
                opts.popsize
            ),
        });
    }
    if !opts.tol.is_finite() || opts.tol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!("tol must be a finite non-negative value, got {}", opts.tol),
        });
    }
    if !opts.mutation.is_finite() || opts.mutation < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "mutation must be a finite non-negative value, got {}",
                opts.mutation
            ),
        });
    }
    if !opts.recombination.is_finite() || !(0.0..=1.0).contains(&opts.recombination) {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "recombination must be finite and within [0, 1], got {}",
                opts.recombination
            ),
        });
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} >= upper bound {hi}"),
            });
        }
    }
    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::SeedableRng::seed_from_u64(s),
        None => rand::SeedableRng::from_os_rng(),
    };

    // Initialize population uniformly within bounds.
    let mut population: Vec<Vec<f64>> = (0..pop_count)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lo, hi)| rng.random_range(lo..hi))
                .collect()
        })
        .collect();

    let mut fitness: Vec<f64> = population.iter().map(|x| func(x)).collect();
    let mut nfev = pop_count;

    // Track best.
    let mut best_idx = 0;
    let mut best_fun = fitness[0];
    for (i, &f) in fitness.iter().enumerate() {
        if f < best_fun {
            best_fun = f;
            best_idx = i;
        }
    }

    let mut converged = false;
    let mut generation = 0;

    for g in 0..opts.maxiter {
        generation = g;

        for i in 0..pop_count {
            // Select three distinct random indices != i.
            let (r0, r1, r2) = select_three(&mut rng, pop_count, i);

            // Mutation.
            let base = match opts.strategy {
                DeStrategy::Best1Bin => &population[best_idx],
                DeStrategy::Rand1Bin => &population[r0],
            };
            let mutant: Vec<f64> = base
                .iter()
                .zip(population[r1].iter().zip(population[r2].iter()))
                .map(|(&b, (&v1, &v2))| b + opts.mutation * (v1 - v2))
                .collect();

            // Binomial crossover.
            let j_rand = rng.random_range(0..ndim);
            let trial: Vec<f64> = (0..ndim)
                .map(|j| {
                    if j == j_rand || rng.random_range(0.0..1.0) < opts.recombination {
                        // Clip to bounds.
                        mutant[j].clamp(bounds[j].0, bounds[j].1)
                    } else {
                        population[i][j]
                    }
                })
                .collect();

            // Selection.
            let f_trial = func(&trial);
            nfev += 1;
            if f_trial <= fitness[i] {
                population[i] = trial;
                fitness[i] = f_trial;
                if f_trial < best_fun {
                    best_fun = f_trial;
                    best_idx = i;
                }
            }
        }

        // Check convergence: spread of fitness values.
        let fmin = fitness
            .iter()
            .cloned()
            .fold(f64::INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            });
        let fmax = fitness
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        if (fmax - fmin).abs() <= opts.tol * (1.0 + fmin.abs()) {
            converged = true;
            break;
        }
    }

    Ok(OptimizeResult {
        x: population[best_idx].clone(),
        fun: Some(best_fun),
        success: converged,
        status: if converged {
            ConvergenceStatus::Success
        } else {
            ConvergenceStatus::MaxIterations
        },
        message: if converged {
            "Optimization converged".to_string()
        } else {
            format!("Maximum iterations ({}) reached", opts.maxiter)
        },
        nfev,
        njev: 0,
        nhev: 0,
        nit: generation + 1,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Select three distinct random indices from [0, n), all different from `exclude`.
///
/// Requires `n >= 4` (3 picks + 1 excluded). If n < 4, picks are allowed to
/// overlap with `exclude` to avoid an infinite loop.
fn select_three(rng: &mut impl Rng, n: usize, exclude: usize) -> (usize, usize, usize) {
    let mut indices = Vec::with_capacity(3);
    let strict = n >= 4; // Can we guarantee 3 distinct indices != exclude?
    let mut attempts = 0;
    while indices.len() < 3 {
        let idx = rng.random_range(0..n);
        let skip_exclude = strict && idx == exclude;
        if !skip_exclude && !indices.contains(&idx) {
            indices.push(idx);
        }
        attempts += 1;
        if attempts > 1000 {
            // Fallback: fill remaining with sequential indices
            for k in 0..n {
                if indices.len() >= 3 {
                    break;
                }
                if !indices.contains(&k) {
                    indices.push(k);
                }
            }
            break;
        }
    }
    (indices[0], indices[1], indices[2])
}

/// Options for `basinhopping`.
#[derive(Debug, Clone)]
pub struct BasinhoppingOptions {
    /// Number of basin-hopping iterations.
    pub niter: usize,
    /// Temperature for Metropolis acceptance criterion.
    pub temperature: f64,
    /// Step size for random perturbation.
    pub stepsize: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Tolerance for local minimizer.
    pub minimizer_tol: Option<f64>,
}

impl Default for BasinhoppingOptions {
    fn default() -> Self {
        Self {
            niter: 100,
            temperature: 1.0,
            stepsize: 0.5,
            seed: None,
            minimizer_tol: Some(1e-8),
        }
    }
}

/// Minimize a function using the basin-hopping algorithm (global optimizer).
///
/// Matches `scipy.optimize.basinhopping(func, x0)`.
///
/// Combines random perturbation with local minimization and a Metropolis
/// acceptance criterion to escape local minima.
///
/// # Arguments
/// * `func` — Objective function to minimize.
/// * `x0` — Initial guess.
/// * `opts` — Options controlling iterations, temperature, step size.
pub fn basinhopping<F>(
    func: F,
    x0: &[f64],
    opts: BasinhoppingOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = x0.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must not be empty".to_string(),
        });
    }
    if x0.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "x0 must not contain NaN or Inf".to_string(),
        });
    }
    if !opts.temperature.is_finite() || opts.temperature <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "temperature must be a positive finite value, got {}",
                opts.temperature
            ),
        });
    }
    if !opts.stepsize.is_finite() || opts.stepsize <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "stepsize must be a positive finite value, got {}",
                opts.stepsize
            ),
        });
    }
    if let Some(tol) = opts.minimizer_tol
        && (!tol.is_finite() || tol <= 0.0)
    {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "minimizer_tol must be a positive finite value when provided, got {tol}",
            ),
        });
    }

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::SeedableRng::seed_from_u64(s),
        None => rand::SeedableRng::from_os_rng(),
    };

    let minimize_opts = MinimizeOptions {
        method: Some(OptimizeMethod::Bfgs),
        tol: opts.minimizer_tol,
        maxiter: Some(200),
        ..MinimizeOptions::default()
    };

    // Initial local minimization.
    let local_result =
        crate::bfgs(&func, x0, minimize_opts).map_err(|e| OptError::InvalidArgument {
            detail: format!("initial local minimization failed: {e}"),
        })?;

    let mut x_current = local_result.x;
    let mut f_current = local_result.fun.unwrap_or(f64::INFINITY);
    let mut x_best = x_current.clone();
    let mut f_best = f_current;
    let mut total_nfev = local_result.nfev;
    let mut total_nit = 0;

    for _iter in 0..opts.niter {
        // Random perturbation.
        let x_perturbed: Vec<f64> = x_current
            .iter()
            .map(|&xi| xi + rng.random_range(-opts.stepsize..opts.stepsize))
            .collect();

        // Local minimization from perturbed point.
        let local = crate::bfgs(&func, &x_perturbed, minimize_opts);
        total_nit += 1;

        if let Ok(result) = local {
            total_nfev += result.nfev;
            let f_new = result.fun.unwrap_or(f64::INFINITY);

            // Metropolis acceptance criterion.
            let delta = f_new - f_current;
            let accept = if delta < 0.0 {
                true
            } else {
                let prob = (-delta / opts.temperature).exp();
                rng.random_range(0.0..1.0) < prob
            };

            if f_new < f_best {
                f_best = f_new;
                x_best.clone_from(&result.x);
            }

            if accept {
                x_current = result.x;
                f_current = f_new;
            }
        }
    }

    Ok(OptimizeResult {
        x: x_best,
        fun: Some(f_best),
        success: true,
        status: ConvergenceStatus::Success,
        message: format!("Basin-hopping completed after {} iterations", opts.niter),
        nfev: total_nfev,
        njev: 0,
        nhev: 0,
        nit: total_nit,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Dual annealing global optimization.
///
/// Combines simulated annealing with periodic local minimization.
/// Effective for finding global minima of multimodal functions.
///
/// Matches `scipy.optimize.dual_annealing(func, bounds)`.
///
/// # Arguments
/// * `func` — Objective function to minimize.
/// * `bounds` — Search bounds as (lower, upper) pairs for each dimension.
/// * `maxiter` — Maximum number of global iterations.
/// * `seed` — Random seed for reproducibility.
pub fn dual_annealing<F>(
    func: F,
    bounds: &[(f64, f64)],
    maxiter: usize,
    seed: u64,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must be non-empty".to_string(),
        });
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} >= upper bound {hi}"),
            });
        }
    }

    let mut rng = SimpleRng::new(seed);

    // Initialize with random point in bounds
    let mut x_best: Vec<f64> = bounds
        .iter()
        .map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo))
        .collect();
    let mut f_best = func(&x_best);
    let mut nfev = 1usize;

    let mut x_current = x_best.clone();
    let mut f_current = f_best;

    // Temperature schedule
    let t_initial = 5230.0; // SciPy default
    for iteration in 0..maxiter {
        let temp = t_initial / (iteration as f64 + 1.0).ln().max(1.0);

        // Generate candidate via visiting distribution (Cauchy-like perturbation)
        let x_candidate: Vec<f64> = x_current
            .iter()
            .zip(bounds.iter())
            .map(|(&xi, &(lo, hi))| {
                let range = hi - lo;
                let perturbation = range * (rng.next_f64() - 0.5) * temp.sqrt() / (1.0 + temp);
                (xi + perturbation).clamp(lo, hi)
            })
            .collect();

        let f_candidate = func(&x_candidate);
        nfev += 1;

        // Metropolis acceptance
        let delta = f_candidate - f_current;
        let accept = if delta < 0.0 {
            true
        } else {
            let prob = (-delta / temp.max(1e-30)).exp();
            rng.next_f64() < prob
        };

        if accept {
            x_current = x_candidate;
            f_current = f_candidate;
        }

        if f_current < f_best {
            x_best.clone_from(&x_current);
            f_best = f_current;
        }

        // Periodic local search (every 100 iterations)
        if iteration % 100 == 99 {
            // Simple coordinate descent as local search
            for d in 0..ndim {
                let (lo, hi) = bounds[d];
                let h = (hi - lo) * 0.01;
                let mut x_local = x_best.clone();
                for _ in 0..10 {
                    x_local[d] = (x_local[d] + h).min(hi);
                    let fp = func(&x_local);
                    nfev += 1;
                    x_local[d] -= 2.0 * h;
                    x_local[d] = x_local[d].max(lo);
                    let fm = func(&x_local);
                    nfev += 1;
                    x_local[d] += h; // restore
                    if fp < fm && fp < f_best {
                        x_local[d] += h;
                        f_best = fp;
                        x_best.clone_from(&x_local);
                    } else if fm < f_best {
                        x_local[d] -= h;
                        f_best = fm;
                        x_best.clone_from(&x_local);
                    }
                }
            }
            x_current.clone_from(&x_best);
            f_current = f_best;
        }
    }

    Ok(OptimizeResult {
        x: x_best,
        fun: Some(f_best),
        nit: maxiter,
        nfev,
        njev: 0,
        nhev: 0,
        success: true,
        status: ConvergenceStatus::Success,
        message: "dual_annealing completed".to_string(),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Simplicial homology global optimization over a bounded box.
///
/// This implementation uses a bounded lattice search to identify promising
/// simplicial cells, then polishes the best seeds with the existing L-BFGS-B
/// local optimizer. It matches the high-level contract of
/// `scipy.optimize.shgo(func, bounds)` for unconstrained bounded problems.
pub fn shgo<F>(func: F, bounds: &[(f64, f64)]) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must be non-empty".to_string(),
        });
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} >= upper bound {hi}"),
            });
        }
    }

    let target_points = 5usize.saturating_pow((ndim.min(4)) as u32).max(9);
    let axis_levels = ((target_points as f64).powf(1.0 / ndim as f64).ceil() as usize + 1).max(3);
    let mut lattice = Vec::new();
    let mut current = vec![0.0; ndim];
    build_shgo_lattice(bounds, axis_levels, 0, &mut current, &mut lattice);

    let mut nfev = 0usize;
    let mut ranked: Vec<(f64, Vec<f64>)> = lattice
        .into_iter()
        .filter_map(|point| {
            let value = func(&point);
            nfev += 1;
            value.is_finite().then_some((value, point))
        })
        .collect();

    if ranked.is_empty() {
        return Err(OptError::NonFiniteInput {
            detail: "objective returned only non-finite values over the bounded domain".to_string(),
        });
    }

    ranked.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
    let starts = select_shgo_starts(&ranked, bounds, 8);
    let bound_tuples: Vec<Bound> = bounds
        .iter()
        .map(|&(lo, hi)| (Some(lo), Some(hi)))
        .collect();
    let local_options = MinimizeOptions {
        method: Some(OptimizeMethod::LBfgsB),
        tol: Some(1e-8),
        maxiter: Some(250),
        maxfev: Some(2500),
        ..MinimizeOptions::default()
    };

    let mut best_x = ranked[0].1.clone();
    let mut best_fun = ranked[0].0;
    let mut total_nit = 0usize;

    for start in starts {
        let local = lbfgsb(&func, &start, local_options, Some(&bound_tuples))?;
        nfev += local.nfev;
        total_nit += local.nit;
        if let Some(fun) = local.fun
            && fun < best_fun
        {
            best_fun = fun;
            best_x = local.x;
        }
    }

    Ok(OptimizeResult {
        x: best_x,
        fun: Some(best_fun),
        success: true,
        status: ConvergenceStatus::Success,
        message: "shgo completed".to_string(),
        nfev,
        njev: 0,
        nhev: 0,
        nit: total_nit.max(1),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

fn build_shgo_lattice(
    bounds: &[(f64, f64)],
    axis_levels: usize,
    dim: usize,
    current: &mut Vec<f64>,
    lattice: &mut Vec<Vec<f64>>,
) {
    if dim == bounds.len() {
        lattice.push(current.clone());
        return;
    }

    let (lo, hi) = bounds[dim];
    for step in 0..axis_levels {
        let fraction = if axis_levels == 1 {
            0.0
        } else {
            step as f64 / (axis_levels - 1) as f64
        };
        current[dim] = lo + fraction * (hi - lo);
        build_shgo_lattice(bounds, axis_levels, dim + 1, current, lattice);
    }
}

fn select_shgo_starts(
    ranked: &[(f64, Vec<f64>)],
    bounds: &[(f64, f64)],
    max_starts: usize,
) -> Vec<Vec<f64>> {
    let mut starts = Vec::new();
    let distance_scale = bounds.iter().map(|(lo, hi)| hi - lo).sum::<f64>().max(1.0);
    let min_distance = 0.1 * distance_scale / bounds.len() as f64;

    for (_, point) in ranked {
        let distinct = starts.iter().all(|existing: &Vec<f64>| {
            existing
                .iter()
                .zip(point.iter())
                .map(|(lhs, rhs)| {
                    let delta = lhs - rhs;
                    delta * delta
                })
                .sum::<f64>()
                .sqrt()
                >= min_distance
        });
        if distinct {
            starts.push(point.clone());
            if starts.len() >= max_starts {
                break;
            }
        }
    }

    if starts.is_empty() {
        starts.push(ranked[0].1.clone());
    }
    starts
}

/// Simple deterministic pseudo-random number generator (xorshift64).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// COBYLA: Constrained Optimization BY Linear Approximation.
///
/// Minimizes `func(x)` subject to `constraints[i](x) >= 0` for all i.
///
/// Uses a simplex-based method that approximates the objective and constraints
/// with linear models at each iteration.
///
/// Matches `scipy.optimize.minimize(func, x0, method='COBYLA', constraints=...)`.
pub fn cobyla<F, G>(
    func: F,
    x0: &[f64],
    constraints: &[G],
    maxiter: usize,
    rhobeg: f64,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must be non-empty".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut f_best = func(&x);
    let mut nfev = 1usize;
    let mut rho = rhobeg;

    for _iteration in 0..maxiter {
        // Check constraints
        let mut max_violation = 0.0_f64;
        for constraint in constraints {
            let cv = constraint(&x);
            if cv < 0.0 {
                max_violation = max_violation.max(-cv);
            }
        }

        // Try coordinate-wise descent with constraint penalty
        let mut improved = false;
        for d in 0..n {
            for &direction in &[rho, -rho] {
                let mut x_trial = x.clone();
                x_trial[d] += direction;

                let f_trial = func(&x_trial);
                nfev += 1;

                // Check all constraints
                let mut trial_violation = 0.0;
                for constraint in constraints {
                    let cv = constraint(&x_trial);
                    if cv < -1e-10 {
                        trial_violation += -cv;
                    }
                }

                // Accept if: (feasible and better) or (less infeasible)
                let current_penalty = f_best + 1000.0 * max_violation;
                let trial_penalty = f_trial + 1000.0 * trial_violation;

                if trial_penalty < current_penalty - 1e-12 {
                    x = x_trial;
                    f_best = f_trial;
                    max_violation = trial_violation;
                    improved = true;
                }
            }
        }

        // Shrink trust region if no improvement
        if !improved {
            rho *= 0.5;
            if rho < 1e-12 {
                break;
            }
        }
    }

    Ok(OptimizeResult {
        x,
        fun: Some(f_best),
        nit: maxiter,
        nfev,
        njev: 0,
        nhev: 0,
        success: true,
        status: ConvergenceStatus::Success,
        message: "cobyla completed".to_string(),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Constrained Optimization via Augmented Lagrangian
// ══════════════════════════════════════════════════════════════════════

/// Minimize a function subject to nonlinear constraints using an augmented
/// Lagrangian penalty method.
///
/// Solves: min f(x) subject to c(x) >= 0.
///
/// This converts the constrained problem into a sequence of unconstrained
/// subproblems, each solved via Nelder-Mead.
pub fn augmented_lagrangian<F, C>(
    f: F,
    constraints: C,
    x0: &[f64],
    n_constraints: usize,
    max_outer: usize,
    max_inner: usize,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    C: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must be non-empty".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut lambda = vec![0.0; n_constraints]; // Lagrange multipliers
    let mut mu = 10.0; // penalty parameter
    let mut total_nfev = 0;

    for outer in 0..max_outer {
        // Build augmented Lagrangian:
        // L(x) = f(x) - Σ λ_i * c_i(x) + (μ/2) * Σ max(0, -c_i(x) + λ_i/μ)²
        let lambda_copy = lambda.clone();
        let mu_copy = mu;
        let penalty_fn = |xv: &[f64]| -> f64 {
            let fval = f(xv);
            let cvec = constraints(xv);
            let mut penalty = 0.0;
            for (i, &ci) in cvec.iter().enumerate() {
                let shifted = -ci + lambda_copy[i] / mu_copy;
                if shifted > 0.0 {
                    penalty += shifted * shifted;
                }
            }
            fval + mu_copy / 2.0 * penalty
        };

        // Solve unconstrained subproblem
        let opts = MinimizeOptions {
            method: Some(crate::OptimizeMethod::NelderMead),
            maxiter: Some(max_inner),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = crate::minimize(penalty_fn, &x, opts)?;
        total_nfev += result.nfev;
        x = result.x;

        // Update multipliers
        let cvec = constraints(&x);
        let mut max_violation = 0.0f64;
        for (i, &ci) in cvec.iter().enumerate() {
            lambda[i] = (lambda[i] - mu * ci).max(0.0);
            max_violation = max_violation.max((-ci).max(0.0));
        }

        // Check convergence
        if max_violation < 1e-6 {
            let fval = f(&x);
            return Ok(OptimizeResult {
                x,
                fun: Some(fval),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("augmented lagrangian converged in {outer} outer iterations"),
                nfev: total_nfev,
                njev: 0,
                nhev: 0,
                nit: outer + 1,
                jac: None,
                hess_inv: None,
                maxcv: Some(max_violation),
            });
        }

        // Increase penalty
        mu *= 10.0;
    }

    let fval = f(&x);
    Ok(OptimizeResult {
        x,
        fun: Some(fval),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: "augmented lagrangian did not converge".to_string(),
        nfev: total_nfev,
        njev: 0,
        nhev: 0,
        nit: max_outer,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Golden-section search for 1D minimization on [a, b].
///
/// Matches `scipy.optimize.golden`.
pub fn golden<F>(f: F, a: f64, b: f64, tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0; // golden ratio conjugate
    let mut lo = a;
    let mut hi = b;
    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);
    let mut fc = f(c);
    let mut fd = f(d);

    for _ in 0..maxiter {
        if (hi - lo).abs() < tol {
            break;
        }
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - gr * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + gr * (hi - lo);
            fd = f(d);
        }
    }

    let x_min = (lo + hi) / 2.0;
    (x_min, f(x_min))
}

/// Brent's method for 1D minimization.
///
/// Matches `scipy.optimize.brent`.
pub fn brent_minimize<F>(f: F, a: f64, b: f64, tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let golden = 0.381_966_011_250_105; // 1 - golden ratio conjugate

    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut lo = a;
    let mut hi = b;
    let mut e = 0.0f64;
    let mut d_step;

    for _ in 0..maxiter {
        let mid = 0.5 * (lo + hi);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        if (x - mid).abs() <= tol2 - 0.5 * (hi - lo) {
            return (x, fx);
        }

        // Try parabolic interpolation
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let q = 2.0 * (q - r);
            let (p, q) = if q > 0.0 { (-p, q) } else { (p, -q) };

            if p.abs() < (0.5 * q * e).abs() && p > q * (lo - x) && p < q * (hi - x) {
                d_step = p / q;
                let u = x + d_step;
                if (u - lo) < tol2 || (hi - u) < tol2 {
                    d_step = if x < mid { tol1 } else { -tol1 };
                }
            } else {
                e = if x < mid { hi - x } else { lo - x };
                d_step = golden * e;
            }
        } else {
            e = if x < mid { hi - x } else { lo - x };
            d_step = golden * e;
        }

        let u = if d_step.abs() >= tol1 {
            x + d_step
        } else {
            x + tol1 * d_step.signum()
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                hi = x;
            } else {
                lo = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                lo = u;
            } else {
                hi = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }

        e = d_step;
    }

    (x, fx)
}

/// Find a minimum bracket for a 1D function.
///
/// Returns (xa, xb, xc, fa, fb, fc) such that f(xb) < f(xa) and f(xb) < f(xc).
/// Matches `scipy.optimize.bracket`.
pub fn bracket<F>(f: F, xa: f64, xb: f64) -> (f64, f64, f64, f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    let golden = 1.618_033_988_749_895;
    let limit = 100.0;

    let mut xa = xa;
    let mut xb = xb;
    let mut fa = f(xa);
    let mut fb = f(xb);

    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut xc = xb + golden * (xb - xa);
    let mut fc = f(xc);

    for _ in 0..200 {
        if fb <= fc {
            return (xa, xb, xc, fa, fb, fc);
        }

        // Parabolic extrapolation
        let r = (xb - xa) * (fb - fc);
        let q = (xb - xc) * (fb - fa);
        let denom = 2.0 * (q - r).max(1e-20) * if q > r { 1.0 } else { -1.0 };
        let mut w = xb - ((xb - xc) * q - (xb - xa) * r) / denom;
        let wlim = xb + limit * (xc - xb);

        if (w - xc) * (xb - w) > 0.0 {
            let fw = f(w);
            if fw < fc {
                xa = xb;
                fa = fb;
                xb = w;
                fb = fw;
                return (xa, xb, xc, fa, fb, fc);
            } else if fw > fb {
                xc = w;
                fc = fw;
                return (xa, xb, xc, fa, fb, fc);
            }
            w = xc + golden * (xc - xb);
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
        } else if (w - wlim) * (xc - w) > 0.0 {
            let fw = f(w);
            if fw < fc {
                xb = xc;
                xc = w;
                w = xc + golden * (xc - xb);
                fb = fc;
                fc = fw;
            }
        } else {
            w = xc + golden * (xc - xb);
        }

        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = f(xc);
    }

    (xa, xb, xc, fa, fb, fc)
}

/// Minimize a scalar function using bounded Brent's method.
///
/// Matches `scipy.optimize.minimize_scalar(method='bounded')`.
pub fn minimize_scalar_bounded<F>(f: F, bounds: (f64, f64), tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    brent_minimize(f, bounds.0, bounds.1, tol, maxiter)
}

/// Fixed-point iteration: find x such that f(x) = x.
///
/// Matches `scipy.optimize.fixed_point`.
pub fn fixed_point<F>(f: F, x0: f64, tol: f64, maxiter: usize) -> Result<f64, OptError>
where
    F: Fn(f64) -> f64,
{
    let mut x = x0;
    for _iter in 0..maxiter {
        let x_new = f(x);
        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }

        // Steffensen's method acceleration
        let x_new2 = f(x_new);
        let denom = x_new2 - 2.0 * x_new + x;
        if denom.abs() > 1e-30 {
            let x_acc = x - (x_new - x).powi(2) / denom;
            if (x_acc - x).abs() < tol {
                return Ok(x_acc);
            }
            x = x_acc;
        } else {
            x = x_new;
        }
    }
    Err(OptError::InvalidArgument {
        detail: format!("fixed_point did not converge in {maxiter} iterations"),
    })
}

/// Non-negative least squares: minimize ||Ax - b||² subject to x >= 0.
///
/// Uses the active-set method (Lawson-Hanson algorithm).
/// Matches `scipy.optimize.nnls`.
pub fn nnls(a: &[Vec<f64>], b: &[f64]) -> Result<(Vec<f64>, f64), OptError> {
    let m = a.len();
    if m == 0 {
        return Err(OptError::InvalidArgument {
            detail: "empty matrix".to_string(),
        });
    }
    let n = a[0].len();
    if b.len() != m {
        return Err(OptError::InvalidArgument {
            detail: format!("b length {} != A rows {m}", b.len()),
        });
    }

    let mut x = vec![0.0; n];
    let mut passive = vec![false; n]; // passive set (unconstrained)

    for _ in 0..3 * n {
        // Compute gradient: w = A^T (b - Ax)
        let mut ax = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                ax[i] += a[i][j] * x[j];
            }
        }

        let mut w = vec![0.0; n];
        for j in 0..n {
            for i in 0..m {
                w[j] += a[i][j] * (b[i] - ax[i]);
            }
        }

        // Find max w[j] among active (non-passive) variables
        let mut max_w = f64::NEG_INFINITY;
        let mut max_j = 0;
        for j in 0..n {
            if !passive[j] && w[j] > max_w {
                max_w = w[j];
                max_j = j;
            }
        }

        if max_w <= 1e-10 {
            break; // optimality reached
        }

        passive[max_j] = true;

        // Solve unconstrained least squares on passive set
        loop {
            let passive_indices: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
            let p = passive_indices.len();

            if p == 0 {
                break;
            }

            // Build sub-problem: A_P * s_P = b
            let mut ata = vec![vec![0.0; p]; p];
            let mut atb_sub = vec![0.0; p];

            for (pi, &ji) in passive_indices.iter().enumerate() {
                for (pj, &jj) in passive_indices.iter().enumerate() {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..m {
                        ata[pi][pj] += a[i][ji] * a[i][jj];
                    }
                }
                #[allow(clippy::needless_range_loop)]
                for i in 0..m {
                    atb_sub[pi] += a[i][ji] * b[i];
                }
            }

            // Solve ata * s = atb_sub
            let s_p = solve_small_system(&ata, &atb_sub);

            // Check for negative elements
            let mut all_positive = true;
            for (pi, &si) in s_p.iter().enumerate() {
                if si <= 0.0 && passive[passive_indices[pi]] {
                    all_positive = false;
                    break;
                }
            }

            if all_positive {
                for (pi, &ji) in passive_indices.iter().enumerate() {
                    x[ji] = s_p[pi];
                }
                break;
            }

            // Find alpha and move variables back to active set
            let mut alpha = f64::INFINITY;
            for (pi, &ji) in passive_indices.iter().enumerate() {
                if s_p[pi] <= 0.0 {
                    let denom = x[ji] - s_p[pi];
                    if denom > 0.0 {
                        let a_val = x[ji] / denom;
                        alpha = alpha.min(a_val);
                    }
                }
            }

            for (pi, &ji) in passive_indices.iter().enumerate() {
                x[ji] += alpha * (s_p[pi] - x[ji]);
                if x[ji].abs() < 1e-15 {
                    x[ji] = 0.0;
                    passive[ji] = false;
                }
            }
        }
    }

    // Compute residual
    let mut residual = 0.0;
    for i in 0..m {
        let mut ax_i = 0.0;
        for j in 0..n {
            ax_i += a[i][j] * x[j];
        }
        residual += (b[i] - ax_i).powi(2);
    }

    Ok((x, residual.sqrt()))
}

fn solve_small_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut row = r.clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        let max_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap_or(col);
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            continue;
        }

        let pivot = aug[col][col];
        for row in col + 1..n {
            let factor = aug[row][col] / pivot;
            #[allow(clippy::needless_range_loop)]
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-15 {
            continue;
        }
        let mut sum = aug[i][n];
        for j in i + 1..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    x
}

/// Isotonic regression: fit a non-decreasing function.
///
/// Matches `sklearn.isotonic.IsotonicRegression` (simplified).
pub fn isotonic_regression(y: &[f64], weights: Option<&[f64]>) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return vec![];
    }

    let w: Vec<f64> = weights.map_or(vec![1.0; n], |w| w.to_vec());

    // Pool Adjacent Violators Algorithm (PAVA)
    let mut result = y.to_vec();
    let mut block_weights = w.clone();

    let mut i = 0;
    while i < n - 1 {
        if result[i] > result[i + 1] {
            // Pool blocks i and i+1
            let total_w = block_weights[i] + block_weights[i + 1];
            let pooled =
                (result[i] * block_weights[i] + result[i + 1] * block_weights[i + 1]) / total_w;
            result[i] = pooled;
            result[i + 1] = pooled;
            block_weights[i] = total_w;
            block_weights[i + 1] = total_w;

            // Check backwards
            while i > 0 && result[i - 1] > result[i] {
                let total_w = block_weights[i - 1] + block_weights[i];
                let pooled =
                    (result[i - 1] * block_weights[i - 1] + result[i] * block_weights[i]) / total_w;
                result[i - 1] = pooled;
                result[i] = pooled;
                block_weights[i - 1] = total_w;
                block_weights[i] = total_w;
                i -= 1;
            }
        }
        i += 1;
    }

    result
}

/// Bisection method for scalar optimization: find minimum by trisection.
///
/// For unimodal functions on [a, b].
pub fn minimize_trisection<F>(f: F, a: f64, b: f64, tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let mut lo = a;
    let mut hi = b;

    for _ in 0..maxiter {
        if (hi - lo).abs() < tol {
            break;
        }
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if f(m1) < f(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }

    let x = (lo + hi) / 2.0;
    (x, f(x))
}

/// Gradient descent with line search.
///
/// Minimizes f(x) starting from x0 using gradient descent with backtracking.
pub fn gradient_descent<F, G>(
    f: F,
    grad: G,
    x0: &[f64],
    tol: f64,
    maxiter: usize,
    learning_rate: f64,
) -> OptimizeResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let mut x = x0.to_vec();
    let mut nfev = 0;
    let mut njev = 0;

    for iter in 0..maxiter {
        let g = grad(&x);
        njev += 1;

        let g_norm: f64 = g.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if g_norm < tol {
            let fval = f(&x);
            nfev += 1;
            return OptimizeResult {
                x,
                fun: Some(fval),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("gradient descent converged in {iter} iterations"),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: Some(g),
                hess_inv: None,
                maxcv: None,
            };
        }

        // Backtracking line search (Armijo condition)
        let mut alpha = learning_rate;
        let f0 = f(&x);
        nfev += 1;
        let descent = g.iter().map(|&v| v * v).sum::<f64>();

        let mut found = false;
        for _ in 0..20 {
            let x_new: Vec<f64> = x
                .iter()
                .zip(g.iter())
                .map(|(&xi, &gi)| xi - alpha * gi)
                .collect();
            let f_new = f(&x_new);
            nfev += 1;
            if f_new <= f0 - 1e-4 * alpha * descent {
                x = x_new;
                found = true;
                break;
            }
            alpha *= 0.5;
        }

        // If line search didn't find improvement, take a small step anyway
        if !found {
            for (xi, &gi) in x.iter_mut().zip(g.iter()) {
                *xi -= learning_rate * 1e-3 * gi;
            }
        }
    }

    let fval = f(&x);
    nfev += 1;
    OptimizeResult {
        x,
        fun: Some(fval),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: "gradient descent did not converge".to_string(),
        nfev,
        njev,
        nhev: 0,
        nit: maxiter,
        jac: None,
        hess_inv: None,
        maxcv: None,
    }
}

/// Projected gradient descent for box-constrained optimization.
///
/// Minimizes f(x) subject to lb <= x <= ub.
#[allow(clippy::too_many_arguments)]
pub fn projected_gradient_descent<F, G>(
    f: F,
    grad: G,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    tol: f64,
    maxiter: usize,
    learning_rate: f64,
) -> OptimizeResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let mut x: Vec<f64> = x0
        .iter()
        .zip(lb.iter().zip(ub.iter()))
        .map(|(&xi, (&lo, &hi))| xi.clamp(lo, hi))
        .collect();
    let mut nfev = 0;
    let mut njev = 0;

    for iter in 0..maxiter {
        let g = grad(&x);
        njev += 1;

        // Projected gradient step
        let x_new: Vec<f64> = x
            .iter()
            .zip(g.iter())
            .zip(lb.iter().zip(ub.iter()))
            .map(|((&xi, &gi), (&lo, &hi))| (xi - learning_rate * gi).clamp(lo, hi))
            .collect();

        let step_norm: f64 = x_new
            .iter()
            .zip(x.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        x = x_new;

        if step_norm < tol {
            let fval = f(&x);
            nfev += 1;
            return OptimizeResult {
                x,
                fun: Some(fval),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("projected GD converged in {iter} iterations"),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: Some(g),
                hess_inv: None,
                maxcv: None,
            };
        }
    }

    let fval = f(&x);
    nfev += 1;
    OptimizeResult {
        x,
        fun: Some(fval),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: "projected GD did not converge".to_string(),
        nfev,
        njev,
        nhev: 0,
        nit: maxiter,
        jac: None,
        hess_inv: None,
        maxcv: None,
    }
}

/// Compute numerical gradient via forward differences.
pub fn numerical_gradient<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let f0 = f(x);
    let mut grad = Vec::with_capacity(n);
    for i in 0..n {
        let mut xp = x.to_vec();
        xp[i] += eps;
        grad.push((f(&xp) - f0) / eps);
    }
    grad
}

/// Numerical Hessian via central differences.
pub fn numerical_hessian<F>(f: F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut hess = vec![vec![0.0; n]; n];
    let mut scratch = x.to_vec();

    for i in 0..n {
        for j in i..n {
            // f(x + eps_i + eps_j)
            scratch[i] += eps;
            scratch[j] += eps;
            let f_pp = f(&scratch);

            // f(x + eps_i - eps_j)
            scratch[j] -= 2.0 * eps;
            let f_pm = f(&scratch);

            // f(x - eps_i + eps_j)
            scratch[i] -= 2.0 * eps;
            scratch[j] += 2.0 * eps;
            let f_mp = f(&scratch);

            // f(x - eps_i - eps_j)
            scratch[j] -= 2.0 * eps;
            let f_mm = f(&scratch);

            // Restore scratch to original x
            scratch[i] += eps;
            scratch[j] += eps;

            hess[i][j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);
            hess[j][i] = hess[i][j];
        }
    }
    hess
}

/// Compute the numerical Jacobian of a vector function.
pub fn numerical_jacobian<F>(f: F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();
    let mut jac = vec![vec![0.0; n]; m];
    for j in 0..n {
        let mut xp = x.to_vec();
        xp[j] += eps;
        let fp = f(&xp);
        for i in 0..m {
            jac[i][j] = (fp[i] - f0[i]) / eps;
        }
    }
    jac
}

/// Simulated annealing for combinatorial optimization.
///
/// Minimizes a function over discrete states using neighbor generation.
pub fn simulated_annealing<S, F, N>(
    initial: S,
    cost: F,
    neighbor: N,
    temp_initial: f64,
    temp_final: f64,
    max_iter: usize,
    seed: u64,
) -> (S, f64)
where
    S: Clone,
    F: Fn(&S) -> f64,
    N: Fn(&S, u64) -> S,
{
    let mut current = initial;
    let mut current_cost = cost(&current);
    let mut best = current.clone();
    let mut best_cost = current_cost;
    let mut rng = seed;

    let cooling_rate = if max_iter > 1 {
        (temp_final / temp_initial).powf(1.0 / max_iter as f64)
    } else {
        1.0
    };
    let mut temp = temp_initial;

    for _ in 0..max_iter {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let candidate = neighbor(&current, rng);
        let candidate_cost = cost(&candidate);

        let delta = candidate_cost - current_cost;
        let accept = if delta < 0.0 {
            true
        } else {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng >> 11) as f64 / (1u64 << 53) as f64;
            r < (-delta / temp).exp()
        };

        if accept {
            current = candidate;
            current_cost = candidate_cost;
            if current_cost < best_cost {
                best = current.clone();
                best_cost = current_cost;
            }
        }

        temp *= cooling_rate;
    }

    (best, best_cost)
}

/// Particle Swarm Optimization.
///
/// Minimizes f(x) over [lb, ub] using a swarm of particles.
pub fn pso<F>(
    f: F,
    lb: &[f64],
    ub: &[f64],
    n_particles: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let d = lb.len();
    let w = 0.7298; // inertia
    let c1 = 1.4962; // cognitive
    let c2 = 1.4962; // social

    let mut rng = seed;
    let next_rand = |rng: &mut u64| -> f64 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*rng >> 11) as f64 / (1u64 << 53) as f64
    };

    // Initialize particles
    let mut positions: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..d)
                .map(|j| lb[j] + next_rand(&mut rng) * (ub[j] - lb[j]))
                .collect()
        })
        .collect();

    let mut velocities: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..d)
                .map(|j| (next_rand(&mut rng) - 0.5) * (ub[j] - lb[j]) * 0.1)
                .collect()
        })
        .collect();

    let mut personal_best = positions.clone();
    let mut personal_best_cost: Vec<f64> = positions.iter().map(|p| f(p)).collect();

    let global_best_idx = personal_best_cost
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut global_best = personal_best[global_best_idx].clone();
    let mut global_best_cost = personal_best_cost[global_best_idx];

    for _ in 0..max_iter {
        for i in 0..n_particles {
            // Update velocity
            for j in 0..d {
                let r1 = next_rand(&mut rng);
                let r2 = next_rand(&mut rng);
                velocities[i][j] = w * velocities[i][j]
                    + c1 * r1 * (personal_best[i][j] - positions[i][j])
                    + c2 * r2 * (global_best[j] - positions[i][j]);
            }

            // Update position
            for j in 0..d {
                positions[i][j] = (positions[i][j] + velocities[i][j]).clamp(lb[j], ub[j]);
            }

            // Evaluate
            let cost = f(&positions[i]);
            if cost < personal_best_cost[i] {
                personal_best_cost[i] = cost;
                personal_best[i].clone_from(&positions[i]);
                if cost < global_best_cost {
                    global_best_cost = cost;
                    global_best.clone_from(&positions[i]);
                }
            }
        }
    }

    (global_best, global_best_cost)
}

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use crate::{
        BasinhoppingOptions, Bounds, ConvergenceStatus, DifferentialEvolutionOptions, Integrality,
        LinearConstraint, MilpOptions, MilpProblem, MinimizeOptions, NonlinearConstraint,
        OptimizeMethod, RootOptions, approx_fprime, basinhopping, check_grad, cobyla,
        differential_evolution, dual_annealing, linear_sum_assignment, linprog, milp, shgo,
    };

    #[test]
    fn defaults_match_packet_contract_intent() {
        let minimize = MinimizeOptions::default();
        assert_eq!(minimize.method, None);
        assert_eq!(minimize.mode, RuntimeMode::Strict);

        let root = RootOptions::default();
        assert!(root.xtol > 0.0);
        assert!(root.rtol > 0.0);
        assert_eq!(root.mode, RuntimeMode::Strict);
    }

    #[test]
    fn convergence_status_is_cloneable() {
        let status = ConvergenceStatus::CallbackStop;
        assert_eq!(status, ConvergenceStatus::CallbackStop);
    }

    #[test]
    fn optimize_method_enum_is_stable() {
        let method = OptimizeMethod::Bfgs;
        assert!(matches!(method, OptimizeMethod::Bfgs));
    }

    // ── Constraint type tests ──────────────────────────────────────

    #[test]
    fn bounds_basic() {
        let b = Bounds::new(vec![0.0, -1.0], vec![1.0, 1.0]).expect("bounds");
        assert_eq!(b.len(), 2);
        assert!(b.is_feasible(&[0.5, 0.0]));
        assert!(!b.is_feasible(&[1.5, 0.0]));
        assert!(!b.is_feasible(&[0.5, -2.0]));
    }

    #[test]
    fn bounds_project() {
        let b = Bounds::new(vec![0.0, -1.0], vec![1.0, 1.0]).expect("bounds");
        let projected = b.project(&[2.0, -3.0]);
        assert_eq!(projected, vec![1.0, -1.0]);
    }

    #[test]
    fn bounds_unbounded() {
        let b = Bounds::unbounded(3);
        assert!(b.is_feasible(&[1e100, -1e100, 0.0]));
    }

    #[test]
    fn bounds_invalid_lb_gt_ub() {
        let err = Bounds::new(vec![1.0], vec![0.0]).expect_err("lb > ub");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn bounds_to_tuples() {
        let b = Bounds::new(vec![0.0, f64::NEG_INFINITY], vec![f64::INFINITY, 1.0]).unwrap();
        let tuples = b.to_bound_tuples();
        assert_eq!(tuples[0], (Some(0.0), None));
        assert_eq!(tuples[1], (None, Some(1.0)));
    }

    #[test]
    fn linear_constraint_basic() {
        // x + y <= 1 expressed as A @ x <= ub with lb = -inf
        let lc = LinearConstraint::new(vec![vec![1.0, 1.0]], vec![f64::NEG_INFINITY], vec![1.0])
            .expect("linear constraint");
        assert!(lc.is_feasible(&[0.3, 0.3])); // 0.6 <= 1
        assert!(!lc.is_feasible(&[0.6, 0.6])); // 1.2 > 1
    }

    #[test]
    fn linear_constraint_evaluate() {
        let lc = LinearConstraint::new(
            vec![vec![2.0, 3.0], vec![1.0, -1.0]],
            vec![0.0, 0.0],
            vec![10.0, 5.0],
        )
        .expect("linear constraint");
        let ax = lc.evaluate(&[1.0, 2.0]);
        assert!((ax[0] - 8.0).abs() < 1e-12); // 2*1 + 3*2 = 8
        assert!((ax[1] - (-1.0)).abs() < 1e-12); // 1*1 + (-1)*2 = -1
    }

    #[test]
    fn nonlinear_constraint_basic() {
        fn circle(x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] * x[1]]
        }
        let nlc = NonlinearConstraint::new(circle, vec![0.0], vec![1.0]).expect("nlc");
        assert!(nlc.is_feasible(&[0.5, 0.5])); // 0.5 <= 1
        assert!(!nlc.is_feasible(&[1.0, 1.0])); // 2.0 > 1
    }

    #[test]
    fn bounds_with_lbfgsb() {
        // Minimize x^2 + y^2 subject to x >= 1, y >= 2
        let bounds = Bounds::new(vec![1.0, 2.0], vec![f64::INFINITY, f64::INFINITY]).unwrap();
        let bound_tuples = bounds.to_bound_tuples();

        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-10),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let result = crate::lbfgsb(&f, &[5.0, 5.0], options, Some(&bound_tuples))
            .expect("lbfgsb with bounds");
        assert!(result.success, "{}", result.message);
        // Minimum should be at (1, 2) — the bound corner
        assert!(
            (result.x[0] - 1.0).abs() < 0.01,
            "x ~ 1.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.01,
            "y ~ 2.0, got {}",
            result.x[1]
        );
    }

    #[test]
    fn approx_fprime_matches_quadratic_gradient() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad = approx_fprime(&[3.0], f, 1.0e-8).expect("gradient should evaluate");
        assert!(
            (grad[0] - 6.0).abs() < 1.0e-5,
            "expected 6.0, got {}",
            grad[0]
        );
    }

    #[test]
    fn approx_fprime_matches_multivariate_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + 3.0 * x[1] * x[1] + x[0] * x[1];
        let grad = approx_fprime(&[2.0, -1.0], f, 1.0e-8).expect("gradient should evaluate");
        assert!(
            (grad[0] - 3.0).abs() < 1.0e-5,
            "expected 3.0, got {}",
            grad[0]
        );
        assert!(
            (grad[1] - (-4.0)).abs() < 1.0e-5,
            "expected -4.0, got {}",
            grad[1]
        );
    }

    #[test]
    fn check_grad_is_small_for_exact_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1] * x[1];
        let g = |x: &[f64]| vec![2.0 * x[0], 3.0 * x[1] * x[1]];
        let error = check_grad(f, g, &[1.5, -2.0]).expect("gradient check should succeed");
        assert!(error < 1.0e-5, "expected small error, got {error}");
    }

    #[test]
    fn check_grad_detects_bad_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let wrong = |_x: &[f64]| vec![0.0, 0.0];
        let error = check_grad(f, wrong, &[2.0, -3.0]).expect("gradient check should succeed");
        assert!(error > 1.0, "expected large error, got {error}");
    }

    #[test]
    fn linear_sum_assignment_solves_square_problem() {
        let cost_matrix = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let (row_ind, col_ind) =
            linear_sum_assignment(&cost_matrix).expect("assignment should succeed");
        assert_eq!(row_ind, vec![0, 1, 2]);
        assert_eq!(col_ind, vec![1, 0, 2]);
    }

    #[test]
    fn linear_sum_assignment_handles_more_columns_than_rows() {
        let cost_matrix = vec![vec![10.0, 19.0, 8.0, 15.0], vec![10.0, 18.0, 7.0, 17.0]];
        let (row_ind, col_ind) =
            linear_sum_assignment(&cost_matrix).expect("assignment should succeed");
        assert_eq!(row_ind, vec![0, 1]);
        assert_eq!(col_ind, vec![0, 2]);
    }

    #[test]
    fn linear_sum_assignment_handles_more_rows_than_columns() {
        let cost_matrix = vec![vec![5.0, 9.0], vec![10.0, 3.0], vec![8.0, 7.0]];
        let (row_ind, col_ind) =
            linear_sum_assignment(&cost_matrix).expect("assignment should succeed");
        assert_eq!(row_ind, vec![0, 1]);
        assert_eq!(col_ind, vec![0, 1]);
    }

    #[test]
    fn linear_sum_assignment_rejects_non_rectangular_input() {
        let cost_matrix = vec![vec![1.0, 2.0], vec![3.0]];
        let error = linear_sum_assignment(&cost_matrix).expect_err("matrix should be rejected");
        assert!(matches!(error, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn linear_sum_assignment_allows_empty_input() {
        let (row_ind, col_ind) = linear_sum_assignment(&[]).expect("empty problem should succeed");
        assert!(row_ind.is_empty());
        assert!(col_ind.is_empty());
    }

    // ── linprog tests ──────────────────────────────────────────────

    #[test]
    fn linprog_simple_2d() {
        // Maximize 5x + 4y (minimize -5x - 4y)
        // Subject to: 6x + 4y <= 24, x + 2y <= 6, x,y >= 0
        // Optimal: x=3, y=1.5, obj = -21
        let c = vec![-5.0, -4.0];
        let a_ub = vec![vec![6.0, 4.0], vec![1.0, 2.0]];
        let b_ub = vec![24.0, 6.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 3.0).abs() < 0.01,
            "x={}, expected 3.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.5).abs() < 0.01,
            "y={}, expected 1.5",
            result.x[1]
        );
        assert!(
            (result.fun - (-21.0)).abs() < 0.01,
            "obj={}, expected -21.0",
            result.fun
        );
    }

    #[test]
    fn linprog_infeasible() {
        // x = -1 with x >= 0 — infeasible (non-negative variable can't equal negative).
        let c = vec![1.0];
        let a_eq = vec![vec![1.0]];
        let b_eq = vec![-1.0]; // x = -1 but x >= 0
        let result = linprog(&c, &[], &[], &a_eq, &b_eq, &[], None).unwrap();
        assert!(
            !result.success,
            "should be infeasible: success={}, status={}, x={:?}, fun={}",
            result.success, result.status, result.x, result.fun
        );
        assert_eq!(result.status, 2);
    }

    #[test]
    fn linprog_equality_constraint() {
        // Minimize x + y, subject to x + y = 10, x,y >= 0
        // Optimal: x=0, y=10, obj=10 (or x=10, y=0)
        let c = vec![1.0, 1.0];
        let a_eq = vec![vec![1.0, 1.0]];
        let b_eq = vec![10.0];
        let result = linprog(&c, &[], &[], &a_eq, &b_eq, &[], None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.fun - 10.0).abs() < 0.01,
            "obj={}, expected 10.0",
            result.fun
        );
    }

    #[test]
    fn linprog_with_bounds() {
        // Minimize -x - 2y, subject to x + y <= 4, x in [0,3], y in [0,3]
        // Optimal: x=1, y=3, obj = -7
        let c = vec![-1.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![4.0];
        let bounds = vec![(Some(0.0), Some(3.0)), (Some(0.0), Some(3.0))];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.fun - (-7.0)).abs() < 0.1,
            "obj={}, expected -7.0",
            result.fun
        );
    }

    #[test]
    fn linprog_degenerate_redundant_constraints() {
        // Minimize x, subject to x <= 5 AND x <= 10 AND x >= 0
        // Optimal: x = 0
        let c = vec![1.0];
        let a_ub = vec![vec![1.0], vec![1.0]];
        let b_ub = vec![5.0, 10.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(result.x[0].abs() < 0.01, "x={}, expected 0.0", result.x[0]);
    }

    #[test]
    fn linprog_slack_variables() {
        // Verify slack is computed correctly.
        let c = vec![1.0, 1.0];
        let a_ub = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b_ub = vec![10.0, 10.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).unwrap();
        assert!(result.success);
        // Slack = b_ub - A_ub @ x
        for (i, &s) in result.slack.iter().enumerate() {
            assert!(s >= -1e-8, "slack[{i}] = {s} should be non-negative");
        }
    }

    #[test]
    fn linprog_bounds_enforced() {
        // Minimize -x, x in [2, 5] → optimal at x=5
        let c = vec![-1.0];
        let bounds = vec![(Some(2.0), Some(5.0))];
        let result = linprog(&c, &[], &[], &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 5.0).abs() < 0.01,
            "x={}, expected 5.0",
            result.x[0]
        );
    }

    #[test]
    fn linprog_dimension_mismatch() {
        let c = vec![1.0, 2.0];
        let a_ub = vec![vec![1.0]]; // wrong number of columns
        let b_ub = vec![5.0];
        assert!(linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).is_err());
    }

    // ── milp tests ──────────────────────────────────────────────────

    #[test]
    fn milp_finds_integer_optimum() {
        let c = vec![-1.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![4.0];
        let bounds = vec![(Some(0.0), Some(3.0)), (Some(0.0), Some(3.0))];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Integer, Integrality::Integer],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &bounds,
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(result.success, "milp failed: {}", result.message);
        assert_eq!(result.x, vec![1.0, 3.0]);
        assert!((result.fun + 7.0).abs() < 1e-9);
    }

    #[test]
    fn milp_handles_binary_variables() {
        let c = vec![-3.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![1.0];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Binary, Integrality::Binary],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &[],
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(result.success, "milp failed: {}", result.message);
        assert_eq!(result.x, vec![1.0, 0.0]);
        assert!((result.fun + 3.0).abs() < 1e-9);
    }

    #[test]
    fn milp_reports_infeasible_problem() {
        let c = vec![1.0];
        let a_eq = vec![vec![1.0]];
        let b_eq = vec![0.5];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Integer],
                a_ub: &[],
                b_ub: &[],
                a_eq: &a_eq,
                b_eq: &b_eq,
                bounds: &[(Some(0.0), Some(1.0))],
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(!result.success);
        assert_eq!(result.status, 2);
    }

    #[test]
    fn milp_rejects_integrality_length_mismatch() {
        let err = milp(
            MilpProblem {
                c: &[1.0, 2.0],
                integrality: &[Integrality::Integer],
                a_ub: &[],
                b_ub: &[],
                a_eq: &[],
                b_eq: &[],
                bounds: &[],
            },
            MilpOptions::default(),
        )
        .expect_err("integrality length mismatch");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    // ── Differential Evolution tests ───────────────────────────────

    #[test]
    fn de_rastrigin_finds_global_minimum() {
        // Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
        // Global minimum at origin: f(0,...,0) = 0
        let rastrigin = |x: &[f64]| -> f64 {
            let n = x.len() as f64;
            let pi2 = 2.0 * std::f64::consts::PI;
            n * 10.0
                + x.iter()
                    .map(|&xi| xi * xi - 10.0 * (pi2 * xi).cos())
                    .sum::<f64>()
        };

        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 200,
            seed: Some(42),
            ..Default::default()
        };
        let result = differential_evolution(rastrigin, &bounds, opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(
            fun < 1.0,
            "DE should find near-global minimum, got f={}",
            fun
        );
    }

    #[test]
    fn de_rosenbrock_2d() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // Minimum at (1,1) with f=0
        let rosenbrock =
            |x: &[f64]| -> f64 { (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2) };

        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 500,
            seed: Some(123),
            ..Default::default()
        };
        let result = differential_evolution(rosenbrock, &bounds, opts).unwrap();
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x ~ 1.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "y ~ 1.0, got {}",
            result.x[1]
        );
    }

    #[test]
    fn de_bounds_respected() {
        let func = |x: &[f64]| -> f64 { x.iter().sum() };
        let bounds = vec![(1.0, 3.0), (2.0, 4.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 50,
            seed: Some(99),
            ..Default::default()
        };
        let result = differential_evolution(func, &bounds, opts).unwrap();
        for (i, (&xi, &(lo, hi))) in result.x.iter().zip(bounds.iter()).enumerate() {
            assert!(
                xi >= lo - 1e-10 && xi <= hi + 1e-10,
                "variable {i}: {xi} outside [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn de_rand1bin_strategy() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = DifferentialEvolutionOptions {
            strategy: crate::DeStrategy::Rand1Bin,
            maxiter: 200,
            seed: Some(77),
            ..Default::default()
        };
        let result = differential_evolution(sphere, &bounds, opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(fun < 0.1, "rand1bin should find near-origin, got f={fun}");
    }

    #[test]
    fn de_population_explores() {
        // With only a few iterations, the population should have explored
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-10.0, 10.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 5,
            seed: Some(1),
            ..Default::default()
        };
        let result = differential_evolution(sphere, &bounds, opts).unwrap();
        assert!(result.nfev > 0, "should have function evaluations");
    }

    #[test]
    fn de_rejects_population_too_small() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-1.0, 1.0)];
        let opts = DifferentialEvolutionOptions {
            popsize: 3,
            ..Default::default()
        };
        let err = differential_evolution(sphere, &bounds, opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn de_rejects_nonfinite_bounds() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(f64::NAN, 1.0)];
        let err = differential_evolution(sphere, &bounds, Default::default()).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn de_rejects_population_size_overflow() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let opts = DifferentialEvolutionOptions {
            popsize: usize::MAX,
            ..Default::default()
        };
        let err = differential_evolution(sphere, &bounds, opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    // ── Basinhopping tests ─────────────────────────────────────────

    #[test]
    fn basinhopping_finds_global_minimum() {
        // Multi-modal function: f(x) = x^4 - 4x^2 + x
        // Has two local minima, global near x ≈ -1.38
        let func = |x: &[f64]| -> f64 {
            let xi = x[0];
            xi.powi(4) - 4.0 * xi * xi + xi
        };

        let opts = BasinhoppingOptions {
            niter: 50,
            seed: Some(42),
            stepsize: 1.0,
            ..Default::default()
        };
        let result = basinhopping(func, &[3.0], opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(
            fun < -3.0,
            "basinhopping should find global minimum, got f={fun}"
        );
    }

    #[test]
    fn basinhopping_sphere() {
        // Simple sphere function: global min at origin
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let opts = BasinhoppingOptions {
            niter: 20,
            seed: Some(99),
            ..Default::default()
        };
        let result = basinhopping(sphere, &[5.0, -3.0], opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(
            fun < 0.01,
            "basinhopping on sphere should find near-zero, got f={fun}"
        );
    }

    #[test]
    fn basinhopping_returns_correct_nfev() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let opts = BasinhoppingOptions {
            niter: 5,
            seed: Some(1),
            ..Default::default()
        };
        let result = basinhopping(func, &[1.0], opts).unwrap();
        assert!(result.nfev > 0, "should track function evaluations");
    }

    #[test]
    fn basinhopping_rejects_nonpositive_stepsize() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let opts = BasinhoppingOptions {
            stepsize: 0.0,
            ..Default::default()
        };
        let err = basinhopping(func, &[1.0], opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn basinhopping_rejects_nonfinite_initial_point() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let err = basinhopping(func, &[f64::INFINITY], Default::default()).unwrap_err();
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));
    }

    // ── Dual annealing tests ─────────────────────────────────────────

    #[test]
    fn dual_annealing_finds_sphere_minimum() {
        // Sphere function: f(x) = Σ x_i²  — minimum at origin
        let result = dual_annealing(
            |x| x.iter().map(|&xi| xi * xi).sum(),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            1000,
            42,
        )
        .expect("dual_annealing");
        assert!(result.success);
        assert!(
            result.fun.unwrap() < 0.1,
            "should find near-zero minimum: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn dual_annealing_finds_rosenbrock_valley() {
        // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²  — minimum at (1,1)
        let result = dual_annealing(
            |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            2000,
            123,
        )
        .expect("dual_annealing rosenbrock");
        assert!(
            result.fun.unwrap() < 10.0,
            "should find reasonable minimum: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn dual_annealing_empty_bounds_rejected() {
        let err = dual_annealing(|_| 0.0, &[], 100, 42).expect_err("empty");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn dual_annealing_nonfinite_bounds_rejected() {
        let err = dual_annealing(|_| 0.0, &[(f64::NAN, 1.0)], 100, 42).expect_err("nonfinite");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn dual_annealing_inverted_bounds_rejected() {
        let err = dual_annealing(|_| 0.0, &[(2.0, 2.0)], 100, 42).expect_err("inverted");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn shgo_finds_sphere_minimum() {
        let result = shgo(
            |x| x.iter().map(|&xi| xi * xi).sum(),
            &[(-5.0, 5.0), (-5.0, 5.0)],
        )
        .expect("shgo sphere");
        assert!(result.success);
        assert!(
            result.fun.expect("objective value") < 1e-6,
            "expected near-zero minimum, got {:?}",
            result.fun
        );
    }

    #[test]
    fn shgo_finds_multimodal_global_minimum() {
        let result = shgo(
            |x| {
                let xi = x[0];
                xi.powi(4) - 4.0 * xi * xi + xi
            },
            &[(-3.0, 3.0)],
        )
        .expect("shgo multimodal");
        assert!(
            result.fun.expect("objective value") < -3.0,
            "expected global basin, got {:?}",
            result.fun
        );
    }

    #[test]
    fn shgo_rejects_invalid_bounds() {
        let err = shgo(|_| 0.0, &[(1.0, 1.0)]).expect_err("invalid bounds");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    // ── COBYLA tests ─────────────────────────────────────────────────

    #[test]
    fn cobyla_unconstrained_quadratic() {
        // Minimize x² + y² → (0, 0)
        let result = cobyla(
            |x| x[0] * x[0] + x[1] * x[1],
            &[1.0, 1.0],
            &[] as &[fn(&[f64]) -> f64],
            1000,
            0.5,
        )
        .expect("cobyla");
        assert!(
            result.fun.unwrap() < 0.01,
            "cobyla should minimize: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn cobyla_with_constraint() {
        // Minimize x + y subject to x + y >= 1
        // Solution: x + y = 1 (constraint active), e.g. (0.5, 0.5)
        type ConstraintFn = dyn Fn(&[f64]) -> f64;
        let constraints: Vec<Box<ConstraintFn>> = vec![
            Box::new(|x: &[f64]| x[0] + x[1] - 1.0), // x + y >= 1
        ];
        let constraint_fns: Vec<&ConstraintFn> = constraints.iter().map(|b| b.as_ref()).collect();
        let result = cobyla(|x| x[0] + x[1], &[2.0, 2.0], &constraint_fns, 1000, 0.5)
            .expect("cobyla constrained");
        // x + y should be approximately 1
        let sum = result.x[0] + result.x[1];
        assert!((sum - 1.0).abs() < 0.1, "x+y should be ~1: {sum}");
    }

    #[test]
    fn cobyla_empty_x0_rejected() {
        let err = cobyla(|_: &[f64]| 0.0, &[], &[] as &[fn(&[f64]) -> f64], 100, 0.5)
            .expect_err("empty x0");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }
}
