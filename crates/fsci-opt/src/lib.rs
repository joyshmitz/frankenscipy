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
    MultivariateRootResult, RootResult, bisect, brenth, brentq, fsolve, ridder, root_scalar,
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
    for index in 0..xk.len() {
        let mut x_plus = xk.to_vec();
        x_plus[index] += epsilon;
        let f_plus = f(&x_plus);
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
            for j in 0..total_vars {
                a_std[i][j] = -a_std[i][j];
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
        for j in 0..phase1_vars + 1 {
            tableau[obj_row][j] -= tableau[i][j];
        }
    }

    let mut basis: Vec<usize> = (total_vars..total_vars + n_art).collect();
    let maxiter = maxiter.unwrap_or(10_000);

    // Simplex iterations for Phase I.
    let phase1_result = simplex_iterate(&mut tableau, &mut basis, maxiter, phase1_vars);
    if let Err(err_code) = phase1_result {
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
            nit: phase1_result.unwrap(),
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
    for j in 0..total_vars {
        tableau2[obj_row][j] = c_std[j];
    }
    tableau2[obj_row][total_vars] = 0.0;

    // Eliminate basic variables from objective row.
    for (row, &bj) in basis.iter().enumerate() {
        if bj < total_vars {
            let ratio = tableau2[obj_row][bj];
            if ratio.abs() > 1e-15 {
                for j in 0..total_vars + 1 {
                    tableau2[obj_row][j] -= ratio * tableau2[row][j];
                }
            }
        }
    }

    let phase2_result = simplex_iterate(&mut tableau2, &mut basis, maxiter, total_vars);
    let nit = phase1_result.unwrap_or(0) + phase2_result.unwrap_or(maxiter);

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
        // Find entering variable: most negative reduced cost (Bland's rule: smallest index).
        let mut pivot_col = None;
        let mut min_cost = -1e-10;
        for j in 0..n_vars {
            if tableau[obj_row][j] < min_cost {
                min_cost = tableau[obj_row][j];
                pivot_col = Some(j);
            }
        }

        let pivot_col = match pivot_col {
            Some(c) => c,
            None => return Ok(iteration), // Optimal
        };

        // Minimum ratio test: find leaving variable.
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            if tableau[i][pivot_col] > 1e-12 {
                let ratio = tableau[i][rhs_col] / tableau[i][pivot_col];
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
        for j in 0..=rhs_col {
            tableau[pivot_row][j] /= pivot_val;
        }

        for i in 0..=obj_row {
            if i != pivot_row {
                let factor = tableau[i][pivot_col];
                if factor.abs() > 1e-15 {
                    for j in 0..=rhs_col {
                        tableau[i][j] -= factor * tableau[pivot_row][j];
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
        let fmin = fitness.iter().cloned().fold(f64::INFINITY, f64::min);
        let fmax = fitness.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if (fmax - fmin).abs() < opts.tol {
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
                x_best = result.x.clone();
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

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use crate::{
        BasinhoppingOptions, Bounds, ConvergenceStatus, DifferentialEvolutionOptions,
        LinearConstraint, MinimizeOptions, NonlinearConstraint, OptimizeMethod, RootOptions,
        approx_fprime, basinhopping, check_grad, differential_evolution, linear_sum_assignment,
        linprog,
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
}
