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
        for (j, &cost) in tableau[obj_row].iter().enumerate().take(n_vars) {
            if cost < min_cost {
                min_cost = cost;
                pivot_col = Some(j);
            }
        }

        let pivot_col = match pivot_col {
            Some(c) => c,
            None => return Ok(iteration), // Optimal
        };
        println!("Iteration {}: pivot_col = {}", iteration, pivot_col);

        // Minimum ratio test: find leaving variable.
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for (i, row) in tableau.iter().enumerate().take(m) {
            if row[pivot_col] > 1e-12 {
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
                for j in 0..=rhs_col {
                    tableau[i][j] -= factor * tableau[pivot_row][j];
                }
            }
        }
        basis[pivot_row] = pivot_col;
    }

    Err(1) // Iteration limit
}

fn main() {
    // A simple problem where multiple columns have negative reduced cost.
    // min -10x1 -20x2
    // st x1 + x2 <= 10
    //    x1 >= 0, x2 >= 0
    // Standard form:
    // min -10x1 -20x2
    // st x1 + x2 + s1 = 10
    //    x1, x2, s1 >= 0
    let mut tableau = vec![
        vec![1.0, 1.0, 1.0, 10.0], // row 0: x1 + x2 + s1 = 10
        vec![-10.0, -20.0, 0.0, 0.0], // obj row
    ];
    let mut basis = vec![2]; // s1 is in basis (index 2)
    
    // Bland's rule should pick x1 (index 0) if it's the FIRST with negative cost.
    // But greedy picks x2 (index 1) because -20 < -10.
    let _ = simplex_iterate(&mut tableau, &mut basis, 10, 3);
}
