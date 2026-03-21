use fsci_linalg::{solve_triangular, SolveOptions, TriangularSolveOptions, TriangularTranspose};
use fsci_runtime::RuntimeMode;

#[test]
fn test_solve_triangular_transpose_backward_error() {
    // A = [[2, 1], [0, 2]]  (Upper Triangular)
    // A^T = [[2, 0], [1, 2]]
    let a = vec![vec![2.0, 1.0], vec![0.0, 2.0]];
    let b = vec![2.0, 4.0];
    // Solve A^T x = b
    // 2x1 = 2 => x1 = 1
    // 1x1 + 2x2 = 4 => 1 + 2x2 = 4 => 2x2 = 3 => x2 = 1.5
    // x = [1, 1.5]
    
    let options = TriangularSolveOptions {
        trans: TriangularTranspose::Transpose,
        lower: false, // a is upper
        ..TriangularSolveOptions::default()
    };
    
    let result = solve_triangular(&a, &b, options).expect("solve_triangular works");
    println!("x = {:?}", result.x);
    println!("backward_error = {:?}", result.backward_error);
    
    // Verify x is correct
    assert!((result.x[0] - 1.0).abs() < 1e-10);
    assert!((result.x[1] - 1.5).abs() < 1e-10);
    
    // Check backward error manually
    // A^T x = [[2, 0], [1, 2]] * [1, 1.5] = [2, 1 + 3] = [2, 4]. Matches b.
    // So backward error should be very small (~1e-16).
    // BUT! If it computes ||Ax - b||:
    // Ax = [[2, 1], [0, 2]] * [1, 1.5] = [2 + 1.5, 3] = [3.5, 3]
    // Ax - b = [3.5 - 2, 3 - 4] = [1.5, -1]
    // ||Ax - b|| = sqrt(1.5^2 + 1^2) = sqrt(2.25 + 1) = sqrt(3.25) ≈ 1.8
    // So if the bug is present, backward_error will be large.
    
    let be = result.backward_error.expect("backward error should be present");
    assert!(be < 1e-10, "backward error should be small for transposed solve, got {}", be);
}
