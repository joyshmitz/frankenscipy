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
pub use root::{RootResult, bisect, brenth, brentq, ridder, root_scalar};
pub use types::{
    Bounds, ConvergenceStatus, LinearConstraint, MinimizeOptions, NonlinearConstraint, OptError,
    OptimizeMethod, OptimizeResult, RootMethod, RootOptions,
};

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use crate::{ConvergenceStatus, MinimizeOptions, OptimizeMethod, RootOptions};

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

    use crate::{Bounds, LinearConstraint, NonlinearConstraint};

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
        let lc = LinearConstraint::new(
            vec![vec![1.0, 1.0]],
            vec![f64::NEG_INFINITY],
            vec![1.0],
        )
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
}
