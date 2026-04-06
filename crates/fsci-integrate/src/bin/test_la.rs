use fsci_integrate::{solve_ivp, SolveIvpOptions, SolverKind, ToleranceValue};
use fsci_runtime::RuntimeMode;

fn main() {
    let opts = SolveIvpOptions {
        t_span: (0.0, f64::NAN),
        y0: &[1.0],
        method: SolverKind::Rk45,
        t_eval: None,
        dense_output: false,
        events: None,
        rtol: 1e-3,
        atol: ToleranceValue::Scalar(1e-6),
        first_step: None,
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
    };
    let res = solve_ivp(
        &mut |_t: f64, y: &[f64]| vec![y[0]],
        &opts
    );
    println!("res with tf=NaN: {:?}", res.is_err());

    let opts2 = SolveIvpOptions {
        t_span: (0.0, 1.0),
        y0: &[f64::NAN],
        method: SolverKind::Rk45,
        t_eval: None,
        dense_output: false,
        events: None,
        rtol: 1e-3,
        atol: ToleranceValue::Scalar(1e-6),
        first_step: None,
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
    };
    let res2 = solve_ivp(
        &mut |_t: f64, y: &[f64]| vec![y[0]],
        &opts2
    );
    println!("res with y0=NaN: {:?}", res2.is_err());
}
