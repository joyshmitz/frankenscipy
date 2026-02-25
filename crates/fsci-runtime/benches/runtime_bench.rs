use criterion::{Criterion, criterion_group, criterion_main};
use fsci_runtime::{
    ConformalCalibrator, DecisionSignals, MatrixConditionState, PolicyController, RuntimeMode,
    SolverPortfolio,
};

fn bench_policy_decide(c: &mut Criterion) {
    let signals = DecisionSignals::new(5.0, 0.3, 0.2);
    c.bench_function("policy_decide_strict", |b| {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
        b.iter(|| ctrl.decide(signals));
    });
    c.bench_function("policy_decide_hardened", |b| {
        let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 64);
        b.iter(|| ctrl.decide(signals));
    });
}

fn bench_solver_select_action(c: &mut Criterion) {
    let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
    for state in &MatrixConditionState::ALL {
        let name = format!("solver_select_{state:?}");
        c.bench_function(&name, |b| {
            b.iter(|| portfolio.select_action(state));
        });
    }
}

fn bench_calibrator_observe(c: &mut Criterion) {
    c.bench_function("calibrator_observe_200", |b| {
        b.iter(|| {
            let mut cal = ConformalCalibrator::new(0.05, 200);
            for i in 0..200 {
                cal.observe(i as f64 * 1e-10);
            }
        });
    });
}

criterion_group!(
    benches,
    bench_policy_decide,
    bench_solver_select_action,
    bench_calibrator_observe
);
criterion_main!(benches);
