use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use fsci_odr::{Data, Model, ODR};

fn make_model() -> Model {
    Model::new(|beta: &[f64], x: &[f64]| {
        x.iter()
            .map(|value| beta[0] + beta[1] * value + beta[2] * (beta[3] * value).sin())
            .collect()
    })
    .with_parameter_count(4)
    .with_scalar_separable(true)
}

fn build_problem(n: usize) -> (Vec<f64>, Vec<f64>) {
    let beta = [0.7, 1.3, 0.4, 0.9];
    let x = (0..n)
        .map(|idx| idx as f64 * 0.05 + (idx % 7) as f64 * 0.001)
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            beta[0]
                + beta[1] * value
                + beta[2] * (beta[3] * value).sin()
                + ((idx % 11) as f64 - 5.0) * 0.001
        })
        .collect::<Vec<_>>();
    (x, y)
}

fn make_solver(n: usize) -> ODR {
    let (x, y) = build_problem(n);
    ODR::new(
        Data::new(x, y).unwrap(),
        make_model(),
        vec![0.5, 1.5, 0.2, 0.8],
    )
    .unwrap()
}

fn bench_odr_structured(c: &mut Criterion) {
    let mut group = c.benchmark_group("odr_structured_scalar");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(2));

    for n in [100usize, 200, 400] {
        group.bench_function(format!("dense_reference_n{n}"), |b| {
            b.iter_batched(
                || make_solver(n),
                |solver| solver.run_dense_reference().unwrap(),
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("structured_n{n}"), |b| {
            b.iter_batched(
                || make_solver(n),
                |solver| solver.run().unwrap(),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Same-binary A/B for hoisting the dense LM path's `JᵀJ`/`Jᵀr` build out of the
/// per-damping-retry loop (build once per Jacobian, re-add `damping` to a diagonal copy
/// per retry). Byte-identical — the fitted `beta` and `sum_square` are asserted bit-equal
/// between arms before timing.
// A far-from-solution initial guess so the dense LM does many damping retries (rebuilds)
// — the ill-conditioned/rejection-heavy regime where the hoist actually saves work.
fn make_solver_far(n: usize) -> ODR {
    let (x, y) = build_problem(n);
    ODR::new(
        Data::new(x, y).unwrap(),
        make_model(),
        vec![6.0, -4.0, 3.0, 3.5],
    )
    .unwrap()
}

fn bench_odr_dense_hoist_ab(c: &mut Criterion) {
    use fsci_odr::ODR_LMSTEP_HOIST_DISABLE;
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("odr_dense_lmstep_hoist_ab");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(3));

    // Rejection-heavy (far init): the regime where the per-retry rebuild is redundant.
    for n in [200usize, 400] {
        ODR_LMSTEP_HOIST_DISABLE.store(false, Ordering::Relaxed);
        let hoisted = make_solver_far(n).run_dense_reference().unwrap();
        ODR_LMSTEP_HOIST_DISABLE.store(true, Ordering::Relaxed);
        let orig = make_solver_far(n).run_dense_reference().unwrap();
        assert!(
            hoisted.beta.len() == orig.beta.len()
                && hoisted
                    .beta
                    .iter()
                    .zip(&orig.beta)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                && hoisted.sum_square.to_bits() == orig.sum_square.to_bits(),
            "odr dense LM hoist (far init) not byte-identical (n={n})"
        );
        group.bench_function(format!("far_current_hoisted_n{n}"), |b| {
            b.iter_batched(
                || make_solver_far(n),
                |solver| {
                    ODR_LMSTEP_HOIST_DISABLE.store(false, Ordering::Relaxed);
                    solver.run_dense_reference().unwrap()
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("far_orig_rebuild_n{n}"), |b| {
            b.iter_batched(
                || make_solver_far(n),
                |solver| {
                    ODR_LMSTEP_HOIST_DISABLE.store(true, Ordering::Relaxed);
                    solver.run_dense_reference().unwrap()
                },
                BatchSize::SmallInput,
            );
        });
    }

    for n in [200usize, 400] {
        ODR_LMSTEP_HOIST_DISABLE.store(false, Ordering::Relaxed);
        let hoisted = make_solver(n).run_dense_reference().unwrap();
        ODR_LMSTEP_HOIST_DISABLE.store(true, Ordering::Relaxed);
        let orig = make_solver(n).run_dense_reference().unwrap();
        assert_eq!(
            hoisted.beta.len(),
            orig.beta.len(),
            "beta length differs (n={n})"
        );
        assert!(
            hoisted
                .beta
                .iter()
                .zip(&orig.beta)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && hoisted.sum_square.to_bits() == orig.sum_square.to_bits(),
            "odr dense LM hoist not byte-identical (n={n})"
        );

        group.bench_function(format!("current_hoisted_n{n}"), |b| {
            b.iter_batched(
                || make_solver(n),
                |solver| {
                    ODR_LMSTEP_HOIST_DISABLE.store(false, Ordering::Relaxed);
                    solver.run_dense_reference().unwrap()
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("orig_rebuild_n{n}"), |b| {
            b.iter_batched(
                || make_solver(n),
                |solver| {
                    ODR_LMSTEP_HOIST_DISABLE.store(true, Ordering::Relaxed);
                    solver.run_dense_reference().unwrap()
                },
                BatchSize::SmallInput,
            );
        });
    }
    ODR_LMSTEP_HOIST_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

criterion_group!(benches, bench_odr_structured, bench_odr_dense_hoist_ab);
criterion_main!(benches);
