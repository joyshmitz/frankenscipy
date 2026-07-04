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

criterion_group!(benches, bench_odr_structured);
criterion_main!(benches);
