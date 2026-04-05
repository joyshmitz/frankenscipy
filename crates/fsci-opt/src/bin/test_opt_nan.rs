use fsci_opt::{MinimizeOptions, OptimizeMethod, minimize};

fn main() {
    let mut opts = MinimizeOptions::default();
    opts.method = Some(OptimizeMethod::NelderMead);
    let res = minimize(
        |x: &[f64]| {
            if x[0].is_nan() || x[0] > 1.0 {
                f64::NAN
            } else {
                x[0] * x[0]
            }
        },
        &[2.0],
        opts,
    );
    println!("{:?}", res);
}
