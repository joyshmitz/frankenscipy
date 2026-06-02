use std::process::ExitCode;

use fsci_interpolate::RectBivariateSpline;

fn grid_1d(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn query_1d(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            0.001 + 0.998 * t
        })
        .collect()
}

fn rect_grid(side: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let x = grid_1d(side);
    let y = grid_1d(side);
    let z = x
        .iter()
        .map(|&xi| {
            y.iter()
                .map(|&yi| (xi * 6.0).sin() * (yi * 4.0).cos())
                .collect()
        })
        .collect();
    (x, y, z)
}

fn print_rect_eval_grid_golden() -> Result<(), String> {
    let (x, y, z) = rect_grid(32);
    let spline = RectBivariateSpline::new(&x, &y, &z, 3, 3)
        .map_err(|err| format!("failed to construct spline: {err}"))?;
    let xi = query_1d(64);
    let yi = query_1d(64);
    let values = spline.eval_grid(&xi, &yi);

    println!("fsci-interpolate rect_eval_grid golden v1");
    println!("input_grid=32 query_grid=64 kx=3 ky=3 order=x-major,y-inner");
    for (i, row) in values.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            println!("{i:02} {j:02} {:016x}", value.to_bits());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "golden".to_string());
    match mode.as_str() {
        "golden" => match print_rect_eval_grid_golden() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("usage: perf_interpolate [golden]");
            ExitCode::from(2)
        }
    }
}
