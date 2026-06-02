use std::process::ExitCode;

use fsci_interpolate::{GriddataMethod, RectBivariateSpline, griddata};

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

fn points_2d(side: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut points = Vec::with_capacity(side * side);
    let mut values = Vec::with_capacity(side * side);
    for iy in 0..side {
        for ix in 0..side {
            let x = (ix as f64 + 0.3 * (iy % 3) as f64) / side as f64;
            let y = (iy as f64 + 0.2 * (ix % 5) as f64) / side as f64;
            points.push(vec![x, y]);
            values.push((x * 5.0).sin() + (y * 3.0).cos());
        }
    }
    (points, values)
}

fn queries_2d(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let x = ((i * 37) % 997) as f64 / 997.0;
            let y = ((i * 53 + 17) % 991) as f64 / 991.0;
            vec![x, y]
        })
        .collect()
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

fn print_griddata_linear_golden() -> Result<(), String> {
    let (points, values) = points_2d(24);
    let queries = queries_2d(1024);
    let values = griddata(&points, &values, &queries, GriddataMethod::Linear)
        .map_err(|err| format!("failed to evaluate griddata linear: {err}"))?;

    println!("fsci-interpolate griddata_linear golden v1");
    println!("points=576 queries=1024 method=linear order=query-input");
    for (i, value) in values.iter().enumerate() {
        println!("{i:04} {:016x}", value.to_bits());
    }
    Ok(())
}

fn main() -> ExitCode {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "golden".to_string());
    match mode.as_str() {
        "griddata-linear" => match print_griddata_linear_golden() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        "golden" => match print_rect_eval_grid_golden() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("usage: perf_interpolate [golden|griddata-linear]");
            ExitCode::from(2)
        }
    }
}
