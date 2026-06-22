#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;
use fsci_special::{SpecialTensor, ellipeinc, ellipkinc, erf, erfc, hyperu, kv};
use std::error::Error;
use std::f64::consts::PI;
use std::hint::black_box;
use std::io::Write;
use std::io::{Error as IoError, ErrorKind};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

type GoldenResult<T = ()> = Result<T, Box<dyn Error>>;

const ERROR_INPUTS: &[f64] = &[
    f64::NEG_INFINITY,
    -3.0,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    3.0,
    f64::INFINITY,
    f64::NAN,
];

fn main() -> GoldenResult {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "perf_special".to_string());
    match args.next().as_deref() {
        Some("golden-error") => print_error_golden(),
        Some("golden-elliptic") => print_elliptic_golden(),
        Some("bench-hyperu") => print_hyperu_benchmark(),
        Some("bench-hyperu-generic") => print_hyperu_generic_benchmark(),
        Some("bench-hyperu-a1-gamma") => print_hyperu_a1_gamma_benchmark(),
        Some("bench-kv") => print_kv_benchmark(),
        _ => Err(IoError::new(
            ErrorKind::InvalidInput,
            format!(
                "usage: {program} <golden-error|golden-elliptic|bench-hyperu|bench-hyperu-generic|bench-hyperu-a1-gamma|bench-kv>"
            ),
        )
        .into()),
    }
}

fn print_kv_benchmark() -> GoldenResult {
    const N: usize = 500_000;
    const ITERS: usize = 3;
    let denom = (N - 1) as f64;
    let z_values: Vec<f64> = (0..N).map(|i| 0.5 + 8.0 * (i as f64) / denom).collect();
    let v = SpecialTensor::RealScalar(2.0);
    let z = SpecialTensor::RealVec(z_values);

    let warm = kv(&v, &z, RuntimeMode::Strict)?;
    black_box(vector_checksum(warm, "kv")?);

    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..ITERS {
        let out = kv(black_box(&v), black_box(&z), RuntimeMode::Strict)?;
        checksum += vector_checksum(out, "kv")?;
    }
    let rust_elapsed = start.elapsed();
    if !checksum.is_finite() {
        return Err(IoError::new(ErrorKind::InvalidData, "non-finite kv checksum").into());
    }

    let scipy_elapsed = scipy_kv_duration(N, ITERS)?;
    let rust_ms = per_iter_ms(rust_elapsed, ITERS);
    let scipy_ms = per_iter_ms(scipy_elapsed, ITERS);
    println!(
        "kv_v2_n{N}_iters{ITERS} rust_ms_per_iter={rust_ms:.6} scipy_ms_per_iter={scipy_ms:.6} ratio={:.6} checksum={checksum:.17e}",
        rust_ms / scipy_ms
    );
    Ok(())
}

fn print_hyperu_benchmark() -> GoldenResult {
    const N: usize = 50_000;
    const ITERS: usize = 5;
    let denom = (N - 1) as f64;
    let x_values: Vec<f64> = (0..N).map(|i| 0.5 + 8.0 * (i as f64) / denom).collect();
    let a = SpecialTensor::RealScalar(1.5);
    let b = SpecialTensor::RealScalar(2.5);
    let x = SpecialTensor::RealVec(x_values);

    let warm = hyperu(&a, &b, &x, RuntimeMode::Strict)?;
    black_box(hyperu_checksum(warm)?);

    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..ITERS {
        let out = hyperu(
            black_box(&a),
            black_box(&b),
            black_box(&x),
            RuntimeMode::Strict,
        )?;
        checksum += hyperu_checksum(out)?;
    }
    let rust_elapsed = start.elapsed();
    if !checksum.is_finite() {
        return Err(IoError::new(ErrorKind::InvalidData, "non-finite hyperu checksum").into());
    }

    let scipy_elapsed = scipy_hyperu_duration(N, ITERS)?;
    let rust_ms = per_iter_ms(rust_elapsed, ITERS);
    let scipy_ms = per_iter_ms(scipy_elapsed, ITERS);
    println!(
        "hyperu_a1.5_b2.5_n{N}_iters{ITERS} rust_ms_per_iter={rust_ms:.6} scipy_ms_per_iter={scipy_ms:.6} ratio={:.6} checksum={checksum:.17e}",
        rust_ms / scipy_ms
    );
    Ok(())
}

fn print_hyperu_generic_benchmark() -> GoldenResult {
    const N: usize = 50_000;
    const ITERS: usize = 5;
    let denom = (N - 1) as f64;
    let x_values: Vec<f64> = (0..N).map(|i| 0.5 + 8.0 * (i as f64) / denom).collect();
    let a = SpecialTensor::RealScalar(1.0);
    let b = SpecialTensor::RealScalar(1.5);
    let x = SpecialTensor::RealVec(x_values);

    let warm = hyperu(&a, &b, &x, RuntimeMode::Strict)?;
    black_box(hyperu_checksum(warm)?);

    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..ITERS {
        let out = hyperu(
            black_box(&a),
            black_box(&b),
            black_box(&x),
            RuntimeMode::Strict,
        )?;
        checksum += hyperu_checksum(out)?;
    }
    let rust_elapsed = start.elapsed();
    if !checksum.is_finite() {
        return Err(IoError::new(ErrorKind::InvalidData, "non-finite hyperu checksum").into());
    }

    let scipy_elapsed = scipy_hyperu_generic_duration(N, ITERS)?;
    let rust_ms = per_iter_ms(rust_elapsed, ITERS);
    let scipy_ms = per_iter_ms(scipy_elapsed, ITERS);
    println!(
        "hyperu_a1_b1.5_n{N}_iters{ITERS} rust_ms_per_iter={rust_ms:.6} scipy_ms_per_iter={scipy_ms:.6} ratio={:.6} checksum={checksum:.17e}",
        rust_ms / scipy_ms
    );
    Ok(())
}

fn print_hyperu_a1_gamma_benchmark() -> GoldenResult {
    const N: usize = 50_000;
    const ITERS: usize = 5;
    let denom = (N - 1) as f64;
    let x_values: Vec<f64> = (0..N).map(|i| 0.5 + 8.0 * (i as f64) / denom).collect();
    let a = SpecialTensor::RealScalar(1.0);
    let b = SpecialTensor::RealScalar(1.25);
    let x = SpecialTensor::RealVec(x_values);

    let warm = hyperu(&a, &b, &x, RuntimeMode::Strict)?;
    black_box(hyperu_checksum(warm)?);

    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..ITERS {
        let out = hyperu(
            black_box(&a),
            black_box(&b),
            black_box(&x),
            RuntimeMode::Strict,
        )?;
        checksum += hyperu_checksum(out)?;
    }
    let rust_elapsed = start.elapsed();
    if !checksum.is_finite() {
        return Err(IoError::new(ErrorKind::InvalidData, "non-finite hyperu checksum").into());
    }

    let scipy_elapsed = scipy_hyperu_a1_gamma_duration(N, ITERS)?;
    let rust_ms = per_iter_ms(rust_elapsed, ITERS);
    let scipy_ms = per_iter_ms(scipy_elapsed, ITERS);
    println!(
        "hyperu_a1_b1.25_n{N}_iters{ITERS} rust_ms_per_iter={rust_ms:.6} scipy_ms_per_iter={scipy_ms:.6} ratio={:.6} checksum={checksum:.17e}",
        rust_ms / scipy_ms
    );
    Ok(())
}

fn hyperu_checksum(output: SpecialTensor) -> GoldenResult<f64> {
    vector_checksum(output, "hyperu")
}

fn vector_checksum(output: SpecialTensor, function: &str) -> GoldenResult<f64> {
    match output {
        SpecialTensor::RealVec(values) => {
            let mid = values.len() / 2;
            let first = values
                .first()
                .copied()
                .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "empty vector output"))?;
            let middle = values
                .get(mid)
                .copied()
                .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "missing vector midpoint"))?;
            let last = values
                .last()
                .copied()
                .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "empty vector output"))?;
            Ok(first + middle + last)
        }
        other => Err(unexpected_tensor(function, other)),
    }
}

fn scipy_kv_duration(n: usize, iters: usize) -> GoldenResult<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
z = np.linspace(0.5, 8.5, n, dtype=np.float64)
sc.kv(2.0, z)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.kv(2.0, z)
    checksum += float(out[0] + out[n // 2] + out[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| IoError::new(ErrorKind::BrokenPipe, "missing scipy stdin"))?
        .write_all(script.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(IoError::other(format!(
            "scipy kv oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let seconds = stdout.trim().parse::<f64>()?;
    Ok(Duration::from_secs_f64(seconds))
}

fn scipy_hyperu_duration(n: usize, iters: usize) -> GoldenResult<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
x = np.linspace(0.5, 8.5, n, dtype=np.float64)
sc.hyperu(1.5, 2.5, x)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.hyperu(1.5, 2.5, x)
    checksum += float(out[0] + out[n // 2] + out[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| IoError::new(ErrorKind::BrokenPipe, "missing scipy stdin"))?
        .write_all(script.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(IoError::other(format!(
            "scipy hyperu oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let seconds = stdout.trim().parse::<f64>()?;
    Ok(Duration::from_secs_f64(seconds))
}

fn scipy_hyperu_generic_duration(n: usize, iters: usize) -> GoldenResult<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
x = np.linspace(0.5, 8.5, n, dtype=np.float64)
sc.hyperu(1.0, 1.5, x)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.hyperu(1.0, 1.5, x)
    checksum += float(out[0] + out[n // 2] + out[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| IoError::new(ErrorKind::BrokenPipe, "missing scipy stdin"))?
        .write_all(script.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(IoError::other(format!(
            "scipy hyperu generic oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let seconds = stdout.trim().parse::<f64>()?;
    Ok(Duration::from_secs_f64(seconds))
}

fn scipy_hyperu_a1_gamma_duration(n: usize, iters: usize) -> GoldenResult<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
x = np.linspace(0.5, 8.5, n, dtype=np.float64)
sc.hyperu(1.0, 1.25, x)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.hyperu(1.0, 1.25, x)
    checksum += float(out[0] + out[n // 2] + out[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| IoError::new(ErrorKind::BrokenPipe, "missing scipy stdin"))?
        .write_all(script.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(IoError::other(format!(
            "scipy hyperu a1 gamma oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let seconds = stdout.trim().parse::<f64>()?;
    Ok(Duration::from_secs_f64(seconds))
}

fn per_iter_ms(duration: Duration, iters: usize) -> f64 {
    duration.as_secs_f64() * 1_000.0 / iters as f64
}

fn print_error_golden() -> GoldenResult {
    for &x in ERROR_INPUTS {
        print_scalar(
            "erf",
            x,
            erf(&SpecialTensor::RealScalar(x), RuntimeMode::Strict),
        )?;
        print_scalar(
            "erfc",
            x,
            erfc(&SpecialTensor::RealScalar(x), RuntimeMode::Strict),
        )?;
    }

    let vector_inputs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    print_vector(
        "erf_vec",
        erf(
            &SpecialTensor::RealVec(vector_inputs.to_vec()),
            RuntimeMode::Strict,
        ),
    )?;
    print_vector(
        "erfc_vec",
        erfc(
            &SpecialTensor::RealVec(vector_inputs.to_vec()),
            RuntimeMode::Strict,
        ),
    )
}

fn print_elliptic_golden() -> GoldenResult {
    let scalar_cases = [(PI / 6.0, 0.0), (PI / 4.0, 0.5), (PI / 3.0, 0.9)];
    for (phi, m) in scalar_cases {
        print_binary_scalar(
            "ellipkinc",
            phi,
            m,
            ellipkinc(
                &SpecialTensor::RealScalar(phi),
                &SpecialTensor::RealScalar(m),
                RuntimeMode::Strict,
            ),
        )?;
        print_binary_scalar(
            "ellipeinc",
            phi,
            m,
            ellipeinc(
                &SpecialTensor::RealScalar(phi),
                &SpecialTensor::RealScalar(m),
                RuntimeMode::Strict,
            ),
        )?;
    }

    print_vector(
        "ellipkinc_broadcast_m",
        ellipkinc(
            &SpecialTensor::RealScalar(PI / 3.0),
            &SpecialTensor::RealVec(vec![0.0, 0.25, 0.5, 0.75]),
            RuntimeMode::Strict,
        ),
    )?;
    print_vector(
        "ellipeinc_pairwise",
        ellipeinc(
            &SpecialTensor::RealVec(vec![PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0 - 0.1]),
            &SpecialTensor::RealVec(vec![0.0, 0.25, 0.5, 0.75]),
            RuntimeMode::Strict,
        ),
    )
}

fn print_scalar(
    function: &str,
    input: f64,
    result: Result<SpecialTensor, fsci_special::SpecialError>,
) -> GoldenResult {
    match result? {
        SpecialTensor::RealScalar(value) => {
            println!(
                "{function} input_bits={:016x} output_bits={:016x} output={value:.17e}",
                input.to_bits(),
                value.to_bits(),
            );
            Ok(())
        }
        other => Err(unexpected_tensor("RealScalar", other)),
    }
}

fn print_binary_scalar(
    function: &str,
    left: f64,
    right: f64,
    result: Result<SpecialTensor, fsci_special::SpecialError>,
) -> GoldenResult {
    match result? {
        SpecialTensor::RealScalar(value) => {
            println!(
                "{function} left_bits={:016x} right_bits={:016x} output_bits={:016x} output={value:.17e}",
                left.to_bits(),
                right.to_bits(),
                value.to_bits(),
            );
            Ok(())
        }
        other => Err(unexpected_tensor("RealScalar", other)),
    }
}

fn print_vector(
    function: &str,
    result: Result<SpecialTensor, fsci_special::SpecialError>,
) -> GoldenResult {
    match result? {
        SpecialTensor::RealVec(values) => {
            print!("{function}");
            for value in values {
                print!(" {:016x}", value.to_bits());
            }
            println!();
            Ok(())
        }
        other => Err(unexpected_tensor("RealVec", other)),
    }
}

fn unexpected_tensor(expected: &str, actual: SpecialTensor) -> Box<dyn Error> {
    IoError::new(
        ErrorKind::InvalidData,
        format!("expected {expected}, got {actual:?}"),
    )
    .into()
}
