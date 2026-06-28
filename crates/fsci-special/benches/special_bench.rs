use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialTensor, beta, ellipe, ellipeinc, ellipk, ellipkinc, erf, erfc, erfinv, gamma, gammainc,
    gammaln, hyperu, j0, j1, jn_zeros, jnjnp_zeros, jnp_zeros, jv, ndtri, rgamma, y0, zeta,
    zeta_scalar,
};
use std::f64::consts::PI;
use std::hint::black_box;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

fn scalar(x: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(x)
}

fn real_vec(values: &[f64]) -> SpecialTensor {
    SpecialTensor::RealVec(values.to_vec())
}

fn real_val(t: &SpecialTensor) -> f64 {
    match t {
        SpecialTensor::RealScalar(v) => *v,
        _ => panic!("expected RealScalar"),
    }
}

fn consume_tensor(t: SpecialTensor) {
    match t {
        SpecialTensor::RealScalar(v) => {
            black_box(v);
        }
        SpecialTensor::RealVec(values) => {
            black_box(values);
        }
        _ => panic!("unexpected tensor shape"),
    }
}

const GAMMA_INPUTS: &[f64] = &[0.5, 1.0, 2.5, 5.0, 10.0, 50.0, 100.0];
const ERF_INPUTS: &[f64] = &[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
const BESSEL_INPUTS: &[f64] = &[0.1, 1.0, 5.0, 10.0, 20.0, 50.0];
const ELLIPTIC_M_INPUTS: &[f64] = &[0.0, 0.5, 0.9];
const ELLIPTIC_INCOMPLETE_INPUTS: &[(f64, f64)] =
    &[(PI / 6.0, 0.0), (PI / 4.0, 0.5), (PI / 3.0, 0.9)];

fn bench_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_gamma");

    for &x in GAMMA_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("gamma", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = gamma(black_box(input), RuntimeMode::Strict).expect("gamma");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_gammaln(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_gammaln");

    for &x in GAMMA_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("gammaln", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = gammaln(black_box(input), RuntimeMode::Strict).expect("gammaln");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_rgamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_rgamma");

    for &x in GAMMA_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("rgamma", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = rgamma(black_box(input), RuntimeMode::Strict).expect("rgamma");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_gammainc(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_gammainc");

    let pairs: &[(f64, f64)] = &[(1.0, 1.0), (2.0, 3.0), (5.0, 5.0), (10.0, 10.0)];
    for &(a, x) in pairs {
        let sa = scalar(a);
        let sx = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("gammainc", format!("a{a}_x{x}")),
            &(sa, sx),
            |b, (sa, sx)| {
                b.iter(|| {
                    let out = gammainc(black_box(sa), black_box(sx), RuntimeMode::Strict)
                        .expect("gammainc");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_erf(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erf");

    for &x in ERF_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("erf", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = erf(black_box(input), RuntimeMode::Strict).expect("erf");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_erfc(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erfc");

    for &x in ERF_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("erfc", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = erfc(black_box(input), RuntimeMode::Strict).expect("erfc");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_erfinv(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erfinv");

    let inputs: &[f64] = &[-0.9, -0.5, 0.0, 0.5, 0.9];
    for &x in inputs {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("erfinv", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = erfinv(black_box(input), RuntimeMode::Strict).expect("erfinv");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn scipy_erfinv_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
y = np.linspace(-0.95, 0.95, n, dtype=np.float64)
sc.erfinv(y)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.erfinv(y)
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
        .spawn()
        .expect("spawn scipy erfinv oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy erfinv oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy erfinv oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy erfinv oracle");
    if !output.status.success() {
        eprintln!(
            "scipy erfinv oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy erfinv timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy erfinv timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_special_erfinv_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erfinv_array");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let n = 100_000usize;
    let denom = (n - 1) as f64;
    let y: Vec<f64> = (0..n).map(|i| -0.95 + 1.9 * (i as f64) / denom).collect();
    let input = real_vec(&y);

    group.bench_function("rust_current_n100000", |b| {
        b.iter(|| {
            let out = erfinv(black_box(&input), RuntimeMode::Strict).expect("erfinv");
            black_box(out);
        });
    });

    if scipy_special_available() {
        group.bench_function("scipy_n100000", |b| {
            b.iter_custom(|iters| {
                scipy_erfinv_duration(n, iters)
                    .expect("scipy erfinv oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping scipy_erfinv_n100000: python3 cannot import scipy.special");
    }

    group.finish();
}

fn bench_beta(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_beta");

    let pairs: &[(f64, f64)] = &[(0.5, 0.5), (1.0, 1.0), (2.0, 3.0), (5.0, 5.0)];
    for &(a, b_val) in pairs {
        let sa = scalar(a);
        let sb = scalar(b_val);
        group.bench_with_input(
            BenchmarkId::new("beta", format!("a{a}_b{b_val}")),
            &(sa, sb),
            |b, (sa, sb)| {
                b.iter(|| {
                    let out =
                        beta(black_box(sa), black_box(sb), RuntimeMode::Strict).expect("beta");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_bessel_jv_array(c: &mut Criterion) {
    // Array J_v(z): scalar order, large real vector — the par_map_indices fan-out
    // path. Head-to-head vs scipy.special.jv(2, z) (~104 ms at n=200k).
    let mut group = c.benchmark_group("special_bessel_jv_array");
    for &n in &[50_000usize, 200_000] {
        let zs: Vec<f64> = (0..n)
            .map(|i| (i as f64 / n as f64) * 50.0 + 0.01)
            .collect();
        let z = SpecialTensor::RealVec(zs);
        let order = SpecialTensor::RealScalar(2.0);
        group.bench_function(BenchmarkId::new("v2", n), |b| {
            b.iter(|| {
                let out = jv(black_box(&order), black_box(&z), RuntimeMode::Strict).expect("jv");
                black_box(out);
            });
        });
    }
    group.finish();
}

fn bench_bessel_j(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_bessel_j");

    for &x in BESSEL_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("j0", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = j0(black_box(input), RuntimeMode::Strict).expect("j0");
                    black_box(real_val(&out));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("j1", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = j1(black_box(input), RuntimeMode::Strict).expect("j1");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_bessel_y0(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_bessel_y");

    for &x in &[0.1, 1.0, 5.0, 10.0, 20.0] {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("y0", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = y0(black_box(input), RuntimeMode::Strict).expect("y0");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_complete_elliptic(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_complete_elliptic");

    for &m in ELLIPTIC_M_INPUTS {
        let input = scalar(m);
        group.bench_with_input(
            BenchmarkId::new("ellipk", format!("m{m}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = ellipk(black_box(input), RuntimeMode::Strict).expect("ellipk");
                    black_box(real_val(&out));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ellipe", format!("m{m}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = ellipe(black_box(input), RuntimeMode::Strict).expect("ellipe");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_incomplete_elliptic(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_incomplete_elliptic");

    for &(phi, m) in ELLIPTIC_INCOMPLETE_INPUTS {
        let ellipkinc_phi_input = scalar(phi);
        let ellipkinc_m_input = scalar(m);
        let ellipeinc_phi_input = scalar(phi);
        let ellipeinc_m_input = scalar(m);
        let case = format!("phi{phi:.3}_m{m:.1}");
        group.bench_with_input(
            BenchmarkId::new("ellipkinc_scalar", &case),
            &(ellipkinc_phi_input, ellipkinc_m_input),
            |b, (phi_input, m_input)| {
                b.iter(|| {
                    let out = ellipkinc(
                        black_box(phi_input),
                        black_box(m_input),
                        RuntimeMode::Strict,
                    )
                    .expect("ellipkinc");
                    black_box(real_val(&out));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ellipeinc_scalar", case),
            &(ellipeinc_phi_input, ellipeinc_m_input),
            |b, (phi_input, m_input)| {
                b.iter(|| {
                    let out = ellipeinc(
                        black_box(phi_input),
                        black_box(m_input),
                        RuntimeMode::Strict,
                    )
                    .expect("ellipeinc");
                    black_box(real_val(&out));
                });
            },
        );
    }

    let broadcast_m = (scalar(PI / 3.0), real_vec(&[0.0, 0.25, 0.5, 0.75]));
    group.bench_with_input(
        BenchmarkId::new("ellipkinc_broadcast_m", "scalar_phi_over_m_vec"),
        &broadcast_m,
        |b, (phi_input, m_input)| {
            b.iter(|| {
                let out = ellipkinc(
                    black_box(phi_input),
                    black_box(m_input),
                    RuntimeMode::Strict,
                )
                .expect("ellipkinc broadcast over m");
                consume_tensor(out);
            });
        },
    );

    let pairwise = (
        real_vec(&[PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0 - 0.1]),
        real_vec(&[0.0, 0.25, 0.5, 0.75]),
    );
    group.bench_with_input(
        BenchmarkId::new("ellipeinc_pairwise_vec", "phi_vec_m_vec"),
        &pairwise,
        |b, (phi_input, m_input)| {
            b.iter(|| {
                let out = ellipeinc(
                    black_box(phi_input),
                    black_box(m_input),
                    RuntimeMode::Strict,
                )
                .expect("ellipeinc pairwise vector");
                consume_tensor(out);
            });
        },
    );

    group.finish();
}

fn scipy_special_available() -> bool {
    let script = "import scipy.special\n";
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn scipy special availability check");
    child
        .stdin
        .as_mut()
        .expect("open scipy special availability stdin")
        .write_all(script.as_bytes())
        .expect("write scipy special availability script");
    child.wait().is_ok_and(|status| status.success())
}

fn scipy_ndtri_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
q = np.linspace(1e-12, 1.0 - 1e-12, n, dtype=np.float64)
sc.ndtri(q)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.ndtri(q)
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
        .spawn()
        .expect("spawn scipy ndtri oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy ndtri oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy ndtri oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy ndtri oracle");
    if !output.status.success() {
        eprintln!(
            "scipy ndtri oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy ndtri timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy ndtri timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_special_ndtri_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_ndtri_array");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let n = 500_000usize;
    let denom = (n - 1) as f64;
    let q: Vec<f64> = (0..n)
        .map(|i| 1.0e-12 + (1.0 - 2.0e-12) * (i as f64) / denom)
        .collect();
    let input = real_vec(&q);

    group.bench_function("rust_current_n500000", |b| {
        b.iter(|| {
            let out = ndtri(black_box(&input), RuntimeMode::Strict).expect("ndtri");
            black_box(out);
        });
    });

    if scipy_special_available() {
        group.bench_function("scipy_n500000", |b| {
            b.iter_custom(|iters| {
                scipy_ndtri_duration(n, iters)
                    .expect("scipy ndtri oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping scipy_ndtri_n500000: python3 cannot import scipy.special");
    }

    group.finish();
}

fn scipy_zeta_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
s = 1.1 + np.arange(n, dtype=np.float64) * (8.9 / max(n - 1, 1))
sc.zeta(s)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.zeta(s)
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
        .spawn()
        .expect("spawn scipy zeta oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy zeta oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy zeta oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy zeta oracle");
    if !output.status.success() {
        eprintln!(
            "scipy zeta oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy zeta timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy zeta timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_special_zeta_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_zeta_array");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let n = 100_000usize;
    let denom = (n - 1).max(1) as f64;
    let values: Vec<f64> = (0..n).map(|i| 1.1 + 8.9 * (i as f64) / denom).collect();
    let input = real_vec(&values);

    group.bench_function("rust_scalar_loop_n100000", |b| {
        b.iter(|| {
            let out: Vec<f64> = values.iter().copied().map(zeta_scalar).collect();
            black_box(out);
        });
    });

    group.bench_function("rust_tensor_n100000", |b| {
        b.iter(|| {
            let out = zeta(black_box(&input), RuntimeMode::Strict).expect("zeta");
            black_box(out);
        });
    });

    if scipy_special_available() {
        group.bench_function("scipy_n100000", |b| {
            b.iter_custom(|iters| {
                scipy_zeta_duration(n, iters)
                    .expect("scipy zeta oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping scipy_zeta_n100000: python3 cannot import scipy.special");
    }

    group.finish();
}

fn scipy_hyperu_a1_gamma_duration(n: usize, iters: u64) -> Option<Duration> {
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
        .spawn()
        .expect("spawn scipy hyperu oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy hyperu oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy hyperu oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy hyperu oracle");
    if !output.status.success() {
        eprintln!(
            "scipy hyperu oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy hyperu timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy hyperu timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_special_hyperu_a1_gamma_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_hyperu_a1_gamma_array");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let n = 50_000usize;
    let denom = (n - 1).max(1) as f64;
    let x_values: Vec<f64> = (0..n).map(|i| 0.5 + 8.0 * (i as f64) / denom).collect();
    let a = scalar(1.0);
    let b_param = scalar(1.25);
    let x = real_vec(&x_values);

    group.bench_function("rust_current_n50000", |b| {
        b.iter(|| {
            let out = hyperu(
                black_box(&a),
                black_box(&b_param),
                black_box(&x),
                RuntimeMode::Strict,
            )
            .expect("hyperu");
            black_box(out);
        });
    });

    if scipy_special_available() {
        group.bench_function("scipy_n50000", |b| {
            b.iter_custom(|iters| {
                scipy_hyperu_a1_gamma_duration(n, iters)
                    .expect("scipy hyperu oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping scipy_hyperu_a1_gamma_n50000: python3 cannot import scipy.special");
    }

    group.finish();
}

fn scipy_jnjnp_zeros_duration(nt: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

nt = int(sys.argv[1])
iters = int(sys.argv[2])
sc.jnjnp_zeros(nt)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    zo, n, m, t = sc.jnjnp_zeros(nt)
    checksum += float(zo[-1]) + float(n[-1]) + float(m[-1]) + float(t[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &nt.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy jnjnp_zeros oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy jnjnp_zeros oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy jnjnp_zeros oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy jnjnp_zeros oracle");
    if !output.status.success() {
        eprintln!(
            "scipy jnjnp_zeros oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy jnjnp_zeros timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy jnjnp_zeros timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn legacy_duplicate_jnjnp_zeros(nt: usize) -> (Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>) {
    if nt == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let mut cands: Vec<(f64, i32, i32, i32)> = Vec::new();
    cands.push((0.0, 0, 0, 1));
    let per = nt + 2;
    let n_max = nt as u32 + 2;
    for n in 0..=n_max {
        for (i, &x) in jn_zeros(n, per).iter().enumerate() {
            cands.push((x, n as i32, (i + 1) as i32, 0));
        }
        let jp = if n == 0 {
            jn_zeros(1, per)
        } else {
            jnp_zeros(n, per)
        };
        for (i, &x) in jp.iter().enumerate() {
            cands.push((x, n as i32, (i + 1) as i32, 1));
        }
    }
    cands.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("Bessel zeros are finite")
            .then(a.3.cmp(&b.3))
    });
    cands.truncate(nt);
    let zo = cands.iter().map(|c| c.0).collect();
    let n = cands.iter().map(|c| c.1).collect();
    let m = cands.iter().map(|c| c.2).collect();
    let t = cands.iter().map(|c| c.3).collect();
    (zo, n, m, t)
}

fn bench_acoco_gauntlet_jnjnp_zeros(c: &mut Criterion) {
    let mut group = c.benchmark_group("acoco_gauntlet_jnjnp_zeros");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    for &nt in &[64_usize, 128_usize] {
        group.bench_function(format!("rust_current_nt{nt}"), |b| {
            b.iter(|| {
                let (zo, n, m, t) = jnjnp_zeros(black_box(nt));
                black_box((zo, n, m, t));
            });
        });
        group.bench_function(format!("rust_legacy_duplicate_nt{nt}"), |b| {
            b.iter(|| {
                let (zo, n, m, t) = legacy_duplicate_jnjnp_zeros(black_box(nt));
                black_box((zo, n, m, t));
            });
        });
        if scipy_special_available() {
            group.bench_function(format!("scipy_nt{nt}"), |b| {
                b.iter_custom(|iters| {
                    scipy_jnjnp_zeros_duration(nt, iters)
                        .expect("scipy jnjnp_zeros oracle should run after availability check")
                });
            });
        } else {
            eprintln!("skipping scipy_nt{nt}: python3 cannot import scipy.special");
        }
    }
    group.finish();
}

/// Array (RealVec) dispatch — the realistic ufunc workload. fsci parallelizes the per-family
/// array path; scipy.special is vectorized single-core C. Head-to-head vs scipy.
fn bench_array(c: &mut Criterion) {
    let xs: Vec<f64> = (0..65536).map(|i| 0.5 + (i as f64) * 0.0001).collect();
    let t = real_vec(&xs);
    let mut group = c.benchmark_group("special_array_65536");
    group.bench_function("gamma", |b| {
        b.iter(|| gamma(black_box(&t), RuntimeMode::Strict).expect("gamma"))
    });
    group.bench_function("erf", |b| {
        b.iter(|| erf(black_box(&t), RuntimeMode::Strict).expect("erf"))
    });
    group.bench_function("j0", |b| {
        b.iter(|| j0(black_box(&t), RuntimeMode::Strict).expect("j0"))
    });
    group.finish();
}

fn bench_ncfdtr(c: &mut Criterion) {
    let mut group = c.benchmark_group("ncfdtr");
    // cost scales with nc: the Poisson(nc/2) mixture spans ~O(√nc) incomplete-beta
    // terms. Evaluate a batch of f for each (dfn, dfd, nc).
    let fs: Vec<f64> = (0..256).map(|i| 0.05 + (i as f64) * 0.02).collect();
    for &(dfn, dfd, nc) in &[(10.0_f64, 10.0_f64, 2.0_f64), (20.0, 30.0, 200.0), (50.0, 50.0, 2000.0)] {
        group.bench_function(BenchmarkId::new("cdf", format!("dfn{dfn}_dfd{dfd}_nc{nc}")), |b| {
            b.iter(|| {
                fs.iter()
                    .map(|&f| fsci_special::ncfdtr(black_box(dfn), black_box(dfd), black_box(nc), black_box(f)))
                    .sum::<f64>()
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ncfdtr,
    bench_array,
    bench_gamma,
    bench_gammaln,
    bench_rgamma,
    bench_gammainc,
    bench_erf,
    bench_erfc,
    bench_erfinv,
    bench_special_erfinv_array,
    bench_special_ndtri_array,
    bench_special_zeta_array,
    bench_special_hyperu_a1_gamma_array,
    bench_beta,
    bench_bessel_jv_array,
    bench_bessel_j,
    bench_bessel_y0,
    bench_complete_elliptic,
    bench_incomplete_elliptic,
    bench_acoco_gauntlet_jnjnp_zeros
);
criterion_main!(benches);
