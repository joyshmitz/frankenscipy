//! Same-binary A/B of savetxt: serial vs parallel row-format with byte identity
//! across matrix size.
use fsci_io::{SAVETXT_FORCE_SERIAL, savetxt};
use std::sync::atomic::Ordering;
use std::time::Instant;

fn time_it(
    rows: usize,
    cols: usize,
    data: &[f64],
    serial: bool,
    reps: usize,
) -> Result<f64, fsci_io::IoError> {
    SAVETXT_FORCE_SERIAL.store(serial, Ordering::Relaxed);
    let _ = savetxt(rows, cols, data, " ")?;
    let t = Instant::now();
    let mut acc = 0usize;
    for _ in 0..reps {
        acc += savetxt(rows, cols, data, " ")?.len();
    }
    std::hint::black_box(acc);
    Ok(t.elapsed().as_secs_f64() * 1000.0 / reps as f64)
}

fn main() -> Result<(), fsci_io::IoError> {
    let mut state = 0x1234u64;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 200.0 - 100.0
    };
    for &(rows, cols) in &[(2000usize, 20usize), (10000, 20), (50000, 20), (200000, 10)] {
        let data: Vec<f64> = (0..rows * cols).map(|_| rng()).collect();
        let reps = if rows >= 50000 { 8 } else { 20 };
        let mut ser = 0.0;
        let mut par = 0.0;
        for _ in 0..3 {
            ser += time_it(rows, cols, &data, true, reps)?;
            par += time_it(rows, cols, &data, false, reps)?;
        }
        ser /= 3.0;
        par /= 3.0;
        SAVETXT_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let s = savetxt(rows, cols, &data, " ")?;
        SAVETXT_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let p = savetxt(rows, cols, &data, " ")?;
        println!(
            "savetxt {rows}x{cols}: serial={ser:>8.2}ms  parallel={par:>7.2}ms  speedup={:>5.1}x  identical={}",
            ser / par,
            s == p
        );
    }
    Ok(())
}
