use std::hint::black_box;
use std::time::Instant;
fn main() {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut r = || { s^=s<<13; s^=s>>7; s^=s<<17; (s>>11) as f64/(1u64<<53) as f64 };
    for &n in &[50_000usize, 500_000, 5_000_000] {
        let base: Vec<f64> = (0..n).map(|_| r()*1e3-500.0).collect();
        let reps = if n<=500_000 {12} else {4};
        let time = |unstable: bool| {
            let mut best = f64::MAX;
            for _ in 0..reps {
                let mut v = base.clone();
                let t = Instant::now();
                if unstable { v.sort_unstable_by(f64::total_cmp); } else { v.sort_by(f64::total_cmp); }
                black_box(&v);
                let e = t.elapsed().as_secs_f64()*1000.0;
                if e<best { best=e; }
            }
            best
        };
        let sb = time(false); let su = time(true);
        println!("n={n:8}: sort_by {sb:8.2}ms -> sort_unstable {su:8.2}ms = {:.2}x", sb/su);
    }
}
