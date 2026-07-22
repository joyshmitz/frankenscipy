// Spawn-vs-pool dispatch cost probe for the cholesky panel loop (thinkstation1).
// Prices (a) thread::scope spawn+join per round, (b) persistent workers woken by
// std::sync::Barrier per round, at T threads with ~W us of fake work each.
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

fn busy(us: u64, sink: &AtomicU64) {
    let start = Instant::now();
    let mut acc = 0u64;
    while (start.elapsed().as_nanos() as u64) < us * 1000 {
        acc = acc.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    sink.fetch_xor(acc, Ordering::Relaxed);
}

fn main() {
    let sink = AtomicU64::new(0);
    let rounds = 300usize;
    for &t in &[4usize, 6, 8, 13, 16] {
        for &work_us in &[0u64, 100, 500] {
            // (a) fresh scope per round
            let mut scope_ns = Vec::with_capacity(rounds);
            for _ in 0..rounds {
                let start = Instant::now();
                std::thread::scope(|s| {
                    for _ in 0..t {
                        s.spawn(|| busy(work_us, &sink));
                    }
                });
                scope_ns.push(start.elapsed().as_nanos() as u64);
            }
            scope_ns.sort_unstable();
            // (b) persistent workers, barrier dispatch per round
            let start_barrier = Barrier::new(t + 1);
            let end_barrier = Barrier::new(t + 1);
            let stop = AtomicBool::new(false);
            let mut pool_ns = Vec::with_capacity(rounds);
            std::thread::scope(|s| {
                for _ in 0..t {
                    s.spawn(|| {
                        loop {
                            start_barrier.wait();
                            if stop.load(Ordering::Acquire) {
                                break;
                            }
                            busy(work_us, &sink);
                            end_barrier.wait();
                        }
                    });
                }
                for _ in 0..rounds {
                    let start = Instant::now();
                    start_barrier.wait();
                    end_barrier.wait();
                    pool_ns.push(start.elapsed().as_nanos() as u64);
                }
                stop.store(true, Ordering::Release);
                start_barrier.wait();
            });
            pool_ns.sort_unstable();
            let med = |v: &[u64]| v[v.len() / 2];
            println!(
                "T={t:2} work_us={work_us:3} scope_median_us={:8.1} pool_median_us={:8.1} overhead_scope_us={:7.1} overhead_pool_us={:7.1}",
                med(&scope_ns) as f64 / 1000.0,
                med(&pool_ns) as f64 / 1000.0,
                med(&scope_ns) as f64 / 1000.0 - work_us as f64,
                med(&pool_ns) as f64 / 1000.0 - work_us as f64,
            );
        }
    }
    println!("sink={}", sink.load(Ordering::Relaxed));
}
