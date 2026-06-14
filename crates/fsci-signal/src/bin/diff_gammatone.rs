// Probe: print fsci gammatone (b,a) for cases matched against scipy.signal.gammatone.
use fsci_signal::{GammatoneType, gammatone};

fn main() {
    let cases: &[(f64, GammatoneType, Option<usize>, Option<usize>, f64)] = &[
        (440.0, GammatoneType::Fir, Some(4), None, 16000.0),
        (1000.0, GammatoneType::Fir, Some(4), None, 16000.0),
        (100.0, GammatoneType::Fir, Some(2), None, 8000.0),
        (440.0, GammatoneType::Iir, None, None, 16000.0),
        (1000.0, GammatoneType::Iir, None, None, 16000.0),
        (250.0, GammatoneType::Iir, None, None, 8000.0),
    ];
    for &(freq, ft, order, numtaps, fs) in cases {
        let c = gammatone(freq, ft, order, numtaps, Some(fs)).unwrap();
        println!("=== {freq} {ft:?} order={order:?} fs={fs}");
        let bs: Vec<String> = c.b.iter().map(|x| format!("{x:.14e}")).collect();
        let as_: Vec<String> = c.a.iter().map(|x| format!("{x:.14e}")).collect();
        println!("b {}", bs.join(" "));
        println!("a {}", as_.join(" "));
    }
}
