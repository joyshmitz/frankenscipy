use fsci_signal::remez;

fn main() {
    let bands = [0.0, 0.2, 0.3, 0.5];
    let desired = [1.0, 0.0];
    let weights = [1.0, 10.0];
    let taps = remez(257, &bands, &desired, Some(&weights)).expect("remez golden should solve");

    println!("case=remez_257_two_band len={}", taps.len());
    for (idx, tap) in taps.iter().enumerate() {
        println!("{idx:03} {tap:.17e}");
    }
}
