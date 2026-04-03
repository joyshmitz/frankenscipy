use fsci_stats::winsorize;

fn main() {
    let result = winsorize(&[1.0, 2.0, 3.0], (0.0, 1.1));
    println!("{:?}", result);
}
