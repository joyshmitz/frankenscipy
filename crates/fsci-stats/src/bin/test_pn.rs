fn main() {
    for x in [1.0_f64, 2.0, 5.0, 10.0] {
        println!("x={x}: tetragamma={:.10} pentagamma={:.10}",
            fsci_special::tetragamma(x), fsci_special::pentagamma(x));
    }
}
