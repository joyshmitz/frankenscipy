fn main() {
    use fsci_stats::fligner;
    let g1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let g2: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let r = fligner(&[&g1, &g2]);
    println!("g2_equal_var: stat={}, p={}", r.statistic, r.pvalue);

    let g3: Vec<f64> = vec![1.0, 5.0, 9.0, 13.0, 17.0];
    let r = fligner(&[&g1, &g3]);
    println!("g2_unequal_var: stat={}, p={}", r.statistic, r.pvalue);

    let g4: Vec<f64> = vec![1.0, 5.0, 9.0];
    let g5: Vec<f64> = vec![2.0, 6.0, 10.0, 14.0];
    let r = fligner(&[&g1, &g4, &g5]);
    println!("g3_mixed: stat={}, p={}", r.statistic, r.pvalue);
}
