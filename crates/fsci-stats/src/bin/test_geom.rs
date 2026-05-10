fn main() {
    use fsci_stats::{geometric_discrepancy, GeometricDiscrepancyMethod};
    let sample = vec![
        0.55130587, 0.15176798,
        0.05130587, 0.81843465,
        0.80130587, 0.48510131,
        0.30130587, 0.26287909,
        0.67630587, 0.92954576,
        0.17630587, 0.59621242,
        0.92630587, 0.04065687,
        0.42630587, 0.70732354,
    ];
    let mindist = geometric_discrepancy(&sample, 2, GeometricDiscrepancyMethod::MinDist).unwrap();
    let mst = geometric_discrepancy(&sample, 2, GeometricDiscrepancyMethod::Mst).unwrap();
    println!("mindist = {:.16}  scipy: 0.25496610764841404", mindist);
    println!("mst     = {:.16}  scipy: 0.3286278710953668", mst);
    println!("mindist diff: {:.2e}", (mindist - 0.25496610764841404_f64).abs());
    println!("mst diff:     {:.2e}", (mst - 0.3286278710953668_f64).abs());
}
