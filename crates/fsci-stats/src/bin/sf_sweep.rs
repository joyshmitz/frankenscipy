use fsci_stats::*;
fn main() {
    // (name, sf(x)) at far-tail points
    let n = Normal::new(0.0, 1.0);
    for x in [8.0, 20.0, 37.0] {
        println!("norm {x} {:.17e}", n.sf(x));
    }
    let t = StudentT::new(5.0);
    for x in [50.0, 300.0] {
        println!("t5 {x} {:.17e}", t.sf(x));
    }
    let c2 = ChiSquared::new(4.0);
    for x in [80.0, 150.0] {
        println!("chi2_4 {x} {:.17e}", c2.sf(x));
    }
    let e = Exponential::new(1.0);
    for x in [40.0, 300.0] {
        println!("expon {x} {:.17e}", e.sf(x));
    }
    let w = Weibull::new(1.5, 1.0);
    for x in [25.0, 60.0] {
        println!("weib_1.5 {x} {:.17e}", w.sf(x));
    }
    let ca = Cauchy::new(0.0, 1.0);
    for x in [1.0e6, 1.0e10] {
        println!("cauchy {x} {:.17e}", ca.sf(x));
    }
    let la = Laplace::new(0.0, 1.0);
    for x in [40.0, 200.0] {
        println!("laplace {x} {:.17e}", la.sf(x));
    }
    let gu = Gumbel::new(0.0, 1.0);
    for x in [40.0, 200.0] {
        println!("gumbel {x} {:.17e}", gu.sf(x));
    }
    let lo = Logistic::new(0.0, 1.0);
    for x in [40.0, 300.0] {
        println!("logistic {x} {:.17e}", lo.sf(x));
    }
    let pa = Pareto::new(2.0, 1.0);
    for x in [1.0e6, 1.0e10] {
        println!("pareto_2 {x} {:.17e}", pa.sf(x));
    }
    let f = FDistribution::new(5.0, 10.0);
    for x in [50.0, 500.0] {
        println!("f_5_10 {x} {:.17e}", f.sf(x));
    }
    let g = GammaDist::new(2.0, 1.0);
    for x in [40.0, 200.0] {
        println!("gamma_2 {x} {:.17e}", g.sf(x));
    }
}
