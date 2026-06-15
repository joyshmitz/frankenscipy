//! Lebedev quadrature probe vs scipy.integrate.lebedev_rule.
//! Lines: `order,i,x,y,z,w`. Dumps every point of every supported order.
use fsci_integrate::lebedev_rule;

fn main() {
    let orders = [
        3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83,
        89, 95, 101, 107, 113, 119, 125, 131,
    ];
    for n in orders {
        let rule = lebedev_rule(n).expect("supported order");
        for (i, (p, &w)) in rule.points.iter().zip(rule.weights.iter()).enumerate() {
            println!(
                "{n},{i},{:.17e},{:.17e},{:.17e},{:.17e}",
                p[0], p[1], p[2], w
            );
        }
    }
}
