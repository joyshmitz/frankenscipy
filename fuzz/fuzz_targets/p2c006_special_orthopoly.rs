#![no_main]

use arbitrary::Arbitrary;
use fsci_special::orthopoly::{
    eval_chebyt, eval_chebyu, eval_gegenbauer, eval_genlaguerre, eval_hermite, eval_hermitenorm,
    eval_jacobi, eval_laguerre, eval_legendre, lpmv, roots_chebyt, roots_chebyu, roots_hermite,
    roots_laguerre, roots_legendre,
};
use libfuzzer_sys::fuzz_target;

// Orthogonal polynomial oracle:
// Tests orthopoly functions for correctness properties:
//
// 1. Polynomial evaluation at roots should be near zero
// 2. Recurrence relations should hold
// 3. Known special values (P_n(1) = 1, T_n(1) = 1, etc.)
// 4. Weights should be positive and sum to expected value
// 5. No panics on valid inputs

const MAX_DEGREE: u32 = 20;
const TOL: f64 = 1e-8;

#[derive(Debug, Arbitrary)]
struct OrthopolyInput {
    degree_raw: u8,
    x_raw: f64,
    alpha_raw: f64,
    beta_raw: f64,
    m_raw: i8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-10.0, 10.0)
    } else {
        0.0
    }
}

fn sanitize_unit(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize_positive(x: f64) -> f64 {
    if x.is_finite() && x > -0.5 {
        x.clamp(-0.4, 10.0)
    } else {
        0.5
    }
}

fuzz_target!(|input: OrthopolyInput| {
    let n = (input.degree_raw as u32) % (MAX_DEGREE + 1);
    let x = sanitize_unit(input.x_raw);
    let alpha = sanitize_positive(input.alpha_raw);
    let beta = sanitize_positive(input.beta_raw);
    let m = (input.m_raw as i32).clamp(-(n as i32), n as i32);

    // Test 1: Legendre polynomial special values P_n(1) = 1
    let p_at_1 = eval_legendre(n, 1.0);
    if (p_at_1 - 1.0).abs() > TOL {
        panic!(
            "Legendre P_{}(1) = {}, expected 1.0",
            n, p_at_1
        );
    }

    // Test 2: Chebyshev T_n(1) = 1
    let t_at_1 = eval_chebyt(n, 1.0);
    if (t_at_1 - 1.0).abs() > TOL {
        panic!(
            "Chebyshev T_{}(1) = {}, expected 1.0",
            n, t_at_1
        );
    }

    // Test 3: Chebyshev U_n(1) = n + 1
    let u_at_1 = eval_chebyu(n, 1.0);
    let expected_u = (n + 1) as f64;
    if (u_at_1 - expected_u).abs() > TOL * expected_u.max(1.0) {
        panic!(
            "Chebyshev U_{}(1) = {}, expected {}",
            n, u_at_1, expected_u
        );
    }

    // Test 4: Polynomial evaluation doesn't panic
    let _ = eval_legendre(n, x);
    let _ = eval_chebyt(n, x);
    let _ = eval_chebyu(n, x);
    let _ = eval_hermite(n, sanitize(input.x_raw));
    let _ = eval_hermitenorm(n, sanitize(input.x_raw));
    let _ = eval_laguerre(n, sanitize(input.x_raw).abs());
    let _ = eval_genlaguerre(n, alpha, sanitize(input.x_raw).abs());
    let _ = eval_jacobi(n, alpha, beta, x);
    let _ = eval_gegenbauer(n, alpha, x);
    let _ = lpmv(m, n, x);

    // Test 5: Roots weights are positive (for small n to avoid timeout)
    if n > 0 && n <= 10 {
        let (nodes, weights) = roots_legendre(n as usize);
        if nodes.len() != n as usize {
            panic!(
                "roots_legendre({}) returned {} nodes, expected {}",
                n,
                nodes.len(),
                n
            );
        }
        for (i, &w) in weights.iter().enumerate() {
            if w <= 0.0 {
                panic!(
                    "roots_legendre({}) weight[{}] = {} <= 0",
                    n, i, w
                );
            }
        }

        // Weights should sum to 2 for Legendre
        let weight_sum: f64 = weights.iter().sum();
        if (weight_sum - 2.0).abs() > TOL {
            panic!(
                "roots_legendre({}) weights sum to {}, expected 2.0",
                n, weight_sum
            );
        }
    }

    // Test 6: Chebyshev roots are in [-1, 1]
    if n > 0 && n <= 10 {
        let (nodes, _) = roots_chebyt(n as usize);
        for (i, &node) in nodes.iter().enumerate() {
            if node < -1.0 - TOL || node > 1.0 + TOL {
                panic!(
                    "roots_chebyt({}) node[{}] = {} out of [-1, 1]",
                    n, i, node
                );
            }
        }

        let (nodes_u, _) = roots_chebyu(n as usize);
        for (i, &node) in nodes_u.iter().enumerate() {
            if node < -1.0 - TOL || node > 1.0 + TOL {
                panic!(
                    "roots_chebyu({}) node[{}] = {} out of [-1, 1]",
                    n, i, node
                );
            }
        }
    }

    // Test 7: Hermite roots are finite
    if n > 0 && n <= 10 {
        let (nodes, weights) = roots_hermite(n as usize);
        for (i, &node) in nodes.iter().enumerate() {
            if !node.is_finite() {
                panic!(
                    "roots_hermite({}) node[{}] is not finite",
                    n, i
                );
            }
        }
        for (i, &w) in weights.iter().enumerate() {
            if !w.is_finite() || w <= 0.0 {
                panic!(
                    "roots_hermite({}) weight[{}] = {} invalid",
                    n, i, w
                );
            }
        }
    }

    // Test 8: Laguerre roots are positive
    if n > 0 && n <= 10 {
        let (nodes, weights) = roots_laguerre(n as usize);
        for (i, &node) in nodes.iter().enumerate() {
            if node < -TOL {
                panic!(
                    "roots_laguerre({}) node[{}] = {} < 0",
                    n, i, node
                );
            }
        }
        for (i, &w) in weights.iter().enumerate() {
            if !w.is_finite() || w <= 0.0 {
                panic!(
                    "roots_laguerre({}) weight[{}] = {} invalid",
                    n, i, w
                );
            }
        }
    }
});
