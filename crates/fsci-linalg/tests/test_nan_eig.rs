use fsci_linalg::{eig, DecompOptions};
use fsci_runtime::RuntimeMode;

#[test]
fn test_nan_eig_hang_or_panic() {
    let opts = DecompOptions {
        mode: RuntimeMode::Strict,
        check_finite: false,
    };
    let matrix = vec![vec![1.0, 2.0], vec![3.0, f64::NAN]];
    println!("Calling eig...");
    let res = eig(&matrix, opts);
    println!("eig done: {:?}", res);
}