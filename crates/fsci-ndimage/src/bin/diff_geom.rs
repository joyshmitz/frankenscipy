//! ndimage geometric-transform probe vs scipy.ndimage (gitignored).
use fsci_ndimage::{BoundaryMode, NdArray, affine_transform, map_coordinates, shift, zoom};
fn emit(name: &str, v: &[f64]) {
    println!(
        "{name},{}",
        v.iter()
            .map(|x| format!("{x:.14e}"))
            .collect::<Vec<_>>()
            .join(";")
    );
}
fn modes() -> Vec<(&'static str, BoundaryMode)> {
    vec![
        ("reflect", BoundaryMode::Reflect),
        ("mirror", BoundaryMode::Mirror),
        ("nearest", BoundaryMode::Nearest),
        ("grid-wrap", BoundaryMode::Wrap),
        ("constant", BoundaryMode::Constant),
    ]
}
fn main() {
    // 5x5 deterministic image
    let data: Vec<f64> = (0..25)
        .map(|i| (i as f64 * 0.37).sin() + 0.1 * i as f64)
        .collect();
    let img = NdArray::new(data.clone(), vec![5, 5]).unwrap();
    for order in [0usize, 1, 3] {
        for (mn, m) in modes() {
            let s = shift(&img, &[0.7, -1.3], order, m, 0.0).unwrap();
            emit(&format!("shift_o{order}_{mn}"), &s.data);
        }
    }
    for order in [1usize, 3] {
        let z = zoom(&img, &[1.6, 1.6], order, BoundaryMode::Reflect, 0.0).unwrap();
        emit(&format!("zoom_o{order}"), &z.data);
    }
    // map_coordinates at scattered points
    let coords = vec![vec![0.3, 1.7, 2.5, 3.9], vec![0.5, 2.2, 1.1, 4.0]];
    for order in [0usize, 1, 3] {
        let mc = map_coordinates(&img, &coords, order, BoundaryMode::Reflect, 0.0).unwrap();
        emit(&format!("mapcoord_o{order}"), &mc);
    }
    // affine: rotate-ish + scale matrix [[a,b,tx],[c,d,ty]]
    let mat = [[0.9, 0.2, 0.5], [-0.15, 1.1, -0.3]];
    for order in [1usize, 3] {
        let a = affine_transform(&img, &mat, order, BoundaryMode::Reflect, 0.0).unwrap();
        emit(&format!("affine_o{order}"), &a.data);
    }
}
