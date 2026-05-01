//! Metamorphic tests for `fsci-ndimage`.
//!
//! Filter shape preservation, morphology subset/superset relations,
//! label invariants, and identity transforms.
//!
//! Run with: `cargo test -p fsci-ndimage --test metamorphic_tests`

use fsci_ndimage::{
    BoundaryMode, NdArray, binary_closing, binary_dilation, binary_erosion, binary_opening,
    convolve, correlate, gaussian_filter, label, median_filter, rotate, shift, sobel, zoom,
};

fn arr_2d(rows: usize, cols: usize, fill: impl Fn(usize, usize) -> f64) -> NdArray {
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(fill(i, j));
        }
    }
    NdArray::new(data, vec![rows, cols]).unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 + 1e-10 * a.abs().max(b.abs()).max(1.0)
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — gaussian_filter preserves shape and finiteness for any sigma.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gaussian_filter_shape_and_finite() {
    let img = arr_2d(10, 12, |i, j| ((i + j) as f64).sin());
    for &sigma in &[0.5_f64, 1.0, 2.0, 5.0] {
        let out = gaussian_filter(&img, sigma, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(
            out.shape, img.shape,
            "MR1 shape changed for sigma={sigma}"
        );
        assert_eq!(out.data.len(), img.data.len());
        for (i, &v) in out.data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "MR1 gaussian non-finite at i={i} sigma={sigma}: {v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — gaussian_filter on a constant image returns the same constant
// (up to round-off).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gaussian_filter_preserves_constant() {
    let img = arr_2d(8, 8, |_, _| 3.7);
    let out = gaussian_filter(&img, 1.5, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, &v) in out.data.iter().enumerate() {
        assert!(
            close(v, 3.7),
            "MR2 gaussian on constant: out[{i}]={v}, expected 3.7"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — sobel of a constant image is zero (derivative of constant).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sobel_constant_image_is_zero() {
    let img = arr_2d(8, 8, |_, _| 5.0);
    for axis in 0..2 {
        let out = sobel(&img, axis, BoundaryMode::Reflect, 0.0).unwrap();
        for (i, &v) in out.data.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "MR3 sobel(axis={axis}) on constant: out[{i}]={v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — binary_erosion ⊆ input ⊆ binary_dilation
//
// For every pixel: erosion[p] <= input[p] and input[p] <= dilation[p].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_morphology_subset_superset() {
    // A small disk in the middle of a 7x7 grid.
    let img = arr_2d(7, 7, |i, j| {
        let di = i as i32 - 3;
        let dj = j as i32 - 3;
        if di * di + dj * dj <= 4 { 1.0 } else { 0.0 }
    });
    let eroded = binary_erosion(&img, 3, 1).unwrap();
    let dilated = binary_dilation(&img, 3, 1).unwrap();
    for (i, &input) in img.data.iter().enumerate() {
        let e = eroded.data[i];
        let d = dilated.data[i];
        // Each value is binary 0 or 1, but compare numerically to be safe.
        assert!(
            e <= input + 1e-12,
            "MR4 erosion superset of input at i={i}: e={e} > input={input}"
        );
        assert!(
            input <= d + 1e-12,
            "MR4 dilation subset of input at i={i}: input={input} > d={d}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — `label` produces labels in {0, 1, ..., num_features} where 0 is
// background and every k in 1..=num_features appears at least once.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_label_consecutive_features() {
    // Three disjoint blobs.
    let img = arr_2d(8, 8, |i, j| {
        let blob1 = i < 2 && j < 2;
        let blob2 = i >= 5 && j < 2;
        let blob3 = i < 3 && j >= 5;
        if blob1 || blob2 || blob3 { 1.0 } else { 0.0 }
    });
    let (labeled, num_features) = label(&img).unwrap();
    assert_eq!(num_features, 3, "expected 3 connected components");
    let mut seen = vec![false; num_features + 1];
    for &v in &labeled.data {
        let k = v as usize;
        assert!(
            (v - k as f64).abs() < 1e-12 && k <= num_features,
            "MR5 non-integer or out-of-range label: {v}"
        );
        seen[k] = true;
    }
    for (k, &was_seen) in seen.iter().enumerate() {
        assert!(was_seen, "MR5 label {k} missing from labeled image");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — shift by zeros is identity (up to f64 round-off).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_shift_by_zero_is_identity() {
    let img = arr_2d(6, 6, |i, j| (i * 7 + j * 11) as f64);
    for order in [0usize, 1, 3] {
        let out = shift(&img, &[0.0, 0.0], order, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(out.shape, img.shape);
        for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
            assert!(
                close(*a, *b),
                "MR6 shift(0) at i={i} order={order}: got={a} expected={b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — median_filter is idempotent on a uniform image.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_median_filter_idempotent_on_uniform() {
    let img = arr_2d(10, 10, |_, _| 2.0);
    let once = median_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    let twice = median_filter(&once, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (a, b)) in once.data.iter().zip(&twice.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR7 median_filter not idempotent on uniform at i={i}: {a} → {b}"
        );
    }
    for &v in &once.data {
        assert!(close(v, 2.0), "MR7 median_filter changed uniform value");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — gaussian_filter is non-expansive on min/max:
// min(input) <= min(output) and max(output) <= max(input) + small slack
// (a Gaussian filter is a convex combination of inputs).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gaussian_non_expansive_extrema() {
    let img = arr_2d(9, 9, |i, j| {
        let d = (i as i32 - 4).pow(2) + (j as i32 - 4).pow(2);
        (-(d as f64) * 0.1).exp() * 5.0
    });
    let lo = img.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = img.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let out = gaussian_filter(&img, 1.0, BoundaryMode::Reflect, 0.0).unwrap();
    let lo_out = out.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi_out = out.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let slack = 1e-9 * hi.abs().max(1.0);
    assert!(
        lo_out >= lo - slack,
        "MR8 gaussian dropped below input min: {lo_out} < {lo}"
    );
    assert!(
        hi_out <= hi + slack,
        "MR8 gaussian exceeded input max: {hi_out} > {hi}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — rotate by 0 degrees with reshape=false is the identity.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotate_by_zero_is_identity() {
    let img = arr_2d(7, 7, |i, j| (i * 11 + j * 13) as f64);
    for &order in &[0_usize, 1, 3] {
        let out = rotate(&img, 0.0, false, order, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(out.shape, img.shape, "MR9 shape changed");
        for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
            assert!(
                close(*a, *b),
                "MR9 rotate(0, order={order}) at i={i}: got {a}, expected {b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — zoom by factor 1.0 is the identity.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_zoom_by_one_is_identity() {
    let img = arr_2d(6, 8, |i, j| (i + j) as f64);
    for &order in &[0_usize, 1, 3] {
        let out = zoom(&img, &[1.0, 1.0], order, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(out.shape, img.shape, "MR10 shape changed");
        for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
            assert!(
                close(*a, *b),
                "MR10 zoom(1, order={order}) at i={i}: got {a}, expected {b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — rotate by 360° (with reshape=false) returns the original image
// to f64-eps tolerance. Previously failed because sub-ULP boundary
// excursions in the half-pixel reflect-coordinate map were leaking
// into the B-spline sampler as out-of-domain (frankenscipy-c7ud);
// fixed by clamping near-boundary excursions back into the spline
// support before evaluating the basis.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotate_by_360_returns_to_original() {
    let img = arr_2d(11, 11, |i, j| {
        let di = i as f64 - 5.0;
        let dj = j as f64 - 5.0;
        (-(di * di + dj * dj) * 0.05).exp()
    });
    let out = rotate(&img, 360.0, false, 1, BoundaryMode::Reflect, 0.0).unwrap();
    assert_eq!(out.shape, img.shape, "MR11 shape changed");
    for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "MR11 rotate(360) drift at i={i}: got {a}, expected {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — convolve and correlate agree when the kernel is symmetric
// (kernel[i, j] = kernel[K-1-i, K-1-j] under flip in every axis).
// Because convolution flips the kernel and correlation does not,
// using a symmetric kernel makes the two operators identical.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_correlate_agree_on_symmetric_kernel() {
    let img = arr_2d(8, 8, |i, j| ((i + j) as f64).sin());
    // 3x3 symmetric averaging kernel.
    let weights = NdArray::new(vec![1.0_f64 / 9.0; 9], vec![3, 3]).unwrap();
    let conv = convolve(&img, &weights, BoundaryMode::Reflect, 0.0).unwrap();
    let corr = correlate(&img, &weights, BoundaryMode::Reflect, 0.0).unwrap();
    assert_eq!(conv.shape, corr.shape, "MR12 shape mismatch");
    for (i, (c, k)) in conv.data.iter().zip(&corr.data).enumerate() {
        assert!(
            close(*c, *k),
            "MR12 conv vs corr at i={i}: {c} vs {k}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — convolve with the 1-element identity kernel returns the
// original image unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_with_identity_kernel_is_input() {
    let img = arr_2d(7, 9, |i, j| (i * 13 + j * 7) as f64);
    let identity = NdArray::new(vec![1.0_f64], vec![1, 1]).unwrap();
    let out = convolve(&img, &identity, BoundaryMode::Reflect, 0.0).unwrap();
    assert_eq!(out.shape, img.shape, "MR13 shape changed");
    for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR13 convolve(image, [1]) at i={i}: got {a}, expected {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — binary_opening = binary_erosion then binary_dilation.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_binary_opening_equals_erode_then_dilate() {
    // Disk-ish blob in a 9x9 grid.
    let img = arr_2d(9, 9, |i, j| {
        let di = i as i32 - 4;
        let dj = j as i32 - 4;
        if di * di + dj * dj <= 6 { 1.0 } else { 0.0 }
    });
    let opened = binary_opening(&img, 3, 1).unwrap();
    let manual = binary_dilation(&binary_erosion(&img, 3, 1).unwrap(), 3, 1).unwrap();
    for (i, (a, b)) in opened.data.iter().zip(&manual.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR14 opening != erode∘dilate at i={i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — binary_closing = binary_dilation then binary_erosion.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_binary_closing_equals_dilate_then_erode() {
    let img = arr_2d(9, 9, |i, j| {
        let di = i as i32 - 4;
        let dj = j as i32 - 4;
        if di * di + dj * dj <= 6 { 1.0 } else { 0.0 }
    });
    let closed = binary_closing(&img, 3, 1).unwrap();
    let manual = binary_erosion(&binary_dilation(&img, 3, 1).unwrap(), 3, 1).unwrap();
    for (i, (a, b)) in closed.data.iter().zip(&manual.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR15 closing != dilate∘erode at i={i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — binary_opening is idempotent: opening(opening(X)) = opening(X).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_binary_opening_idempotent() {
    let img = arr_2d(9, 9, |i, j| {
        let di = i as i32 - 4;
        let dj = j as i32 - 4;
        if di * di + dj * dj <= 6 { 1.0 } else { 0.0 }
    });
    let once = binary_opening(&img, 3, 1).unwrap();
    let twice = binary_opening(&once, 3, 1).unwrap();
    for (i, (a, b)) in once.data.iter().zip(&twice.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR16 opening not idempotent at i={i}: {a} vs {b}"
        );
    }
}
