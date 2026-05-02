//! Metamorphic tests for `fsci-ndimage`.
//!
//! Filter shape preservation, morphology subset/superset relations,
//! label invariants, and identity transforms.
//!
//! Run with: `cargo test -p fsci-ndimage --test metamorphic_tests`

use fsci_ndimage::{
    BoundaryMode, NdArray, argmax, argmin, binary_closing, binary_dilation, binary_erosion,
    binary_fill_holes, binary_opening, black_tophat, convolve, correlate, cumsum_array,
    distance_transform_edt, equal_within, flatten, full, gaussian_filter, grey_dilation,
    grey_erosion, label, laplace, masked_select, maximum_filter, mean_labels, median_filter,
    minimum_filter, morphological_gradient, ones, percentile_filter, prewitt, reshape, rotate,
    shift, sobel, sum_labels, uniform_filter, variance_labels, where_cond, white_tophat, zoom,
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

// ─────────────────────────────────────────────────────────────────────
// MR17 — maximum_filter dominates the input pointwise.
// (For each pixel, the local max ≥ the pixel itself.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_maximum_filter_dominates_input() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let m = maximum_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, mx)) in img.data.iter().zip(&m.data).enumerate() {
        assert!(
            *mx + 1e-12 >= *x,
            "MR17 maximum_filter at i={i}: max = {mx} < input = {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — minimum_filter is dominated by input pointwise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimum_filter_dominated_by_input() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let m = minimum_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, mn)) in img.data.iter().zip(&m.data).enumerate() {
        assert!(
            *mn <= *x + 1e-12,
            "MR18 minimum_filter at i={i}: min = {mn} > input = {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — prewitt of a constant image is zero (no gradient).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_prewitt_constant_image_is_zero() {
    let img = arr_2d(8, 10, |_, _| 7.5);
    for axis in 0..2 {
        let g = prewitt(&img, axis, BoundaryMode::Reflect, 0.0).unwrap();
        for (i, &v) in g.data.iter().enumerate() {
            assert!(
                v.abs() < 1e-9,
                "MR19 prewitt(axis={axis}) at i={i}: {v} != 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — laplace of a linear image is zero (∂²/∂x² + ∂²/∂y² = 0 for
// f(x, y) = ax + by + c).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_laplace_linear_image_is_zero() {
    // f(i, j) = 2*i + 3*j + 1
    let img = arr_2d(8, 10, |i, j| 2.0 * i as f64 + 3.0 * j as f64 + 1.0);
    let lap = laplace(&img, BoundaryMode::Reflect, 0.0).unwrap();
    // Check the strict interior to avoid boundary mode artefacts.
    let cols = 10;
    for i in 1..7 {
        for j in 1..9 {
            let v = lap.data[i * cols + j];
            assert!(
                v.abs() < 1e-9,
                "MR20 laplace(linear)[{i}, {j}] = {v}, expected 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — uniform_filter preserves a constant image.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_uniform_filter_preserves_constant() {
    let img = arr_2d(6, 8, |_, _| 3.25);
    for &size in &[1usize, 3, 5] {
        let f = uniform_filter(&img, size, BoundaryMode::Reflect, 0.0).unwrap();
        for (i, &v) in f.data.iter().enumerate() {
            assert!(
                close(v, 3.25),
                "MR21 uniform_filter(size={size}) at i={i}: {v} != 3.25"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — distance_transform_edt is non-negative, and is exactly 0 on
// every background pixel (input == 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_distance_transform_edt_zero_on_background() {
    // Build a 7×7 image with a 3×3 foreground cluster near the centre.
    let img = arr_2d(7, 7, |i, j| {
        if (2..=4).contains(&i) && (2..=4).contains(&j) {
            1.0
        } else {
            0.0
        }
    });
    let dt = distance_transform_edt(&img, None).unwrap();
    let cols = 7;
    for i in 0..7 {
        for j in 0..7 {
            let d = dt.data[i * cols + j];
            assert!(d >= -1e-12, "MR22 EDT[{i}, {j}] = {d} < 0");
            if img.data[i * cols + j] == 0.0 {
                assert!(
                    d.abs() < 1e-12,
                    "MR22 EDT[{i}, {j}] = {d}, expected 0 on background"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — sum_labels of a constant 1 image with k disjoint regions
// returns the area of each region.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sum_labels_counts_pixels() {
    // 6×6 image, region 1 occupies top-left 2×2; region 2 occupies
    // bottom-right 3×3.
    let img = arr_2d(6, 6, |_, _| 1.0);
    let labels = arr_2d(6, 6, |i, j| {
        if i < 2 && j < 2 {
            1.0
        } else if i >= 3 && j >= 3 {
            2.0
        } else {
            0.0
        }
    });
    let sums = sum_labels(&img, &labels, 2).unwrap();
    assert_eq!(sums.len(), 2, "MR23 sum_labels length");
    assert!((sums[0] - 4.0).abs() < 1e-12, "MR23 region 1 area = {}", sums[0]);
    assert!((sums[1] - 9.0).abs() < 1e-12, "MR23 region 2 area = {}", sums[1]);
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — mean_labels of a constant value c yields c for every region.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_mean_labels_constant_input() {
    let c = 5.5_f64;
    let img = arr_2d(5, 5, |_, _| c);
    let labels = arr_2d(5, 5, |i, j| {
        if (i + j) % 2 == 0 { 1.0 } else { 2.0 }
    });
    let means = mean_labels(&img, &labels, 2).unwrap();
    for (k, &m) in means.iter().enumerate() {
        assert!(
            (m - c).abs() < 1e-12,
            "MR24 mean_labels region {k} = {m}, expected {c}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — variance_labels is non-negative for every region.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_variance_labels_nonneg() {
    let img = arr_2d(5, 5, |i, j| (i * 5 + j) as f64);
    let labels = arr_2d(5, 5, |i, j| if (i + j) % 3 == 0 { 1.0 } else { 2.0 });
    let v = variance_labels(&img, &labels, 2).unwrap();
    for (k, &val) in v.iter().enumerate() {
        assert!(
            val >= -1e-12,
            "MR25 variance_labels region {k} = {val} < 0"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — binary_fill_holes never removes foreground pixels: result ≥
// input pointwise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_binary_fill_holes_dominates_input() {
    // Donut shape: outer ring at radius ≤ 4, hole at radius ≤ 2.
    let img = arr_2d(9, 9, |i, j| {
        let di = i as i32 - 4;
        let dj = j as i32 - 4;
        let r2 = di * di + dj * dj;
        if r2 <= 16 && r2 > 4 { 1.0 } else { 0.0 }
    });
    let filled = binary_fill_holes(&img).unwrap();
    for (i, (x, y)) in img.data.iter().zip(&filled.data).enumerate() {
        assert!(
            *y + 1e-12 >= *x,
            "MR26 fill_holes lost foreground at i={i}: {y} < {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — grey_dilation dominates input pointwise (analogous to maximum
// filter for continuous values).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_grey_dilation_dominates_input() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64 - 5.0);
    let g = grey_dilation(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, y)) in img.data.iter().zip(&g.data).enumerate() {
        assert!(
            *y + 1e-12 >= *x,
            "MR27 grey_dilation at i={i}: {y} < input {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — grey_erosion is dominated by input pointwise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_grey_erosion_dominated_by_input() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64 - 5.0);
    let g = grey_erosion(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, y)) in img.data.iter().zip(&g.data).enumerate() {
        assert!(
            *y <= *x + 1e-12,
            "MR28 grey_erosion at i={i}: {y} > input {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — reshape then flatten preserves the data.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_reshape_flatten_roundtrip() {
    let img = arr_2d(4, 5, |i, j| (i * 5 + j) as f64);
    let reshaped = reshape(&img, vec![20]).unwrap();
    assert_eq!(reshaped.shape, vec![20], "MR29 reshape shape");
    let flat = flatten(&img);
    assert_eq!(flat.size(), 20, "MR29 flatten size");
    for (i, (a, b)) in flat.data.iter().zip(&reshaped.data).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "MR29 flatten/reshape data mismatch at {i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — argmax of a strictly increasing array points at the last
// element; argmin points at the first.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_argmax_argmin_increasing() {
    let img = arr_2d(3, 4, |i, j| (i * 4 + j) as f64);
    let n = img.size();
    assert_eq!(argmax(&img), n - 1, "MR30 argmax");
    assert_eq!(argmin(&img), 0, "MR30 argmin");
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — cumsum_array last entry equals the sum of all entries.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cumsum_array_final_equals_total() {
    let img = arr_2d(3, 5, |i, j| 0.5 * ((i * 5 + j) as f64));
    let cum = cumsum_array(&img);
    let total: f64 = img.data.iter().sum();
    assert!(
        (cum.data[img.size() - 1] - total).abs() < 1e-12,
        "MR31 cumsum last = {} vs total = {total}",
        cum.data[img.size() - 1]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — where_cond with all-true mask returns `a`.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_where_cond_all_true_returns_a() {
    let a = arr_2d(3, 4, |i, j| (i + j) as f64);
    let b = arr_2d(3, 4, |i, j| (i * j) as f64 - 5.0);
    let cond = ones(vec![3, 4]);
    let r = where_cond(&cond, &a, &b).unwrap();
    for (i, (x, y)) in r.data.iter().zip(&a.data).enumerate() {
        assert!(
            (x - y).abs() < 1e-12,
            "MR32 where_cond all-true at i={i}: {x} vs {y}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — equal_within(a, a, 0) returns all 1s (reflexivity).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_equal_within_reflexive() {
    let a = arr_2d(4, 4, |i, j| 0.1 * ((i * 4 + j) as f64));
    let r = equal_within(&a, &a, 0.0).unwrap();
    for (i, &v) in r.data.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-12,
            "MR33 equal_within(a, a)[{i}] = {v}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — where_cond with all-false mask (zeros) returns `b`.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_where_cond_all_false_returns_b() {
    let a = arr_2d(3, 4, |i, j| (i + j) as f64);
    let b = arr_2d(3, 4, |i, j| (i * j) as f64 - 5.0);
    let cond = arr_2d(3, 4, |_, _| 0.0);
    let r = where_cond(&cond, &a, &b).unwrap();
    for (i, (x, y)) in r.data.iter().zip(&b.data).enumerate() {
        assert!(
            (x - y).abs() < 1e-12,
            "MR34 where_cond all-false at i={i}: {x} vs {y}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — morphological_gradient is non-negative everywhere (= max − min
// over the structuring element).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_morphological_gradient_nonneg() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let g = morphological_gradient(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, &v) in g.data.iter().enumerate() {
        assert!(
            v >= -1e-12,
            "MR35 morphological_gradient[{i}] = {v} < 0"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — white_tophat is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_white_tophat_nonneg() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let w = white_tophat(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, &v) in w.data.iter().enumerate() {
        assert!(
            v >= -1e-9,
            "MR36 white_tophat[{i}] = {v} < 0"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — black_tophat is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_black_tophat_nonneg() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let b = black_tophat(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, &v) in b.data.iter().enumerate() {
        assert!(
            v >= -1e-9,
            "MR37 black_tophat[{i}] = {v} < 0"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — percentile_filter at p=50 equals median_filter.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_percentile_filter_p50_equals_median() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let p = percentile_filter(&img, 50.0, 3, BoundaryMode::Reflect, 0.0).unwrap();
    let m = median_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (a, b)) in p.data.iter().zip(&m.data).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "MR38 percentile_filter(p=50) vs median at i={i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — full(shape, c) returns an array of constant value c with the
// requested shape.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_full_constant_array() {
    for &c in &[0.0_f64, 1.5, -3.0, 100.0] {
        let arr = full(vec![3, 5], c);
        assert_eq!(arr.shape, vec![3, 5], "MR39 full shape");
        for (i, &v) in arr.data.iter().enumerate() {
            assert!(
                (v - c).abs() < 1e-15,
                "MR39 full({c})[{i}] = {v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — masked_select returns a vector of length sum(mask).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_masked_select_length_equals_mask_sum() {
    let img = arr_2d(4, 5, |i, j| (i * 5 + j) as f64);
    // Mask the diagonal (4 entries).
    let mask = arr_2d(4, 5, |i, j| if i == j && i < 4 { 1.0 } else { 0.0 });
    let v = masked_select(&img, &mask);
    let mask_sum: f64 = mask.data.iter().sum();
    assert_eq!(
        v.len(),
        mask_sum as usize,
        "MR40 masked_select length = {} vs mask_sum = {}",
        v.len(),
        mask_sum as usize
    );
}




