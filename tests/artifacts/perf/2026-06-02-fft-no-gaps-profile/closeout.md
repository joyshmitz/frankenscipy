# Closeout: frankenscipy-cj2is

## Target

- Bead: `frankenscipy-cj2is`
- Crate: `fsci-fft`
- Hot path: `fft2/512x512` via `apply_axis_transform`
- One shipped lever: hoist a per-axis radix-2 plan for Cooley-Tukey power-of-two axes.

## Baseline and Profile

- RCH broad profile before: `criterion_fft_broad_rch.txt`
  - `baseline_fft2/fft2/512x512`: 9.3870 ms median on `vmi1149989`
  - `baseline_rfft/rfft/262144`: 3.0660 ms median
  - `baseline_fft/fft/262144`: 5.7246 ms median
- Focused RCH baseline before: `baseline_fft2_512_rch.txt`
  - `baseline_fft2/fft2/512x512`: 8.2364 ms median on `vmi1149989`
- Local upstream reference: `scipy_fft_reference_local.txt`
  - SciPy `fft2/512x512`: 2.496414 ms median

## Result

- Focused RCH after: `after_fft2_512_axis_plan_rch.txt`
  - `baseline_fft2/fft2/512x512`: 6.9579 ms median on `vmi1149989`
  - Delta: 15.4% faster than the same-worker focused baseline.
- Broad RCH reprofile after: `reprofile_fft_broad_after_axis_plan_rch.txt`
  - `baseline_fft2/fft2/512x512`: 7.8781 ms median on `vmi1149989`
  - Remaining top rows include `baseline_fft/fft/262144` at 5.7284 ms and `baseline_rfft/rfft/262144` at 3.9773 ms.
- Score: Impact 3.0 x Confidence 4.0 / Effort 2.0 = 6.0.

## Isomorphism Proof

- Query/data ordering: unchanged, because `apply_axis_transform` iterates axes, outer blocks, offsets, and fiber indices in the same order.
- Tie-breaking/order: unchanged, because bit-reversal swaps are generated in the same ascending-`i` order as the old inline loop.
- Floating point: unchanged, because the same twiddle table values and the same butterfly operations are used in the same order.
- RNG: no RNG surface.
- Validation SHA: `79b17591371e8b8472ce6e9a89264628b960ae9fcde9f0e5c6b0c6de57bba5d8`
- Golden compare: `GOLDEN_FFT2_BEFORE_AFTER_AXIS_PLAN_CMP_EXIT:0`

## Validation

- `RCH_FSCI_FFT_CHECK_ALL_TARGETS_EXIT:0`
- `RCH_FSCI_FFT_CLIPPY_NO_DEPS_ALL_TARGETS_EXIT:0`
- `RCH_FSCI_FFT_TEST_LIB_EXIT:0`
- `CARGO_FMT_CHECK_FSCI_FFT_FINAL_EXIT:0`
- `UBS_FSCI_FFT_CHANGED_FINAL_EXIT:0`

## Notes

- `stride == 1` direct row transform was tested and rejected: it preserved output bits but regressed to 14.343 ms median.
- Global bit-reversal cache and twiddle-only hoist were tested and rejected as insufficient or regressive on their focused RCH runs.
- The residual upstream gap is not accepted as architectural. The next target from the reprofile is either large 1D complex FFT or large RFFT, depending on the next same-worker upstream comparison.
