# frankenscipy-vd5m2 closeout

## Target

- Crate: `fsci-interpolate`
- Profile-backed hotspot: `rbf_rect/rect_eval_grid/32x32_to_64x64`
- Broad baseline: `21.747 ms 22.158 ms 22.589 ms` on `vmi1153651`
- Focused baseline: `21.328 ms 21.599 ms 21.867 ms` on `vmi1153651`

## One lever

`RectBivariateSpline::eval_grid` now evaluates the x-direction spline rows once
per `x` grid row, stores the intermediate y coefficients, and evaluates the
y-direction spline for each `y` query using shared scratch buffers. The shared
`BSpline` evaluation logic was factored into slice-based helpers so the grid path
can avoid constructing temporary `BSpline` objects for every point.

## Result

- Focused after: `178.99 us 183.06 us 187.92 us` on `vmi1149989`
- Broad reprofile after: `362.97 us 376.19 us 391.83 us` on `vmi1153651`
- Same-worker comparison: `21.599 ms -> 376.19 us` median, 57.42x faster
- Original broad comparison: `22.158 ms -> 376.19 us` median, 58.90x faster
- Post-change top hotspot: `scattered_2d/griddata_linear/576x1024`, median `14.429 ms`; filed as `frankenscipy-4czwo`

## Behavior proof

- Golden before sha256: `e93dbc630b58cea606f24f7bda912a92e4b81baf1bfd3da9120eb5d4b870e2c2`
- Golden after sha256: `e93dbc630b58cea606f24f7bda912a92e4b81baf1bfd3da9120eb5d4b870e2c2`
- Byte comparison: `GOLDEN_RECT_BEFORE_AFTER_FINAL_CMP_EXIT:0`
- Isomorphism:
  - Outer `xi` traversal order and inner `yi` traversal order are unchanged.
  - Output remains x-major with y-inner rows.
  - Non-finite `x` and `y` values still produce `NaN` in the same positions.
  - Bounds clamping is unchanged.
  - The same Cox-de Boor recurrence is used for x and y spline evaluation.
  - Floating-point combination order changes only by removing temporary object construction; golden output is bit-identical for the profiled grid.
  - No RNG is used by this routine.

## Validation

- `rch exec -- cargo check -p fsci-interpolate --all-targets --locked`: pass on `vmi1153651`
- `rch exec -- cargo clippy --no-deps -p fsci-interpolate --all-targets --locked -- -D warnings`: pass on `vmi1153651`
- `rch exec -- cargo test -p fsci-interpolate --lib --locked`: pass on `vmi1156319`, 120 passed, 0 failed
- `cargo fmt -p fsci-interpolate --check`: pass
- `ubs crates/fsci-interpolate/src/lib.rs crates/fsci-interpolate/src/bin/perf_interpolate.rs crates/fsci-interpolate/benches/interpolate_bench.rs`: pass, 0 critical
- Full clippy with workspace dependency lints failed on existing `fsci-sparse` lowercase SciPy alias warnings; filed as `frankenscipy-6946y`

## Score

- Impact: 5.0
- Confidence: 4.5
- Effort: 1.2
- Score: 18.75
