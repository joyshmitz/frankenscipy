# Optimization Protocol

One-page checklist for performance optimization PRs in FrankenSciPy.

## Pre-Optimization Checklist

1. **Profile first.** Run `cargo bench` to identify the hot path. Do not optimize speculatively.
2. **Capture baseline.** `cargo bench -- --save-baseline before`
3. **Verify conformance.** `cargo test --workspace --lib` must pass before any change.

## During Optimization

4. **Single concern.** Each optimization PR addresses one bottleneck.
5. **Measure continuously.** Re-run `cargo bench -- --baseline before` after each change to track progress.
6. **Preserve behavior.** Run the full differential + metamorphic test suite to verify behavior isomorphism:
   ```bash
   cargo test -p fsci-runtime --test differential_metamorphic
   cargo test -p fsci-runtime --test casp_comprehensive
   ```

## Post-Optimization Checklist

7. **Capture after baseline.** `cargo bench -- --save-baseline after`
8. **Compare.** `cargo bench -- --baseline before` — verify improvement, flag any >5% p95 regressions in unrelated benchmarks.
9. **Update baselines.** If accepted, update `fixtures/artifacts/baselines/baseline_initial.json` with new numbers.
10. **Conformance diff.** Full test suite must still pass:
    ```bash
    cargo test --workspace --lib
    cargo clippy --workspace --lib
    ```

## Regression Detection

A benchmark regression is flagged when p50 latency increases by more than 5% compared to the stored baseline. Compare using:

```bash
cargo bench -- --output-format bencher 2>&1 | grep "^test "
```

Then compare against `fixtures/artifacts/baselines/baseline_initial.json`.

## Benchmark Inventory

| Crate | Benchmark | Function |
|-------|-----------|----------|
| fsci-runtime | `runtime_bench` | policy_decide, solver_select, calibrator_observe |
| fsci-integrate | `integrate_bench` | solve_ivp (exponential, Lorenz), validate_tol |
| fsci-linalg | `linalg_bench` | solve, inv (4x4, 16x16, 64x64) |
