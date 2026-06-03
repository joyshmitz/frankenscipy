# Conclusion

Bead: `frankenscipy-8l8r1.14`

Verdict: KEPT.

One lever kept:
- `solve` and `solve_with_audit` now use solve-internal condition diagnostics that skip the non-observable Cholesky positive-definite probe.
- Public `condition_diagnostics` still runs the full SPD probe and preserves its observable report.

Five-pass optimization loop:
1. Profile and baseline: RCH Criterion `baseline_solve/1000x1000` baseline on `vmi1156319` was `[545.06 ms, 555.58 ms, 566.84 ms]`.
2. Golden and isomorphism: RCH `perf_solve golden` normalized sha256 before edit was `5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd`; remote SciPy reference solve tests passed before edit.
3. Primitive selection: used the alien-graveyard profile-contract rule to remove a non-observable O(n^3) solve diagnostic probe, with public diagnostic behavior preserved.
4. Implementation: one source lever in `crates/fsci-linalg/src/lib.rs`; clippy-only test cleanup changed `x_true` from `Vec<Vec<f64>>` to an array of rows.
5. Gate: after RCH Criterion on `vmi1149989` was `[98.835 ms, 100.42 ms, 102.02 ms]`, a 5.53x median speedup versus the fresh RCH baseline. Score: `12.0 = impact 4 * confidence 3 / effort 1`, keep threshold `>= 2.0`.

Behavior proof:
- Normalized golden sha256 before and after edit matched: `5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd`.
- RCH SciPy reference tests after edit passed: `solve_matches_scipy_reference_values`, `lu_solve_matches_scipy_reference_values`, and `cho_solve_matches_scipy_reference_values`.
- Ordering, tie-breaking, RNG, and floating-point solve arithmetic are unchanged. The skipped work was a diagnostic-only SPD probe; public diagnostics still evaluates SPD status.

Validation:
- `cargo fmt -p fsci-linalg --check`: passed.
- `rch exec -- cargo check -p fsci-linalg --all-targets --locked`: passed.
- `rch exec -- cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`: passed after the test-only `useless_vec` cleanup.

Artifacts:
- `baseline_criterion_baseline_solve_1000_rch.txt`
- `golden_before_perf_solve_rch.txt`
- `golden_before_perf_solve_normalized.txt`
- `golden_before_perf_solve_normalized.sha256`
- `golden_before_scipy_reference_tests_rch_remote.txt`
- `golden_after_perf_solve_rch.txt`
- `golden_after_perf_solve_normalized.txt`
- `golden_compare_before_after.sha256`
- `golden_after_scipy_reference_tests_rch.txt`
- `after_criterion_baseline_solve_1000_rch.txt`
- `cargo_fmt_fsci_linalg_check_final.txt`
- `cargo_check_fsci_linalg_all_targets_rch.txt`
- `cargo_clippy_fsci_linalg_all_targets_rch_retry.txt`
