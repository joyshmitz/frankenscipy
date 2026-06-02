# fsci-integrate RK45 Error Norm Closeout

Bead: `frankenscipy-28tzt`

## Profile-Backed Target

Fresh RCH Criterion profile before source edits:

- Artifact: `criterion_broad_rch.txt`
- Worker: `vmi1153651`
- `solve_ivp_lorenz_rk45`: `[59.867 us, 61.155 us, 62.594 us]`
- `solve_ivp_exponential_rk45`: `[44.430 us, 46.572 us, 49.197 us]`
- `validate_tol_vector_100`: `[11.983 us, 12.181 us, 12.408 us]`

Source evidence: each attempted RK step materialized `scale: Vec<f64>` and
`err: Vec<f64>` before computing the RMS error norm.

## One Lever

Kept one source lever in `crates/fsci-integrate/src/rk.rs`: compute the same
RMS error norm directly from solver state without allocating the intermediate
error and scale vectors.

## Behavior Proof

- Golden before sha256: `327b936597b6df9a5eb5d181a7f545a7c946458febea97652c73343781aa1eff`
- Golden after final sha256: `327b936597b6df9a5eb5d181a7f545a7c946458febea97652c73343781aa1eff`
- Byte comparison: `golden_before.txt` and `golden_after_final2.txt` matched with `cmp -s`.
- Ordering/tie-breaking: no public ordering or tie-breaking path changed; solver step acceptance and output vector ordering remain unchanged.
- Floating point: for each component, the new helper accumulates `e_s * k[s][i]` in the same ascending stage order, multiplies by `h`, computes the same `atol + max(abs(y), abs(y_new)) * rtol` scale, and preserves the old `(err / scale) * (err / scale)` expression and component-order sum. The denominator remains `self.n`.
- RNG: no RNG path exists in this RK45 step path.

## Benchmarks

Focused Lorenz baseline:

- Artifact: `baseline_lorenz_rk45_focused_rch.txt`
- Worker: `vmi1156319`
- `solve_ivp_lorenz_rk45`: `[53.224 us, 54.274 us, 55.385 us]`

Final focused Lorenz after:

- Artifact: `after_final2_lorenz_rk45_focused_rch.txt`
- Worker: `vmi1293453`
- `solve_ivp_lorenz_rk45`: `[26.946 us, 27.571 us, 28.123 us]`

Final broad re-profile:

- Artifact: `reprofile_after_final2_broad_rch.txt`
- Worker: `vmi1149989`
- `solve_ivp_exponential_rk45`: `[14.466 us, 14.871 us, 15.308 us]`
- `solve_ivp_lorenz_rk45`: `[26.118 us, 27.217 us, 28.284 us]`
- `validate_tol_vector_100`: `[5.5778 us, 5.8356 us, 6.0814 us]`
- `validate_tol_scalar`: `[314.39 ns, 324.22 ns, 334.96 ns]`

RCH worker selection was not pinnable, so worker IDs are recorded beside every
row. The source-relevant solve rows show a large win after removing per-attempt
allocations; unrelated tolerance rows also vary across workers and are not used
as source-impact evidence.

Score: `7.0 = impact 3.5 * confidence 3.0 / effort 1.5`. Kept because the
profile target is real, the source lever is small, exact golden output is
unchanged, and repeated final solve rows show a clear improvement despite RCH
worker variance.

## Validation

- RCH `cargo test -p fsci-integrate --lib --locked`: `189 passed`
- RCH `cargo clippy -p fsci-integrate --all-targets --locked -- -D warnings`: passed
- `cargo fmt -p fsci-integrate --check`: passed
- `ubs crates/fsci-integrate/src/rk.rs`: exit `0`, critical `0`
