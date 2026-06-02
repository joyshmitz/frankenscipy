# fsci-special erf/erfc real scalar perf closeout

Bead: frankenscipy-perf-special-erf-real-ndwa1

Fresh profile target:
- RCH worker vmi1156319, artifact `../2026-06-02-special-fresh-profile/baseline_special_bench_rch.txt`.
- `special_erf/erf/-3`: 712.09 ns mean.
- `special_erf/erf/3`: 730.80 ns mean.
- `special_erfc/erfc/-3`: 723.03 ns mean.
- `special_erfc/erfc/3`: 718.27 ns mean.

Lever:
- Keep finite real `erf_scalar` and `erfc_scalar` on a real-valued path instead of constructing `Complex64` and discarding `.re`.
- Preserve the old complex-real division shape with `complex_real_div(lhs, rhs) = (lhs * rhs) / (rhs * rhs)` so real-axis asymptotic values keep the previous bit pattern.
- No RNG, allocation ordering, or tie-breaking source changed.

Focused Criterion results:

| row | baseline | after | delta |
| --- | ---: | ---: | ---: |
| `special_erf/erf/-3` | 461.28 ns | 265.35 ns | 42.5% faster |
| `special_erf/erf/3` | 482.65 ns | 283.11 ns | 41.3% faster |
| `special_erfc/erfc/-3` | 465.63 ns | 269.30 ns | 42.2% faster |
| `special_erfc/erfc/3` | 432.00 ns | 274.95 ns | 36.4% faster |

Artifacts:
- Baseline: `baseline_special_er_family_rch.txt` on RCH worker vmi1149989.
- After: `after_special_er_family_rch.txt` on RCH worker vmi1153651.

Isomorphism proof:
- Ordering and tie-breaking: same NaN and infinity returns, same negative-argument complement rules, same series/asymptotic split, same monotone-term and epsilon break predicates.
- Floating point: finite real series uses the same recurrence. The asymptotic path preserves the former `Complex64` real-axis division evaluation order via `complex_real_div`.
- RNG: not used.
- Golden output: `golden_before.txt` and `golden_after.txt` match byte-for-byte; sha256 `e094a3f0cec2d4754da4f37f776cee6af607dd8e1a4e2a482a538fad826ab7f1`.

Validation:
- `cargo fmt -p fsci-special --check`: `cargo_fmt_check_fsci_special_final4.txt`, exit 0.
- `rch exec -- cargo test -p fsci-special --lib --locked`: `cargo_test_fsci_special_lib_final2_rch.txt`, 937 passed, exit 0.
- `rch exec -- cargo clippy -p fsci-special --all-targets --locked -- -D warnings`: `cargo_clippy_fsci_special_all_targets_final3_rch.txt`, exit 0.
- `ubs crates/fsci-special/src/error.rs`: `ubs_error_rs.txt`, exit 0; existing warnings only, no critical issues.

Score:
- Impact 5, confidence 5, effort 3 => 8.3. Keep.
