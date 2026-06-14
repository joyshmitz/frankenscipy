# frankenscipy-7lbyj Evidence

## Target
- Bead: `frankenscipy-7lbyj`
- Area: `fsci-opt` BFGS
- Profile-backed target: BFGS Rosenbrock still spent the live route in finite-difference gradient evaluation after the CG finite-diff keeps. BronzeDove's current RCH artifact reported `bfgs/rosenbrock/10` p50 `72.782 us` on `vmi1227854`; fresh pre-edit RubyWaterfall baseline on `vmi1167313` measured `[197.94 us, 207.41 us, 216.32 us]` for the same row.

## Lever
BFGS now honors `MinimizeOptions::gradient` by routing both initial and next-step gradient evaluations through the existing `evaluate_minimize_gradient` helper. The `gradient: None` path remains the same finite-difference implementation through that helper, and the line-search, BFGS update, convergence, callback, and result-shaping logic are unchanged.

## Behavior Proof
- Default minimize golden payload before SHA-256: `f02b24201c2844e1cb1577159ebb29535e2d16a8ccd3676279670e8b6fffad27`
- Default minimize golden payload after SHA-256: `f02b24201c2844e1cb1577159ebb29535e2d16a8ccd3676279670e8b6fffad27`
- Ordering and tie-breaking: no changes to BFGS step acceptance, Armijo line search, Hessian update, callback order, or convergence checks.
- Floating point contract: the default `gradient: None` path still calls central finite-difference gradient evaluation with the same epsilon and objective evaluation order. The exact-gradient path is opt-in and uses the caller-provided callback only where BFGS previously asked for a gradient.
- RNG contract: no RNG involved.
- Validation: exact-gradient callback shape and nonfinite output reuse the existing typed gradient validator and return `InvalidInput` / `NanEncountered`.

## Benchmarks
Same-run RCH Criterion comparison on `vmi1156319`, sample size 20, warm-up 1s, measurement 3s:

| Row | finite-diff p50 | exact-gradient p50 | ratio |
| --- | ---: | ---: | ---: |
| `bfgs/rosenbrock/2` | `45.417 us` | `27.880 us` | `1.63x` |
| `bfgs/rosenbrock/5` | `115.65 us` | `79.902 us` | `1.45x` |
| `bfgs/rosenbrock/10` | `239.25 us` | `200.77 us` | `1.19x` |

Score: `Impact 1.6 * Confidence 4.0 / Effort 2.0 = 3.2`, keep.

## Gates
- `rch exec -- cargo check -p fsci-opt --lib --locked`: pass on `vmi1149989`.
- `rch exec -- cargo clippy -p fsci-opt --lib --locked --no-deps -- -D warnings`: pass on `vmi1149989`.
- `rch exec -- cargo check -j 1 -p fsci-opt --all-targets --locked`: pass on `vmi1153651`.
- `rch exec -- cargo clippy -j 1 -p fsci-opt --all-targets --locked -- -D warnings`: pass on `vmi1227854`.
- `cargo fmt -p fsci-opt -- --check`: pass.
- `cargo fmt --check -p fsci-opt`: pass.
- `rch exec -- cargo test -p fsci-opt bfgs --lib --locked -- --nocapture`: pass on `vmi1149989`, `22 passed`.
- `ubs crates/fsci-opt/src/minimize.rs crates/fsci-opt/benches/optimize_bench.rs`: non-zero on pre-existing scanner findings in the changed whole files, including test-only `expect` / intentional poison panic and false-positive enum equality as secret comparison. UBS embedded fmt/clippy/check/test-build sections were clean.

## Artifacts
- `baseline_bfgs_rosenbrock10_rch.txt`
- `after_pair_bfgs_rosenbrock_rch.txt`
- `golden_minimize_before_payload_lf.txt`
- `golden_minimize_after_payload_lf.txt`
- `golden_minimize_before_rch_retry_vmi1167313.txt`
- `golden_minimize_after_rch.txt`
- `check_fsci_opt_lib_rch.txt`
- `clippy_fsci_opt_lib_rch.txt`
- `test_bfgs_all_rch.txt`
- `ubs_fsci_opt_bfgs_gradient.txt`
