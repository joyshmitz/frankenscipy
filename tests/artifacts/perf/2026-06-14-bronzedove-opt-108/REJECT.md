# frankenscipy-8l8r1.108 rejection

Verdict: REJECT, Score 0.0.

Target: post-`.107` CG derivative-interface primitive in `fsci-opt`.

Candidate tried: fused `value_gradient` callback for CG/Wolfe that evaluates objective value and gradient together and returns the accepted Wolfe gradient without a separate gradient callback allocation.

Fresh baseline evidence:
- Default CG `cg/rosenbrock/10`, RCH `vmi1152480`: `[359.61 us, 387.33 us, 424.15 us]`.
- Exact-gradient CG `cg/rosenbrock_exact_gradient/10`, RCH `vmi1152480`: `[263.22 us, 280.77 us, 300.69 us]`.

Candidate and comparator evidence:
- Exploratory fused `cg/rosenbrock_value_gradient/10`, RCH `vmi1264463`: `[647.66 us, 781.22 us, 998.20 us]`; not used for the keep gate because it is cross-worker.
- Same-worker exact-gradient comparator, RCH `vmi1152480`: `[159.14 us, 164.47 us, 169.27 us]`.
- Same-worker fused value-gradient candidate, RCH `vmi1152480`: `[248.32 us, 256.70 us, 268.99 us]`.

Behavior proof during probe:
- RCH focused tests passed while the candidate was applied: `cargo test -p fsci-opt --lib --locked value_gradient -- --nocapture`, 2 tests passed on `vmi1152480`.
- The test checked bit-identical final success/status/nit/nfev/objective/x/jac against the exact-gradient path and invalid callback shape rejection.

Restore proof:
- Candidate source edits were restored manually. `git diff --exit-code -- crates/fsci-opt/src/types.rs crates/fsci-opt/src/linesearch.rs crates/fsci-opt/src/minimize.rs crates/fsci-opt/benches/optimize_bench.rs` passed.
- Restored source SHA-256:
  - `crates/fsci-opt/src/types.rs`: `2f2aeeb0521b675ecd3d585c253acaf3a960cb3000b47f2156a7a10a1d1a010f`
  - `crates/fsci-opt/src/linesearch.rs`: `f438a9a3a0b698793e201bf4af95c64eab1a334c5ccc25c657bed1926ee3bd74`
  - `crates/fsci-opt/src/minimize.rs`: `1e1c6b83fe73aee66f6150de5f9ccbe45e0229632e10e8fcd7549484cbf69128`
  - `crates/fsci-opt/benches/optimize_bench.rs`: `175fbe59c24e1309b02c3e372f570370e3198017d7d28ddaccf8ad0b2bf37c01`

Rationale:
- The fused callback did not beat the existing exact-gradient route on the same worker; it was `256.70 us` median versus `164.47 us` median.
- Even relative to the initial exact-gradient baseline, the best observed fused median improvement was too small for the added API and line-search complexity.

Next route:
- Do not retry fused public callback spelling or line-search gradient-materialization micro-levers.
- Attack a deeper derivative primitive: internal dual-number or AD-tape objective artifact that computes Rosenbrock-family value and gradient through a reusable tape/arena with no public API expansion, or a CG specialization that changes the derivative representation rather than callback plumbing.
