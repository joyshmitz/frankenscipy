# frankenscipy-acoco jnjnp_zeros Gauntlet Evidence

- Agent: cod-a / MistyBirch
- Date: 2026-06-19
- Crate: `fsci-special`
- Lever under verification: `8680af1b` `jnjnp_zeros` bracket reuse.
- Artifact command:
  `env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot`
- Rust benchmark group: `acoco_gauntlet_jnjnp_zeros`
- SciPy oracle: Python 3.13.7, NumPy 2.4.3, SciPy 1.17.1,
  `scipy.special.jnjnp_zeros(nt)`.

Criterion point-estimate means:

| Workload | Rust current | Rust legacy duplicate route | SciPy original | Current vs SciPy | Current vs legacy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `jnjnp_zeros(nt=64)` | 80.728603 ms | 101.762454 ms | 0.493655 ms | 163.53x slower | 1.26x faster |
| `jnjnp_zeros(nt=128)` | 410.059973 ms | 544.006333 ms | 0.924456 ms | 443.57x slower | 1.33x faster |

Decision:

- KEEP the bracket-reuse optimization: the current route is materially faster
  than the benchmark-only recreation of the previous duplicate-bracketing path.
- Record the overall routine as a SciPy LOSS: Rust remains 163.53x to 443.57x
  slower than the original SciPy routine on these realistic zero-enumeration
  workloads.
- Do not retry the same duplicate-`jn_zeros` lever without a new profile.
  Future work should target the root-finding/enumeration algorithm itself.

Validation:

- PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo check -p fsci-special --benches`
- PASS: `rustfmt --edition 2024 --check crates/fsci-special/benches/special_bench.rs`
- PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-special jnyn_and_jnjnp_zeros_match_scipy -- --nocapture`
- PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-special derivative_bessel_zeros_match_scipy_reference_points -- --nocapture`
- BLOCKED: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo clippy -p fsci-special --benches -- -D warnings`
  stopped on existing dependency lints in `fsci-integrate/src/api.rs`,
  `fsci-integrate/src/rk.rs`, and `fsci-linalg/src/lib.rs` before this
  benchmark file was linted.
