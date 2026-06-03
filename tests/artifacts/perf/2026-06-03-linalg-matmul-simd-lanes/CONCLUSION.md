# frankenscipy-8l8r1.22 conclusion

## Verdict

Kept. The one production lever adds safe Rust portable SIMD lanes to the full
4x8 tile path in the large flat-workspace GEMM kernel.

## Baseline and after

Fresh RCH baseline on `vmi1156319`:

| row | baseline median |
| --- | ---: |
| `matmul/256x256` | `11.363 ms` |
| `matmul/512x512` | `94.399 ms` |
| `matmul/768x768` | `543.23 ms` |
| `matmul/1024x1024` | `817.34 ms` |

RCH after run on `vmi1293453`:

| row | after median | ratio |
| --- | ---: | ---: |
| `matmul/256x256` | `5.2061 ms` | `2.18x` |
| `matmul/512x512` | `39.680 ms` | `2.38x` |
| `matmul/768x768` | `132.01 ms` | `4.12x` |
| `matmul/1024x1024` | `219.74 ms` | `3.72x` |

Score: `7.0 = impact 4 * confidence 3.5 / effort 2`.

## Isomorphism proof

- Ordering preserved: yes. Public output remains row-major `Vec<Vec<f64>>`.
- Tie-breaking unchanged: yes. GEMM has no tie-breaking surface.
- Floating-point preserved: yes. Each output lane still accumulates
  `k = 0..ka` monotonically with explicit multiply followed by add.
- RNG preserved: N/A. No RNG surface exists.
- Golden tests: before and after sorted test-line SHA-256 both
  `61e12eb58f34ccba1dcedd29425ff3292fd7df5769f7411352cd2a617a58d6c7`.

## Gates

- `cargo fmt -p fsci-linalg --check`: pass.
- `ubs crates/fsci-linalg/src/lib.rs`: exit `0`, introduced critical `0`.
- RCH `cargo test -p fsci-linalg --release --locked matmul -- --nocapture`:
  pass.
- RCH `cargo check -p fsci-linalg --all-targets --locked`: pass on
  `vmi1227854`.
- RCH `cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`:
  pass on `vmi1153651`.

## Reprofile

RCH linalg reprofile on `vmi1227854` still ranks `matmul/1024x1024` first at
median `216.09 ms`, followed by `lstsq/512x256` at `141.68 ms`,
`matmul/768x768` at `131.13 ms`, `pinv/512x256` at `119.47 ms`, and
`baseline_solve/1000x1000` at `99.318 ms`.

Next target: another deeper GEMM primitive, not a ceiling call.
