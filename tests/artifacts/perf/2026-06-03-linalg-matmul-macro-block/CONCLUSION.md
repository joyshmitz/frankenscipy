# frankenscipy-8l8r1.20 conclusion

## Verdict

Kept. The one production lever adds a 64-row macro block around the large
flat-workspace GEMM traversal.

## Baseline and after

Same-worker focused RCH Criterion on `vmi1156319`:

| row | before median | after median | ratio |
| --- | ---: | ---: | ---: |
| `matmul/512x512` | `100.62 ms` | `103.54 ms` | `0.97x` |
| `matmul/768x768` | `713.35 ms` | `537.60 ms` | `1.33x` |
| `matmul/1024x1024` | `1.1256 s` | `982.93 ms` | `1.15x` |

The optimized path remains gated at `m >= 1024`, `k >= 1024`, and `n >= 1024`,
so `1024x1024` is the production keep row. The `512` and `768` rows are included
as bench-context rows; they do not enter the production gate.

Score: `2.25 = impact 1.5 * confidence 3 / effort 2`.

## Isomorphism proof

- Ordering preserved: yes. Public output remains row-major `Vec<Vec<f64>>`.
- Tie-breaking unchanged: yes. GEMM has no tie-breaking surface.
- Floating-point preserved: yes. Each output cell still accumulates `k = 0..ka`
  monotonically with separate `acc += a * b` updates.
- RNG preserved: N/A. No RNG surface exists.
- Golden tests: before and after sorted test-line SHA-256 both
  `61e12eb58f34ccba1dcedd29425ff3292fd7df5769f7411352cd2a617a58d6c7`.

## Gates

- `cargo fmt -p fsci-linalg --check`: pass.
- `ubs crates/fsci-linalg/src/lib.rs`: critical `0`.
- RCH `cargo test -p fsci-linalg --release --locked matmul -- --nocapture`:
  pass.
- RCH `cargo check -p fsci-linalg --all-targets --locked`: pass on
  `vmi1149989`.
- RCH `cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`:
  pass on `vmi1153651`.

## Reprofile

RCH linalg reprofile on `vmi1153651` still ranks `matmul/1024x1024` first at
median `899.37 ms`, followed by `matmul/768x768` at `638.31 ms`,
`baseline_solve/1000x1000` at `343.72 ms`, `lstsq/512x256` at `157.74 ms`,
and `pinv/512x256` at `120.06 ms`.

Next target: a deeper GEMM primitive again, likely a recursive/cache-oblivious
or packed-A/B panel algorithm with stricter same-worker paired measurement.
