# perf: anderson (Anderson-Darling, dist="norm") — dedup double-CDF + parallel CDF/log table

## Lever (ONE)
The serial `anderson` A² loop computes
`A² = -n - (1/n)·Σ_i (2i-1)·[ln F(Y_i) + ln(1 - F(Y_{n+1-i}))]`,
evaluating the erf-based standard-normal CDF **twice per i** — once at `Y_i` and
once at `Y_{n+1-i}` — so over the whole loop it evaluates the CDF at every
sorted point TWICE. Replace with a single per-point table
`table[k] = (ln F(Y_k), ln(1 - F(Y_k)))` (each clamped to `[1e-15, 1-1e-15]`),
filled once (in parallel across threads for n ≥ 2048), then sum **serially in
the exact original i order** reading `table[i].0` and `table[n-1-i].1`.

This (a) halves the CDF evaluations (2n → n) and (b) moves all transcendentals
(the CDF + both `ln`s) off the serial critical path into a parallel table fill.

## Isomorphism / parity proof — BIT-IDENTICAL
- Each table entry is computed with the identical expression the serial loop
  used (`cdf(z).clamp(1e-15, 1-1e-15)`, then `.ln()` / `(1-f).ln()`), written to
  its own slot in input order → each per-point value is bit-identical.
- The reduction reads the same two table values the serial loop computed inline
  for that `i` (`ln F(Y_i)` and `ln(1-F(Y_{n-1-i}))`), multiplies by the same
  `(2(i+1)-1)` coefficient, and accumulates in the **same left-to-right i
  order** → identical floating-point summation, no reassociation.
- Golden statistic bits (`%.17e`) are IDENTICAL between the serial baseline build
  and the table+parallel build for n ∈ {50, 2047, 2048, 8000} (straddling the
  2048 parallel threshold). See `golden_payload.txt`.
  sha256 = 27f6cb30639f7b54b50a051831a71df8bd6b1d1bb25bc54146ecb427e2d6db10
  (same sha for BOTH builds).

## Timing — rch remote, 64 cores, `--profile release-perf`, reps=10
Same machine, back-to-back (serial baseline measured by stashing the change).

| n         | baseline (serial) | table + parallel | speedup |
|-----------|-------------------|------------------|---------|
| 20000     | 6.168 ms          | 3.59 ms          | 1.72x   |
| 200000    | 61.502 ms         | 13.70 ms         | 4.49x   |
| 2000000   | 631.317 ms        | 112.67 ms        | 5.60x   |

Score ≥ 2.0 cleared at n ≥ 200000 (the regime the parallel path targets); the
n < 2048 path stays serial and unchanged.

Harness: `crates/fsci-stats/src/bin/perf_anderson.rs`
Run: `cargo run --profile release-perf -p fsci-stats --bin perf_anderson`

## Notes
- The earlier WIP in the tree used a *reduction* fan-out that reindexed the
  upper-tail term and summed in a different order (tolerance-parity only, and its
  inline comment falsely claimed bit-identity). This commit replaces it with the
  table approach, which keeps the exact original summation order and is therefore
  truly bit-identical.
- A first benchmark run landed on a contended worker (near-serial, ~1.3x = the
  dedup-only floor); the 64-core numbers above are the representative result and
  were confirmed across two runs.
