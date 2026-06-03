# Pass 3 - Alien Primitive Selection: Halton 4D Batch

## Recommendation Card

Bead: `frankenscipy-rlcl4`  
Parent: `frankenscipy-8l8r1`  
Target: `fsci-stats` Halton 4D sampling, `qmc_sampling/halton_4d/4096`

Measured baseline:

- Command: `rch exec -- cargo bench -p fsci-stats --bench stats_bench --locked -- qmc_sampling/halton_4d/4096 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`
- Worker: `vmi1227854`
- Criterion interval: `[238.55 us, 243.66 us, 248.22 us]`
- Baseline median: `243.66 us`
- Golden before sha: `b8483708b0ed4f5becc8f7a11560adb063b7926c98fabf54fe8c95bfff4f2a44`
- Current hot shape: `HaltonSampler::sample` loops rows, iterates `self.primes`, and calls `radical_inverse(idx, prime)` for each coordinate.

Selected primitive:

- Name: constant-prime 4D Halton batch micro-kernel.
- Graveyard lineage: vectorized execution / batch-specialized data-plane primitive, with certified-rewrite discipline for a tiny numerical hot loop.
- Alien-artifact form: a monomorphized, safe-Rust, fixed-base radical-inverse kernel for bases `2, 3, 5, 7`, selected only for `self.primes == [2, 3, 5, 7]`.
- No-gap compliance: pure safe Rust, no C/BLAS/MKL/XLA, no unsafe, no external runtime dependency.

Score:

| Primitive | Impact | Confidence | Effort | Score |
| --- | ---: | ---: | ---: | ---: |
| Constant-prime 4D Halton batch micro-kernel | 3 | 4 | 2 | 6.0 |

Rationale:

- Impact `3`: likely 10-25% on the measured target by removing dynamic prime iteration in the 4D path and letting LLVM strength-reduce `%` and `/` by compile-time constants in the digit loops.
- Confidence `4`: the RCH Criterion target is exactly the Halton 4D batch path; the code shape is dominated by repeated small-base radical-inverse calls. Confidence is not `5` because `perf stat`/callgraph counters are unavailable under `perf_event_paranoid=4`.
- Effort `2`: small, isolated branch plus const-base helpers and focused proof tests. No API or data-layout migration required.

Implementation sketch:

1. Keep `HaltonSampler::sample` as the only public behavior surface.
2. Add a private fixed-dimension branch:
   - if `self.primes.as_slice() == [2, 3, 5, 7]`, dispatch to a private `sample_halton_4d_batch(start, n)` helper;
   - otherwise retain the current generic loop unchanged.
3. In the 4D helper, preserve row-major order:
   - for each local `idx`, push base-2, base-3, base-5, base-7 radical inverses in that exact coordinate order.
4. Use const-base helpers that preserve the current arithmetic order:
   - `inv_prime = 1.0_f64 / BASE as f64`;
   - `f = inv_prime`;
   - while `index > 0`: compute `digit = index % BASE`, then `result += digit as f64 * f`, then `index /= BASE`, then `f *= inv_prime`.
5. Preserve `next_index` by carrying a local `idx` with `idx = idx.saturating_add(1)` once per emitted row, then assigning it back after the loop.

Rejected variants for this pass:

- Bit-reversal base-2 shortcut: attractive, but it changes the floating-point construction path and needs a separate bit-for-bit proof across high-index cases.
- Incremental carry-state Halton updates: attractive, but it changes accumulation order and risks non-identical `f64` bits.
- Static lookup tables for 4096 rows: too benchmark-specific and would not preserve general `next_index` and `sample(n)` semantics.

## Isomorphism Proof Obligations

Required before implementation can be kept:

- Ordering preserved: output remains row-major, `out[i * 4 + 0..4] == bases [2, 3, 5, 7]`.
- Tie-breaking unchanged: N/A; no comparisons or ordering decisions are introduced.
- Floating-point preserved: each coordinate uses the same digit order and the same `result += digit as f64 * f; f *= inv_prime` recurrence as `radical_inverse`.
- RNG/global state preserved: N/A; Halton 4D is deterministic and does not touch RNG or global state.
- `next_index` preserved: one saturating increment per emitted row, including the `u64::MAX` repeat behavior that follows from saturating arithmetic.
- Error behavior preserved: constructor validation for dimensions `0` and `>32` is untouched; non-4D dimensions use the generic path.
- Golden output preserved: after implementation, RCH `perf_stats halton4-golden | sha256sum` must still produce `b8483708b0ed4f5becc8f7a11560adb063b7926c98fabf54fe8c95bfff4f2a44`.
- Focused tests: compare optimized 4D output bit-for-bit against a generic reference for starts including `0`, `1`, `4095`, `4096`, a large non-power-of-base index, and `u64::MAX - 2`.

## Rejection and Fallback Rule

Reject this primitive and keep the existing generic `HaltonSampler::sample` loop if any of the following happens in the implementation pass:

- the golden sha changes from `b8483708b0ed4f5becc8f7a11560adb063b7926c98fabf54fe8c95bfff4f2a44`;
- RCH Criterion median improvement for `qmc_sampling/halton_4d/4096` is less than `3%`, or the rescored `Impact x Confidence / Effort` drops below `2.0`;
- focused `fsci-stats` tests, `cargo fmt --check`, or crate-scoped clippy fail for touched files;
- any non-4D Halton behavior changes.

Fallback trigger: if this branch fails any gate above, abandon the fixed-4D specialization and re-profile the generic radical-inverse loop before selecting a riskier primitive. Do not switch to bit-reversal or incremental carry-state in the same change.
