# w2pp0 compact-WY panel-generation rejection

Bead: `frankenscipy-w2pp0`
Agent: `RubyWaterfall`
Worktree: `/data/projects/.scratch/frankenscipy-rubywaterfall-psn7x-20260615-1850`
Base commit: `7284b661`
Date: 2026-06-15

## Profile-backed target

This was a follow-up to the compact-WY/two-stage native `eigh` route. Existing
private replay evidence showed compact-WY full-to-band replay could beat scalar
full-to-band replay, but production integration still needed a true in-panel
reflector generator: each panel reflector must see the matrix state after prior
panel reflectors, without falling back to full scalar two-sided replay.

The candidate intentionally targeted only the full-to-band generator, not public
`eigh` routing.

## Baseline

`baseline_compact_wy_full_to_band_replay_rch.txt` was run on RCH worker
`vmi1227854` before the generation source probe:

| shape | scalar replay | compact-WY replay | replay speedup |
| ---: | ---: | ---: | ---: |
| 256x256 | 9.887517 ms | 8.132467 ms | 1.215808x |
| 512x512 | 110.193609 ms | 74.289096 ms | 1.483308x |

That baseline proves the already-existing replay primitive remains promising,
but it does not prove the new generator.

## Candidate

Temporary source probe:

- generate a panel of Householder reflectors using cross-block scalar updates,
- delay the compact-WY symmetric trailing update until the panel is complete,
- zero eliminated full-to-band entries during panel construction,
- mirror the lower cross-block into the upper cross-block after each panel,
- keep public `eigh` dispatch unchanged.

The audit verdict was that this bundled generator work with a replay-order
change. The source probe was restored to zero `crates/fsci-linalg/src/lib.rs`
diff after the proof failed.

## Proof and benchmark gate

`proof_compact_wy_generation_matches_scalar_rch_retry.txt`:

- RCH worker `vmi1227854`.
- Command: `cargo test -j 1 -p fsci-linalg --lib compact_wy_full_to_band_generation_matches_scalar_generation --release --locked -- --nocapture --test-threads=1`.
- Result: failed before benchmark scoring.
- Failure: reflector 4 `tau` differed by bits from the scalar generator:
  `4637095629383278180` vs `4637095629383279111`.

`after_compact_wy_generation_perf_rch.txt`:

- RCH worker `ovh-a`.
- Command: ignored release perf probe for compact-WY full-to-band generation.
- Result: failed at n=256 with `compact-WY full-to-band generation drift 3.29738995146337199e0`.

Score: `Impact 0.0 * Confidence 4.0 / Effort 2.0 = 0.0`.

Verdict: REJECT. The generator does not preserve the scalar reflector sequence
or reduced matrix.

## Isomorphism proof summary

- Ordering preserved: public `eigh` was not routed through the candidate.
- Tie-breaking unchanged: no public sorting/tie code shipped.
- Floating-point: not preserved inside the candidate; reflector `tau` diverged
  and full-to-band reduction drifted.
- RNG: unchanged; deterministic fixtures only.
- Golden outputs: not promoted because the private generator failed first.

## Next route

Do not retry delayed panel generation with stale cross-block state, replay-order
changes, row-block Givens replay, row-major Givens replay, or slice/index
spelling variants.

The next admissible primitive must replace the tridiagonal eigensolver burden
directly:

- divide-and-conquer symmetric tridiagonal eigensolver,
- MRRR-style relatively robust representations,
- or a structurally different band-to-tridiagonal plus eigenvector backtransform
  route with a fresh scalar-oracle proof.
