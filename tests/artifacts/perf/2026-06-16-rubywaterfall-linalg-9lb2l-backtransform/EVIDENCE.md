# frankenscipy-9lb2l column-local backtransform replay rejection

Bead: `frankenscipy-9lb2l`
Agent: RubyWaterfall
Date: 2026-06-16

## Target

The production-aligned native symmetric `eigh` stage split made backtransform a profile-backed target after `frankenscipy-8ty4p` corrected the stale full-mirror stage probe. This pass tested exactly one reflector-replay locality lever.

## Baseline

RCH worker: `vmi1227854`

Production-aligned stage split:

- 400x400: reduction `15.578955 ms`, tridiagonal `10.183960 ms`, backtransform `6.744406 ms`, sort `0.589714 ms`, digest `0x0dbbde75b75c8612`
- 800x800: reduction `112.402899 ms`, tridiagonal `40.607628 ms`, backtransform `43.612981 ms`, sort `2.448268 ms`, digest `0x4461962827bdb038`
- 1200x1200: reduction `362.706503 ms`, tridiagonal `90.459377 ms`, backtransform `142.309442 ms`, sort `6.517907 ms`, digest `0x2fc45e1f18ceb0ab`

## Candidate

Temporary source lever: changed only `apply_left_reflectors_to_column_chunk` so each worker chunk replayed all reflectors for one column before moving to the next column. The original code replayed one reflector over all columns before moving to the next reflector.

Contract: for each column, reflector order stayed reverse chronological, dot-product offset order stayed unchanged, update offset order stayed unchanged, worker partitioning stayed unchanged, RNG/fallback/threshold behavior stayed unchanged, and no arithmetic formula changed.

## Proof

RCH `vmi1227854`:

- `symmetric_eigh_backtransform_parallel_matches_serial_bits` passed.

The existing proof compares serial reflector replay against `apply_left_reflectors_column_chunks` element-by-element using `f64::to_bits()`.

## Rebench

After RCH stage split on `vmi1227854`:

- 400x400: backtransform `6.744406 -> 8.142566 ms`, digest unchanged `0x0dbbde75b75c8612`
- 800x800: backtransform `43.612981 -> 42.337788 ms`, digest unchanged `0x4461962827bdb038`
- 1200x1200: backtransform `142.309442 -> 150.877702 ms`, digest unchanged `0x2fc45e1f18ceb0ab`

Score: `Impact 0.0 * Confidence 4.0 / Effort 1.0 = 0.0`

Verdict: REJECT. Source restored; `git diff -- crates/fsci-linalg/src/lib.rs` is empty after restore.

## Isomorphism

- Ordering preserved: yes. Each column saw reflectors in the same reverse order.
- Tie-breaking preserved: yes. Eigenvalue sort/order was not touched.
- Floating-point preserved: yes in proof. Dot and update loops used the same offset order for each reflector and column; bitwise serial/parallel proof passed.
- RNG preserved: yes. No RNG code or seeds changed.
- Golden outputs: stage-profile value digests stayed unchanged for 400/800/1200.

## Next Route

Do not retry column-local replay or worker-count retuning. The next primitive should be structurally different: compact/batched reflector replay with a proven block transform, or route to the next profile-backed bead if another agent already claimed the backtransform successor.
