# frankenscipy-9lb2l evidence log

This artifact preserves two independent one-lever attempts against the same
profile-backed backtransform target. The first was rejected and restored before
remote integration. The second is the kept source lever landed by this commit.

## Rejected pass: column-local backtransform replay

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
## Kept pass: native eigh backtransform unroll

# frankenscipy-9lb2l keep: native eigh backtransform unroll

Agent: RubyWaterfall
Crate: `fsci-linalg`
Base commit: `24fd40f3`
Bead: `frankenscipy-9lb2l`

## Scope

Lever: unroll the dot and update loops inside
`apply_left_reflectors_to_column_chunk`, the native symmetric `eigh`
backtransform replay kernel.

The unroll preserves the exact ascending offset order for both the dot product
and the element updates. It does not change worker counts, spawn strategy,
reflector order, sorting, fallback behavior, RNG behavior, or public dispatch.
No `unsafe` and no external BLAS/LAPACK/MKL/XLA linkage were added.

## Baseline

RCH worker: `vmi1149989`

Public route baseline:

| n | before routed native | before nalgebra | speedup vs nalgebra | values digest |
|---:|---:|---:|---:|---|
| 400 | 76.971212 ms | 84.200121 ms | 1.093917x | `0x4b8334c92ce624eb` |
| 800 | 330.803788 ms | 509.824830 ms | 1.541170x | `0xad8a7e5fa1980bfb` |
| 1200 | 990.072464 ms | 1977.708723 ms | 1.997539x | `0x181b3486089d0e4a` |

Transcript: `baseline_public_native_route_rch.txt`

Production-aligned stage baseline:

| n | reduction | tridiagonal_eigen | backtransform | sort |
|---:|---:|---:|---:|---:|
| 400 | 14.904058 ms | 9.767578 ms | 14.577668 ms | 0.433793 ms |
| 800 | 125.714032 ms | 59.501891 ms | 40.818767 ms | 1.755201 ms |
| 1200 | 384.827471 ms | 82.846703 ms | 303.985756 ms | 5.693017 ms |

Transcript: `baseline_stage_profile_rch.txt`

## Rebench

RCH worker: `vmi1149989`

Public route after:

| n | before | after | speedup | values digest |
|---:|---:|---:|---:|---|
| 400 | 76.971212 ms | 53.676934 ms | 1.433982x | `0x4b8334c92ce624eb` |
| 800 | 330.803788 ms | 172.561917 ms | 1.916992x | `0xad8a7e5fa1980bfb` |
| 1200 | 990.072464 ms | 481.770373 ms | 2.055072x | `0x181b3486089d0e4a` |

Transcript: `after_unroll_public_native_route_rch.txt`

Stage after:

| n | before backtransform | after backtransform | speedup |
|---:|---:|---:|---:|
| 400 | 14.577668 ms | 6.485639 ms | 2.247683x |
| 800 | 40.818767 ms | 49.827003 ms | 0.819206x |
| 1200 | 303.985756 ms | 157.843030 ms | 1.925874x |

The n=800 isolated stage slice regressed, but the public n=800 route improved
substantially on the same worker and the n=1200 target improved in both stage
and public-route measurements.

Transcript: `after_unroll_stage_profile_rch.txt`

## Behavior Proof

- Bitwise reflector replay proof passed:
  `symmetric_eigh_backtransform_parallel_matches_serial_bits`.
- Public route values digests were unchanged for n=`400/800/1200`.
- Public golden digest remained `0x287a5d3679a8bc6a` via
  `eigh_index_sort_matches_materialized_pair_sort_bits`.
- Native symmetric `eigh` proof against nalgebra passed:
  `symmetric_eigh_native_matches_nalgebra_and_timing`.
- Ordering/tie behavior is unchanged; sort and public dispatch are untouched.
- Floating-point order inside each reflector dot/update is unchanged by the
  unroll: offsets are accumulated and applied in the same ascending sequence.
- RNG behavior is unchanged; probes use deterministic fixtures only.

## Gates

- `cargo fmt -p fsci-linalg -- --check`: passed.
- `ubs crates/fsci-linalg/src/lib.rs`: exit 0, critical issues 0.
- RCH `cargo check -j 1 -p fsci-linalg --lib --locked`: passed.
- RCH `cargo clippy -j 1 -p fsci-linalg --lib --no-deps --locked -- -D warnings`:
  blocked by upstream lints in recently added `rsf2csf` and `matrix_balance`
  code (`manual_is_multiple_of`, `needless_range_loop`, `mut_range_bound`).
  These are outside the backtransform lever; the unroll hunk did not introduce a
  clippy finding.

Both RCH compile gates emitted the known dependency warning in
`fsci-fft/src/helpers.rs:58`.

## Score

Impact `4.0` x Confidence `4.0` / Effort `1.5` = `10.67`.

Verdict: KEEP. The public route clears the score gate with directly comparable
same-worker RCH evidence, unchanged digests, bitwise reflector replay proof, and
golden-output proof.

## Next Route

Re-profile after this keep. If reduction is again dominant, attack a different
reduction primitive; if backtransform remains material, escalate to a stronger
reflector replay primitive rather than worker-count retuning.
