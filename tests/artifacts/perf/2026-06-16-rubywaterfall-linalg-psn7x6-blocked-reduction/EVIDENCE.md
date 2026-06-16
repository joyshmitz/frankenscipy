# frankenscipy-psn7x.6 rejection: SIMD lower-storage rank-2 reduction

Agent: RubyWaterfall
Crate: `fsci-linalg`
Base commit: `1410b8e9`
Bead: `frankenscipy-psn7x.6`

## Scope

Lever tested: portable-SIMD vectorization of the contiguous lower-storage
rank-2 reduction kernel in `apply_symmetric_householder_trailing_rank2_lower_storage`.

The candidate vectorized the contiguous half of `p += A*v` and the lower-triangle
rank-2 update. It did not change public dispatch, sorting, fallback behavior,
worker counts, RNG behavior, or unsafe-code policy. Source was restored after
the failed score gate.

## Baseline

RCH worker: `vmi1149989`

| n | reduction | tridiagonal_eigen | backtransform | sort | values digest |
|---:|---:|---:|---:|---:|---|
| 400 | 14.933936 ms | 9.824466 ms | 6.558160 ms | 0.560814 ms | `0x0dbbde75b75c8612` |
| 800 | 101.649628 ms | 40.370640 ms | 47.769386 ms | 2.671451 ms | `0x4461962827bdb038` |
| 1200 | 312.980943 ms | 82.613878 ms | 151.573276 ms | 4.242936 ms | `0x2fc45e1f18ceb0ab` |

Transcript: `baseline_stage_profile_rch.txt`

## Proof

RCH `symmetric_rank2_lower_storage_matches_full_update_lower_bits` passed.

The proof compares lower-storage p/w workspaces and the lower triangle against
the full symmetric update by `f64::to_bits()`, so the candidate preserved the
existing lower-storage bitwise contract for the focused proof fixture.

Transcript: `proof_simd_lower_storage_bits_rch.txt`

## Rebench

RCH worker: `vmi1149989`

| n | before reduction | after reduction | speedup | values digest |
|---:|---:|---:|---:|---|
| 400 | 14.933936 ms | 54.556754 ms | 0.273737x | `0x0dbbde75b75c8612` |
| 800 | 101.649628 ms | 167.037724 ms | 0.608544x | `0x4461962827bdb038` |
| 1200 | 312.980943 ms | 497.829908 ms | 0.628704x | `0x2fc45e1f18ceb0ab` |

Transcript: `after_simd_lower_storage_stage_profile_rch.txt`

## Verdict

Rejected/no-ship. Bitwise proof passed and values digests stayed unchanged, but
the same-worker reduction stage regressed at all profiled sizes.

Score: Impact `0.0` x Confidence `4.0` / Effort `1.0` = `0.0`.

Source status after restoration: `git diff -- crates/fsci-linalg/src/lib.rs`
is empty.

## Next Route

Do not retry vector spelling of the scalar lower-storage rank-2 loop. Route to a
structurally different two-stage or panel-blocked symmetric reduction primitive:
blocked full-to-band followed by band-to-tridiagonal/backtransform integration,
or a true compact-WY panel reduction with explicit proof obligations.
