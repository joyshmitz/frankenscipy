# frankenscipy-psn7x.2 blocked mirror-materialization rejection

Agent: `RubyWaterfall`
Date: 2026-06-15
Crate: `fsci-linalg`
Worker: RCH `ovh-a`

## Target

The profile-backed target remained the native symmetric `eigh` Householder
reduction wall after the kept `frankenscipy-psn7x.1` rank-2 index/fused-dot
lever. Current-head public route baselines still showed large-shape native
`eigh` spending hundreds of milliseconds in the Householder path.

Alien primitive harvested: communication-avoiding/block Householder families
from the graveyard notes. This pass tested the smallest bit-proofable memory
schedule slice: keep the exact scalar rank-2 math, update the lower triangle
contiguously, then materialize the upper mirror by destination column instead
of interleaving scattered mirror stores with the lower update.

No production source was retained after the performance gate failed.

## Baseline

Criterion (`baseline_eigh_dense_criterion_ovh_a_rch.txt`):

| bench | mean range |
| --- | ---: |
| `eigh_dense/256x256` | `10.891 ms` - `11.009 ms` |
| `eigh_dense/512x512` | `93.944 ms` - `94.729 ms` |

Public native route (`baseline_public_native_route_ovh_a_rch.txt`):

| n | routed native | nalgebra | speedup |
| ---: | ---: | ---: | ---: |
| 400 | `46.813422 ms` | `41.760757 ms` | `0.892068x` |
| 800 | `204.479441 ms` | `347.971005 ms` | `1.701741x` |
| 1200 | `603.412375 ms` | `1060.210706 ms` | `1.757025x` |

Native direct (`baseline_native_vs_nalgebra_ovh_a_rch.txt`):

| n | native | nalgebra | ratio |
| ---: | ---: | ---: | ---: |
| 400 | `30.8 ms` | `43.5 ms` | `1.41x` |
| 800 | `189.1 ms` | `324.8 ms` | `1.72x` |
| 1200 | `587.0 ms` | `1062.6 ms` | `1.81x` |

## Candidate

Single source lever: thresholded two-phase mirror materialization inside
`apply_symmetric_householder_trailing_rank2`.

- Small active suffixes kept the existing interleaved lower/mirror update.
- Large suffixes first updated only the lower triangle, preserving every
  floating-point operation used to compute each lower-triangle value.
- A second pass mirrored the lower triangle into the upper triangle with
  destination-column-contiguous stores.

## Proof

- `proof_rank2_two_phase_bits_ovh_a_rch.txt`: the touched rank-2 update matched
  the rowwise reference bit-for-bit for `p`, `w`, and every matrix entry.
- `proof_public_eigh_behavior_golden_ovh_a_rch.txt`: 16 `eigh` behavior tests
  passed; public materialized-pair golden digest stayed
  `0x287a5d3679a8bc6a`.
- Ordering/tie behavior: unchanged by the public materialized-pair golden proof.
- Floating point: lower-triangle update values were bit-proven against the
  existing rowwise reference before any timing decision.
- RNG: deterministic fixtures/probes only; no randomness added.

## Rebench

Public native route (`after_two_phase_public_native_route_ovh_a_rch.txt`):

| n | baseline routed | candidate routed | ratio |
| ---: | ---: | ---: | ---: |
| 400 | `46.813422 ms` | `45.611726 ms` | `1.026348x` |
| 800 | `204.479441 ms` | `206.208800 ms` | `0.991612x` |
| 1200 | `603.412375 ms` | `611.440761 ms` | `0.986870x` |

Native direct (`after_two_phase_native_vs_nalgebra_ovh_a_rch.txt`):

| n | baseline native | candidate native | ratio |
| ---: | ---: | ---: | ---: |
| 400 | `30.8 ms` | `38.9 ms` | `0.791774x` |
| 800 | `189.1 ms` | `217.9 ms` | `0.867829x` |
| 1200 | `587.0 ms` | `630.7 ms` | `0.930712x` |

Score: `Impact 0.0 * Confidence 4.0 / Effort 2.0 = 0.0`.

Verdict: REJECT. The separated mirror pass preserved behavior but increased
large-shape time. The lost locality from rereading lower-triangle values and the
extra pass outweighed any benefit from contiguous upper writes.

## Next Primitive

Do not retry lower-storage-only mirror suppression, interleaved-vs-separated
mirror materialization, scalar/slice spelling variants, worker-count retuning,
or per-step thread spawning.

Next route: a fundamentally deeper packed-panel primitive. Build an exact
packed active suffix for each Householder step or panel so `p = tau*A*v` reads
contiguous panel data without branchy symmetric loads, while the live `DMatrix`
is updated only after a bit-proven panel computation. The proof target must
remain exact scalar `p/w` values, public digest `0x287a5d3679a8bc6a`, and
same-worker RCH timing before any keep.
