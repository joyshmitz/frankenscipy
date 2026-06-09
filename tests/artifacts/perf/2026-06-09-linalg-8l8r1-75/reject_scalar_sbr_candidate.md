# frankenscipy-8l8r1.75 scalar SBR dense-eigh candidate rejection

Date: 2026-06-09
Agent: BlackThrush
Worktree: `/data/projects/.scratch/frankenscipy-mifdz-codex-20260609-2059`

## Target

`frankenscipy-8l8r1.75` followed the `.73` rejection and tested a different
dense symmetric `eigh` primitive: a private two-stage successive band reduction
probe. The probe reduced dense symmetric `A` to a fixed half-bandwidth-32 band
with symmetric Householder transforms, called the existing safe-Rust
`eig_banded` path, then backtransformed eigenvectors.

The implementation was test-only and was removed after measurement. Public
`eigh` stayed on the existing nalgebra route.

## Baseline

RCH Criterion requested `ovh-a`; RCH selected `vmi1227854`.

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo bench -p fsci-linalg --bench linalg_bench -- eigh_dense --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Measured rows:

```text
eigh_dense/256x256      time:   [12.453 ms 12.972 ms 13.447 ms]
eigh_dense/512x512      time:   [94.785 ms 97.292 ms 100.52 ms]
```

Focused public golden proof requested `vmi1227854`; RCH selected `ovh-a`.

```text
eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a
test result: ok. 1 passed; 0 failed
```

## Candidate Evidence

RCH candidate probe requested `vmi1227854`; RCH selected `ovh-a`.

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo test -p fsci-linalg --release --lib --locked public_eigh_sbr_candidate_perf_probe -- --ignored --nocapture --test-threads=1
```

Same-process probe rows:

```text
256x256:
public_eigh_ms=12.517715
sbr_candidate_ms=137.218141
speedup=0.091225
eigenvalue_max_abs_diff=1.08002495835535228e-11
reconstruction_max_abs=8.18545231595635414e-12
orthogonality_max_abs=7.66053886991358013e-15
public_digest=0xb2c0047b36b19aa2
candidate_digest=0x2d26c99d16279439
candidate_raw_public_bits_equal=false

512x512:
public_eigh_ms=86.914909
sbr_candidate_ms=1151.224983
speedup=0.075498
eigenvalue_max_abs_diff=2.91038304567337036e-11
reconstruction_max_abs=1.81898940354585648e-11
orthogonality_max_abs=1.13242748511765967e-14
public_digest=0x4729412f23d9f9e4
candidate_digest=0x143d117ed9baae66
candidate_raw_public_bits_equal=false
```

## Isomorphism Decision

The candidate preserved validation shape, used ascending `f64::total_cmp` after
the candidate backend, and introduced no RNG, unsafe code, or external
BLAS/LAPACK/MKL/XLA surface. Numerical invariants passed: eigenvalue drift,
reconstruction error, and orthogonality error stayed within the current
tolerance contract.

The candidate failed both retention gates:

- It was much slower in same-process measurement: `0.091225x` at `256x256` and
  `0.075498x` at `512x512`.
- Raw public eigenvector bits/digests differed. This is expected for an
  independent symmetric eigensolver basis, but it means it cannot replace public
  `eigh` under the current raw-output contract.

## Verdict

Rejected and source restored.

Do not repeat scalar full-to-band Householder reduction plus existing
`eig_banded`. The next dense-`eigh` primitive must avoid both scalar stages:
use a compact-WY blocked panel reducer with BLAS-3-style far updates and a
direct band-to-tridiagonal/bulge-chasing stage, with stage timings before any
public-route attempt.
