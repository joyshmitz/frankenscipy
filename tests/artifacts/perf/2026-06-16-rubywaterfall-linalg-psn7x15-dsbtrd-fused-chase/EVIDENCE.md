# frankenscipy-psn7x.15 Evidence - Fused Compact-Band DSBTRD Chase Harness

## Target

- Bead: `frankenscipy-psn7x.15`
- Crate: `fsci-linalg`
- Profile-backed route: successor to `frankenscipy-psn7x.14`, which kept compact-band Q metadata replay and showed dense-Q materialization/multiply vs replay speedups up to `18.90x` locally on the rebased head.
- Environment: local cargo + hyperfine with `RCH_REQUIRE_REMOTE=0`; `ts1` RCH path remains offline.

## Baseline

Baseline before source edits used the existing committed compact-envelope and Q-replay proof slices.

Envelope frontier command:

```text
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenscipy-rubywaterfall-opt-20260616-1955/.local-target hyperfine --warmup 1 --runs 5 --show-output 'cargo test -j 1 -p fsci-linalg --lib lower_band_envelope_frontier_rotation_perf_probe --release --locked -- --ignored --nocapture'
```

Envelope baseline ranges:

| shape | bandwidth | dense ms | envelope ms | speedup range |
| --- | ---: | ---: | ---: | ---: |
| 128x128 | 32 | `0.107804`-`0.179780` | `0.060224`-`0.095851` | `1.790050x`-`1.918322x` |
| 256x256 | 32 | `0.806377`-`1.097539` | `0.134344`-`0.188096` | `5.834994x`-`6.215889x` |
| 512x512 | 32 | `5.899313`-`7.363697` | `0.402663`-`0.513333` | `12.510055x`-`15.561260x` |

Q replay command:

```text
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenscipy-rubywaterfall-opt-20260616-1955/.local-target hyperfine --warmup 1 --runs 5 --show-output 'cargo test -j 1 -p fsci-linalg --lib lower_band_frontier_q_metadata_replay_perf_probe --release --locked -- --ignored --nocapture'
```

Q replay baseline ranges:

| shape | bandwidth | dense Q ms | replay ms | speedup range |
| --- | ---: | ---: | ---: | ---: |
| 128x128 | 32 | `8.534424`-`12.347877` | `0.644210`-`1.006106` | `12.272938x`-`16.340317x` |
| 256x256 | 32 | `21.735388`-`29.836110` | `1.623284`-`3.008939` | `9.711179x`-`14.487054x` |
| 512x512 | 32 | `65.175133`-`76.417196` | `4.043739`-`6.725719` | `9.995729x`-`18.897658x` |

Public `eig_banded` baseline before the edit remained:

| shape | bandwidth | values digest | vectors digest |
| --- | ---: | --- | --- |
| 128x128 | 32 | `0xd6dbb9200f65bd92` | `0x6cf3573b5b50c275` |
| 256x256 | 32 | `0x09ed4d367faab431` | `0xc32797c0d224a75a` |

## Lever

One source lever was kept: add an isolated fused compact-band frontier chase harness.

The harness compares:

- Dense oracle: apply frontier rotations to dense `A`, materialize dense `Q`, emit D/E from transformed dense storage, and compute `Q * V`.
- Compact candidate: apply the same frontier rotations to compact envelope storage, emit D/E directly from the envelope, and replay the stored rotation metadata directly to eigenvector rows.

This is proof/probe code only. It does not change public `eig_banded`, thresholds, sorting, tie-breaking, fallback selection, RNG behavior, or any user-visible numerical route.

## Proof

Command:

```text
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenscipy-rubywaterfall-opt-20260616-1955/.local-target cargo test -j 1 -p fsci-linalg --lib lower_band_fused_frontier_chase_matches_dense_oracle --release --locked -- --nocapture
```

Proof output:

```text
lower_band_fused_chase n=18 bandwidth=4 cols=7 rotations=3 transformed_drift=7.10542735760100186e-15 eigenvector_drift=3.46944695195361419e-18 compact_digest=0x1c3ef0d9d3dc1083
lower_band_fused_chase n=37 bandwidth=8 cols=13 rotations=4 transformed_drift=2.84217094304040074e-14 eigenvector_drift=2.22044604925031308e-16 compact_digest=0x8a89e08b8c6dc6dd
lower_band_fused_chase n=64 bandwidth=12 cols=17 rotations=5 transformed_drift=5.68434188608080149e-14 eigenvector_drift=6.93889390390722838e-18 compact_digest=0xf1534a90714d38a0
```

Golden output artifact:

```text
tests/artifacts/perf/2026-06-16-rubywaterfall-linalg-psn7x15-dsbtrd-fused-chase/lower-band-fused-frontier-chase-golden-output.txt
```

Golden output sha256: `1f66e9e7130c39dcacc815bd7f2c5344359948fcc6812474201281577458875d`

Proof obligations:

- Compact transformed storage matched the dense frontier oracle.
- Compact D/E emission matched dense D/E emission through the tridiagonal extraction contract.
- Replayed eigenvector block matched dense `Q * V`.
- Dense `Q^T A Q` projection and Q orthogonality passed.
- Ordering/tie-breaking unchanged: no public eigenvalue/eigenvector ordering path changed.
- Floating point: public route unchanged; proof has explicit dense-oracle tolerances.
- RNG unchanged: deterministic fixtures only.
- Safety unchanged: safe Rust only; no C BLAS/LAPACK/MKL/XLA linkage.

## Rebench

Command:

```text
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenscipy-rubywaterfall-opt-20260616-1955/.local-target hyperfine --warmup 1 --runs 5 --show-output 'cargo test -j 1 -p fsci-linalg --lib lower_band_fused_frontier_chase_perf_probe --release --locked -- --ignored --nocapture'
```

Hyperfine wall time: `313.7 ms +/- 11.9 ms` over 5 runs.

Internal probe ranges:

| shape | bandwidth | cols | dense ms | compact ms | speedup range | transformed drift | eigenvector drift | dense digest | compact digest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 128x128 | 32 | 64 | `10.405989`-`13.434075` | `4.727884`-`6.003030` | `2.010626x`-`2.766834x` | `1.13686837721616030e-13` | `6.93889390390722838e-18` | `0x9105700991057009` | `0x325d2bcbcda2d434` |
| 256x256 | 32 | 96 | `23.816479`-`27.678083` | `14.625411`-`16.786184` | `1.482065x`-`1.834741x` | `1.13686837721616030e-13` | `1.11022302462515654e-16` | `0xef3001c2de10cf1d` | `0xcf9442ea75930712` |
| 512x512 | 32 | 128 | `46.673731`-`52.840271` | `26.713896`-`32.641624` | `1.547150x`-`1.820491x` | `2.27373675443232059e-13` | `1.11022302462515654e-16` | `0xd14b98dc70ded6a6` | `0xd5488334a24b80bc` |

Public smoke after the lever remained unchanged:

| shape | bandwidth | values digest | vectors digest |
| --- | ---: | --- | --- |
| 128x128 | 32 | `0xd6dbb9200f65bd92` | `0x6cf3573b5b50c275` |
| 256x256 | 32 | `0x09ed4d367faab431` | `0xc32797c0d224a75a` |

## Score

- Impact: `2.0` (fused compact chase replaces dense frontier + dense Q materialization in the harness and wins at every measured size).
- Confidence: `4.0` (dense oracle proof, D/E extraction proof, Q projection/orthogonality, deterministic digests, repeated local hyperfine).
- Effort: `1.0`.
- Score: `8.0`.

Verdict: KEEP.

## Next Profile Route

Re-profile after this proof slice. The next primitive should move from proof-slice frontier rotations to a real bulge chase sequence with progressive off-band annihilation and compact Q metadata replay, still before public `eig_banded` wiring.
