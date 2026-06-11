# frankenscipy-8l8r1.87 keep decision

## Scope

This keeps one private safe-Rust primitive in `crates/fsci-linalg/src/lib.rs`: a compact-WY symmetric panel update kernel for dense full-to-band reduction work.

The public `eigh`/`eigvalsh` dispatch remains unchanged. No external BLAS/LAPACK/MKL/XLA linkage and no unsafe code were introduced.

## Baseline and target

- Bead: `frankenscipy-8l8r1.87`
- Worker: `vmi1227854`
- Fresh profile-backed baseline: `eigh_dense/512x512` mean `108.63 ms`
- Prior rejected route: scalar-panel full-to-band probe regressed at `512x512` (`166.288718 ms` candidate vs `105.830953 ms` public path, `0.636429x`)
- This slice target: compact-WY panel trailing update versus scalar reflector replay on deterministic symmetric active blocks

## Proof

- Compact-WY proof: `proof_compact_wy_panel_rch_retry1.txt`
  - `compact_wy_panel_rejects_invalid_shapes` passed
  - `compact_wy_symmetric_update_matches_scalar_replay` passed
  - digests:
    - `n=12 start=2 k=2 digest=0x8dde440efdff1393`
    - `n=33 start=4 k=5 digest=0xd02bb04badff9732`
    - `n=65 start=8 k=8 digest=0x2bd7d17c005a6a87`
- Public golden proof after source edit: `proof_public_eigh_golden_after_compact_wy_rch_retry3_eigh_filter.txt`
  - 13 eig/eigh tests passed
  - `eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a`, unchanged from the pre-edit golden artifact
- Zero-test artifacts from narrower filters are retained as admission/filter diagnostics and are not used as behavior proof.

## Same-worker release probe

Artifact: `after_compact_wy_perf_probe_rch.txt`

| Shape | Scalar replay | Compact-WY | Speedup | Max abs diff | Symmetry drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `256x256` | `0.645397 ms` | `0.402654 ms` | `1.602858x` | `1.70530256582424045e-12` | `5.68434188608080149e-14` |
| `512x512` | `2.881077 ms` | `1.884514 ms` | `1.528817x` | `7.95807864051312208e-12` | `1.13686837721616030e-13` |

The digest differs from scalar replay because the compact-WY batched update changes floating-point association inside a private proof-gated primitive. Public output ordering/tie behavior is covered by the unchanged `eigh_index_sort_public_golden_digest`.

## Score

`(Impact 3.5 * Confidence 4) / Effort 2.5 = 5.6`

Verdict: KEEP. The measured same-worker stage win clears the `Score >= 2.0` threshold and replaces the failed scalar-panel replay family with a BLAS-3-style compact-WY primitive.

## Gates

- `rustfmt --edition 2024 --check crates/fsci-linalg/src/lib.rs`: passed
- `ubs crates/fsci-linalg/src/lib.rs`: exit 0; no critical findings, broad pre-existing inventory warnings
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fsci-linalg --lib --locked`: passed on `vmi1227854`
- Broad `cargo clippy -p fsci-linalg --lib --locked -- -D warnings`: blocked by unrelated `fsci-fft` path dependency lint in `crates/fsci-fft/src/transforms.rs:2744`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p fsci-linalg --lib --locked --no-deps -- -D warnings`: passed on `vmi1227854`

## Next route

Next non-repeating primitive: wire this compact-WY `V/T` panel state into a full-to-band blocked symmetric reduction slice, then compare full reduction plus reconstruction against the public `eigh_dense/512x512` baseline before any public dispatch.
