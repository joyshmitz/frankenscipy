# frankenscipy-r7te6 rejection: zeroed generated compact-WY panel

## Target

- Bead: `frankenscipy-r7te6`
- Worker: `vmi1152480`
- Target: `fsci-linalg` native symmetric `eigh` full-to-band reduction.
- Profile-backed hotspot: after the `psn7x.8` backtransform keep, the direct reduction still dominates large native `eigh` cases.

## Baseline

Command:

```text
rch exec -- cargo test -p fsci-linalg symmetric_eigh_native_stage_breakdown_probe --release -- --ignored --nocapture
```

Same-worker baseline:

| shape | reduction_ms | tridiagonal_eigen_ms | backtransform_ms | sort_ms | max_abs_diff | values_digest |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 400x400 | 20.399006 | 15.233532 | 26.621937 | 0.988595 | 5.24025267623073887e-13 | `0x0dbbde75b75c8612` |
| 800x800 | 174.336668 | 49.244159 | 106.755178 | 2.455152 | 1.71418435002124170e-12 | `0x4461962827bdb038` |
| 1200x1200 | 464.094159 | 104.148309 | 281.030846 | 5.612751 | 1.79412040779425297e-12 | `0x2fc45e1f18ceb0ab` |

Baseline transcript SHA-256:

```text
5a447fa3e6e46487b7580028bb43a8e98c1824ed7b30c2870f29d95846f8cf4c  baseline_stage_profile_rch.txt
```

## Lever

One attempted source lever: zero eliminated full-to-band columns immediately after each generated reflector while constructing compact-WY panels from raw panel reflectors.

This was a narrow corrective attempt after `psn7x.7`: prevent later cross-block right updates from observing stale below-band entries before the compact panel update is replayed.

## Behavior proof

Command:

```text
rch exec -- cargo test -p fsci-linalg compact_wy_zeroed_full_to_band_generation_matches_scalar_reduction -- --nocapture
```

Result: failed. The generated compact-WY reduction diverged from the scalar deterministic full-to-band reduction:

```text
zeroed compact-WY full-to-band drift 6.72055843216352450e-2 for n=37, bandwidth=8, panel_width=4
```

The smaller fixture printed a different reduced-matrix digest before the failing case:

```text
n=18 bandwidth=4 panel_width=3 scalar=0x75dfccbb72f486ff compact=0x7f5bb780f9fcc3cc
```

Proof transcript SHA-256:

```text
19bdb6a2ea7d8d51cc79783b5f7ea91a3e3ccc838d3c1b45c153260154559465  zeroed_generation_proof_rch.txt
```

## Decision

- Score: 0.0. Behavior is not isomorphic to the scalar full-to-band reference, so the source lever was restored and no performance rebench is admissible.
- Ordering/tie-breaking/RNG: no accepted behavior change. The failed candidate was removed before commit.
- Floating point: the rejected candidate changed the reduction itself, not only rounding order, and failed the proof threshold.
- Golden output: the accepted tree remains at the baseline value digests listed above.

Next primitive: stop retrying raw-reflector generated panels. Attack either a proper transformed-panel compact-WY generator, where subsequent panel vectors are updated in the live panel basis before forming `T`, or a full safe-Rust two-stage band-to-tridiagonal bulge chaser with accumulated orthogonal transforms.
