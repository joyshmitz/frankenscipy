# psn7x post-back-transform native-eigh stage profile

Bead: `frankenscipy-psn7x`
Agent: `RubyWaterfall`
Base commit: `1feb36d2`
Date: 2026-06-15

## Profile

RCH worker: `vmi1152480`
Probe: temporary ignored stage-breakdown test, removed before any source commit.

Current native `symmetric_eigh_native` stage timing for `n=1200`:

| stage | time |
| --- | ---: |
| Householder reduction | 903.217 ms |
| Tridiagonal QR/eigenvectors | 1089.798 ms |
| Back-transform | 296.175 ms |
| Eigenpair sort/copy | 27.073 ms |
| Total | 2317.048 ms |

The next optimization target is therefore the tridiagonal QR/eigenvector stage, not
the already-kept back-transform path.

## Artifact

- `stage_breakdown_vmi1152480_rch.txt`
- SHA-256: `d2a7da6e58b1f5f53b553daad642217607af8a80260369d1533a8a7b6299c2bc`
