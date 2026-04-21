# fsci-special Implementation Sequence (P2C-006-D)

This file is the packet-D execution map for `fsci-special`.

## Module Boundaries

1. `gamma.rs`
Functions: `gamma`, `gammaln`, `digamma`, `polygamma`, `rgamma`

2. `beta.rs`
Functions: `beta`, `betaln`, `betainc`

3. `bessel.rs`
Functions: `jv`, `yv`, `iv`, `kv`, `hankel1`, `hankel2`

4. `error.rs`
Functions: `erf`, `erfc`, `erfinv`, `erfcinv`

5. `hyper.rs`
Functions: `hyp0f1`, `hyp1f1`, `hyp2f1`

6. `types.rs`
Shared tensor abstraction, dispatch-plan metadata, and strict/hardened error taxonomy.

## Contract Alignment Inputs

- Legacy anchor map: `crates/fsci-conformance/fixtures/artifacts/P2C-006/anchor/behavior_ledger.json`
- Contract table: `crates/fsci-conformance/fixtures/artifacts/P2C-006/contracts/contract_table.json`
- Threat model: `crates/fsci-conformance/fixtures/artifacts/P2C-006/threats/threat_matrix.json`

## Dispatch Strategy By Family

### Gamma family
- Primary regimes: `reflection -> recurrence -> asymptotic`
- Key edge handling:
  - nonpositive integer poles
  - signed-zero pole semantics (`gamma(+0.0)=+inf`, `gamma(-0.0)=-inf`)
  - near-pole cancellation warnings in hardened mode

### Beta family
- Primary regimes: `logspace composition -> continued fraction / recurrence`
- Key edge handling:
  - gamma-pole inheritance for `beta`
  - underflow-sensitive large-parameter regimes route through `betaln`
  - incomplete beta endpoint parity at `x=0` and `x=1`

### Bessel family
- Primary regimes: `series -> recurrence -> asymptotic`, with backend delegates for branch-heavy regions
- Key edge handling:
  - near-origin singular neighborhoods for `yv` and `kv`
  - negative-order phase/sign identities for `jv`, `hankel1`, `hankel2`
  - overflow-sensitive large-order modified Bessel paths (`iv`, `kv`)

### Error-function family
- Primary regimes: `series -> asymptotic` plus dedicated complement/inverse pathways
- Key edge handling:
  - endpoint saturation: `erf(+/-inf)=+/-1`
  - cancellation-safe `erfc` kernel (`1-erf` is forbidden in deep tails)
  - inverse-domain guards (`erfinv`: `[-1,1]`, `erfcinv`: `[0,2]`)

### Hypergeometric family
- Primary regimes: `series -> continued fraction -> recurrence -> asymptotic`
- Key edge handling:
  - denominator-parameter pole exclusions for `hyp0f1` / `hyp1f1` / `hyp2f1`
  - `hyp2f1` branch behavior near `z=1`
  - bounded iteration policy in hardened mode

## Strict vs Hardened Mode Seams

- Strict mode:
  - target SciPy-observable outputs and nonfinite semantics.
  - preserve signed infinities, NaN propagation, and branch conventions.

- Hardened mode:
  - apply fail-closed diagnostics for malformed domains and dangerous branch regions.
  - maintain strict outward values where contract says behavior is defined.
  - bound recurrence depth/iteration to prevent hostile-input resource blowups.

## D -> D2 Handoff Checklist

1. Replace placeholder `NotYetImplemented` bodies with numerical kernels per module.
2. Add per-function fixture-driven parity tests from P2C-006 contract rows.
3. Wire structured logs for domain/overflow/cancellation events.
4. Confirm `fsci-special` quality gates (fmt/check/clippy/tests) before opening P2C-006-E.
