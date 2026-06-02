# Residual Clone Split - Pass 2

**Bead:** `frankenscipy-perf-linalg-lu-scalar-2y3wp`
**Related bead:** `frankenscipy-perf-linalg-backward-error-clone-qrufd`
**Skill loop:** `extreme-software-optimization`, pass 2 of 5
**Verdict:** rejected; no Rust code change kept.

## Contract Read

`SolveResult.backward_error` is observable API state and is recorded into solver
portfolio evidence. Hardened full-validation also consults it when present, so
the pass could not silently remove or make `backward_error` opt-in.

The only acceptable lever was therefore to keep `backward_error: Some(_)` while
avoiding the pristine `DMatrix` retained only for
`compute_backward_error(&matrix, &x, &rhs)`.

## Baseline

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- hyperfine --setup 'env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_residual_clone_pass2 RUSTFLAGS="-C force-frame-pointers=yes" cargo build -p fsci-linalg --profile release-perf --bin perf_solve' --warmup 3 --runs 10 --export-json tests/artifacts/perf/2026-06-01-linalg-solve/pass2_residual_clone_baseline_rch.json '/tmp/rch_target_fsci_linalg_residual_clone_pass2/release-perf/perf_solve solve 1000 1 42'
```

Result:

| metric | value |
|--------|-------|
| mean +- sigma | 131.8 ms +- 10.8 ms |
| median | 129.6 ms |
| min / max | 114.2 / 150.8 ms |
| user / system | 101.4 / 28.8 ms |

## Candidate Tested

Candidate shape:

- split solve-side diagnostics so `matrix.clone().lu()` could become
  ownership-consuming `matrix.lu()` for `solve`;
- keep cached `LU` reuse;
- compute the residual/backward error from `effective_a` rows instead of a
  retained pristine `DMatrix`.

No candidate code was kept.

## Candidate Benchmark

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- hyperfine --setup 'env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_residual_clone_pass2_after RUSTFLAGS="-C force-frame-pointers=yes" cargo build -p fsci-linalg --profile release-perf --bin perf_solve' --warmup 3 --runs 10 --export-json tests/artifacts/perf/2026-06-01-linalg-solve/pass2_residual_clone_after_rch.json '/tmp/rch_target_fsci_linalg_residual_clone_pass2_after/release-perf/perf_solve solve 1000 1 42'
```

Result:

| metric | baseline | candidate | delta |
|--------|----------|-----------|-------|
| mean +- sigma | 131.8 +- 10.8 ms | 127.1 +- 7.1 ms | -3.5% |
| median | 129.6 ms | 126.7 ms | -2.3% |
| min / max | 114.2 / 150.8 ms | 115.4 / 143.2 ms | mixed |
| user / system | 101.4 / 28.8 ms | 102.2 / 23.7 ms | system -17.5% |

The timing direction was favorable but not enough to override behavior drift.

## Isomorphism Proof

Golden command:

```bash
/tmp/rch_target_fsci_linalg_residual_clone_pass2_after/release-perf/perf_solve golden > tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_residual_clone_pass2_after.txt
```

Golden comparison:

```text
5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd  golden/golden_before.txt
6d5b35d108866da526ea26bbc553152df32b317797df8f5afc76d1ee5ac1cb7f  golden/golden_residual_clone_pass2_after.txt
```

Result:

- Ordering preserved: yes; solver action order and fallback order unchanged.
- Tie-breaking unchanged: yes; no selection comparison changed.
- Floating-point: failed; solution vector bits matched in inspected drift lines,
  but `backward_error` changed by 1-2 ULP in multiple golden cases.
- RNG seeds: unchanged; `perf_solve` deterministic seeds unchanged.
- Golden outputs: failed sha256 comparison.

## Opportunity Score

| lever | impact | confidence | effort | score | decision |
|-------|--------|------------|--------|-------|----------|
| row residual without retained DMatrix | 2 | 0 | 2 | 0.0 | reject |

Confidence is zero because the observable `backward_error` bit pattern changed.

## Checked And Rejected

- Dropping `backward_error`: rejected because it is observable `SolveResult`
  state and recorded into CASP evidence.
- Making `backward_error` opt-in: rejected for this pass because it would alter
  default `solve` semantics.
- Computing backward error from `effective_a` rows: rejected because it changed
  `backward_error` bit patterns even though solution bits stayed identical in
  inspected golden diffs.
- Reconstructing residual from LU factors: rejected for this pass because LU
  stores pivoted/mutated factors, not pristine `A`; using it would be a larger
  numerical semantics proof, not the residual clone split.

## Additional Preserved Evidence

A concurrent subagent addendum produced and left these related files in the same
artifact directory; they are preserved with this pass because the shared progress
entry references them:

- `baseline_residual_clone_pass2_before.json`
- `stage_residual_clone_pass2_before.json`
- `golden/golden_residual_clone_pass2_before.txt`
- `golden/golden_residual_clone_pass2_after_exact.txt`
- `golden/golden_residual_clone_pass2_restored.txt`

Their sha256 evidence matches the same rejection pattern: before/restored golden
outputs are `5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd`,
while the exact row-residual attempt is
`374443b7902ede8c01df577a24d72b756a0d08d9679e55a74edfec2dcad6f1a7`.

No code or Cargo configuration change is retained.
