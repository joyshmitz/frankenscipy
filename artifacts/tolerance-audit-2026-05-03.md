# Tolerance Audit — 2026-05-03

Initial baseline measurement of `tolerance_lint`'s gate output. This is the starting datum
that bead `frankenscipy-9rg2o` (slice of `frankenscipy-e9z9y` M4 hardening) leaves on disk so
the eventual CI gate has a number to ratchet down from.

The lint binary `tolerance_lint` (added in this commit at
`crates/fsci-conformance/src/bin/tolerance_lint.rs`) walks every fixture in
`crates/fsci-conformance/fixtures/FSCI-P2C-*.json` and, for each case with a numeric scalar
`(rtol, atol)` pair, compares `rtol` to the per-packet baseline tier from
`artifacts/TOLERANCE_POLICY.md` §2. Cases that are **looser than baseline AND lack a
`rationale` field** are flagged.

## 1. Summary

- **361** violations across **11** packets out of 17 (the other 6 are Tnone/structural).
- **0** cases currently carry a `rationale` field — the policy was published 2026-05-03 but no
  fixture has been retrofitted yet. Every relaxation today is implicit.
- The single largest violator class is **FSCI-P2C-006 (special)** with 191 cases, dominated
  by Bessel/Bessel-derivative tail cases (`diff_y0_*`, `diff_jvp_*`, `diff_yvp_*`) where
  scipy itself documents low-precision branches.

## 2. Per-packet breakdown

`×N` columns count cases falling in the bucket "rtol ÷ baseline ∈ [lo, hi)":

| Packet                | Violations | ×1–10 | ×10–100 | ×100–1k | ×1k–1M | >×1M |
|-----------------------|----------:|------:|--------:|--------:|-------:|-----:|
| FSCI-P2C-002 (linalg) |        17 |     0 |       0 |       6 |      8 |    3 |
| FSCI-P2C-004 (sparse) |         7 |     0 |       0 |       0 |      6 |    1 |
| FSCI-P2C-005 (fft)    |        39 |     0 |      26 |      13 |      0 |    0 |
| FSCI-P2C-006 (special)|       191 |     2 |       1 |     112 |     47 |   29 |
| FSCI-P2C-009 (cluster)|        14 |     0 |      14 |       0 |      0 |    0 |
| FSCI-P2C-010 (spatial)|         1 |     0 |       0 |       0 |      1 |    0 |
| FSCI-P2C-011 (signal) |        45 |     0 |      12 |       6 |      3 |   24 |
| FSCI-P2C-012 (stats)  |        14 |     0 |       0 |       9 |      5 |    0 |
| FSCI-P2C-013 (integ.) |        20 |     0 |       0 |       5 |      5 |   10 |
| FSCI-P2C-014 (interp) |        10 |     0 |       0 |       4 |      6 |    0 |
| FSCI-P2C-016 (const)  |         3 |     0 |       1 |       2 |      0 |    0 |
| **Total**             |   **361** | **2** | **54**  | **157** | **81** | **67**|

The packets not in this table (FSCI-P2C-001, -003, -007, -008, -015, -017) have no numeric
baseline (Tnone/structural-only) and therefore can't produce violations under the lint rule.

## 3. Worst offenders (top 10 by `rtol / baseline`)

These are the first cases that should either receive an explicit `rationale` field or be
tightened to baseline:

| Packet       | case_id                                          | rtol  | ×baseline   |
|--------------|--------------------------------------------------|------:|------------:|
| FSCI-P2C-006 | `diff_y0_one`                                    | 8e-3  | ×8e+09      |
| FSCI-P2C-004 | `eigs_uppertri_4x4_top2`                         | 1e-1  | ×1e+09      |
| FSCI-P2C-013 | `ivp_lsoda_robertson_short`                      | 5e-2  | ×5e+08      |
| FSCI-P2C-011 | `remez_lp_11_passband_0p2_stopband_0p3`          | 5e-2  | ×5e+08      |
| FSCI-P2C-011 | `remez_bp_15_3band`                              | 5e-2  | ×5e+08      |
| FSCI-P2C-011 | `firwin2_lp_11_freqsamp`                         | 5e-2  | ×5e+08      |
| FSCI-P2C-011 | `firwin2_bp_15_3band`                            | 5e-2  | ×5e+08      |
| FSCI-P2C-006 | `diff_yvp_v0_x1p75_n1`                           | 5e-4  | ×5e+08      |
| FSCI-P2C-006 | `diff_jvp_v0_x1p25_n2`                           | 5e-4  | ×5e+08      |
| FSCI-P2C-006 | `diff_jvp_v0_x1p25_n1`                           | 5e-4  | ×5e+08      |

Most of these are already accounted for in `TOLERANCE_POLICY.md` §5 ("Active exceptions") —
they need a `rationale: "frankenscipy-1i92b"` (or whichever bead they cite) added to the
fixture entry to clear the lint.

## 4. Reduction plan

1. **Phase 1 — annotate (no behavior change).** For every case in the top 10, add the
   matching exception bead id to the case's `rationale` field. Re-run the lint; expect
   ~10–20 violations to clear immediately.
2. **Phase 2 — bulk annotate.** For each packet, sweep the documented exception classes from
   `TOLERANCE_POLICY.md` §2 (e.g. P2C-006 Bessel `diff_*` cases → `frankenscipy-1i92b`;
   P2C-011 `firwin2_*`/`remez_*` → `frankenscipy-b6z3m`) and apply rationales en bloc. Expect
   the count to drop below 100.
3. **Phase 3 — tighten.** Cases with `×1–100` baseline violation (56 today) are candidates
   for tightening to baseline rather than adding a rationale. These represent ~5 ulp slack
   that is probably no longer needed.
4. **Phase 4 — gate.** Once the count is below the agreed ratchet (say ≤50), wire
   `tolerance_lint --max-violations 50` into the CI workflow as G9. Subsequent ratchet
   reductions happen by lowering `--max-violations` in the workflow file.

## 5. CLI reference

```bash
# Run with default fixture dir, fail on any violation:
cargo run --release -p fsci-conformance --bin tolerance_lint

# Allow up to N violations during ratchet-down:
cargo run --release -p fsci-conformance --bin tolerance_lint -- --max-violations 361

# Emit JSON for downstream tooling (dashboards, drift reports):
cargo run --release -p fsci-conformance --bin tolerance_lint -- --json
```

Exit codes: 0 = within budget, 1 = over threshold, 2 = IO/parse error.

## 6. Honesty notes

- The per-packet baseline tier is currently hardcoded inside `tolerance_lint.rs`. Source of
  truth still lives in `artifacts/TOLERANCE_POLICY.md` §2 — when those tables change, the
  binary must be updated. A future improvement is parsing the markdown directly, but the
  hardcode is intentional today so the linter ships ready-to-run.
- The `rationale` field doesn't yet exist on any fixture case. Adding it to existing fixtures
  is a regenerate-step (the harness consumes the fixture but doesn't author it; fixtures are
  typically regenerated from `fixture_regen` / oracle-capture pipelines). Wiring rationale
  through `fixture_regen` is part of the CI gate work tracked under `frankenscipy-fmsdy`.
- Vector-valued tolerances (P2C-001 `validate_tol`) are skipped — the baseline tier in
  `TOLERANCE_POLICY.md` §1 explicitly classifies them as Txn (non-uniform per element), and
  the linter has no scalar to compare against.
