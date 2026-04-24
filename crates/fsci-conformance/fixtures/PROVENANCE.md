# Fixture Provenance

Every fixture records how it was generated so a future regeneration
can be diffed against a known baseline. If fixture values silently
change, this file is what tells us whether the change is a scipy
upgrade, a generator bug, or an intentional expansion.

## Generator baseline

All `fsci-conformance/fixtures/FSCI-P2C-*.json` were last regenerated
against:

```
python3   : 3.13
numpy     : 2.3.x
scipy     : 1.15.x
generator : crates/fsci-conformance/python_oracle/scipy_<family>_oracle.py
command   : python3 scipy_<family>_oracle.py \
                --fixture fixtures/FSCI-P2C-<id>_<family>.json \
                --output  fixtures/artifacts/P2C-<id>/oracle_capture.json
```

Most fixture `expected` values were *seeded* by running the reference
scipy implementation on the fixture's `args`, then pasting the output
into `expected.value`. The per-family oracle script can reproduce the
capture at any time.

## Per-family versions

| Family | Packet | Generator | Last touched |
|--------|--------|-----------|-------------|
| linalg      | P2C-002 | scipy_linalg_oracle.py    | 2026-04-13 |
| optimize    | P2C-003 | scipy_optimize_oracle.py  | 2026-04-21 |
| sparse_ops  | P2C-004 | scipy_sparse_oracle.py    | 2026-04-14 |
| fft         | P2C-005 | scipy_fft_oracle.py       | 2026-04-14 |
| special     | P2C-006 | scipy_special_oracle.py   | 2026-04-22 |
| arrayapi    | P2C-007 | (property-based, no oracle) | 2026-03-12 |
| runtime_casp| P2C-008 | (property-based)          | 2026-03-13 |
| cluster     | P2C-009 | scipy_cluster_oracle.py   | 2026-04-14 |
| spatial     | P2C-010 | scipy_spatial_oracle.py   | 2026-04-21 |
| signal      | P2C-011 | scipy_signal_oracle.py    | 2026-04-14 |
| stats       | P2C-012 | scipy_stats_oracle.py     | 2026-04-22 |
| integrate   | P2C-013 | scipy_integrate_oracle.py | 2026-04-18 |
| interpolate | P2C-014 | scipy_interpolate_oracle.py | 2026-04-24 |
| ndimage     | P2C-015 | scipy_ndimage_oracle.py | 2026-04-24 |
| constants   | P2C-016 | (property-based)          | 2026-03-31 |

Dates correspond to the last modification of the fixture file on disk
and do *not* guarantee every case was regenerated at that date — only
that some subset was touched.

## Regenerating a fixture's oracle capture

```bash
# Example: spatial
python3 crates/fsci-conformance/python_oracle/scipy_spatial_oracle.py \
  --fixture crates/fsci-conformance/fixtures/FSCI-P2C-010_spatial_core.json \
  --output  crates/fsci-conformance/fixtures/artifacts/P2C-010/oracle_capture.json

# Then compare against the fixture's embedded expected values:
diff <(jq '.cases[]|{case_id, expected}' FSCI-P2C-010_spatial_core.json) \
     <(jq '.case_outputs[]|{case_id, result}' artifacts/P2C-010/oracle_capture.json)
```

If expected and captured values disagree beyond the fixture tolerance,
either the scipy version changed, or a case was hand-edited; update
DISCREPANCIES.md accordingly.

## Dependency pinning

`Cargo.toml` pins `rand = "0.10"` (not 0.9 or 0.11) and `nalgebra = "0.34"`
— numerical regressions from silent minor-version churn are what drove
br-otdp and the `chore(deps)` series of commits. Update this file
when the pins move.

## Related

- `DISCREPANCIES.md` — numbered divergences from scipy reference.
- `COVERAGE.md` — function-surface coverage matrix.
- `python_oracle/` — the six scripts that produce these outputs.
