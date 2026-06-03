# Linalg Reprofile After Packed-B Panel GEMM

- Timestamp: 2026-06-03T17:03:20-04:00
- Worker: `vmi1293453`
- Head: `45073e21`
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- 'matmul|lstsq/512x256|pinv/512x256|baseline_solve/1000x1000|baseline_lstsq/1000x500|baseline_pinv/1000x500' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Exit: `0`

## Median Ranking

| rank | benchmark | median |
| ---: | --- | ---: |
| 1 | `baseline_pinv/1000x500` | `379.32 ms` |
| 2 | `baseline_lstsq/1000x500` | `369.98 ms` |
| 3 | `matmul/1024x1024` | `240.76 ms` |
| 4 | `matmul/768x768` | `164.10 ms` |
| 5 | `lstsq/512x256` | `145.11 ms` |
| 6 | `pinv/512x256` | `130.13 ms` |
| 7 | `baseline_solve/1000x1000` | `112.20 ms` |
| 8 | `matmul/512x512` | `43.773 ms` |
| 9 | `matmul/256x256` | `5.0838 ms` |

## Target Implication

The packed-B panel moved large GEMM out of the top slot. The next profile-backed target is rectangular `fsci_linalg::pinv`, specifically `baseline_pinv/1000x500` at `379.32 ms`.

Source evidence points at pseudoinverse materialization: `pseudo_inverse_from_svd` builds a dense diagonal `sigma_pinv` matrix and multiplies `V * Sigma * U^T`. The next candidate is diagonal-operator elimination by directly scaling `V` columns, preserving threshold/rank/certificate semantics and proving output equivalence by golden SHA.
