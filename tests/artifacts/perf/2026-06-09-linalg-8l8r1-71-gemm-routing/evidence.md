# frankenscipy-8l8r1.72 evidence

## Target

Profile-backed medium GEMM dispatch under the no-gaps linalg parent.

Note: the artifact directory name contains `8l8r1-71` because this pass started
before `origin/main` advanced and took `.71` for the eigvalsh route bead. The
tracker record for this GEMM keep is `frankenscipy-8l8r1.72`.

Post-keep reprofile after `frankenscipy-mifdz` showed the hot rows:

- `matmul/512x512`: 43.094 ms median on `vmi1227854`
- `matmul/768x768`: 143.24 ms median on `vmi1227854`
- `matmul/1024x1024`: 70.832 ms median on `vmi1227854`
- `eigh_dense/512x512`: 107.50 ms median on `vmi1227854`
- `baseline_solve/2000x2000`: 508.10 ms median on `vmi1227854`

Existing parent notes identified the next non-repeat GEMM primitive as a B-panel/column-panel packed route on top of the register microkernel. The shipped flat-workspace GEMM already used a packed B panel and parallel row slices, but public `matmul` only dispatched to it at all dimensions >=1024.

## Lever

One lever: lower the public flat-workspace route gate from `1024` to `512` and factor the dimension predicate into `matmul_flat_workspace_candidate_dims`.

No other algorithmic behavior changed.

## Matched RCH benchmark

Both matched baseline and after were run on `ovh-a` with:

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked matmul
```

| row | baseline median | after median | speedup |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | 4.6852 ms | 4.6485 ms | 1.008x |
| `matmul/512x512` | 38.139 ms | 9.6456 ms | 3.953x |
| `matmul/768x768` | 127.24 ms | 30.167 ms | 4.218x |
| `matmul/1024x1024` | 51.322 ms | 50.107 ms | 1.024x |

Score: Impact 5 x Confidence 5 / Effort 2 = 12.5. Keep.

## Isomorphism proof

- Ordering/tie-breaking: not applicable to GEMM; row/column output ordering is unchanged.
- Floating-point: unchanged per output element. The new route uses the existing flat-workspace helper, whose per-cell reduction still accumulates `k` in monotonic `0..ka` order.
- RNG: none.
- Public route proof: `matmul_medium_flat_workspace_route_golden_digest` compares the public 512x512 route against direct `matmul_flat_workspace` output bit-for-bit and freezes digest `0x5fd37bf053d54fb0`.
- Existing small/tile proofs still pass: `matmul_ikj_is_bit_identical_to_naive_ijk`, `matmul_microkernel_is_bit_identical_to_flat_ikj`, `matmul_microkernel_golden_digest`, `matmul_flat_workspace_is_bit_identical_to_naive_ijk`, and `matmul_flat_compute_rows_row_split_is_bit_identical`.

## Validation

- RCH ignored route proof, release: passed.
- RCH non-ignored `matmul` unit proofs: passed.
- RCH `cargo check -p fsci-linalg --lib --locked`: passed.
- RCH `cargo clippy -p fsci-linalg --lib --locked --no-deps -- -D warnings`: passed.
- `cargo fmt -p fsci-linalg --check`: passed.
- `ubs crates/fsci-linalg/src/lib.rs`: 0 critical findings; remaining findings are pre-existing file-wide warning inventory.
