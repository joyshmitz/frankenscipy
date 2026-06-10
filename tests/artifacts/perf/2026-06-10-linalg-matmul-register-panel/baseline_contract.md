# frankenscipy-cckrw matmul baseline contract

Date: 2026-06-10
Worktree: `/data/projects/.scratch/frankenscipy-blackthrush-matmul-20260610`
Revision: `origin/main` / `a923fb66509ec0869f82b09aab6709ac90503b2c`
Bead: `frankenscipy-cckrw` in progress, parent `frankenscipy-8l8r1`
Scope: documentation and contract only. No production code, `.beads`, or `.skill-loop-progress.md` edits in this pass.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo bench -j 1 -p fsci-linalg --bench linalg_bench --locked -- matmul
```

Artifact:

```text
tests/artifacts/perf/2026-06-10-linalg-matmul-register-panel/baseline_matmul_criterion_rch.txt
sha256: 62d9ffc74fa324d0f1fc8d26684227d0cd8fe63781c050eaca65969cc02fae89
worker: vmi1227854
```

Criterion estimates in the baseline artifact:

| Benchmark | Low | Mean | High | Route |
| --- | ---: | ---: | ---: | --- |
| `matmul/256x256` | 4.7710 ms | 4.9797 ms | 5.1451 ms | public register micro-kernel; below `MATMUL_FLAT_WORKSPACE_MIN_DIM` |
| `matmul/512x512` | 27.717 ms | 31.611 ms | 37.148 ms | `matmul_flat_workspace` |
| `matmul/768x768` | 83.827 ms | 91.072 ms | 95.094 ms | `matmul_flat_workspace` |
| `matmul/1024x1024` | 252.57 ms | 281.04 ms | 310.53 ms | `matmul_flat_workspace` |

The benchmark source fixes `MATMUL_SIZES = &[256, 512, 768, 1024]`, uses deterministic square matrices, sets `sample_size(10)`, and measures each matmul group for 5 seconds.

## Hotspot and Route

The public `matmul` dispatch enters `matmul_flat_workspace` only when all dimensions are at least 512 and both input matrices are rectangular. This makes 512x512, 768x768, and 1024x1024 the affected benchmark rows; 256x256 is a sentinel for the smaller public register micro-kernel and should not be used as evidence that the flat-workspace candidate worked.

Current flat-workspace staging:

1. Flatten `A` into `a_flat`.
2. Flatten `B` into `b_flat`.
3. Pack every complete 8-column B panel from `b_flat` into `packed_b`.
4. Compute full 8- or 16-column panels from `packed_b`.
5. Compute scalar column tails from `b_flat`.

The candidate for a later pass is exactly one lever: remove the redundant `b_flat` materialization in `matmul_flat_workspace`. Pack full B panels directly from the original `b[k][j0..j0 + NR]` rows, and use the original `b[k][j]` row values for scalar tails. Keep `a_flat`, `packed_b`, row splitting, the dispatch gate, and the public API unchanged.

## Alien-Graveyard Lineage

Mapped lineage:

- `alien_cs_graveyard` section 9.6, Communication-Avoiding Algorithms: minimize data movement for linear algebra and keep dense submatrix work in BLAS-3 style kernels.
- Dense-kernel sub-lineage: tiled/register-panel matrix multiply is the inner kernel that communication-avoiding LU/QR/SVD families rely on.

This candidate is intentionally not a new numerical algorithm. It is a narrow data-movement cleanup inside the existing register/panel GEMM route: avoid copying B into one row-major flat buffer only to immediately repack complete panels. It preserves the current dense-kernel arithmetic while reducing staging traffic and allocation pressure.

## Recommendation Contract

Change:
Remove `b_flat` from `matmul_flat_workspace`; direct-pack complete B panels from `b`, and read scalar tails from `b`.

Hotspot evidence:
The clean RCH baseline shows the affected flat-workspace rows are the dominant matmul costs in this bead: 31.611 ms, 91.072 ms, and 281.04 ms means for 512/768/1024. Source inspection confirms those rows enter `matmul_flat_workspace`, where `B` is currently copied once into `b_flat` and then copied again into `packed_b`.

Mapped graveyard sections:
Communication-Avoiding Algorithms section 9.6; BLAS-3 dense-kernel inner-loop discipline.

EV score:

```text
EV = (Impact * Confidence * Reuse) / (Effort * AdoptionFriction)
Impact = 2      # O(n^2) staging removed from an O(n^3) kernel; modest but targeted
Confidence = 4  # source route and proof obligation are straightforward
Reuse = 3       # applies to public 512+ matmul and internal users of matmul_flat_workspace
Effort = 2      # one helper signature/body change plus tests
AdoptionFriction = 1 # no dependency, API, unsafe, or behavior surface change
EV = 12.0
```

Extreme-optimization opportunity score:

```text
Impact * Confidence / Effort = 2 * 4 / 2 = 4.0
```

Priority tier:
A. The lever is small, proofable, and directly on the measured route, but it is not a deep algorithmic replacement.

Adoption wedge:
Private helper-only implementation behind the existing `matmul` dispatch gate. No new public API, no new dependency, no unsafe code, no compatibility shim.

Budgeted mode:
One production lever in the later implementation pass. No second optimization in the same pass. If the direct-pack implementation needs new abstractions beyond a small helper/signature adjustment, stop and keep the current `b_flat` design until a separate deeper bead exists.

Expected-loss model:

```text
States:
- proof_clean_win: bit proofs pass and same-worker affected-row geomean improves
- proof_clean_noise: bit proofs pass but same-worker performance is within noise
- proof_clean_regression: bit proofs pass but any affected row regresses beyond the gate
- proof_fail: any bit-identical proof or golden digest changes

Actions:
- keep_candidate
- reject_candidate_restore_b_flat
- route_deeper_to_panel/blocking/cache work

Loss:
- proof_fail: infinite; reject immediately
- proof_clean_regression: high; reject
- proof_clean_noise: medium; reject and route deeper
- proof_clean_win: low; keep only with same-worker evidence
```

Calibration and fallback trigger:
Keep only if a post-change run uses the same command on `vmi1227854` or another explicitly comparable same-worker rerun pair, the 512/768/1024 geomean mean improves by at least 3%, no affected size regresses by more than 2%, and 256x256 does not regress by more than 2% as the unaffected sentinel. Cross-worker results are routing evidence only. Any proof failure, golden digest drift, panic, ragged-shape behavior change, or build/clippy regression falls back to the current `b_flat` staging.

Isomorphism proof plan:

- Ordering: N/A for observable row/column ordering; output shape and row order are unchanged.
- Tie-breaking: N/A.
- RNG: N/A; deterministic tests and benchmarks keep the same matrix generators.
- Floating point: must remain bit-identical. For each output `c[i][j]`, the compute kernel must still accumulate `k` in monotonic `0..ka` order and execute the same multiply/add sequence. Direct packing may change where each B scalar is loaded from, but not the sequence of values loaded for a fixed `(k, j)`.
- Packed-panel invariant: for each full panel `jb` and inner index `k`, `packed_b[jb * ka * NR + k * NR + dj]` must equal the pre-change `b_flat[k * n + jb * NR + dj]`, which is the same scalar as `b[k][jb * NR + dj]`.
- Scalar-tail invariant: for any tail `j`, the pre-change scalar tail load `b_flat[k * n + j]` must be replaced only by `b[k][j]`.
- Threading invariant: row chunks remain disjoint; only B staging changes.
- Golden digest: existing matmul tests must remain unchanged, especially the medium flat-workspace route digest.

p50/p95/p99 before/after target:
This artifact records Criterion low/mean/high estimates, not p50/p95/p99. For this bead, use the recorded Criterion means and confidence intervals as the baseline comparator. Do not label future Criterion low/mean/high rows as p50/p95/p99 unless a separate percentile capture is added.

Primary failure risk and countermeasure:

- Risk: off-by-one or panel-index drift while direct-packing B, especially at `n % NR != 0`.
- Countermeasure: prove full panels and scalar tails independently through the existing bit-identical helper tests, including row-split coverage with non-multiple-of-8 `n`.
- Risk: accidental semantic broadening for ragged B rows.
- Countermeasure: keep the existing public rectangularity gate before entering `matmul_flat_workspace`; no new fallback or compatibility path.
- Risk: constants kill the win because the removed O(n^2) copy is small next to O(n^3) compute.
- Countermeasure: same-worker benchmark gate; reject if the affected-row geomean does not clear the threshold.

Repro artifact pack:

- Existing baseline artifact: `baseline_matmul_criterion_rch.txt`
- Existing baseline SHA-256: `62d9ffc74fa324d0f1fc8d26684227d0cd8fe63781c050eaca65969cc02fae89`
- Required later candidate artifact: `candidate_matmul_direct_b_pack_criterion_rch.txt`
- Required later candidate summary: include worker, command, git revision, artifact SHA-256, and affected-row delta table.
- Legal/provenance: no external implementation code; high-level lineage only from communication-avoiding / BLAS-3 dense-kernel literature.

Primary paper status:
Hypothesis only for this pass. The later candidate does not depend on implementing a paper algorithm; it uses the communication-avoiding lineage as a data-movement design constraint.

Interference test status:
Not required; this is not a composition of adaptive controllers or concurrent policies. The row-split bit-identity test is still required because the existing kernel can split rows across threads.

Demo linkage:
N/A; private numerical kernel.

Rollback:
If the later implementation is committed and fails any gate, revert that single candidate commit. Until then, fallback is simply to leave `b_flat` materialization in place.

Baseline comparator:
The baseline comparator is `origin/main` at `a923fb66509ec0869f82b09aab6709ac90503b2c` with the baseline artifact and SHA-256 above.

## Exact Verification Commands For Later Pass

Do not run these in this documentation-only pass. Run them after the candidate code change.

Baseline artifact integrity:

```bash
printf '%s  %s\n' \
  62d9ffc74fa324d0f1fc8d26684227d0cd8fe63781c050eaca65969cc02fae89 \
  tests/artifacts/perf/2026-06-10-linalg-matmul-register-panel/baseline_matmul_criterion_rch.txt \
  | sha256sum -c -
```

Focused matmul proof tests:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --locked matmul_flat_workspace_is_bit_identical_to_naive_ijk -- --nocapture
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --locked matmul_flat_compute_rows_row_split_is_bit_identical -- --nocapture
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --locked matmul_microkernel_golden_digest -- --nocapture
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --release --locked matmul_medium_flat_workspace_route_golden_digest -- --ignored --nocapture
```

Crate-level quality gates:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo check -j 1 -p fsci-linalg --all-targets --locked
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo clippy -j 1 -p fsci-linalg --all-targets --locked -- -D warnings
cargo fmt --check
```

Post-change same-command benchmark:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo bench -j 1 -p fsci-linalg --bench linalg_bench --locked -- matmul \
  2>&1 | tee tests/artifacts/perf/2026-06-10-linalg-matmul-register-panel/candidate_matmul_direct_b_pack_criterion_rch.txt
sha256sum tests/artifacts/perf/2026-06-10-linalg-matmul-register-panel/candidate_matmul_direct_b_pack_criterion_rch.txt
```

Acceptance delta:

- Compare only same-worker or explicitly paired reruns.
- Compute affected-row geomean over 512x512, 768x768, and 1024x1024 means.
- Treat 256x256 as an unaffected sentinel.
- Keep only if the calibration gate above passes and all proof commands are green.
