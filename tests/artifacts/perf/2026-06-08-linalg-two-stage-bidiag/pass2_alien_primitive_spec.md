# Pass 2 Alien Primitive Spec

Bead: `frankenscipy-8l8r1.52`

## 1. Source and Profile Target

Line ranges are from the live workspace inspected during Pass 2. Source,
`.beads`, and `.skill-loop-progress.md` were read only.

Primary source target:

- `crates/fsci-linalg/src/lib.rs:7376-7439`:
  `golub_kahan_bidiagonal_reduction`. This is the current dense reduction
  kernel to replace experimentally. It clones the input, then for each column
  runs one left Householder, one right Householder, immediate trailing updates,
  and finally materializes `diagonal`, `superdiagonal`, and `bidiagonal`.
- `crates/fsci-linalg/src/lib.rs:7242-7244`:
  `deterministic_thin_svd` calls the reducer and then converts the reduction to
  the deterministic thin SVD.
- `crates/fsci-linalg/src/lib.rs:7283-7290`:
  `public_bidiag_thin_svd_candidate` is the public-route candidate gate for the
  bidiag path.
- `crates/fsci-linalg/src/lib.rs:7331-7369`:
  `public_bidiag_svd_accepts` enforces rank-gap, singular-value ordering, shape,
  and reconstruction tolerance.

Relevant implementation building blocks:

- `crates/fsci-linalg/src/lib.rs:6202-6220`:
  `HouseholderReflector` and `BidiagonalReduction` storage.
- `crates/fsci-linalg/src/lib.rs:6261-6276`:
  `BidiagonalReduction::left_product_transpose` and `right_product` replay
  stored reflectors.
- `crates/fsci-linalg/src/lib.rs:6352-6381`:
  `make_householder_reflector`.
- `crates/fsci-linalg/src/lib.rs:6383-6409`:
  `apply_householder_left`.
- `crates/fsci-linalg/src/lib.rs:6477-6522`:
  `apply_householder_right_with_workspace`.
- `crates/fsci-linalg/src/lib.rs:6526-6587`:
  `apply_bidiag_fused_rank_k_update`, an existing packed-update shape
  `A -= V*Y + X*U`. This is useful as a storage/loop-shape reference only; the
  Pass 3 lever must not be another single-step fusion retry.

Relevant tests/probes:

- `crates/fsci-linalg/src/lib.rs:13239-13246`:
  `bidiag_deterministic_matrix`.
- `crates/fsci-linalg/src/lib.rs:13296-13312`:
  `assert_upper_bidiagonal`.
- `crates/fsci-linalg/src/lib.rs:13337-13394`:
  rowwise reference reducer.
- `crates/fsci-linalg/src/lib.rs:13432-13467`:
  bit-identical reduction comparison.
- `crates/fsci-linalg/src/lib.rs:13469-13495`:
  `bidiag_reduction_digest`.
- `crates/fsci-linalg/src/lib.rs:13838-13862`:
  `bidiag_golub_kahan_golden_payload`.
- `crates/fsci-linalg/src/lib.rs:13865-13870`:
  right-workspace bit proof.
- `crates/fsci-linalg/src/lib.rs:13873-13894`:
  `bidiag_large_reduction_perf_probe`, the 1024x512 reduction benchmark.
- `crates/fsci-linalg/src/lib.rs:14549-14605` and `14791-14860`:
  public route proof and perf probes.

Existing `.52` artifacts:

- `pass1_baseline_contract.md`: baseline contract and admissible family score.
- `bidiag_large_reduction_perf_probe_rch_attempt2.txt`: RCH worker
  `vmi1167313`, `elapsed_ms=414.461569`, digest
  `0x90cdd3f8f71ed2c1`.
- `public_svd_lstsq_pinv_golden_payload_rch.txt`: local fallback payload
  produced the expected public values.
- `public_svd_lstsq_pinv_golden_payload_rch_remote.txt`: remote-required
  public-golden rerun was refused due no admissible RCH worker.
- `br_show_frankenscipy-8l8r1.52.json`: bead explicitly asks for a
  two-stage/packed-panel route and forbids single-step pass fusion, replay
  ordering, dense compact-WY composition, and thread fanout micro-levers.

Baseline anchor:

- Worker: `vmi1167313`.
- Probe: `bidiag_large_reduction_perf_probe`.
- Current time: `414.461569 ms`.
- Current reduction digest: `0x90cdd3f8f71ed2c1`.
- Public golden SHA-256:
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.
- Remote public golden must be rerun before any keep decision.

## 2. Alien Primitive Selected

Selected primitive: sequential communication-avoiding packed-panel band
bidiagonalization, implemented as Stage 1 of a two-stage reducer.

Local graveyard basis:

- `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md:3388-3401`
  describes communication-avoiding dense linear algebra, including panel
  factorization and BLAS-3 dense submatrix updates. The parallel tree portions
  are not selected for this pass.
- `/data/projects/alien_cs_graveyard/high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md:2400-2401`
  reinforces profile-first optimization, opportunity-matrix gates, and graceful
  degradation.

Practical form for this codebase:

1. Add a private large-matrix experimental reducer that works in fixed-width
   panels, initially `panel_width = 16`.
2. For panel `j..j+panel_width`, continue using the existing deterministic
   Householder sign convention from `make_householder_reflector`.
3. Apply each newly generated left/right reflector only to the active panel and
   the near band needed to generate subsequent panel reflectors.
4. Accumulate four packed column-major panels for the far trailing update:
   `V` and `X` shaped `row_count x k`, plus `Y` and `U` shaped `col_count x k`.
5. After the panel is complete, update the far trailing matrix once using the
   GEMM-shaped formula:

   ```text
   A22 -= V * Y^T + X * U^T
   ```

   In implementation terms this is a cache-tiled safe-Rust loop over the
   column-major `DMatrix` backing slice. It must use a fixed loop order and no
   thread fanout.
6. The Stage 1 output is a band-bidiagonal work matrix plus the packed panels
   and reflectors needed for a later deterministic Stage 2 band-to-bidiagonal
   bulge chase.

This is fundamentally different from prior rejected work because it changes the
algorithmic blocking boundary. It is not an index cleanup, not replay cleanup,
not scalar DLABRD retry, not dense compact-WY composition, and not thread
parallelism.

## 3. One-Lever Boundary for Pass 3

Implement exactly this in Pass 3:

- A private Stage 1 packed-panel band reducer for `DMatrix<f64>` in
  `crates/fsci-linalg/src/lib.rs`.
- A small private result struct, for example
  `PackedPanelBidiagonalBandReduction`, storing:
  `rows`, `cols`, `panel_width`, the band work matrix, left/right reflectors,
  and enough packed-panel metadata to replay or verify the Stage 1 transform.
- A fixed safe-Rust packed trailing-update helper with the same mathematical
  shape as `A22 -= V*Y^T + X*U^T`, but generated from actual panel reflectors.
- Unit coverage that the Stage 1 transform reconstructs the original matrix
  through stored reflectors within an explicit tolerance and that it is
  deterministic for fixed input.
- One ignored perf probe for the Stage 1 path, separate from public routing.

Do not include in Pass 3:

- No public route switch in `svd`, `svdvals`, `lstsq`, or `pinv`.
- No change to `deterministic_thin_svd` default behavior.
- No Stage 2 band-to-bidiagonal bulge chase.
- No dense compact-WY product construction.
- No scalar DLABRD panel retry.
- No single-step `golub_kahan_bidiagonal_reduction` fusion or indexing cleanup.
- No thread fanout, rayon, tokio, or async work.
- No RCH keep/reject claim without rerunning the baseline and public golden on
  an admissible remote worker.

The Pass 3 output is therefore a measured experimental primitive, not a kept
public implementation.

## 4. Behavior Proof Obligations

Ordering:

- Stage 1 may change floating-point operation order internally, so it must not
  replace the public route in Pass 3.
- Within the new helper, every loop order must be fixed: panels in increasing
  `j`, reflectors in increasing panel step, columns in increasing order, rows in
  increasing order inside each tile.
- Stored reflectors must replay deterministically. If a later Stage 2 route
  replaces the public path, its exposed singular-value ordering must still pass
  `public_bidiag_svd_accepts`.

Tie-breaking:

- Do not change sign canonicalization, singular ordering, rank thresholding, or
  clustered-spectrum rejection.
- Preserve the public route's tie-gap policy from
  `public_bidiag_svd_accepts`.

Floating point:

- Pass 3 Stage 1 is allowed to differ from the exact
  `0x90cdd3f8f71ed2c1` reduction digest because a true packed-panel update
  changes summation order.
- That difference is acceptable only while the primitive is private and
  opt-in. Public outputs must remain unchanged because the route is not used.
- Required Stage 1 proof: reconstruct the input from stored transforms and the
  band matrix with `max_abs_error <= 1e-8 * max(1, max_abs(input)) * sqrt(cols)`
  on deterministic 128x64 and 1024x512 probes, plus orthogonality checks for
  replayed factors.

RNG:

- Unchanged. The deterministic matrix generator and the reduction path use no
  RNG.

Golden SHA:

- Pass 3 must keep the public payload SHA
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.
- Because the Pass 1 remote-required golden rerun was blocked by RCH admission,
  rerun it on an admissible remote worker before any keep decision.

## 5. Opportunity Score and Target Ratio

Selected primitive score:

| Candidate | Impact | Confidence | Effort | Score |
|---|---:|---:|---:|---:|
| Sequential packed-panel Stage 1 band bidiagonalization | 5 | 3 | 4 | 3.75 |

This clears the `Score >= 2.0` gate.

Target ratios:

- Eventual public-route keep target:
  `bidiag_large_reduction_perf_probe <= 331.569255 ms` on the same worker as
  the 414.461569 ms baseline, i.e. at least `1.25x`.
- Pass 3 primitive target:
  Stage 1 packed-panel probe must show a credible path to at least `1.20x`
  total reduction improvement on 1024x512, or the Stage 2 follow-up should not
  be opened from this primitive.
- Public behavior target: public golden SHA unchanged; no tolerance broadening
  in public acceptance gates.

## 6. Rejection Criteria and Rollback Condition

Reject the Pass 3 lever if any of these occur:

- It requires routing public APIs before Stage 2 exists.
- It changes source outside `crates/fsci-linalg/src/lib.rs` or creates new
  source files.
- It reintroduces a banned family: single-step fusion, replay/index cleanup,
  scalar DLABRD retry, dense compact-WY composition, or thread fanout.
- Stage 1 reconstruction or orthogonality proof misses the stated tolerance.
- The primitive is nondeterministic for fixed input.
- Public golden SHA changes before an intentional migration contract exists.
- Same-worker perf evidence cannot support at least a credible `1.20x` route to
  the total reduction target.

Rollback condition:

- If any rejection criterion triggers after Pass 3 source edits, revert the
  single Pass 3 implementation commit and leave this Pass 2 artifact as the
  no-keep rationale. Do not delete files as rollback.
