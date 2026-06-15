# frankenscipy-psn7x.1 Evidence

## Target

- Bead: `frankenscipy-psn7x.1`
- Kernel: native symmetric-eigh Householder trailing rank-2 update
- Worker: RCH `ovh-a`
- Lever: replace iterator-adapter hot loops with explicit index loops and fuse the `p *= tau` pass with `v dot p`, preserving the column-major accumulation order and the scalar lower-triangle update formula.

## Baseline

- `baseline_public_eigh_route_rch.txt`
  - 400x400 public `eigh`: `44.115212 ms`, digest `0x4b8334c92ce624eb`
  - 800x800 public `eigh`: `231.176244 ms`, digest `0xad8a7e5fa1980bfb`
  - 1200x1200 public `eigh`: `694.001290 ms`, digest `0x181b3486089d0e4a`
- `baseline_native_vs_nalgebra_rch.txt`
  - 400x400 native: `35.1 ms`
  - 800x800 native: `231.3 ms`
  - 1200x1200 native: `722.4 ms`

## Proof

- `after_rank2_column_update_bits_rch.txt`: `symmetric_rank2_column_update_matches_rowwise_bits` passed, proving the touched kernel remains bit-identical to the rowwise reference fixture.
- `after_eigh_behavior_rch.txt`: 16 `eigh` behavior tests passed, including native correctness and `eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a`.
- Public route digests stayed fixed for all measured shapes after the lever:
  - 400x400: `0x4b8334c92ce624eb`
  - 800x800: `0xad8a7e5fa1980bfb`
  - 1200x1200: `0x181b3486089d0e4a`

## Rebench

- `after_public_eigh_route_rch.txt`
  - 400x400 public `eigh`: `46.615584 ms` (below the native threshold; small nalgebra-route noise)
  - 800x800 public `eigh`: `197.849506 ms` (`1.168x` faster than baseline)
  - 1200x1200 public `eigh`: `596.825623 ms` (`1.163x` faster than baseline)
- `after_native_vs_nalgebra_rch.txt`
  - 400x400 native: `33.2 ms` (`1.057x` faster than baseline)
  - 800x800 native: `192.5 ms` (`1.202x` faster than baseline)
  - 1200x1200 native: `592.5 ms` (`1.219x` faster than baseline)

## Gates

- `cargo fmt -p fsci-linalg -- --check`: passed.
- `check_fsci_linalg_lib_rch.txt`: `cargo check -j 1 -p fsci-linalg --lib --locked` passed.
- `clippy_fsci_linalg_lib_rch.txt`: first clippy run failed on the deliberate index-loop spelling.
- `clippy_fsci_linalg_lib_rch_retry.txt`: `cargo clippy -j 1 -p fsci-linalg --lib --no-deps --locked -- -D warnings` passed after the allow was scoped to the hot kernel.
- `final_integrated_check_fsci_linalg_lib_rch.txt`: after cherry-picking onto current `origin/main`, `cargo check -j 1 -p fsci-linalg --lib --locked` passed.
- `ubs crates/fsci-linalg/src/lib.rs`: completed with `Critical issues: 0`; warnings were broad pre-existing file-wide inventory.

## Score

- Impact: `2.0` (large native `eigh` route improved by ~16-22% on same-worker probes)
- Confidence: `4.0` (bitwise kernel proof, public golden digest, public route digests, and same-worker timing agree)
- Effort: `1.0`
- Score: `8.0`
- Verdict: KEEP
