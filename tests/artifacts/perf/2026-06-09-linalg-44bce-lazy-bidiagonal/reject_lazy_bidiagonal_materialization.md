## frankenscipy-44bce: lazy dense-bidiagonal materialization rejected

### Target

- Profile-backed residual: deterministic thin SVD `512x512` reduction stage remains the largest stage after V replay.
- Candidate: remove eager dense `rows x cols` bidiagonal matrix materialization from `BidiagonalReduction`; keep diagonal, superdiagonal, and reflectors, and materialize the dense bidiagonal only in private tests.
- Primitive class: data-layout/materialization elision.

### Isomorphism

- Ordering preserved: yes. Golub-Kahan reflector generation and update loops are untouched.
- Tie-breaking unchanged: yes. SVD backend, singular ordering, rank policy, and sign policy are untouched.
- Floating-point: intended identical for public SVD; removed work is private dense debug/test materialization after the reduction values are already computed.
- RNG seeds: N/A.
- Golden/output proof: not promoted to full golden validation because same-worker perf evidence was mixed.

### Same-worker evidence

Positive vmi sample:

- Baseline artifact: `baseline_stage_vmi1227854_rch.txt`
- After artifact: `after_stage_breakdown_rch.txt`
- Worker: `vmi1227854`
- `worker_count`: `10 -> 10`
- `reduction_ms`: `130.747 -> 101.706`
- Speedup: `1.285536x`

Negative ovh sample:

- Baseline artifact: `baseline_stage_ovh_a_repeat_rch.txt`
- After artifact: `after_stage_breakdown_vmi_repeat_rch.txt` (RCH selected `ovh-a`)
- Worker: `ovh-a`
- `worker_count`: `16 -> 16`
- `reduction_ms`: `137.296 -> 146.014`
- Speed ratio: `0.940293x`

### Decision

Reject. The candidate is behavior-plausible and one same-worker sample improved, but the matching `ovh-a` pair regressed. Confidence is too low for the campaign keep gate, so no source is retained.

Next route: skip materialization-only and thread-fanout families. The next pass needs a different numerical primitive with a stronger expected ratio: bidiagonal backend algorithm replacement, cache-blocked reflector storage/replay that changes memory traffic materially, or a new child bead for a communication-avoiding reduction design with a dedicated same-run A/B probe.
