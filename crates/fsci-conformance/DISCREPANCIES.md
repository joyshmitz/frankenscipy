# Conformance Divergences (DISCREPANCIES)

Every intentional divergence from the reference SciPy/NumPy behavior
lives here. Every entry has a DISC-NNN ID, a status
(ACCEPTED / INVESTIGATING / WILL-FIX), the tests affected, and the last
review date. Stale divergences (>= 6 months since review) are flagged
at the bottom for re-evaluation.

> **Rule:** Tests that exercise accepted divergences must use
> `#[ignore]` or a dedicated XFAIL marker — never plain `#[test]` that
> silently passes on the divergent side.

## DISC-001 — rand 0.10 numerical drift in stats.rvs

- **Reference:** scipy.stats.<dist>.rvs(seed=42) on numpy's PCG64.
- **Our impl:** fsci-stats uses the rand 0.10 StdRng (ChaCha12).
- **Impact:** Sample streams differ bit-for-bit; mean/variance converge
  but per-sample values never match scipy.
- **Resolution:** ACCEPTED — porting StdRng to PCG64 is a separate
  project. Statistical invariants (sample mean within tolerance)
  suffice for conformance; sample-by-sample parity is not a MUST.
- **Tests affected:** `fsci-stats::tests::rvs_*` (seeded under br-otdp).
- **Review date:** 2026-04-23

## DISC-002 — Complex Matrix Market rejection

- **Reference:** scipy.io.mmread returns a complex-valued sparse matrix
  when the header declares `field complex`.
- **Our impl:** fsci-io returns an error on `field complex`.
- **Impact:** Files with complex matrices cannot round-trip.
- **Resolution:** ACCEPTED for Phase 2 — fsci-sparse does not yet
  expose a complex CSR type. Re-enable once sparse is complex-capable.
- **Tests affected:** `e2e_io::mm_read_complex_rejected`.
- **Review date:** 2026-04-23
- **Related beads:** frankenscipy-r5tv

## DISC-003 — cubature bounds dimensionality guard

- **Reference:** scipy.integrate.cubature is tolerant of mismatched
  lower/upper bound lengths when ndim > 1 (broadcast-style).
- **Our impl:** fsci_integrate::cubature hard-errors on mismatched
  bound dimensionality to prevent silent misintegration.
- **Resolution:** ACCEPTED — stricter validation is safer than scipy's
  permissive broadcast, and no real workload we've seen relies on it.
- **Tests affected:** fixture case `cubature_rejects_mismatched_bounds`
  (P2C-013, Hardened mode).
- **Review date:** 2026-04-23

## DISC-004 — Cluster labels exact-equality → permutation-invariant

- **Reference:** scipy.cluster.hierarchy.fcluster / leaves_list emit
  labels in a specific numeric order determined by their internal
  linkage walk.
- **Our impl:** fsci_cluster produces the same partition but no
  contract guarantees identical numeric IDs. The conformance
  comparator now canonicalizes by first-occurrence order
  (br-7eaq), so partitions match regardless of label assignment.
- **Resolution:** ACCEPTED — label numeric value is not semantic.
  Fixture cases that rely on exact label IDs opt in via
  `expected.deterministic_labels: true`.
- **Tests affected:** P2C-009 cluster fixture, all `kind: "labels"` cases.
- **Review date:** 2026-04-23
- **Related beads:** frankenscipy-7eaq

## DISC-005 — e2e-only families are excluded from oracle parity counts

- **Reference:** scipy.interpolate / scipy.ndimage / scipy.io expose
  observable behavior that should eventually be compared by P2C fixture
  packets and Python oracle captures.
- **Our impl:** fsci-interpolate, fsci-ndimage, and fsci-io currently
  have crate-level e2e tests but no `fixtures/FSCI-P2C-*.json` packet
  lane in `fsci-conformance`, so the dashboard must not report
  oracle-backed parity percentages for those families yet.
- **Impact:** Users reading parity numbers see fixture-backed families
  only; e2e-only coverage remains real test coverage but not a SciPy
  oracle-backed parity score.
- **Resolution:** ACCEPTED for the current dashboard. Promote each
  family into a P2C fixture + oracle lane before adding it to the
  oracle-backed score.
- **Tests affected:** `e2e_interpolate.rs`, `e2e_ndimage.rs`,
  `e2e_io.rs`.
- **Review date:** 2026-04-24
- **Related beads:** frankenscipy-di9p

## Stale / needs re-review

_None yet — first revision._

---

## Adding a new divergence

1. Pick the next DISC-NNN number.
2. Fill all fields (reference / our impl / impact / resolution /
   tests affected / review date).
3. Link back from the affected test or fixture case by the DISC ID in
   a comment.
4. Never remove an entry; mark it `RESOLVED` with the commit SHA when
   the divergence disappears.
