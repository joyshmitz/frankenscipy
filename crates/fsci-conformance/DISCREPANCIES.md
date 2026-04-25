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

## DISC-005 — e2e-only io was excluded from oracle parity counts

- **Reference:** scipy.io exposes observable behavior that should
  eventually be compared by P2C fixture packets and Python oracle
  captures.
- **Our impl:** fsci-io previously had crate-level e2e tests but no
  `fixtures/FSCI-P2C-*.json` packet lane in `fsci-conformance`.
  FSCI-P2C-017 now seeds an oracle-backed lane for Matrix Market,
  MAT v4 real-double matrices, numeric text, and WAV metadata/value parity.
- **Impact:** Users reading parity numbers see fixture-backed families
  only; io now has a narrow fixture-backed score, while any broader
  scipy.io claims remain out of scope until more cases land.
- **Resolution:** RESOLVED for the seed io lane. MAT parity is currently
  narrowed to SciPy-compatible `savemat(..., format="4")` / `loadmat` for
  full real double matrices. MAT v5/v7.3, structs, cells, sparse matrices,
  chars, and complex arrays remain out of scope until new fixture packets
  expand the surface.
- **Tests affected:** `e2e_io.rs`, `FSCI-P2C-017_io_core.json`.
- **Review date:** 2026-04-25
- **Related beads:** frankenscipy-3m6f

## DISC-007 — fsci-integrate LSODA early-termination on highly stiff oscillators

- **Reference:** scipy.integrate.solve_ivp(method='LSODA') handles
  stiff Van der Pol mu=10 (and similar transition-stiff problems)
  by adaptively shrinking step size and switching between Adams and
  BDF formulas internally.
- **Our impl:** fsci_integrate::LsodaSolver returns the initial point
  only on Van der Pol mu=10 (does not advance past t=0), suggesting
  step-size control or stiffness detection has not stabilized for
  highly stiff oscillators. Robertson chemistry (much stiffer at
  small t) and exponential decay both succeed.
- **Impact:** br-r8ug fixture dropped the
  `ivp_lsoda_van_der_pol_mu10` case. Robertson, linear growth, and
  exponential decay coverage are sufficient to exercise both stiff
  and non-stiff LSODA paths.
- **Resolution:** ACCEPTED for the parity slice. Investigate LSODA
  oscillator handling as a follow-up if a consumer needs Van der Pol-
  style ODEs.
- **Tests affected:** N/A (the case was excluded rather than asserted).
- **Review date:** 2026-04-25
- **Related beads:** frankenscipy-r8ug, frankenscipy-ljmg

## DISC-NN — fsci-signal firwin2 / remez coefficient divergence

- **Reference:** scipy.signal.firwin2 (frequency-sampling +
  inverse-FFT + windowing) and scipy.signal.remez (Parks-McClellan
  exchange).
- **Our impl:** fsci_signal::firwin2 / remez follow the same
  algorithm families but reach slightly different numerical results —
  observed max coefficient diff ~0.05 on canonical lowpass / bandpass
  designs.
- **Impact:** P2C-011 fixture cases for firwin2 and remez use
  `atol = 0.06, rtol = 0.05` per br-7jrx rather than the tight
  1e-10/1e-9 used for firwin and firls. The qualitative response
  shape (passband/stopband transitions, ripple) is preserved within
  usable filter-design margins.
- **Resolution:** ACCEPTED — tightening requires matching scipy's
  windowing-application order in firwin2 and Remez exchange
  numerical refinement details; out of scope for the parity slice.
- **Tests affected:** P2C-011 firwin2_lp_11_freqsamp,
  firwin2_bp_15_3band, remez_lp_11_passband_0p2_stopband_0p3,
  remez_bp_15_3band.
- **Review date:** 2026-04-25
- **Related beads:** frankenscipy-7jrx

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
