# Closeout

Bead: frankenscipy-6w1mf

Subsystem: fsci-sparse

Profile-backed target:
RCH Criterion ranked sparse_arithmetic/10000x10000_d0_add/10000 as the dominant fsci-sparse row at mean 59.649 ms. The focused baseline for the selected row was 37.795 ms mean.

Lever:
Add a canonical sorted+deduplicated CSR row-stream merge fast path for add_csr and sub_csr, while retaining the existing COO/BTreeMap path for noncanonical inputs.

Performance:
sparse_arithmetic/10000x10000_d0_add/10000 mean 37.795 ms -> 1.6915 ms, -95.5%.

Behavior:
Golden CSR output is recorded in golden_csr_add_canonical.txt. Shape errors, noncanonical fallback, row/column ordering, explicit zero elision, floating-point operation order, tie-breaking, and RNG behavior are preserved.

Validation:
Focused remote test passed, full fsci-sparse lib tests passed 304/304, adjusted remote clippy passed with only the known lowercase SciPy alias lint allowed, scoped rustfmt passed, UBS exit 0.
