# Closeout

Bead: frankenscipy-hip0w

Subsystem: fsci-arrayapi

Profile-backed target:
RCH Criterion ranked arrayapi_broadcast/promote_and_broadcast/10000 as the dominant fsci-arrayapi row at mean 232.06 us in the broad profile. Focused same-worker baseline for the selected row was 925.90 us mean.

Lever:
Specialize rank-2 singleton-axis broadcasts in CoreArrayBackend::broadcast_to with direct copy loops for already-shaped, scalar-2d, row-vector, and column-vector cases.

Performance:
arrayapi_broadcast/promote_and_broadcast/10000 mean 925.90 us -> 752.40 us, -18.7%.

Behavior:
Golden row-major outputs are recorded in golden_rank2_broadcast_values.txt. The code copies ScalarValue entries and preserves dtype, shape, MemoryOrder, ordering, tie-breaking, floating-point values, and RNG behavior.

Validation:
Focused remote test passed, full fsci-arrayapi lib tests passed 54/54, remote clippy passed, UBS exit 0. Package fmt is blocked by pre-existing formatting drift in crates/fsci-arrayapi/src/creation.rs.
