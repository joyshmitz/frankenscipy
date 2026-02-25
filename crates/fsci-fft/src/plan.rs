use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::{Normalization, TransformKind};

/// How planning heuristics are produced for a transform key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PlanningStrategy {
    /// Static estimate only; cheapest and deterministic.
    #[default]
    EstimateOnly,
    /// Measure candidate paths and persist chosen plan metadata.
    MeasureAndPersist,
}

/// Admission mode controlling what enters the plan cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CacheAdmissionPolicy {
    Disabled,
    #[default]
    CostWeightedLru,
    AlwaysInsert,
}

/// Stable cache key for FFT planning decisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlanKey {
    pub kind: TransformKind,
    pub shape: Vec<usize>,
    pub axes: Vec<usize>,
    pub normalization: Normalization,
    pub real_input: bool,
}

impl PlanKey {
    #[must_use]
    pub fn new(
        kind: TransformKind,
        shape: Vec<usize>,
        axes: Vec<usize>,
        normalization: Normalization,
        real_input: bool,
    ) -> Self {
        Self {
            kind,
            shape,
            axes,
            normalization,
            real_input,
        }
    }
}

/// Fingerprint proving how a concrete FFT plan was selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanFingerprint {
    pub radix_path: Vec<usize>,
    pub estimated_flops: u64,
    pub scratch_bytes: usize,
}

/// Persistent metadata associated with a cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanMetadata {
    pub key: PlanKey,
    pub fingerprint: PlanFingerprint,
    pub generated_by: PlanningStrategy,
}

/// Control-plane configuration for plan caching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanCacheConfig {
    pub capacity: usize,
    pub max_working_set_bytes: usize,
    pub planning_strategy: PlanningStrategy,
    pub admission_policy: CacheAdmissionPolicy,
}

impl Default for PlanCacheConfig {
    fn default() -> Self {
        Self {
            capacity: 128,
            max_working_set_bytes: 64 * 1024 * 1024,
            planning_strategy: PlanningStrategy::EstimateOnly,
            admission_policy: CacheAdmissionPolicy::CostWeightedLru,
        }
    }
}

/// Storage interface to decouple planning from cache implementation details.
pub trait PlanCacheBackend {
    fn lookup(&self, key: &PlanKey) -> Option<PlanMetadata>;
    fn store(&mut self, metadata: PlanMetadata) -> bool;
    fn config(&self) -> &PlanCacheConfig;
}

static SHARED_PLAN_CACHE: OnceLock<Mutex<HashMap<PlanKey, PlanMetadata>>> = OnceLock::new();

fn shared_cache() -> &'static Mutex<HashMap<PlanKey, PlanMetadata>> {
    SHARED_PLAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[must_use]
pub fn lookup_shared_plan(key: &PlanKey) -> Option<PlanMetadata> {
    shared_cache()
        .lock()
        .ok()
        .and_then(|cache| cache.get(key).cloned())
}

pub fn store_shared_plan(metadata: PlanMetadata) {
    if let Ok(mut cache) = shared_cache().lock() {
        cache.insert(metadata.key.clone(), metadata);
    }
}

#[must_use]
pub fn shared_plan_cache_len() -> usize {
    shared_cache().lock().map_or(0, |cache| cache.len())
}

pub fn clear_shared_plan_cache() {
    if let Ok(mut cache) = shared_cache().lock() {
        cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PlanCacheConfig, PlanFingerprint, PlanKey, PlanMetadata, PlanningStrategy,
        clear_shared_plan_cache, lookup_shared_plan, shared_plan_cache_len, store_shared_plan,
    };
    use crate::{Normalization, TransformKind};

    #[test]
    fn default_cache_config_is_bounded_and_deterministic() {
        let config = PlanCacheConfig::default();
        assert_eq!(config.capacity, 128);
        assert_eq!(config.max_working_set_bytes, 64 * 1024 * 1024);
    }

    #[test]
    fn plan_key_captures_contract_surface() {
        let key = PlanKey::new(
            TransformKind::Fftn,
            vec![32, 32, 16],
            vec![0, 2],
            Normalization::Backward,
            false,
        );
        assert_eq!(key.kind, TransformKind::Fftn);
        assert_eq!(key.axes, vec![0, 2]);
    }

    #[test]
    fn shared_cache_roundtrip_works() {
        clear_shared_plan_cache();
        let key = PlanKey::new(
            TransformKind::Fft,
            vec![64],
            vec![0],
            Normalization::Backward,
            false,
        );
        let metadata = PlanMetadata {
            key: key.clone(),
            fingerprint: PlanFingerprint {
                radix_path: vec![2, 2, 2, 2, 2, 2],
                estimated_flops: 64 * 64,
                scratch_bytes: 64 * 16,
            },
            generated_by: PlanningStrategy::EstimateOnly,
        };
        store_shared_plan(metadata);
        assert!(shared_plan_cache_len() >= 1);
        assert!(lookup_shared_plan(&key).is_some());
    }
}
