#![no_main]

use arbitrary::Arbitrary;
use fsci_datasets::{
    Dataset, DatasetDType, canonical_fixtures, clear_cache, download_all, load_fixture,
};
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct DatasetsInput {
    selector: u8,
    probe_a: u32,
    probe_b: u32,
}

fn assert_u8_probe_is_addressable(data: &[u8], first: usize, second: usize) {
    assert!(data.get(first).or_else(|| data.get(second)).is_some());
}

fn assert_f64_finite(data: &[f64], first: usize, second: usize) {
    if let Some(value) = data.get(first).or_else(|| data.get(second)) {
        assert!(value.is_finite());
    }
}

fuzz_target!(|input: DatasetsInput| {
    let fixtures = canonical_fixtures();
    assert_eq!(fixtures.len(), 10);
    let fixture = fixtures[usize::from(input.selector) % fixtures.len()];
    let loaded = match load_fixture(fixture.name) {
        Ok(loaded) => loaded,
        Err(error) => {
            assert_eq!(error.to_string(), "registered fixture should load");
            return;
        }
    };

    assert_eq!(loaded.name(), fixture.name);
    assert_eq!(loaded.dtype(), fixture.dtype);
    assert_eq!(loaded.shape(), fixture.shape);
    assert_eq!(loaded.element_count(), fixture.element_count);

    let first = input.probe_a as usize % fixture.element_count.max(1);
    let second = input.probe_b as usize % fixture.element_count.max(1);
    match loaded {
        Dataset::ImageU8(image) => {
            assert_eq!(image.dtype(), DatasetDType::U8);
            assert_u8_probe_is_addressable(image.data(), first, second);
            assert!(image.channels() == 1 || image.channels() == 3);
        }
        Dataset::SignalF64(signal) => {
            assert_eq!(signal.dtype(), DatasetDType::F64);
            assert_f64_finite(signal.data(), first, second);
            assert_eq!(signal.sample_rate_hz(), 360.0);
        }
    }

    assert_eq!(download_all().downloaded, 0);
    assert_eq!(clear_cache().removed, 0);
});
