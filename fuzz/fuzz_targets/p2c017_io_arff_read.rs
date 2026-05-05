#![no_main]

use fsci_io::read_arff;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT: usize = 16_384;

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() > MAX_INPUT {
        return;
    }
    // ARFF is text; reject non-UTF-8 inputs cheaply rather than panicking.
    let Ok(text) = std::str::from_utf8(bytes) else {
        return;
    };

    // Property: read_arff must not panic on any UTF-8 input.
    let _ = read_arff(text);
});
