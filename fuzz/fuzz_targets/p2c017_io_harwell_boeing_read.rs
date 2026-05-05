#![no_main]

use fsci_io::read_harwell_boeing;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT: usize = 16_384;

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() > MAX_INPUT {
        return;
    }
    // Harwell-Boeing is text. Reject non-UTF-8 cheaply.
    let Ok(text) = std::str::from_utf8(bytes) else {
        return;
    };
    // Property: read_harwell_boeing must not panic on any UTF-8 input.
    let _ = read_harwell_boeing(text);
});
