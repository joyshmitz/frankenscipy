#![no_main]

use fsci_io::{FortranEndian, read_fortran_unformatted, write_fortran_record};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT: usize = 16_384;

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() > MAX_INPUT {
        return;
    }

    // Property 1: must never panic on arbitrary input. Both endians.
    let _ = read_fortran_unformatted(bytes, FortranEndian::Little);
    let _ = read_fortran_unformatted(bytes, FortranEndian::Big);

    // Property 2: round-trip via write_fortran_record. Treat bytes as
    // a single payload, frame it with both endians, and verify the
    // reader recovers the original payload exactly.
    if bytes.len() <= 8192 {
        for endian in [FortranEndian::Little, FortranEndian::Big] {
            let framed = write_fortran_record(bytes, endian);
            let records = read_fortran_unformatted(&framed, endian)
                .expect("framed record must parse");
            assert_eq!(records.len(), 1, "single-frame round-trip yields one record");
            assert_eq!(records[0], bytes, "round-trip byte-equal payload");
        }
    }
});
