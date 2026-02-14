#![no_main]

use arbitrary::Arbitrary;
use fsci_opt::add;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct AddInput {
    left: u64,
    right: u64,
}

fuzz_target!(|input: AddInput| {
    let _ = add(input.left, input.right);
});
