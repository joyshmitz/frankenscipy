#![no_main]

use arbitrary::Arbitrary;
use fsci_opt::add;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct AddPropertiesInput {
    a: u64,
    b: u64,
    c: u64,
}

fuzz_target!(|input: AddPropertiesInput| {
    let ab = add(input.a, input.b);
    let ba = add(input.b, input.a);
    if ab != ba {
        panic!("commutativity violation");
    }

    let assoc_left = add(add(input.a, input.b), input.c);
    let assoc_right = add(input.a, add(input.b, input.c));
    if assoc_left != assoc_right {
        panic!("associativity violation");
    }
});
