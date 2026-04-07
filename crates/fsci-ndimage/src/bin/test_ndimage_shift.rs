use fsci_ndimage::{BoundaryMode, NdArray, shift};
fn main() {
    let input = NdArray::new(vec![0.0, 10.0, 20.0, 30.0], vec![4]).unwrap();
    let cubic_shift = shift(&input, &[0.5], 3, BoundaryMode::Nearest, 0.0).unwrap();
    println!("{:?}", cubic_shift.data);
}
