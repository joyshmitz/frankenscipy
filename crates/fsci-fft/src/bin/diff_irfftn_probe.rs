use fsci_fft::{irfft2, rfft2, Complex64, FftOptions};
fn main() {
    let o = FftOptions::default();
    // real 3x4 -> rfft2 -> (3,3) complex half-spectrum
    let sig2: Vec<f64> = (0..12).map(|k| { let t=k as f64; (0.5*t).cos()+0.2*t-0.03*t*t+0.9 }).collect();
    let spec = rfft2(&sig2, (3,4), &o).unwrap();
    for (i,c) in spec.iter().enumerate() { println!("rfft2,{i},re,{:.17e}", c.0); println!("rfft2,{i},im,{:.17e}", c.1); }
    // irfft2 of that half-spectrum back to (3,4)
    let back = irfft2(&spec, (3,4), &o).unwrap();
    for (i,&x) in back.iter().enumerate() { println!("irfft2,{i},re,{:.17e}", x); }
    // irfft2 of an arbitrary (conj) spectrum, shape (3,4)
    let spec2: Vec<Complex64> = (0..9).map(|k| { let t=k as f64; ((0.4*t).cos()+0.5, -(0.3*t).sin()*0.6) }).collect();
    let r = irfft2(&spec2, (3,4), &o).unwrap();
    for (i,&x) in r.iter().enumerate() { println!("irfft2_arb,{i},re,{:.17e}", x); }
}
