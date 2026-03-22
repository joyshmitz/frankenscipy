use fsci_fft::{dct, dct_i, dct_iii, dct_iv, idct, FftOptions};

fn main() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let opts = FftOptions::default();
    
    println!("dct1: {:?}", dct_i(&x, &opts).unwrap());
    println!("dct2: {:?}", dct(&x, &opts).unwrap());
    println!("dct3: {:?}", dct_iii(&x, &opts).unwrap());
    println!("dct4: {:?}", dct_iv(&x, &opts).unwrap());
    println!("idct2: {:?}", idct(&x, &opts).unwrap());
}
