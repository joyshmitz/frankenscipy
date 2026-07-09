use fsci_ndimage::{NdArray, affine_transform, BoundaryMode, NDIMAGE_ZOOM_SEPARABLE_DISABLE};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut s=1u64; let mut r=||{s^=s<<13;s^=s>>7;s^=s<<17;(s>>11)as f64/(1u64<<53)as f64};
    let side=512usize;
    let data:Vec<f64>=(0..side*side).map(|_|r()).collect();
    let img=NdArray::new(data, vec![side,side]).unwrap();
    let m=[[0.6f64,0.0,7.0],[0.0,0.8,-4.0]];  // diagonal scale+translate
    for &order in &[2usize,3,5] {
        let run=|dis:bool|{NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(dis,Ordering::Relaxed); affine_transform(&img,&m,order,BoundaryMode::Reflect,0.0).unwrap()};
        let new=run(false); let orig=run(true);
        let md=new.data.iter().zip(&orig.data).map(|(a,b)|(a-b).abs()).fold(0.0f64,f64::max);
        let bench=|dis:bool|{NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(dis,Ordering::Relaxed); let f=||affine_transform(&img,&m,order,BoundaryMode::Reflect,0.0).unwrap().data.len(); let _=black_box(f()); let reps=4; let t=Instant::now(); for _ in 0..reps{black_box(f());} t.elapsed().as_secs_f64()/reps as f64*1000.0};
        let (mut a,mut b)=(f64::MAX,f64::MAX); for _ in 0..3{b=b.min(bench(true));a=a.min(bench(false));}
        println!("order={order} diag: perpixel {b:.2}ms separable {a:.2}ms {:.2}x maxdiff={md:.1e}", b/a);
    }
}
