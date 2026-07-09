use fsci_ndimage::{NdArray, zoom, BoundaryMode, NDIMAGE_ZOOM_SEPARABLE_DISABLE};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut s=1u64; let mut r=||{s^=s<<13;s^=s>>7;s^=s<<17;(s>>11)as f64/(1u64<<53)as f64};
    let side=512usize;
    let data:Vec<f64>=(0..side*side).map(|_|r()).collect();
    let img=NdArray::new(data, vec![side,side]).unwrap();
    for &order in &[1usize,2,3,4,5] {
        for &mode in &[BoundaryMode::Reflect, BoundaryMode::Mirror] {
            let run=|dis:bool|{NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(dis,Ordering::Relaxed); zoom(&img,&[2.0,2.0],order,mode,0.0).unwrap()};
            let sep=run(false); let per=run(true);
            let md=sep.data.iter().zip(&per.data).map(|(a,b)|(a-b).abs()).fold(0.0f64,f64::max);
            let bench=|dis:bool|{NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(dis,Ordering::Relaxed); let f=||zoom(&img,&[2.0,2.0],order,mode,0.0).unwrap().data.len(); let _=black_box(f()); let reps=4; let t=Instant::now(); for _ in 0..reps{black_box(f());} t.elapsed().as_secs_f64()/reps as f64*1000.0};
            let (mut a,mut b)=(f64::MAX,f64::MAX); for _ in 0..3{b=b.min(bench(true));a=a.min(bench(false));}
            println!("order={order} {mode:?}: perpixel {b:.2}ms separable {a:.2}ms {:.2}x maxdiff={md:.1e}", b/a);
        }
    }
}
