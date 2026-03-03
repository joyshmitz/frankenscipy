use crate::backend::ArrayApiBackend;
use crate::error::{ArrayApiError, ArrayApiErrorKind, ArrayApiResult};
use crate::types::Shape;

pub fn broadcast_shapes(shapes: &[Shape]) -> ArrayApiResult<Shape> {
    if shapes.is_empty() {
        return Ok(Shape::scalar());
    }

    let rank = shapes.iter().map(Shape::rank).max().unwrap_or(0);
    let mut out = vec![1usize; rank];

    for shape in shapes {
        let len = shape.dims.len();
        for (offset, dim) in shape.dims.iter().enumerate() {
            let idx = rank - len + offset;
            let current = out[idx];
            if current == 1 {
                out[idx] = *dim;
                continue;
            }
            if *dim == 1 || *dim == current {
                continue;
            }
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::BroadcastIncompatible,
                "Array shapes are incompatible for broadcasting.",
            ));
        }
    }

    Ok(Shape::new(out))
}

pub fn promote_and_broadcast<B: ArrayApiBackend>(
    backend: &B,
    arrays: &[&B::Array],
    force_floating: bool,
) -> ArrayApiResult<Vec<B::Array>> {
    if arrays.is_empty() {
        return Ok(Vec::new());
    }

    let dtypes = arrays
        .iter()
        .map(|array| backend.dtype_of(array))
        .collect::<Vec<_>>();
    let target_dtype = backend.result_type(&dtypes, force_floating)?;
    let target_shape = broadcast_shapes(
        &arrays
            .iter()
            .map(|array| backend.shape_of(array))
            .collect::<Vec<_>>(),
    )?;

    arrays
        .iter()
        .map(|array| {
            let cast = backend.astype(array, target_dtype)?;
            backend.broadcast_to(&cast, &target_shape)
        })
        .collect()
}
