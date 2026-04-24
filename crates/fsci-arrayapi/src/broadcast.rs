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

pub fn broadcast_shapes_with_audit(
    shapes: &[Shape],
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> ArrayApiResult<Shape> {
    let result = broadcast_shapes(shapes);
    if let Err(err) = &result {
        crate::audit::record_array_api_error(
            ledger,
            "broadcast_shapes",
            format!("{shapes:?}").as_bytes(),
            err.kind,
        );
    }
    result
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ArrayApiArray, ArrayApiBackend, CoreArrayBackend};
    use crate::types::{DType, ExecutionMode, MemoryOrder, ScalarValue};

    #[test]
    fn broadcast_shapes_covers_scalar_compatible_and_incompatible_inputs() {
        assert_eq!(
            broadcast_shapes(&[]).expect("empty shape list should broadcast to scalar"),
            Shape::scalar()
        );
        assert_eq!(
            broadcast_shapes(&[Shape::new(vec![2, 1]), Shape::new(vec![1, 3])])
                .expect("compatible shapes should broadcast"),
            Shape::new(vec![2, 3])
        );

        let err = broadcast_shapes(&[Shape::new(vec![2, 3]), Shape::new(vec![3, 2])])
            .expect_err("incompatible shapes should fail");
        assert_eq!(err.kind, ArrayApiErrorKind::BroadcastIncompatible);
    }

    #[test]
    fn promote_and_broadcast_promotes_dtype_and_targets_common_shape() {
        let backend = CoreArrayBackend::new(ExecutionMode::Strict);
        let left = backend
            .array_from_slice(
                &[ScalarValue::F64(1.0), ScalarValue::F64(2.0)],
                &Shape::new(vec![2, 1]),
                DType::Float32,
                MemoryOrder::C,
            )
            .expect("left array should build");
        let right = backend
            .array_from_slice(
                &[
                    ScalarValue::ComplexF64 { re: 3.0, im: 1.0 },
                    ScalarValue::ComplexF64 { re: 4.0, im: -2.0 },
                ],
                &Shape::new(vec![1, 2]),
                DType::Complex128,
                MemoryOrder::C,
            )
            .expect("right array should build");

        let result =
            promote_and_broadcast(&backend, &[&left, &right], false).expect("promotion succeeds");
        assert_eq!(result.len(), 2);
        for array in result {
            assert_eq!(array.shape(), &Shape::new(vec![2, 2]));
            assert_eq!(array.dtype(), DType::Complex128);
            assert_eq!(array.size(), 4);
        }
    }

    #[test]
    fn promote_and_broadcast_propagates_shape_mismatch_errors() {
        let backend = CoreArrayBackend::new(ExecutionMode::Strict);
        let left = backend
            .array_from_slice(
                &[
                    ScalarValue::F64(1.0),
                    ScalarValue::F64(2.0),
                    ScalarValue::F64(3.0),
                    ScalarValue::F64(4.0),
                ],
                &Shape::new(vec![2, 2]),
                DType::Float64,
                MemoryOrder::C,
            )
            .expect("left array should build");
        let right = backend
            .array_from_slice(
                &[
                    ScalarValue::F64(5.0),
                    ScalarValue::F64(6.0),
                    ScalarValue::F64(7.0),
                    ScalarValue::F64(8.0),
                    ScalarValue::F64(9.0),
                    ScalarValue::F64(10.0),
                ],
                &Shape::new(vec![2, 3]),
                DType::Float64,
                MemoryOrder::C,
            )
            .expect("right array should build");

        let err = promote_and_broadcast(&backend, &[&left, &right], false)
            .expect_err("shape mismatch should bubble up");
        assert_eq!(err.kind, ArrayApiErrorKind::BroadcastIncompatible);
    }
}
