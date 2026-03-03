use crate::broadcast::broadcast_shapes;
use crate::dtype::default_float_dtype;
use crate::error::{ArrayApiError, ArrayApiErrorKind, ArrayApiResult};
use crate::types::{DType, ExecutionMode, IndexExpr, MemoryOrder, ScalarValue, Shape, SliceSpec};
use nalgebra::{DMatrix, DVector};
use std::sync::Mutex;

pub trait ArrayApiArray {
    fn shape(&self) -> &Shape;

    fn dtype(&self) -> DType;

    fn ndim(&self) -> usize {
        self.shape().rank()
    }

    fn size(&self) -> usize {
        self.shape().element_count().unwrap_or(0)
    }
}

pub trait ArrayApiBackend {
    type Array: ArrayApiArray;

    fn namespace_name(&self) -> &'static str;

    fn shape_of(&self, array: &Self::Array) -> Shape;

    fn dtype_of(&self, array: &Self::Array) -> DType;

    fn asarray(
        &self,
        value: ScalarValue,
        dtype: Option<DType>,
        copy: Option<bool>,
    ) -> ArrayApiResult<Self::Array>;

    fn zeros(&self, shape: &Shape, dtype: DType, order: MemoryOrder)
    -> ArrayApiResult<Self::Array>;

    fn ones(&self, shape: &Shape, dtype: DType, order: MemoryOrder) -> ArrayApiResult<Self::Array>;

    fn empty(&self, shape: &Shape, dtype: DType, order: MemoryOrder)
    -> ArrayApiResult<Self::Array>;

    fn full(
        &self,
        shape: &Shape,
        fill_value: ScalarValue,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array>;

    fn arange(
        &self,
        start: ScalarValue,
        stop: ScalarValue,
        step: ScalarValue,
        dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array>;

    fn linspace(
        &self,
        start: ScalarValue,
        stop: ScalarValue,
        num: usize,
        endpoint: bool,
        dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array>;

    fn getitem(&self, array: &Self::Array, index: &IndexExpr) -> ArrayApiResult<Self::Array>;

    fn broadcast_to(&self, array: &Self::Array, shape: &Shape) -> ArrayApiResult<Self::Array>;

    fn astype(&self, array: &Self::Array, dtype: DType) -> ArrayApiResult<Self::Array>;

    fn result_type(&self, dtypes: &[DType], force_floating: bool) -> ArrayApiResult<DType>;

    fn array_from_slice(
        &self,
        values: &[ScalarValue],
        shape: &Shape,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let _ = (values, shape, dtype, order);
        Err(ArrayApiError::new(
            ArrayApiErrorKind::NotYetImplemented,
            "from_slice not implemented for backend",
        ))
    }

    fn reshape(&self, array: &Self::Array, new_shape: &Shape) -> ArrayApiResult<Self::Array> {
        let _ = (array, new_shape);
        Err(ArrayApiError::new(
            ArrayApiErrorKind::NotYetImplemented,
            "reshape not implemented for backend",
        ))
    }

    fn transpose(&self, array: &Self::Array) -> ArrayApiResult<Self::Array> {
        let _ = array;
        Err(ArrayApiError::new(
            ArrayApiErrorKind::NotYetImplemented,
            "transpose not implemented for backend",
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DTypeDispatchLog {
    pub requested_dtype: Option<DType>,
    pub resolved_dtype: DType,
    pub mode: ExecutionMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeMismatchLog {
    pub operation: &'static str,
    pub expected_shape: Shape,
    pub actual_shape: Shape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoreArray {
    shape: Shape,
    dtype: DType,
    values: Vec<ScalarValue>,
    order: MemoryOrder,
}

impl CoreArray {
    #[must_use]
    pub fn from_dmatrix(matrix: &DMatrix<f64>) -> Self {
        let shape = Shape::new(vec![matrix.nrows(), matrix.ncols()]);
        let mut values = Vec::with_capacity(matrix.nrows().saturating_mul(matrix.ncols()));
        for row in 0..matrix.nrows() {
            for col in 0..matrix.ncols() {
                values.push(ScalarValue::F64(matrix[(row, col)]));
            }
        }
        Self {
            shape,
            dtype: DType::Float64,
            values,
            order: MemoryOrder::C,
        }
    }

    pub fn to_dmatrix(&self) -> ArrayApiResult<DMatrix<f64>> {
        if self.shape.rank() != 2 {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "DMatrix conversion requires a 2D array",
            ));
        }

        let rows = self.shape.dims[0];
        let cols = self.shape.dims[1];
        let mut data = Vec::with_capacity(rows.saturating_mul(cols));
        for value in &self.values {
            data.push(scalar_to_f64(*value)?);
        }
        Ok(DMatrix::from_row_slice(rows, cols, &data))
    }

    #[must_use]
    pub fn from_dvector(vector: &DVector<f64>) -> Self {
        let shape = Shape::new(vec![vector.len()]);
        let values = vector
            .iter()
            .map(|value| ScalarValue::F64(*value))
            .collect();
        Self {
            shape,
            dtype: DType::Float64,
            values,
            order: MemoryOrder::C,
        }
    }

    pub fn to_dvector(&self) -> ArrayApiResult<DVector<f64>> {
        if self.shape.rank() != 1 {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "DVector conversion requires a 1D array",
            ));
        }
        let mut data = Vec::with_capacity(self.values.len());
        for value in &self.values {
            data.push(scalar_to_f64(*value)?);
        }
        Ok(DVector::from_vec(data))
    }

    #[must_use]
    pub fn values(&self) -> &[ScalarValue] {
        &self.values
    }
}

impl ArrayApiArray for CoreArray {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

#[derive(Debug)]
pub struct CoreArrayBackend {
    mode: ExecutionMode,
    dtype_dispatch_logs: Mutex<Vec<DTypeDispatchLog>>,
    shape_mismatch_logs: Mutex<Vec<ShapeMismatchLog>>,
}

impl CoreArrayBackend {
    #[must_use]
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            mode,
            dtype_dispatch_logs: Mutex::new(Vec::new()),
            shape_mismatch_logs: Mutex::new(Vec::new()),
        }
    }

    #[must_use]
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    #[must_use]
    pub fn dtype_dispatch_logs(&self) -> Vec<DTypeDispatchLog> {
        self.dtype_dispatch_logs
            .lock()
            .expect("dtype dispatch log mutex should not be poisoned")
            .clone()
    }

    #[must_use]
    pub fn shape_mismatch_logs(&self) -> Vec<ShapeMismatchLog> {
        self.shape_mismatch_logs
            .lock()
            .expect("shape mismatch log mutex should not be poisoned")
            .clone()
    }

    fn record_dtype_dispatch(&self, requested_dtype: Option<DType>, resolved_dtype: DType) {
        self.dtype_dispatch_logs
            .lock()
            .expect("dtype dispatch log mutex should not be poisoned")
            .push(DTypeDispatchLog {
                requested_dtype,
                resolved_dtype,
                mode: self.mode,
            });
    }

    fn record_shape_mismatch(
        &self,
        operation: &'static str,
        expected_shape: Shape,
        actual_shape: Shape,
    ) {
        self.shape_mismatch_logs
            .lock()
            .expect("shape mismatch log mutex should not be poisoned")
            .push(ShapeMismatchLog {
                operation,
                expected_shape,
                actual_shape,
            });
    }

    fn resolve_supported_dtype(&self, requested: Option<DType>) -> ArrayApiResult<DType> {
        let resolved = requested.unwrap_or(default_float_dtype(self.mode));
        if !is_scoped_dtype(resolved) {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::UnsupportedDtype,
                "only f32/f64/complex64/complex128 are supported",
            ));
        }
        self.record_dtype_dispatch(requested, resolved);
        Ok(resolved)
    }
}

impl Default for CoreArrayBackend {
    fn default() -> Self {
        Self::new(ExecutionMode::Strict)
    }
}

impl ArrayApiBackend for CoreArrayBackend {
    type Array = CoreArray;

    fn namespace_name(&self) -> &'static str {
        "array_api"
    }

    fn shape_of(&self, array: &Self::Array) -> Shape {
        array.shape.clone()
    }

    fn dtype_of(&self, array: &Self::Array) -> DType {
        array.dtype
    }

    fn asarray(
        &self,
        value: ScalarValue,
        dtype: Option<DType>,
        _copy: Option<bool>,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(dtype)?;
        let cast = cast_scalar_to_dtype(value, resolved_dtype)?;
        Ok(CoreArray {
            shape: Shape::scalar(),
            dtype: resolved_dtype,
            values: vec![cast],
            order: MemoryOrder::C,
        })
    }

    fn zeros(
        &self,
        shape: &Shape,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let size = checked_size(shape)?;
        let values = filled_values(size, ScalarValue::F64(0.0), resolved_dtype)?;
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values,
            order,
        })
    }

    fn ones(&self, shape: &Shape, dtype: DType, order: MemoryOrder) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let size = checked_size(shape)?;
        let values = filled_values(size, ScalarValue::F64(1.0), resolved_dtype)?;
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values,
            order,
        })
    }

    fn empty(
        &self,
        shape: &Shape,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let size = checked_size(shape)?;
        let values = filled_values(size, ScalarValue::F64(0.0), resolved_dtype)?;
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values,
            order,
        })
    }

    fn full(
        &self,
        shape: &Shape,
        fill_value: ScalarValue,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let size = checked_size(shape)?;
        let values = filled_values(size, fill_value, resolved_dtype)?;
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values,
            order,
        })
    }

    fn arange(
        &self,
        start: ScalarValue,
        stop: ScalarValue,
        step: ScalarValue,
        dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(dtype)?;
        let start_v = scalar_to_f64(start)?;
        let stop_v = scalar_to_f64(stop)?;
        let step_v = scalar_to_f64(step)?;
        if step_v == 0.0 {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidStep,
                "step must be nonzero",
            ));
        }

        let mut values = Vec::new();
        let mut current = start_v;
        if step_v > 0.0 {
            while current < stop_v {
                values.push(cast_scalar_to_dtype(
                    ScalarValue::F64(current),
                    resolved_dtype,
                )?);
                current += step_v;
            }
        } else {
            while current > stop_v {
                values.push(cast_scalar_to_dtype(
                    ScalarValue::F64(current),
                    resolved_dtype,
                )?);
                current += step_v;
            }
        }

        Ok(CoreArray {
            shape: Shape::new(vec![values.len()]),
            dtype: resolved_dtype,
            values,
            order: MemoryOrder::C,
        })
    }

    fn linspace(
        &self,
        start: ScalarValue,
        stop: ScalarValue,
        num: usize,
        endpoint: bool,
        dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(dtype)?;
        let start_v = scalar_to_f64(start)?;
        let stop_v = scalar_to_f64(stop)?;
        if num == 0 {
            return Ok(CoreArray {
                shape: Shape::new(vec![0]),
                dtype: resolved_dtype,
                values: Vec::new(),
                order: MemoryOrder::C,
            });
        }

        let step = if endpoint {
            if num == 1 {
                0.0
            } else {
                (stop_v - start_v) / (num.saturating_sub(1) as f64)
            }
        } else {
            (stop_v - start_v) / (num as f64)
        };

        let mut values = Vec::with_capacity(num);
        for idx in 0..num {
            let value = if endpoint && idx == num - 1 {
                stop_v
            } else {
                start_v + (idx as f64) * step
            };
            values.push(cast_scalar_to_dtype(
                ScalarValue::F64(value),
                resolved_dtype,
            )?);
        }

        Ok(CoreArray {
            shape: Shape::new(vec![values.len()]),
            dtype: resolved_dtype,
            values,
            order: MemoryOrder::C,
        })
    }

    fn getitem(&self, array: &Self::Array, index: &IndexExpr) -> ArrayApiResult<Self::Array> {
        match index {
            IndexExpr::Basic { slices } => basic_getitem(array, slices, self.mode),
            IndexExpr::Advanced { indices } => advanced_getitem(array, indices, self.mode),
            IndexExpr::BooleanMask { mask_shape } => {
                if *mask_shape != array.shape {
                    self.record_shape_mismatch(
                        "boolean_mask",
                        array.shape.clone(),
                        mask_shape.clone(),
                    );
                    return Err(ArrayApiError::new(
                        ArrayApiErrorKind::InvalidShape,
                        "boolean index did not match indexed array shape",
                    ));
                }
                Ok(array.clone())
            }
        }
    }

    fn broadcast_to(&self, array: &Self::Array, shape: &Shape) -> ArrayApiResult<Self::Array> {
        let target_shape = broadcast_shapes(&[array.shape.clone(), shape.clone()])?;
        if target_shape != *shape {
            self.record_shape_mismatch("broadcast_to", shape.clone(), array.shape.clone());
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::BroadcastIncompatible,
                "operands could not be broadcast together with the requested shape",
            ));
        }

        let output_size = checked_size(shape)?;
        if output_size == 0 {
            return Ok(CoreArray {
                shape: shape.clone(),
                dtype: array.dtype,
                values: Vec::new(),
                order: array.order,
            });
        }

        let out_rank = shape.rank();
        let in_rank = array.shape.rank();
        let mut values = Vec::with_capacity(output_size);
        for linear in 0..output_size {
            let out_coords = unravel_index(linear, &shape.dims);
            let mut in_coords = vec![0usize; in_rank];
            for (in_dim_idx, in_dim) in array.shape.dims.iter().enumerate() {
                let out_dim_idx = out_rank - in_rank + in_dim_idx;
                in_coords[in_dim_idx] = if *in_dim == 1 {
                    0
                } else {
                    out_coords[out_dim_idx]
                };
            }
            let in_linear = ravel_index(&in_coords, &array.shape.dims);
            values.push(array.values[in_linear]);
        }

        Ok(CoreArray {
            shape: shape.clone(),
            dtype: array.dtype,
            values,
            order: array.order,
        })
    }

    fn astype(&self, array: &Self::Array, dtype: DType) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let mut values = Vec::with_capacity(array.values.len());
        for value in &array.values {
            values.push(cast_scalar_to_dtype(*value, resolved_dtype)?);
        }
        Ok(CoreArray {
            shape: array.shape.clone(),
            dtype: resolved_dtype,
            values,
            order: array.order,
        })
    }

    fn result_type(&self, dtypes: &[DType], force_floating: bool) -> ArrayApiResult<DType> {
        if dtypes.is_empty() {
            return self.resolve_supported_dtype(None);
        }
        let mut rank = 0usize;
        for dtype in dtypes {
            rank = rank.max(dtype_rank(*dtype, force_floating)?);
        }
        let resolved = match rank {
            0 => DType::Float32,
            1 => DType::Float64,
            2 => DType::Complex64,
            _ => DType::Complex128,
        };
        self.resolve_supported_dtype(Some(resolved))
    }

    fn array_from_slice(
        &self,
        values: &[ScalarValue],
        shape: &Shape,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let expected = checked_size(shape)?;
        if expected != values.len() {
            self.record_shape_mismatch("from_slice", shape.clone(), Shape::new(vec![values.len()]));
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "input slice length does not match target shape",
            ));
        }
        let mut cast_values = Vec::with_capacity(values.len());
        for value in values {
            cast_values.push(cast_scalar_to_dtype(*value, resolved_dtype)?);
        }
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values: cast_values,
            order,
        })
    }

    fn reshape(&self, array: &Self::Array, new_shape: &Shape) -> ArrayApiResult<Self::Array> {
        let expected = checked_size(new_shape)?;
        let actual = checked_size(&array.shape)?;
        if expected != actual {
            self.record_shape_mismatch("reshape", new_shape.clone(), array.shape.clone());
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "cannot reshape array because element counts differ",
            ));
        }
        Ok(CoreArray {
            shape: new_shape.clone(),
            dtype: array.dtype,
            values: array.values.clone(),
            order: array.order,
        })
    }

    fn transpose(&self, array: &Self::Array) -> ArrayApiResult<Self::Array> {
        if array.shape.rank() != 2 {
            self.record_shape_mismatch("transpose", Shape::new(vec![2]), array.shape.clone());
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "transpose requires a rank-2 array",
            ));
        }

        let rows = array.shape.dims[0];
        let cols = array.shape.dims[1];
        let mut values = Vec::with_capacity(array.values.len());
        for col in 0..cols {
            for row in 0..rows {
                values.push(array.values[row * cols + col]);
            }
        }
        Ok(CoreArray {
            shape: Shape::new(vec![cols, rows]),
            dtype: array.dtype,
            values,
            order: array.order,
        })
    }
}

fn is_scoped_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Float32 | DType::Float64 | DType::Complex64 | DType::Complex128
    )
}

fn checked_size(shape: &Shape) -> ArrayApiResult<usize> {
    shape.element_count().ok_or_else(|| {
        ArrayApiError::new(ArrayApiErrorKind::Overflow, "shape element count overflow")
    })
}

fn filled_values(
    size: usize,
    fill_value: ScalarValue,
    dtype: DType,
) -> ArrayApiResult<Vec<ScalarValue>> {
    let fill = cast_scalar_to_dtype(fill_value, dtype)?;
    Ok(vec![fill; size])
}

fn scalar_to_f64(value: ScalarValue) -> ArrayApiResult<f64> {
    match value {
        ScalarValue::Bool(v) => Ok(if v { 1.0 } else { 0.0 }),
        ScalarValue::I64(v) => Ok(v as f64),
        ScalarValue::U64(v) => Ok(v as f64),
        ScalarValue::F64(v) => Ok(v),
        ScalarValue::ComplexF64 { re, im } => {
            if im == 0.0 {
                Ok(re)
            } else {
                Err(ArrayApiError::new(
                    ArrayApiErrorKind::UnsupportedDtype,
                    "cannot coerce complex value with nonzero imaginary part to real dtype",
                ))
            }
        }
    }
}

fn cast_scalar_to_dtype(value: ScalarValue, dtype: DType) -> ArrayApiResult<ScalarValue> {
    match dtype {
        DType::Float32 => Ok(ScalarValue::F64((scalar_to_f64(value)? as f32) as f64)),
        DType::Float64 => Ok(ScalarValue::F64(scalar_to_f64(value)?)),
        DType::Complex64 => {
            let (re, im) = scalar_to_complex_components(value)?;
            Ok(ScalarValue::ComplexF64 {
                re: (re as f32) as f64,
                im: (im as f32) as f64,
            })
        }
        DType::Complex128 => {
            let (re, im) = scalar_to_complex_components(value)?;
            Ok(ScalarValue::ComplexF64 { re, im })
        }
        _ => Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedDtype,
            "only f32/f64/complex64/complex128 are supported",
        )),
    }
}

fn scalar_to_complex_components(value: ScalarValue) -> ArrayApiResult<(f64, f64)> {
    match value {
        ScalarValue::Bool(v) => Ok((if v { 1.0 } else { 0.0 }, 0.0)),
        ScalarValue::I64(v) => Ok((v as f64, 0.0)),
        ScalarValue::U64(v) => Ok((v as f64, 0.0)),
        ScalarValue::F64(v) => Ok((v, 0.0)),
        ScalarValue::ComplexF64 { re, im } => Ok((re, im)),
    }
}

fn dtype_rank(dtype: DType, force_floating: bool) -> ArrayApiResult<usize> {
    match dtype {
        DType::Float32 => Ok(0),
        DType::Float64 => Ok(1),
        DType::Complex64 => Ok(2),
        DType::Complex128 => Ok(3),
        DType::Bool
        | DType::Int8
        | DType::Int16
        | DType::Int32
        | DType::Int64
        | DType::UInt8
        | DType::UInt16
        | DType::UInt32
        | DType::UInt64 => {
            if force_floating {
                Ok(1)
            } else {
                Err(ArrayApiError::new(
                    ArrayApiErrorKind::UnsupportedDtype,
                    "integer and boolean dtypes are not implemented in this scope",
                ))
            }
        }
    }
}

fn normalize_single_index(index: isize, len: usize, mode: ExecutionMode) -> ArrayApiResult<usize> {
    let len_i = len as isize;
    if mode == ExecutionMode::Hardened && (index < -len_i || index >= len_i) {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidIndex,
            "index out of bounds in hardened mode",
        ));
    }
    let mut normalized = index;
    if normalized < 0 {
        normalized += len_i;
    }
    if normalized < 0 || normalized >= len_i {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidIndex,
            "index out of bounds",
        ));
    }
    Ok(normalized as usize)
}

fn basic_slice_indices(
    spec: &SliceSpec,
    len: usize,
    mode: ExecutionMode,
) -> ArrayApiResult<Vec<usize>> {
    if spec.step == 0 {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidStep,
            "slice step cannot be zero",
        ));
    }

    let len_i = len as isize;
    if mode == ExecutionMode::Hardened {
        for endpoint in [spec.start, spec.stop] {
            if let Some(value) = endpoint
                && (value < -len_i || value > len_i)
            {
                return Err(ArrayApiError::new(
                    ArrayApiErrorKind::InvalidIndex,
                    "slice bounds out of range in hardened mode",
                ));
            }
        }
    }

    let step = spec.step;
    let mut start = spec.start.unwrap_or(if step > 0 { 0 } else { len_i - 1 });
    let mut stop = spec.stop.unwrap_or(if step > 0 { len_i } else { -1 });

    if start < 0 {
        start += len_i;
    }
    if stop < 0 {
        stop += len_i;
    }

    let mut out = Vec::new();
    if step > 0 {
        start = start.clamp(0, len_i);
        stop = stop.clamp(0, len_i);
        let mut cur = start;
        while cur < stop {
            out.push(cur as usize);
            cur += step;
        }
        return Ok(out);
    }

    start = start.clamp(-1, len_i.saturating_sub(1));
    stop = stop.clamp(-1, len_i.saturating_sub(1));
    let mut cur = start;
    while cur > stop {
        if cur >= 0 {
            out.push(cur as usize);
        }
        cur += step;
    }
    Ok(out)
}

fn basic_getitem(
    array: &CoreArray,
    slices: &[SliceSpec],
    mode: ExecutionMode,
) -> ArrayApiResult<CoreArray> {
    match array.shape.rank() {
        1 => {
            if slices.len() != 1 {
                return Err(ArrayApiError::new(
                    ArrayApiErrorKind::InvalidIndex,
                    "1D indexing requires exactly one slice",
                ));
            }
            let indices = basic_slice_indices(&slices[0], array.shape.dims[0], mode)?;
            let mut values = Vec::with_capacity(indices.len());
            for idx in &indices {
                values.push(array.values[*idx]);
            }
            Ok(CoreArray {
                shape: Shape::new(vec![indices.len()]),
                dtype: array.dtype,
                values,
                order: array.order,
            })
        }
        2 => {
            if slices.len() != 2 {
                return Err(ArrayApiError::new(
                    ArrayApiErrorKind::InvalidIndex,
                    "2D indexing requires two slices",
                ));
            }
            let rows = basic_slice_indices(&slices[0], array.shape.dims[0], mode)?;
            let cols = basic_slice_indices(&slices[1], array.shape.dims[1], mode)?;
            let mut values = Vec::with_capacity(rows.len().saturating_mul(cols.len()));
            let ncols = array.shape.dims[1];
            for row in &rows {
                for col in &cols {
                    values.push(array.values[row * ncols + col]);
                }
            }
            Ok(CoreArray {
                shape: Shape::new(vec![rows.len(), cols.len()]),
                dtype: array.dtype,
                values,
                order: array.order,
            })
        }
        _ => Err(ArrayApiError::new(
            ArrayApiErrorKind::NotYetImplemented,
            "basic slicing is currently scoped to rank-1 and rank-2 arrays",
        )),
    }
}

fn advanced_getitem(
    array: &CoreArray,
    indices: &[Vec<isize>],
    mode: ExecutionMode,
) -> ArrayApiResult<CoreArray> {
    if array.shape.rank() != 1 || indices.len() != 1 {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::NotYetImplemented,
            "advanced indexing is currently scoped to 1D arrays",
        ));
    }
    let len = array.shape.dims[0];
    let mut values = Vec::with_capacity(indices[0].len());
    for index in &indices[0] {
        let normalized = normalize_single_index(*index, len, mode)?;
        values.push(array.values[normalized]);
    }
    Ok(CoreArray {
        shape: Shape::new(vec![values.len()]),
        dtype: array.dtype,
        values,
        order: array.order,
    })
}

fn unravel_index(mut index: usize, dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut coords = vec![0usize; dims.len()];
    for pos in (0..dims.len()).rev() {
        let dim = dims[pos];
        if dim == 0 {
            coords[pos] = 0;
        } else {
            coords[pos] = index % dim;
            index /= dim;
        }
    }
    coords
}

fn ravel_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for pos in (0..dims.len()).rev() {
        index += coords[pos] * stride;
        stride = stride.saturating_mul(dims[pos].max(1));
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{CreationRequest, from_slice, full, zeros};
    use crate::indexing::{IndexRequest, IndexingMode, getitem, reshape, transpose};

    fn strict_backend() -> CoreArrayBackend {
        CoreArrayBackend::new(ExecutionMode::Strict)
    }

    fn hardened_backend() -> CoreArrayBackend {
        CoreArrayBackend::new(ExecutionMode::Hardened)
    }

    #[test]
    fn array_accessors_and_creation_flow() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![2, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };

        let array = zeros(&backend, &request).expect("zeros should succeed");
        assert_eq!(array.shape(), &Shape::new(vec![2, 2]));
        assert_eq!(array.dtype(), DType::Float64);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.size(), 4);

        let log = backend.dtype_dispatch_logs();
        assert!(!log.is_empty());
        assert_eq!(log[0].requested_dtype, Some(DType::Float64));
        assert_eq!(log[0].resolved_dtype, DType::Float64);
        assert_eq!(log[0].mode, ExecutionMode::Strict);
    }

    #[test]
    fn from_slice_reshape_and_transpose_round_trip() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![2, 3]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = [
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(3.0),
            ScalarValue::F64(4.0),
            ScalarValue::F64(5.0),
            ScalarValue::F64(6.0),
        ];

        let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");
        let transposed = transpose(&backend, &array).expect("transpose should succeed");
        assert_eq!(transposed.shape(), &Shape::new(vec![3, 2]));

        let reshaped =
            reshape(&backend, &array, &Shape::new(vec![3, 2])).expect("reshape should succeed");
        assert_eq!(reshaped.shape(), &Shape::new(vec![3, 2]));
    }

    #[test]
    fn strict_vs_hardened_slice_bounds_behavior() {
        let strict = strict_backend();
        let hardened = hardened_backend();

        let request = CreationRequest {
            shape: Shape::new(vec![5]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = [
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(3.0),
            ScalarValue::F64(4.0),
            ScalarValue::F64(5.0),
        ];
        let strict_array =
            from_slice(&strict, &values, &request).expect("strict from_slice should succeed");
        let hardened_array =
            from_slice(&hardened, &values, &request).expect("hardened from_slice should succeed");

        let request = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![SliceSpec {
                    start: Some(0),
                    stop: Some(20),
                    step: 1,
                }],
            },
        };

        let strict_out = getitem(&strict, &strict_array, &request).expect("strict slice clamps");
        assert_eq!(strict_out.shape(), &Shape::new(vec![5]));

        let hardened_error =
            getitem(&hardened, &hardened_array, &request).expect_err("hardened should reject");
        assert_eq!(hardened_error.kind, ArrayApiErrorKind::InvalidIndex);
    }

    #[test]
    fn nalgebra_dmatrix_and_dvector_round_trip() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let array = CoreArray::from_dmatrix(&matrix);
        let matrix_back = array.to_dmatrix().expect("dmatrix conversion");
        assert_eq!(matrix_back, matrix);

        let vector = DVector::from_vec(vec![10.0, 20.0, 30.0]);
        let vector_array = CoreArray::from_dvector(&vector);
        let vector_back = vector_array.to_dvector().expect("dvector conversion");
        assert_eq!(vector_back, vector);
    }

    #[test]
    fn shape_mismatch_logs_are_emitted() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![2, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = [
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(3.0),
            ScalarValue::F64(4.0),
        ];
        let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");
        let err = reshape(&backend, &array, &Shape::new(vec![3, 2])).expect_err("reshape mismatch");
        assert_eq!(err.kind, ArrayApiErrorKind::InvalidShape);

        let logs = backend.shape_mismatch_logs();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].operation, "reshape");
        assert_eq!(logs[0].expected_shape, Shape::new(vec![3, 2]));
        assert_eq!(logs[0].actual_shape, Shape::new(vec![2, 2]));
    }

    #[test]
    fn strict_dtype_mismatch_errors_are_reported() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![2]),
            dtype: DType::Int64,
            order: MemoryOrder::C,
        };
        let err = zeros(&backend, &request).expect_err("int dtype is unsupported in this scope");
        assert_eq!(err.kind, ArrayApiErrorKind::UnsupportedDtype);

        let full_request = crate::creation::FullRequest {
            fill_value: ScalarValue::Bool(true),
            dtype: DType::Complex128,
            order: MemoryOrder::C,
        };
        let ok = full(&backend, &Shape::new(vec![1]), &full_request);
        assert!(ok.is_ok());
    }
}
