use crate::error::ArrayApiResult;
use crate::types::{DType, IndexExpr, MemoryOrder, ScalarValue, Shape};

pub trait ArrayApiBackend {
    type Array;

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
}
