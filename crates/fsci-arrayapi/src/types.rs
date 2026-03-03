#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    K,
    A,
    C,
    F,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Complex64,
    Complex128,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    ComplexF64 { re: f64, im: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    #[must_use]
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    #[must_use]
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    #[must_use]
    pub fn element_count(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceSpec {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: isize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexExpr {
    Basic { slices: Vec<SliceSpec> },
    Advanced { indices: Vec<Vec<isize>> },
    BooleanMask { mask_shape: Shape },
}
