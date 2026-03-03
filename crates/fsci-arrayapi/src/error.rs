use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayApiErrorKind {
    NotYetImplemented,
    InvalidShape,
    InvalidIndex,
    InvalidStep,
    UnsupportedDtype,
    BroadcastIncompatible,
    NonFiniteInput,
    NamespaceMismatch,
    Overflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayApiError {
    pub kind: ArrayApiErrorKind,
    pub message: &'static str,
}

impl ArrayApiError {
    #[must_use]
    pub const fn new(kind: ArrayApiErrorKind, message: &'static str) -> Self {
        Self { kind, message }
    }
}

impl Display for ArrayApiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message)
    }
}

impl Error for ArrayApiError {}

pub type ArrayApiResult<T> = Result<T, ArrayApiError>;
