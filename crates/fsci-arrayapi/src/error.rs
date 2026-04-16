use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayApiErrorKind {
    InvalidShape,
    UnsupportedShape,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_api_error_exposes_kind_and_display_message() {
        let err = ArrayApiError::new(
            ArrayApiErrorKind::InvalidIndex,
            "index expression does not match requested indexing mode",
        );
        assert_eq!(err.kind, ArrayApiErrorKind::InvalidIndex);
        assert_eq!(
            err.to_string(),
            "index expression does not match requested indexing mode"
        );
    }

    #[test]
    fn array_api_result_alias_behaves_like_result() {
        fn pass() -> ArrayApiResult<usize> {
            Ok(7)
        }

        fn fail() -> ArrayApiResult<usize> {
            Err(ArrayApiError::new(
                ArrayApiErrorKind::Overflow,
                "shape element count overflow",
            ))
        }

        assert_eq!(pass().expect("success value should pass through"), 7);
        let err = fail().expect_err("error variant should pass through");
        assert_eq!(err.kind, ArrayApiErrorKind::Overflow);
    }

    #[test]
    fn array_api_error_supports_unsupported_shape_kind() {
        let err = ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedShape,
            "operation is outside the currently supported rank scope",
        );
        assert_eq!(err.kind, ArrayApiErrorKind::UnsupportedShape);
        assert_eq!(
            err.to_string(),
            "operation is outside the currently supported rank scope"
        );
    }
}
