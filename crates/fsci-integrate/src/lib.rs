#![forbid(unsafe_code)]

pub mod api;
pub mod solver;
pub mod step_size;
pub mod validation;

pub use api::{EventFn, OdeSolution, SolveIvpOptions, SolveIvpResult, SolverKind, solve_ivp};
pub use solver::{OdeSolver, OdeSolverState, StepFailure, StepOutcome};
pub use step_size::{InitialStepRequest, StepRhsFn, select_initial_step};
pub use validation::{
    EPS, IntegrateValidationError, MIN_RTOL, ToleranceValue, ToleranceWarning, ValidatedTolerance,
    validate_first_step, validate_max_step, validate_tol,
};
