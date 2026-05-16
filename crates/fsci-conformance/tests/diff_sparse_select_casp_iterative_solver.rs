#![forbid(unsafe_code)]
//! Branch coverage for fsci_sparse::select_casp_iterative_solver.
//!
//! Resolves [frankenscipy-k7i1k]. Exercise each documented routing
//! branch and verify the expected solver is chosen.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{
    CaspIterativeSolveOptions, CaspIterativeSolver, CaspMatvecCost, CooMatrix,
    FormatConvertible, Shape2D, select_casp_iterative_solver,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    expected: String,
    actual: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create casp_iter_select diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn build_csr(rows: usize, cols: usize, trips: &[(usize, usize, f64)]) -> fsci_sparse::CsrMatrix {
    let mut data = Vec::new();
    let mut rs = Vec::new();
    let mut cs = Vec::new();
    for &(r, c, v) in trips {
        data.push(v);
        rs.push(r);
        cs.push(c);
    }
    let coo = CooMatrix::from_triplets(Shape2D::new(rows, cols), data, rs, cs, true).unwrap();
    coo.to_csr().unwrap()
}

#[test]
fn diff_sparse_select_casp_iterative_solver() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    // Helper for non-default options
    let mk_opts = |preconditioner: bool, low_memory: bool, cost: CaspMatvecCost| {
        CaspIterativeSolveOptions {
            preconditioner_available: preconditioner,
            prefer_low_memory: low_memory,
            matrix_vector_cost: cost,
            ..CaspIterativeSolveOptions::default()
        }
    };

    // 1. Rectangular overdetermined (rows > cols) → Lsqr
    let r1 = build_csr(5, 3, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 0, 0.5), (4, 1, 0.3)]);
    let b1 = vec![1.0_f64; 5];
    let d1 = select_casp_iterative_solver(&r1, &b1, None, mk_opts(false, false, CaspMatvecCost::Auto)).unwrap();
    diffs.push(CaseDiff {
        case_id: "rect_over".into(),
        expected: format!("{:?}", CaspIterativeSolver::Lsqr),
        actual: format!("{:?}", d1.selected_solver),
        pass: d1.selected_solver == CaspIterativeSolver::Lsqr,
    });

    // 2. Rectangular underdetermined (rows < cols) → Lsmr
    let r2 = build_csr(3, 5, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (0, 3, 0.5), (1, 4, 0.3)]);
    let b2 = vec![1.0_f64; 3];
    let d2 = select_casp_iterative_solver(&r2, &b2, None, mk_opts(false, false, CaspMatvecCost::Auto)).unwrap();
    diffs.push(CaseDiff {
        case_id: "rect_under".into(),
        expected: format!("{:?}", CaspIterativeSolver::Lsmr),
        actual: format!("{:?}", d2.selected_solver),
        pass: d2.selected_solver == CaspIterativeSolver::Lsmr,
    });

    // 3. Symmetric, positive diag, row-dominant → Cg (tridiagonal)
    let mut tri = Vec::new();
    for i in 0..5 {
        tri.push((i, i, 4.0_f64));
        if i + 1 < 5 {
            tri.push((i, i + 1, -1.0));
            tri.push((i + 1, i, -1.0));
        }
    }
    let m3 = build_csr(5, 5, &tri);
    let b3 = vec![1.0_f64; 5];
    let d3 = select_casp_iterative_solver(&m3, &b3, None, mk_opts(false, false, CaspMatvecCost::Auto)).unwrap();
    diffs.push(CaseDiff {
        case_id: "spd_cg".into(),
        expected: format!("{:?}", CaspIterativeSolver::Cg),
        actual: format!("{:?}", d3.selected_solver),
        pass: d3.selected_solver == CaspIterativeSolver::Cg,
    });

    // 4. Symmetric but NOT positive diag → Minres (sym indefinite via negative diag)
    let sym_indef = vec![
        (0, 0, -1.0_f64), (0, 1, 0.5),
        (1, 0, 0.5), (1, 1, 2.0),
    ];
    let m4 = build_csr(2, 2, &sym_indef);
    let b4 = vec![1.0_f64, 2.0];
    let d4 = select_casp_iterative_solver(&m4, &b4, None, mk_opts(false, false, CaspMatvecCost::Auto)).unwrap();
    diffs.push(CaseDiff {
        case_id: "sym_minres".into(),
        expected: format!("{:?}", CaspIterativeSolver::Minres),
        actual: format!("{:?}", d4.selected_solver),
        pass: d4.selected_solver == CaspIterativeSolver::Minres,
    });

    // 5. Nonsymmetric with preconditioner → Lgmres
    let asym = vec![
        (0, 0, 2.0_f64), (0, 1, 1.0),
        (1, 0, -1.0), (1, 1, 3.0),
    ];
    let m5 = build_csr(2, 2, &asym);
    let b5 = vec![1.0_f64, 1.0];
    let d5 = select_casp_iterative_solver(&m5, &b5, None, mk_opts(true, false, CaspMatvecCost::Auto)).unwrap();
    diffs.push(CaseDiff {
        case_id: "asym_precond_lgmres".into(),
        expected: format!("{:?}", CaspIterativeSolver::Lgmres),
        actual: format!("{:?}", d5.selected_solver),
        pass: d5.selected_solver == CaspIterativeSolver::Lgmres,
    });

    // 6. Nonsymmetric, low memory → Bicgstab
    let d6 = select_casp_iterative_solver(&m5, &b5, None, mk_opts(false, true, CaspMatvecCost::Auto)).unwrap();
    diffs.push(CaseDiff {
        case_id: "asym_low_mem_bicgstab".into(),
        expected: format!("{:?}", CaspIterativeSolver::Bicgstab),
        actual: format!("{:?}", d6.selected_solver),
        pass: d6.selected_solver == CaspIterativeSolver::Bicgstab,
    });

    // 7. Default nonsymmetric small dense → Gmres
    let d7 = select_casp_iterative_solver(&m5, &b5, None, mk_opts(false, false, CaspMatvecCost::Cheap)).unwrap();
    diffs.push(CaseDiff {
        case_id: "asym_default_gmres".into(),
        expected: format!("{:?}", CaspIterativeSolver::Gmres),
        actual: format!("{:?}", d7.selected_solver),
        pass: d7.selected_solver == CaspIterativeSolver::Gmres,
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_select_casp_iterative_solver".into(),
        category: "fsci_sparse::select_casp_iterative_solver branch coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("casp_iter_select mismatch: {} got {} (expected {})", d.case_id, d.actual, d.expected);
        }
    }

    assert!(
        all_pass,
        "casp_iter_select branch coverage failed: {} cases",
        diffs.len(),
    );
}
