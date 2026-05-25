# SciPy Parity Coverage Report

Generated: 2026-05-25

## Overall Coverage: 68.0%

866 of 1274 SciPy callable symbols have FrankenSciPy equivalents.

## Module-Level Coverage

| Module | scipy | covered | Coverage |
|--------|-------|---------|----------|
| ndimage | 75 | 75 | 100.0% |
| linalg | 98 | 77 | 78.6% |
| fft | 41 | 31 | 75.6% |
| special | 357 | 259 | 72.5% |
| spatial | 18 | 12 | 66.7% |
| stats | 301 | 194 | 64.5% |
| signal | 156 | 99 | 63.5% |
| interpolate | 57 | 33 | 57.9% |
| optimize | 71 | 39 | 54.9% |
| integrate | 33 | 18 | 54.5% |
| io | 14 | 6 | 42.9% |
| sparse | 53 | 23 | 43.4% |

## Out-of-Scope Items

The following scipy features are intentionally out-of-scope for V1:

1. **BLAS/LAPACK internals**: `get_blas_funcs`, `get_lapack_funcs`, `find_best_blas_type` - FrankenSciPy uses pure Rust implementations
2. **Plotting utilities**: `convex_hull_plot_2d`, `voronoi_plot_2d`, `delaunay_plot_2d` - visualization out of scope
3. **Deprecated functions**: Functions marked deprecated in scipy
4. **Test utilities**: `test` functions, internal testing infrastructure

## High-Priority Missing Functions

### sparse (20.3% coverage)
- Matrix constructors: `csr_matrix`, `csc_matrix`, `coo_matrix`, `bsr_matrix`, `dia_matrix`, `dok_matrix`, `lil_matrix` - **Have formats but scipy-style constructors missing**
- Array variants: `csr_array`, `csc_array`, etc.

### io (28.6% coverage)  
- `loadmat`/`savemat` - **Implemented but not matching exact scipy API**
- `readsav` - IDL sav files
- `arff` - ARFF format

### integrate (48.6% coverage)
- `quad_vec` - **Implemented**
- `simpson` variants - **Implemented** 
- Missing: `cumulative_simpson`, `quadrature`, `romberg` with full options

### optimize (52.4% coverage)
- Most minimizers implemented
- Missing: `OptimizeResult` class structure, `show_options`, `fmin_*` legacy interfaces

## Notes

- Coverage counts include classes, functions, and type aliases
- Some functions have different names in FrankenSciPy (e.g., `solve_with_casp` vs `solve`)
- Many "missing" items are aliases or thin wrappers around implemented functionality
