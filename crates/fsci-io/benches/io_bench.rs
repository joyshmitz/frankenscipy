use criterion::{Criterion, criterion_group, criterion_main};
use fsci_io::{
    MatArray, NetcdfDimension, NetcdfFile, NetcdfValue, NetcdfVariable, SAVEMAT_TEXT_FORCE_SERIAL,
    SAVETXT_FORCE_SERIAL, WRITE_JSON_FORCE_SERIAL, WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING,
    loadtxt, mmread, mmwrite, savemat_text, savetxt, write_csv, write_json_array,
    write_netcdf_classic,
};
use std::sync::atomic::Ordering;

fn matrix(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols).map(|i| i as f64 * 0.001 + 1.0).collect()
}

fn row_matrix(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| (r * cols + c) as f64 * 0.001 + 1.0)
                .collect()
        })
        .collect()
}

/// Literal pre-streaming general-array reader used only as the same-binary
/// benchmark control. It intentionally retains both staging allocations.
fn mmread_general_array_staged_reference(content: &str, rows: usize, cols: usize) -> Vec<f64> {
    let mut lines = content.lines();
    let _header = lines.next().expect("Matrix Market header");
    let _size = lines
        .find(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('%')
        })
        .expect("Matrix Market size");
    let positions: Vec<(usize, usize)> = (0..cols)
        .flat_map(|col| (0..rows).map(move |row| (row, col)))
        .collect();
    let values: Vec<f64> = lines
        .filter_map(|line| {
            let trimmed = line.trim();
            (!trimmed.is_empty() && !trimmed.starts_with('%')).then_some(trimmed)
        })
        .map(|value| value.parse().expect("Matrix Market value"))
        .collect();
    assert_eq!(values.len(), positions.len());

    let mut data = vec![0.0; rows * cols];
    for (&(row, col), &value) in positions.iter().zip(&values) {
        data[row * cols + col] = value;
    }
    data
}

/// Whitespace-delimited text round trip. savetxt writes cells with `write!` into
/// the buffer (frankenscipy-d1uxy); loadtxt's serial path parses straight into the
/// output buffer (frankenscipy-fwnb1) — both exercised here at <4096 rows.
fn bench_text_io(c: &mut Criterion) {
    let (rows, cols) = (500usize, 20usize);
    let data = matrix(rows, cols);
    let text = savetxt(rows, cols, &data, " ").expect("savetxt");
    let mut group = c.benchmark_group("text_io");
    group.bench_function("savetxt/500x20", |b| {
        b.iter(|| savetxt(rows, cols, &data, " "))
    });
    group.bench_function("loadtxt/500x20", |b| b.iter(|| loadtxt(&text)));
    group.finish();
}

/// MatrixMarket array round trip. mmwrite uses `writeln!` per entry; mmread parses
/// off the split iterator (frankenscipy-1f4yh for coordinate).
fn bench_matrix_market(c: &mut Criterion) {
    let (rows, cols) = (100usize, 100usize);
    let data = matrix(rows, cols);
    let mm = mmwrite(rows, cols, &data).expect("mmwrite");
    let (large_rows, large_cols) = (1_000usize, 1_000usize);
    let large_data = matrix(large_rows, large_cols);
    let large_mm = mmwrite(large_rows, large_cols, &large_data).expect("large mmwrite");
    let mut group = c.benchmark_group("matrix_market");
    group.bench_function("mmwrite/100x100", |b| b.iter(|| mmwrite(rows, cols, &data)));
    group.bench_function("mmread/100x100", |b| b.iter(|| mmread(&mm)));
    group.bench_function("mmread/1000x1000", |b| b.iter(|| mmread(&large_mm)));
    group.finish();
}

/// Same-binary A/B for streaming general array values directly into their
/// row-major destination versus the former positions-plus-values staging.
fn bench_mmread_general_array_stream_ab(c: &mut Criterion) {
    let (rows, cols) = (256usize, 256usize);
    let mm = mmwrite(rows, cols, &matrix(rows, cols)).expect("mmwrite");
    let current = mmread(&mm).expect("mmread").data;
    let staged = mmread_general_array_staged_reference(&mm, rows, cols);
    assert!(
        current
            .iter()
            .zip(&staged)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
    );

    let mut group = c.benchmark_group("mmread_general_array_stream_ab");
    group.bench_function("current_stream/256x256", |b| {
        b.iter(|| mmread(&mm).expect("mmread").data)
    });
    group.bench_function("original_staged/256x256", |b| {
        b.iter(|| mmread_general_array_staged_reference(&mm, rows, cols))
    });
    group.finish();
}

/// CSV/JSON write helpers previously materialized per-cell/per-value `String`s
/// before joining them; keep these in the IO bench so the allocation-free write
/// path has a stable per-crate gate.
fn bench_write_helpers(c: &mut Criterion) {
    let rows = row_matrix(500, 20);
    let flat = matrix(10_000, 1);
    let mut group = c.benchmark_group("write_helpers");
    group.bench_function("write_csv/500x20", |b| {
        b.iter(|| write_csv(None, &rows, ','))
    });
    group.bench_function("write_json_array/10000", |b| {
        b.iter(|| write_json_array(&flat))
    });
    group.finish();
}

/// Same-binary A/B for the row-parallel `savetxt` formatter against the legacy
/// serial path.
fn bench_savetxt_parallel_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("savetxt_parallel_ab");
    for &(rows, cols) in &[(10_000usize, 20usize), (50_000, 20)] {
        let data = matrix(rows, cols);
        group.bench_function(format!("current_parallel/{rows}x{cols}"), |b| {
            b.iter(|| {
                SAVETXT_FORCE_SERIAL.store(false, Ordering::Relaxed);
                savetxt(rows, cols, &data, " ")
            })
        });
        group.bench_function(format!("orig_serial/{rows}x{cols}"), |b| {
            b.iter(|| {
                SAVETXT_FORCE_SERIAL.store(true, Ordering::Relaxed);
                savetxt(rows, cols, &data, " ")
            })
        });
    }
    SAVETXT_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

/// Same-binary A/B for the value-parallel JSON array formatter against the
/// legacy serial path.
fn bench_write_json_parallel_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_json_parallel_ab");
    for &n in &[200_000usize, 1_000_000] {
        let data = matrix(n, 1);
        group.bench_function(format!("current_parallel/{n}"), |b| {
            b.iter(|| {
                WRITE_JSON_FORCE_SERIAL.store(false, Ordering::Relaxed);
                write_json_array(&data)
            })
        });
        group.bench_function(format!("orig_serial/{n}"), |b| {
            b.iter(|| {
                WRITE_JSON_FORCE_SERIAL.store(true, Ordering::Relaxed);
                write_json_array(&data)
            })
        });
    }
    WRITE_JSON_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

/// Same-binary A/B for the row-parallel MATLAB text formatter against the
/// legacy serial path.
fn bench_savemat_text_parallel_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("savemat_text_parallel_ab");
    for &(rows, cols) in &[(10_000usize, 20usize), (50_000, 20)] {
        let arrays = vec![MatArray {
            name: "A".to_string(),
            rows,
            cols,
            data: matrix(rows, cols),
        }];
        group.bench_function(format!("current_parallel/{rows}x{cols}"), |b| {
            b.iter(|| {
                SAVEMAT_TEXT_FORCE_SERIAL.store(false, Ordering::Relaxed);
                savemat_text(&arrays)
            })
        });
        group.bench_function(format!("orig_serial/{rows}x{cols}"), |b| {
            b.iter(|| {
                SAVEMAT_TEXT_FORCE_SERIAL.store(true, Ordering::Relaxed);
                savemat_text(&arrays)
            })
        });
    }
    SAVEMAT_TEXT_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

/// Same-binary A/B for calculating NetCDF variable byte sizes from type and
/// element count rather than encoding every payload twice while building the
/// placeholder and final headers.
fn bench_write_netcdf_header_size_ab(c: &mut Criterion) {
    let (rows, cols) = (512usize, 256usize);
    let file = NetcdfFile {
        dimensions: vec![
            NetcdfDimension {
                name: "rows".to_string(),
                len: Some(rows),
            },
            NetcdfDimension {
                name: "cols".to_string(),
                len: Some(cols),
            },
        ],
        attributes: Vec::new(),
        variables: vec![NetcdfVariable {
            name: "samples".to_string(),
            dim_ids: vec![0, 1],
            attributes: Vec::new(),
            data: NetcdfValue::Double(matrix(rows, cols)),
        }],
    };

    WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING.store(true, Ordering::Relaxed);
    let redundant = write_netcdf_classic(&file).expect("redundant header payload encodes");
    WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING.store(false, Ordering::Relaxed);
    let current = write_netcdf_classic(&file).expect("length-only header sizing");
    assert_eq!(current, redundant);

    let mut group = c.benchmark_group("write_netcdf_header_size_ab");
    group.bench_function("current_length_only/512x256", |b| {
        b.iter(|| {
            WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING.store(false, Ordering::Relaxed);
            write_netcdf_classic(&file)
        })
    });
    group.bench_function("original_redundant_encode/512x256", |b| {
        b.iter(|| {
            WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING.store(true, Ordering::Relaxed);
            write_netcdf_classic(&file)
        })
    });
    WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING.store(false, Ordering::Relaxed);
    group.finish();
}

criterion_group!(
    benches,
    bench_text_io,
    bench_matrix_market,
    bench_mmread_general_array_stream_ab,
    bench_write_helpers,
    bench_savetxt_parallel_ab,
    bench_write_json_parallel_ab,
    bench_savemat_text_parallel_ab,
    bench_write_netcdf_header_size_ab
);
criterion_main!(benches);
