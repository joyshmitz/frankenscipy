#![forbid(unsafe_code)]

use blake3::hash;
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const VECTORS_PER_FAMILY: usize = 100;
const CLASSES: [&str; 4] = ["nan_inf", "extreme_dimensions", "empty_inputs", "max_size"];

#[derive(Debug, Serialize)]
struct AdversarialVector {
    id: String,
    packet_family: String,
    attack_surface: String,
    class: String,
    payload: Value,
    blake3: String,
}

#[derive(Debug, Serialize)]
struct FamilyManifest {
    packet_family: String,
    attack_surface: String,
    vector_count: usize,
    class_counts: BTreeMap<String, usize>,
    corpus_file: String,
    corpus_blake3: String,
}

#[derive(Debug, Serialize)]
struct CorpusManifest {
    schema_version: u32,
    generated_at_epoch_seconds: u64,
    generator: String,
    families: Vec<FamilyManifest>,
}

fn now_epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn corpus_root() -> PathBuf {
    repo_root().join("crates/fsci-conformance/fixtures/adversarial")
}

fn canonical_hash(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    hash(&bytes).to_hex().to_string()
}

fn attack_surface_for(packet_family: &str) -> &'static str {
    match packet_family {
        "P2C-001" => "tolerance_validation",
        "P2C-002" => "matrix_operations",
        "P2C-003" => "format_conversions",
        _ => "unknown",
    }
}

fn special_float_token(seed: usize) -> &'static str {
    match seed % 3 {
        0 => "NaN",
        1 => "Inf",
        _ => "-Inf",
    }
}

fn payload_for(packet_family: &str, class: &str, index: usize) -> Value {
    match (packet_family, class) {
        ("P2C-001", "nan_inf") => json!({
            "rtol": special_float_token(index),
            "atol": [special_float_token(index + 1), 1e-12_f64, 0.0_f64],
            "n": (index % 7) + 1
        }),
        ("P2C-001", "extreme_dimensions") => {
            let n = [0_u64, 1, 2, 1024, 65535][index % 5];
            let vector_len = [0_u64, 1, 3, 32, 4096][index % 5];
            let mode_hint = if index.is_multiple_of(2) {
                "strict"
            } else {
                "hardened"
            };
            json!({ "n": n, "vector_len": vector_len, "mode_hint": mode_hint })
        }
        ("P2C-001", "empty_inputs") => json!({
            "rtol": 1e-6_f64,
            "atol": [],
            "n": 0,
            "first_step": 0.0_f64
        }),
        ("P2C-001", "max_size") => json!({
            "n": 1_000_000_u64,
            "atol_repr": "sparse-vector",
            "max_step": "Inf",
            "metadata_budget_bytes": 8_388_608_u64
        }),
        ("P2C-002", "nan_inf") => json!({
            "matrix": [["NaN", 1.0_f64], [2.0_f64, special_float_token(index)]],
            "rhs": [1.0_f64, special_float_token(index + 1)]
        }),
        ("P2C-002", "extreme_dimensions") => {
            let rows = [0_u64, 1, 2, 512, 4096][index % 5];
            let cols = [0_u64, 1, 2, 512, 4096][(index + 2) % 5];
            let bw = [0_u64, 1, 2, 128][index % 4];
            json!({ "rows": rows, "cols": cols, "bandwidth_hint": bw })
        }
        ("P2C-002", "empty_inputs") => {
            let op = ["solve", "inv", "det", "lstsq"][index % 4];
            json!({ "matrix": [], "rhs": [], "operation_hint": op })
        }
        ("P2C-002", "max_size") => {
            let scale = [1e300_f64, 1e-300][index % 2];
            json!({ "rows": 10_000_u64, "cols": 10_000_u64, "nnz_hint": 100_000_000_u64, "value_scale": scale })
        }
        ("P2C-003", "nan_inf") => json!({
            "objective_value": special_float_token(index),
            "gradient_value": special_float_token(index + 1),
            "x0": [0.0_f64, 1.0_f64, special_float_token(index + 2)]
        }),
        ("P2C-003", "extreme_dimensions") => {
            let pc = [0_u64, 1, 2, 128, 8192][index % 5];
            let ib = [0_u64, 1, 10, 100, 1_000_000][index % 5];
            json!({ "parameter_count": pc, "iteration_budget": ib })
        }
        ("P2C-003", "empty_inputs") => {
            let method = ["BFGS", "CG", "Powell", "brentq", "bisect"][index % 5];
            json!({ "objective": "empty", "bounds": [], "constraints": [], "method": method })
        }
        ("P2C-003", "max_size") => {
            let scale = [1e308_f64, 1e-308][index % 2];
            json!({ "parameter_count": 1_000_000_u64, "value_scale": scale, "storage_hint": "chunked" })
        }
        _ => json!({"unsupported": true}),
    }
}

fn build_vectors(packet_family: &str) -> Vec<AdversarialVector> {
    let attack_surface = attack_surface_for(packet_family).to_owned();
    let mut vectors = Vec::with_capacity(VECTORS_PER_FAMILY);
    for index in 0..VECTORS_PER_FAMILY {
        let class = CLASSES[index % CLASSES.len()].to_owned();
        let payload = payload_for(packet_family, &class, index);
        vectors.push(AdversarialVector {
            id: format!("{packet_family}-{index:03}"),
            packet_family: packet_family.to_owned(),
            attack_surface: attack_surface.clone(),
            class,
            blake3: canonical_hash(&payload),
            payload,
        });
    }
    vectors
}

fn write_jsonl(path: &Path, vectors: &[AdversarialVector]) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    for vector in vectors {
        serde_json::to_writer(&mut file, vector)
            .map_err(|error| io::Error::other(error.to_string()))?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn collect_class_counts(vectors: &[AdversarialVector]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for vector in vectors {
        *counts.entry(vector.class.clone()).or_insert(0) += 1;
    }
    counts
}

fn parse_args() -> PathBuf {
    let mut args = std::env::args().skip(1);
    if let Some(flag) = args.next() {
        if flag == "--output-root" {
            if let Some(path) = args.next() {
                return PathBuf::from(path);
            }
            eprintln!("missing value for --output-root");
            std::process::exit(2);
        }
        eprintln!("unrecognized argument: {flag}");
        std::process::exit(2);
    }
    corpus_root()
}

fn write_family(
    root: &Path,
    packet_family: &str,
) -> Result<FamilyManifest, Box<dyn std::error::Error>> {
    let family_dir = root.join(packet_family);
    fs::create_dir_all(&family_dir)?;

    let vectors = build_vectors(packet_family);
    let corpus_path = family_dir.join("corpus.jsonl");
    write_jsonl(&corpus_path, &vectors)?;

    let corpus_bytes = fs::read(&corpus_path)?;
    let corpus_hash = hash(&corpus_bytes).to_hex().to_string();
    let class_counts = collect_class_counts(&vectors);

    Ok(FamilyManifest {
        packet_family: packet_family.to_owned(),
        attack_surface: attack_surface_for(packet_family).to_owned(),
        vector_count: vectors.len(),
        class_counts,
        corpus_file: corpus_path
            .strip_prefix(root)
            .unwrap_or(&corpus_path)
            .to_string_lossy()
            .to_string(),
        corpus_blake3: corpus_hash,
    })
}

fn write_top_manifest(root: &Path, families: Vec<FamilyManifest>) -> Result<(), io::Error> {
    let manifest = CorpusManifest {
        schema_version: 1,
        generated_at_epoch_seconds: now_epoch_seconds(),
        generator: "adversarial_corpus.rs".to_owned(),
        families,
    };
    let path = root.join("manifest.json");
    let mut document = Map::new();
    document.insert(
        "schema_version".to_owned(),
        Value::from(manifest.schema_version),
    );
    document.insert(
        "generated_at_epoch_seconds".to_owned(),
        Value::from(manifest.generated_at_epoch_seconds),
    );
    document.insert("generator".to_owned(), Value::from(manifest.generator));
    document.insert(
        "families".to_owned(),
        serde_json::to_value(manifest.families).unwrap_or(Value::Array(Vec::new())),
    );
    let payload = serde_json::to_string_pretty(&Value::Object(document))
        .map_err(|error| io::Error::other(error.to_string()))?;
    fs::write(path, payload)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_root = parse_args();
    fs::create_dir_all(&output_root)?;

    let mut families = Vec::new();
    for packet_family in ["P2C-001", "P2C-002", "P2C-003"] {
        families.push(write_family(&output_root, packet_family)?);
    }
    write_top_manifest(&output_root, families)?;

    println!(
        "generated adversarial corpus at {}",
        output_root.to_string_lossy()
    );
    Ok(())
}
