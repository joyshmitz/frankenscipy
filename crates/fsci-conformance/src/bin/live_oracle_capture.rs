#![forbid(unsafe_code)]

use fsci_conformance::{
    DifferentialOracleConfig, HarnessConfig, LiveOracleCaptureReport,
    default_zero_drift_thresholds, evaluate_drift_gate, run_live_oracle_capture_lane,
};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Clone)]
struct CliArgs {
    fixture_root: PathBuf,
    oracle_root: PathBuf,
    python_path: PathBuf,
    timeout_secs: u64,
    packet_filter: Option<String>,
    output_path: Option<PathBuf>,
    allow_missing_oracle: bool,
    allow_drift: bool,
}

#[derive(Debug, Clone)]
enum CliParseError {
    Help,
    Message(String),
}

fn parse_cli_args(args: &[String]) -> Result<CliArgs, CliParseError> {
    let defaults = HarnessConfig::default_paths();
    let oracle_defaults = DifferentialOracleConfig::default();
    let mut fixture_root = defaults.fixture_root;
    let mut oracle_root = defaults.oracle_root;
    let mut python_path = oracle_defaults.python_path;
    let mut timeout_secs = oracle_defaults.timeout_secs;
    let mut packet_filter = None;
    let mut output_path = None;
    let mut allow_missing_oracle = false;
    let mut allow_drift = false;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(CliParseError::Help),
            "--fixture-root" => {
                fixture_root = next_path(args, index, "--fixture-root")?;
                index += 2;
            }
            "--oracle-root" => {
                oracle_root = next_path(args, index, "--oracle-root")?;
                index += 2;
            }
            "--python" | "--python-path" => {
                python_path = next_path(args, index, "--python")?;
                index += 2;
            }
            "--timeout-secs" => {
                timeout_secs =
                    next_value(args, index, "--timeout-secs")?
                        .parse()
                        .map_err(|error| {
                            CliParseError::Message(format!("invalid --timeout-secs: {error}"))
                        })?;
                index += 2;
            }
            "--packet" | "--packet-filter" => {
                packet_filter = Some(next_value(args, index, "--packet")?.to_owned());
                index += 2;
            }
            "--output" => {
                output_path = Some(next_path(args, index, "--output")?);
                index += 2;
            }
            "--allow-missing-oracle" => {
                allow_missing_oracle = true;
                index += 1;
            }
            "--allow-drift" => {
                allow_drift = true;
                index += 1;
            }
            unknown => {
                return Err(CliParseError::Message(format!(
                    "unrecognized argument `{unknown}`"
                )));
            }
        }
    }

    Ok(CliArgs {
        fixture_root,
        oracle_root,
        python_path,
        timeout_secs,
        packet_filter,
        output_path,
        allow_missing_oracle,
        allow_drift,
    })
}

fn next_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str, CliParseError> {
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| CliParseError::Message(format!("missing value for {flag}")))
}

fn next_path(args: &[String], index: usize, flag: &str) -> Result<PathBuf, CliParseError> {
    next_value(args, index, flag).map(PathBuf::from)
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} [--fixture-root <path>] [--oracle-root <path>] [--python <path>] [--packet <filter>] [--output <path>] [--timeout-secs <n>] [--allow-missing-oracle] [--allow-drift]"
    );
}

fn write_summary(
    report: &LiveOracleCaptureReport,
    output_path: Option<&PathBuf>,
) -> Result<(), ExitCode> {
    let bytes = match serde_json::to_vec_pretty(report) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to serialize live oracle report: {error}");
            return Err(ExitCode::from(2));
        }
    };

    if let Some(path) = output_path {
        if let Err(error) = std::fs::write(path, bytes) {
            eprintln!("failed to write {}: {error}", path.display());
            return Err(ExitCode::from(2));
        }
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }

    Ok(())
}

fn main() -> ExitCode {
    let argv = std::env::args().collect::<Vec<_>>();
    let program = argv
        .first()
        .cloned()
        .unwrap_or_else(|| String::from("live_oracle_capture"));
    let args = match parse_cli_args(&argv[1..]) {
        Ok(args) => args,
        Err(CliParseError::Help) => {
            print_usage(&program);
            return ExitCode::SUCCESS;
        }
        Err(CliParseError::Message(message)) => {
            eprintln!("{message}");
            print_usage(&program);
            return ExitCode::from(2);
        }
    };

    let harness = HarnessConfig {
        oracle_root: args.oracle_root,
        fixture_root: args.fixture_root,
        strict_mode: true,
    };
    let oracle = DifferentialOracleConfig {
        python_path: args.python_path,
        script_path: DifferentialOracleConfig::default().script_path,
        timeout_secs: args.timeout_secs,
        required: !args.allow_missing_oracle,
    };

    let report =
        match run_live_oracle_capture_lane(&harness, &oracle, args.packet_filter.as_deref()) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("live oracle capture failed: {error}");
                return ExitCode::from(2);
            }
        };

    if let Err(code) = write_summary(&report, args.output_path.as_ref()) {
        return code;
    }
    if !args.allow_missing_oracle && report.oracle_unavailable_packets > 0 {
        eprintln!(
            "oracle unavailable for {} packet(s)",
            report.oracle_unavailable_packets
        );
        return ExitCode::from(1);
    }
    if !args.allow_drift {
        let drift_gate = evaluate_drift_gate(&report, &default_zero_drift_thresholds());
        if !drift_gate.passed() {
            eprintln!(
                "detected {} drifted case(s) above threshold across {} packet(s)",
                drift_gate.total_drifted_cases, drift_gate.failed_packets
            );
            for packet in drift_gate.packets.iter().filter(|packet| !packet.passed) {
                eprintln!(
                    "{}: {} drifted case(s), max allowed {}",
                    packet.packet_id, packet.drifted_cases, packet.max_allowed_drifted_cases
                );
            }
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}
