#![forbid(unsafe_code)]

use fsci_conformance::e2e::{E2eOrchestratorConfig, run_orchestrator};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Clone)]
struct CliArgs {
    artifact_root: PathBuf,
    fixture_root: PathBuf,
    packet_filter: Option<String>,
    scenario_filter: Option<String>,
    run_id: Option<String>,
}

#[derive(Debug, Clone)]
enum CliParseError {
    Help,
    Message(String),
}

fn parse_cli_args(args: &[String]) -> Result<CliArgs, CliParseError> {
    let defaults = E2eOrchestratorConfig::default();
    let mut artifact_root = defaults.artifact_root;
    let mut fixture_root = defaults.fixture_root;
    let mut packet_filter = None;
    let mut scenario_filter = None;
    let mut run_id = None;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(CliParseError::Help),
            "--artifact-root" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(CliParseError::Message(String::from(
                        "missing value for --artifact-root",
                    )));
                };
                artifact_root = PathBuf::from(value);
                index += 2;
            }
            "--fixture-root" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(CliParseError::Message(String::from(
                        "missing value for --fixture-root",
                    )));
                };
                fixture_root = PathBuf::from(value);
                index += 2;
            }
            "--packet" | "--packet-filter" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(CliParseError::Message(String::from(
                        "missing value for --packet",
                    )));
                };
                packet_filter = Some(value.clone());
                index += 2;
            }
            "--scenario" | "--scenario-filter" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(CliParseError::Message(String::from(
                        "missing value for --scenario",
                    )));
                };
                scenario_filter = Some(value.clone());
                index += 2;
            }
            "--run-id" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(CliParseError::Message(String::from(
                        "missing value for --run-id",
                    )));
                };
                run_id = Some(value.clone());
                index += 2;
            }
            unknown => {
                return Err(CliParseError::Message(format!(
                    "unrecognized argument `{unknown}`"
                )));
            }
        }
    }

    Ok(CliArgs {
        artifact_root,
        fixture_root,
        packet_filter,
        scenario_filter,
        run_id,
    })
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} [--artifact-root <path>] [--fixture-root <path>] [--packet <id>] [--scenario <id>] [--run-id <id>]"
    );
    eprintln!("  --artifact-root <path>  root containing packet artifact directories");
    eprintln!("  --fixture-root <path>   root containing fixture json files");
    eprintln!("  --packet <id>           run only scenarios for one packet");
    eprintln!("  --scenario <id>         run only one scenario id");
    eprintln!("  --run-id <id>           explicit run id (default uses timestamp)");
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let program = argv
        .first()
        .cloned()
        .unwrap_or_else(|| String::from("e2e_orchestrator"));

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

    let config = E2eOrchestratorConfig {
        artifact_root: args.artifact_root,
        fixture_root: args.fixture_root,
        packet_filter: args.packet_filter,
        scenario_filter: args.scenario_filter,
        run_id: args.run_id,
    };

    match run_orchestrator(&config) {
        Ok(summary) => {
            eprintln!(
                "run_id={} total={} passed={} failed={}",
                summary.run_id,
                summary.total_scenarios,
                summary.passed_scenarios,
                summary.failed_scenarios
            );
            for scenario in &summary.scenario_results {
                let status = if scenario.passed { "PASS" } else { "FAIL" };
                eprintln!(
                    "{status} packet={} scenario={} bundle={}",
                    scenario.packet_id,
                    scenario.scenario_id,
                    scenario.run_bundle_dir.display()
                );
                if !scenario.passed {
                    eprintln!("replay_command={}", scenario.replay_command);
                }
            }
            if summary.failed_scenarios > 0 {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            eprintln!("orchestrator error: {error}");
            ExitCode::from(2)
        }
    }
}
