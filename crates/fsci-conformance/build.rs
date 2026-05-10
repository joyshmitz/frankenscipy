#![forbid(unsafe_code)]

fn main() {
    println!("cargo:rerun-if-env-changed=FSCI_REQUIRE_SCIPY_ORACLE");
    println!("cargo:rerun-if-env-changed=RCH_ENV_ALLOWLIST");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_DIR");

    let rch_remote_target = std::env::var("CARGO_TARGET_DIR")
        .is_ok_and(|target_dir| target_dir.split('/').any(|part| part == ".rch-target"));

    if rch_remote_target || std::env::var_os("FSCI_REQUIRE_SCIPY_ORACLE").is_some() {
        println!("cargo:rustc-env=FSCI_REQUIRE_SCIPY_ORACLE=1");
    }
}
