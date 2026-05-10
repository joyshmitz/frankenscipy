#![forbid(unsafe_code)]

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn main() {
    println!("cargo:rerun-if-env-changed=FSCI_REQUIRE_SCIPY_ORACLE");
    println!("cargo:rerun-if-env-changed=RCH_ENV_ALLOWLIST");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_DIR");

    if env_truthy("FSCI_REQUIRE_SCIPY_ORACLE") {
        println!("cargo:rustc-env=FSCI_REQUIRE_SCIPY_ORACLE=1");
    }
}
