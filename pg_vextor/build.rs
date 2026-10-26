use std::process::Command;

fn main() {
    let config = std::env::var("PGRX_PG_CONFIG_PATH").unwrap_or_else(|_| "pg_config".into());
    let output = Command::new(config)
        .arg("--includedir-server")
        .output()
        .unwrap();
    assert!(output.status.success());
    let include = String::from_utf8(output.stdout).unwrap();
    cc::Build::new()
        .file("src/bridge.c")
        .file("../src/data_path.c")
        .define("VEXTOR_PLAN_ONLY", None)
        .include(include.trim())
        .include("../include")
        .flag("-std=gnu11")
        // PostgreSQL inline headers contain unused parameters.
        .flag("-Wno-unused-parameter")
        .compile("vextor_pg_bridge");
    println!("cargo:rerun-if-changed=src/bridge.c");
    println!("cargo:rerun-if-changed=../src/data_path.c");
    println!("cargo:rerun-if-changed=../include/vextor_protocol.h");
    println!("cargo:rerun-if-changed=../include/vextor_data_path.h");
}
