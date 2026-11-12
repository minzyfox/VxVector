use std::process::Command;

fn main() {
    let config = std::env::var("PGRX_PG_CONFIG_PATH").unwrap_or_else(|_| "pg_config".into());
    let output = Command::new(config)
        .arg("--includedir-server")
        .output()
        .unwrap();
    assert!(output.status.success());
    let include = String::from_utf8(output.stdout).unwrap();
    // vxvector_plan_cpu_v1 calls the plan in the Vx object. The object needs
    // only abort and write from libc, not the Vx runtime library.
    let kernel = std::env::var("VXVECTOR_KERNEL_OBJECT")
        .unwrap_or_else(|_| "../build/linux/kmeans.o".into());
    assert!(
        std::path::Path::new(&kernel).exists(),
        "{kernel} is missing. Run scripts/build_linux_worker.sh first."
    );
    cc::Build::new()
        .file("src/bridge.c")
        .file("../src/data_path.c")
        .object(&kernel)
        .define("VXVECTOR_PLAN_ONLY", None)
        .include(include.trim())
        .include("../include")
        .flag("-std=gnu11")
        // PostgreSQL inline headers contain unused parameters.
        .flag("-Wno-unused-parameter")
        .compile("vxvector_pg_bridge");
    println!("cargo:rerun-if-changed=src/bridge.c");
    println!("cargo:rerun-if-changed=../src/data_path.c");
    println!("cargo:rerun-if-changed={kernel}");
    println!("cargo:rerun-if-env-changed=VXVECTOR_KERNEL_OBJECT");
    println!("cargo:rerun-if-changed=../include/vxvector_protocol.h");
    println!("cargo:rerun-if-changed=../include/vxvector_data_path.h");
}
