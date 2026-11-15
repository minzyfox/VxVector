use std::process::Command;

fn main() {
    let config = std::env::var("PGRX_PG_CONFIG_PATH").unwrap_or_else(|_| "pg_config".into());
    let output = Command::new(&config)
        .arg("--includedir-server")
        .output()
        .expect("could not run pg_config --includedir-server");
    assert!(
        output.status.success(),
        "pg_config --includedir-server failed"
    );
    let include = String::from_utf8(output.stdout).unwrap();
    let output = Command::new(&config)
        .arg("--cppflags")
        .output()
        .expect("could not run pg_config --cppflags");
    assert!(output.status.success(), "pg_config --cppflags failed");
    let cppflags = String::from_utf8(output.stdout).unwrap();
    // vxvector_plan_cpu_v1 calls the plan in the Vx object. The object needs
    // only abort and write from libc, not the Vx runtime library.
    let default_kernel = if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        "../build/kmeans.o"
    } else {
        "../build/linux/kmeans.o"
    };
    let kernel = std::env::var("VXVECTOR_KERNEL_OBJECT").unwrap_or_else(|_| default_kernel.into());
    assert!(
        std::path::Path::new(&kernel).exists(),
        "{kernel} is missing. Build the Vx k-means kernel for this platform first."
    );
    let mut build = cc::Build::new();
    let mut flags = cppflags.split_whitespace();
    while let Some(flag) = flags.next() {
        build.flag(flag);
        if flag == "-isysroot" {
            build.flag(
                flags
                    .next()
                    .expect("pg_config returned -isysroot without a path"),
            );
        }
    }
    build
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
    println!("cargo:rerun-if-env-changed=PGRX_PG_CONFIG_PATH");
    println!("cargo:rerun-if-changed=../include/vxvector_protocol.h");
    println!("cargo:rerun-if-changed=../include/vxvector_data_path.h");
}
