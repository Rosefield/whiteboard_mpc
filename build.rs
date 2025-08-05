fn main() {
    cxx_build::bridge("src/ffi.rs")
        .file("dependencies/mpc_runner.cpp")
        .flag("-std=c++17")
        .flag("-lcrypto")
        .flag("-O2")
        .flag("-march=native")
        // There is a bunch of hackery with pointer casting to SIMD structs
        .flag("-Wno-ignored-attributes")
        .compile("mpc_runner");

    println!("cargo:rerun-if-changed=dependencies/emp_agmpc/abitmp.h");
    println!("cargo:rerun-if-changed=dependencies/emp_agmpc/fpremp.h");
    println!("cargo:rerun-if-changed=dependencies/emp_agmpc/mpc.h");
    println!("cargo:rerun-if-changed=dependencies/mpc_ffi.h");
    println!("cargo:rerun-if-changed=dependencies/mpc_runner.cpp");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rustc-link-lib=crypto");
}
