use std::env;
use std::path::PathBuf;

fn probe_opus_pkg_config() -> Option<Vec<PathBuf>> {
    match pkg_config::Config::new()
        .atleast_version("1.3")
        .probe("opus")
    {
        Err(e) => {
            eprintln!("pkg_config: {}", e);
            None
        }
        Ok(lib) => Some(lib.include_paths),
    }
}

fn probe_opus_path_patch(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    // MacOs homebrew Opus specific patch
    // we get path like /opt/homebrew/Cellar/opus/1.6.1/include/opus ,
    // but it should be /opt/homebrew/Cellar/opus/1.6.1/include

    for path in paths.iter() {
        if path.join("opus/opus.h").exists() {
            return paths;
        }
    }

    let mut op_extra_path: Option<PathBuf> = None;
    for path in paths.iter() {
        if let Some(parent) = path.parent() {
            if parent.join("opus/opus.h").exists() {
                op_extra_path = Some(parent.into());
                break;
            }
        }
    }

    let Some(extra_path) = op_extra_path else {
        return paths;
    };

    let mut paths = paths;
    paths.push(extra_path);
    paths
}

fn do_opus() {
    println!("cargo:rerun-if-changed=./interop/opus_wrapper.h");
    println!("cargo:rerun-if-changed=./interop/opus_wrapper.c");

    let include_paths = if env::var_os("VCPKG_ROOT").is_some() {
        env::set_var("VCPKGRS_DYNAMIC", "1");
        let pkg = vcpkg::Config::new()
            .find_package("opus")
            .expect("cant't find opus package");
        pkg.include_paths
    } else {
        let path = probe_opus_pkg_config().expect("Couldn't find opus");
        probe_opus_path_patch(path)
    };

    let bindgen_builder = bindgen::Builder::default()
        .trust_clang_mangling(false)
        .size_t_is_usize(true)
        .header("opus_wrapper.h");

    let mut cc_builder = cc::Build::new();
    cc_builder.file("opus_wrapper.c");

    for elm in include_paths.iter() {
        cc_builder.include(elm);
    }

    let bindings = bindgen_builder
        .allowlist_function("_opus_.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("opus_bindings.rs"))
        .expect("Couldn't write opus_bindings!");

    cc_builder.compile("opus-wrapper")
}

fn main() {
    do_opus();
}
