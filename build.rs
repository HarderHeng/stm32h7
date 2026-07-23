use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = &PathBuf::from(env::var_os("OUT_DIR").unwrap());

    /* memory.x */

    File::create(out.join("memory.x"))
        .unwrap()
        .write_all(include_bytes!("memory.x"))
        .unwrap();

    /* sections.x */

    File::create(out.join("sections.x"))
        .unwrap()
        .write_all(include_bytes!("sections.x"))
        .unwrap();

    println!("cargo:rustc-link-search={}", out.display());

    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=sections.x");
    println!("cargo:rerun-if-changed=weights/manifest.json");
    println!("cargo:rerun-if-env-changed=STRICT_WEIGHTS");

    /* Weight validation gate: warns on synthetic weights; hard-fails when STRICT_WEIGHTS
 * is one of: 1, true, yes, on. Any other value (unset, empty, "0", "false") keeps
 * dev mode where only a warning is printed. */

    let manifest_bytes = include_bytes!("weights/manifest.json");
    let manifest = std::str::from_utf8(manifest_bytes).unwrap_or("");
    let validation_passed = manifest.contains("\"validation_passed\": true")
        || manifest.contains("\"validation_passed\":true");
    let strict = matches!(
        env::var("STRICT_WEIGHTS").as_deref(),
        Ok("1") | Ok("true") | Ok("yes") | Ok("on")
    );

    if !validation_passed {
        eprintln!(
            "=================================================================\n\
             WARNING: weights/manifest.json reports validation_passed=false.\n\
             This means the embedded INT8 weights are SYNTHETIC (random\n\
             initialization). Flashing this firmware will produce meaningless\n\
             motor commands — DO NOT run on hardware.\n\n\
             Fix: re-run tools/quantize_onnx.py against a trained policy.onnx:\n\
               python3 tools/quantize_onnx.py --input <policy.onnx> \\\n\
       --output weights/ --strict\n\n\
             To override in dev: leave STRICT_WEIGHTS unset (or =0/false/no/off).\n\
             To enforce strict mode: STRICT_WEIGHTS=1 cargo build\n\
             ================================================================="
        );
        if strict {
            panic!(
                "Refusing to build: weights failed validation. Unset STRICT_WEIGHTS (or =0/false) to override."
            );
        }
    }

    /* linker args */

    println!("cargo:rustc-link-arg-bins=--nmagic");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    println!("cargo:rustc-link-arg-bins=-Tdefmt.x");
    println!("cargo:rustc-link-arg-bins=-Tsections.x");
}