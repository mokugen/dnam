#![forbid(unsafe_code)]

//! CLI entrypoint for compiling, validating, and drift-checking SONA v3.0 WebGPU shaders.

use anyhow::{Result, bail};
use sona_shaders::compiler::{ShaderPaths, build_shaders, check_drift};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let paths = ShaderPaths::workspace_default();
    match args.as_slice() {
        [cmd] if cmd == "build" => {
            let manifest = build_shaders(&paths)?;
            println!(
                "Built and validated {} portable + {} subgroup WGSL kernels into {}",
                manifest.portable.len(),
                manifest.subgroups.len(),
                paths.wgsl_dir.display()
            );
            Ok(())
        }
        [cmd] if cmd == "check-drift" => {
            check_drift(&paths)?;
            println!(
                "Verified zero WGSL/manifest drift in {}",
                paths.wgsl_dir.display()
            );
            Ok(())
        }
        _ => bail!("usage: sona-shaders-gen <build|check-drift>"),
    }
}
