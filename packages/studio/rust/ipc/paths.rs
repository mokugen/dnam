#![forbid(unsafe_code)]
//! Native backend detection, OS user config resolution, and private export writers.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{StudioError, StudioResult};

static ATOMIC_TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Returns the active native SIMD backend label on the current CPU architecture.
#[must_use]
pub const fn detect_native_backend_str() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "ARM64 NEON"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64 AVX2+FMA"
    } else {
        "Scalar"
    }
}

fn resolve_os_config_dir_from_vars(
    home: PathBuf,
    appdata: Option<&str>,
    xdg: Option<&str>,
    target_os: &str,
) -> PathBuf {
    match target_os {
        "macos" => home
            .join("Library")
            .join("Application Support")
            .join("dnam"),
        "windows" => match appdata.map(str::trim).filter(|s| !s.is_empty()) {
            Some(dir) => PathBuf::from(dir).join("dnam"),
            None => home.join("AppData").join("Roaming").join("dnam"),
        },
        _ => match xdg.map(str::trim).filter(|s| !s.is_empty()) {
            Some(dir) => PathBuf::from(dir).join("dnam"),
            None => home.join(".config").join("dnam"),
        },
    }
}

/// Returns the OS-appropriate user configuration directory for `dnam-studio`.
#[must_use]
pub fn default_user_config_dir() -> PathBuf {
    if let Ok(env_dir) = env::var("DNAM_CONFIG_DIR") {
        let trimmed = env_dir.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let appdata = env::var("APPDATA").ok().filter(|s| !s.trim().is_empty());
    let xdg = env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let home = env::var("HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            env::var("USERPROFILE")
                .ok()
                .filter(|s| !s.trim().is_empty())
        });
    match home {
        Some(h) => resolve_os_config_dir_from_vars(
            PathBuf::from(h.trim()),
            appdata.as_deref(),
            xdg.as_deref(),
            env::consts::OS,
        ),
        None => PathBuf::from(".dnam-config"),
    }
}

/// Atomically writes an export or report file into `export_dir` (`0o700` on Unix),
/// rejecting path traversal and symlinks (`O_CREAT | O_EXCL`).
pub fn write_private_studio_export_file(
    export_dir: &Path,
    filename: &str,
    content: &[u8],
    overwrite: bool,
) -> StudioResult<PathBuf> {
    use std::io::Write as _;
    use std::path::Component;

    if filename.is_empty()
        || filename != filename.trim()
        || filename.contains(['/', '\\', '\0', ':'])
        || export_dir
            .components()
            .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(StudioError::Validation(format!(
            "Invalid export path or filename '{filename}'"
        )));
    }
    let comps: Vec<Component<'_>> = Path::new(filename).components().collect();
    if comps.len() != 1 || !matches!(comps[0], Component::Normal(_)) {
        return Err(StudioError::Validation(format!(
            "Invalid export filename '{filename}'"
        )));
    }

    if let Ok(meta) = fs::symlink_metadata(export_dir)
        && meta.file_type().is_symlink()
    {
        return Err(StudioError::Validation(format!(
            "Refusing symlink export directory: {}",
            export_dir.display()
        )));
    }
    fs::create_dir_all(export_dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(export_dir, fs::Permissions::from_mode(0o700))?;
    }

    let target = export_dir.join(filename);
    match fs::symlink_metadata(&target) {
        Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => {
            return Err(StudioError::Validation(format!(
                "Export target is a symlink or non-regular file: {}",
                target.display()
            )));
        }
        Ok(_) if !overwrite => return Ok(target),
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }

    let seq = ATOMIC_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp_path = target.with_extension(format!("tmp.{}_{}", std::process::id(), seq));
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    let mut file = opts.open(&tmp_path)?;
    if let Err(err) = file.write_all(content) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err.into());
    }
    drop(file);
    if let Err(err) = fs::rename(&tmp_path, &target) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err.into());
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_ipc_paths_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn config_dir_and_private_export_writer_enforce_safety() {
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), None, None, "macos"),
            PathBuf::from("/u/Library/Application Support/dnam")
        );
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), Some("/ad"), None, "windows"),
            PathBuf::from("/ad/dnam")
        );
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), None, None, "windows"),
            PathBuf::from("/u/AppData/Roaming/dnam")
        );
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), None, Some("/xdg"), "linux"),
            PathBuf::from("/xdg/dnam")
        );
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), None, None, "linux"),
            PathBuf::from("/u/.config/dnam")
        );
        assert!(!default_user_config_dir().as_os_str().is_empty());

        let dir = temp_dir("exports");
        let export_dir = dir.join("private_out");
        let p1 = write_private_studio_export_file(&export_dir, "model.dnam", b"v1", false)
            .expect("write new export");
        assert_eq!(fs::read(&p1).unwrap(), b"v1");
        let p2 = write_private_studio_export_file(&export_dir, "model.dnam", b"v2", false)
            .expect("keep existing when overwrite=false");
        assert_eq!(fs::read(&p2).unwrap(), b"v1");
        let p3 = write_private_studio_export_file(&export_dir, "model.dnam", b"v2", true)
            .expect("overwrite when overwrite=true");
        assert_eq!(fs::read(&p3).unwrap(), b"v2");

        assert!(write_private_studio_export_file(&export_dir, "", b"x", true).is_err());
        assert!(write_private_studio_export_file(&export_dir, "../bad", b"x", true).is_err());
        assert!(write_private_studio_export_file(&export_dir, "a/b.dnam", b"x", true).is_err());
        assert!(
            write_private_studio_export_file(Path::new("../bad"), "ok.dnam", b"x", true).is_err()
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
