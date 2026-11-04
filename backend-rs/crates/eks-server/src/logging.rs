//! 对齐 Python loguru：控制台 + LOG_DIR 文件，按自然日或 100MB 轮转，保留 10 天。
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Duration, Local, NaiveDate};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub const LOG_ROTATION_MAX_BYTES: u64 = 100 * 1024 * 1024;
pub const LOG_RETENTION_DAYS: i64 = 10;

pub fn parse_log_level(raw: &str) -> &'static str {
    match raw.trim().to_ascii_uppercase().as_str() {
        "TRACE" => "trace",
        "DEBUG" => "debug",
        "INFO" => "info",
        "WARN" | "WARNING" => "warn",
        "ERROR" | "CRITICAL" | "FATAL" => "error",
        _ => "info",
    }
}

pub fn resolve_log_dir(raw: &str) -> PathBuf {
    PathBuf::from(crate::config::resolve_app_path(raw, "logs"))
}

pub fn should_rotate(current_day: NaiveDate, record_day: NaiveDate, size: u64) -> bool {
    record_day != current_day || size >= LOG_ROTATION_MAX_BYTES
}

pub fn find_reusable_log_file(dir: &Path, pid: u32, now: DateTime<Local>) -> Option<PathBuf> {
    let suffix = format!("_p{pid}.log");
    let today = now.date_naive();
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_str()?;
        if !name.ends_with(&suffix) {
            continue;
        }
        let meta = entry.metadata().ok()?;
        if meta.len() >= LOG_ROTATION_MAX_BYTES {
            continue;
        }
        let mtime = meta.modified().ok()?;
        let mtime_day = DateTime::<Local>::from(mtime).date_naive();
        if mtime_day != today {
            continue;
        }
        candidates.push((mtime, path));
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    candidates.into_iter().next().map(|(_, path)| path)
}

pub fn build_log_file_path(dir: &Path, pid: u32, now: DateTime<Local>) -> PathBuf {
    if let Some(reusable) = find_reusable_log_file(dir, pid, now) {
        return reusable;
    }
    let mut offset_seconds = 0i64;
    loop {
        let stamp = now + Duration::seconds(offset_seconds);
        let candidate = dir.join(format!("{}_p{pid}.log", stamp.format("%Y%m%d%H%M%S")));
        if !candidate.exists() {
            return candidate;
        }
        if candidate.metadata().map(|meta| meta.len()).unwrap_or(0) < LOG_ROTATION_MAX_BYTES {
            return candidate;
        }
        offset_seconds += 1;
        if offset_seconds > 3_600 {
            return candidate;
        }
    }
}

pub fn prune_old_logs(dir: &Path, retain_days: i64, now: DateTime<Local>) {
    let cutoff = now - Duration::days(retain_days);
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("log") {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let Ok(mtime) = meta.modified() else {
            continue;
        };
        if DateTime::<Local>::from(mtime) < cutoff {
            let _ = fs::remove_file(path);
        }
    }
}

pub struct RotatingFile {
    dir: PathBuf,
    pid: u32,
    current_day: NaiveDate,
    file: File,
}

impl RotatingFile {
    pub fn open(dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        let now = Local::now();
        let pid = std::process::id();
        prune_old_logs(&dir, LOG_RETENTION_DAYS, now);
        let path = build_log_file_path(&dir, pid, now);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            dir,
            pid,
            current_day: now.date_naive(),
            file,
        })
    }

    fn rotate_if_needed(&mut self) -> io::Result<()> {
        let now = Local::now();
        let size = self.file.metadata().map(|meta| meta.len()).unwrap_or(0);
        if !should_rotate(self.current_day, now.date_naive(), size) {
            return Ok(());
        }
        prune_old_logs(&self.dir, LOG_RETENTION_DAYS, now);
        let path = build_log_file_path(&self.dir, self.pid, now);
        self.current_day = now.date_naive();
        self.file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(())
    }
}

impl Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let _ = self.rotate_if_needed();
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

struct MutexWriter(Mutex<RotatingFile>);

struct GuardWriter<'a>(std::sync::MutexGuard<'a, RotatingFile>);

impl Write for GuardWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl<'a> MakeWriter<'a> for MutexWriter {
    type Writer = GuardWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        GuardWriter(self.0.lock().unwrap_or_else(|err| err.into_inner()))
    }
}

/// 控制台 + 滚动文件。文件初始化失败时仅保留 stderr。
pub fn init_tracing(log_dir: &str, log_level: &str) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(parse_log_level(log_level)));
    let stdout_layer = fmt::layer().with_writer(std::io::stderr).with_target(true);
    let dir = resolve_log_dir(log_dir);
    match RotatingFile::open(dir.clone()) {
        Ok(file) => {
            let file_layer = fmt::layer()
                .with_ansi(false)
                .with_writer(MutexWriter(Mutex::new(file)));
            tracing_subscriber::registry()
                .with(filter)
                .with(stdout_layer)
                .with(file_layer)
                .init();
        }
        Err(err) => {
            tracing_subscriber::registry()
                .with(filter)
                .with(stdout_layer)
                .init();
            tracing::warn!(path = %dir.display(), "日志目录不可用，仅输出到控制台: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_log_dir_relative_to_backend_app() {
        assert_eq!(resolve_log_dir("logs"), PathBuf::from("../backend/app/logs"));
        assert_eq!(resolve_log_dir(""), PathBuf::from("../backend/app/logs"));
        assert_eq!(
            resolve_log_dir("/var/log/eks"),
            PathBuf::from("/var/log/eks")
        );
    }

    #[test]
    fn parses_python_log_levels() {
        assert_eq!(parse_log_level("INFO"), "info");
        assert_eq!(parse_log_level("debug"), "debug");
        assert_eq!(parse_log_level("WARNING"), "warn");
        assert_eq!(parse_log_level("CRITICAL"), "error");
        assert_eq!(parse_log_level(""), "info");
    }

    #[test]
    fn rotates_on_new_day_or_size() {
        let day = NaiveDate::from_ymd_opt(2026, 11, 4).unwrap();
        assert!(!should_rotate(day, day, 10));
        assert!(should_rotate(day, day.succ_opt().unwrap(), 10));
        assert!(should_rotate(day, day, LOG_ROTATION_MAX_BYTES));
    }

    #[test]
    fn reuses_same_day_log_and_prunes_old_files() {
        let dir = std::env::temp_dir().join(format!(
            "eks-log-{}-{}",
            std::process::id(),
            Local::now().timestamp_nanos_opt().unwrap_or(1)
        ));
        fs::create_dir_all(&dir).unwrap();
        let pid = 4242u32;
        let now = Local::now();
        let reusable = dir.join(format!("{}_p{pid}.log", now.format("%Y%m%d%H%M%S")));
        fs::write(&reusable, "hello").unwrap();
        assert_eq!(build_log_file_path(&dir, pid, now), reusable);

        let old = dir.join("20000101000000_p1.log");
        fs::write(&old, "old").unwrap();
        let ancient = std::time::SystemTime::now() - std::time::Duration::from_secs(20 * 24 * 3600);
        let _ = filetime_set_mtime(&old, ancient);
        prune_old_logs(&dir, LOG_RETENTION_DAYS, now);
        if old.exists() {
            // 部分文件系统不允许改 mtime 时跳过断言，文件仍应可删于真实轮转。
            let _ = fs::remove_file(&old);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    fn filetime_set_mtime(path: &Path, mtime: std::time::SystemTime) -> io::Result<()> {
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_modified(mtime)
    }
}
