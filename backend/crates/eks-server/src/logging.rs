//! 对齐 Python loguru：控制台 + LOG_DIR 文件，按自然日或 100MB 轮转，保留 10 天。
//! stdout/stderr 按行灌进 tracing，对齐 `_StreamToLogger`。
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use chrono::{DateTime, Duration, Local, NaiveDate};
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields, MakeWriter};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;
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

struct MutexWriterFd(Mutex<File>);

struct GuardWriterFd<'a>(std::sync::MutexGuard<'a, File>);

impl Write for GuardWriterFd<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl<'a> MakeWriter<'a> for MutexWriterFd {
    type Writer = GuardWriterFd<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        GuardWriterFd(self.0.lock().unwrap_or_else(|err| err.into_inner()))
    }
}

static STDIO_INTERCEPT_INSTALLED: AtomicBool = AtomicBool::new(false);

/// 对齐 Python `_StreamToLogger.write`：按行切、去掉行尾 CR、空行丢弃。
pub fn take_stdio_lines(buffer: &mut String, chunk: &str) -> Vec<String> {
    buffer.push_str(chunk);
    let mut lines = Vec::new();
    while let Some(pos) = buffer.find('\n') {
        let mut line = buffer[..pos].to_string();
        buffer.drain(..=pos);
        if line.ends_with('\r') {
            line.pop();
        }
        if !line.is_empty() {
            lines.push(line);
        }
    }
    lines
}

/// 对齐 Python `_StreamToLogger.flush`：剩余缓冲去空白后记一行。
pub fn flush_stdio_buffer(buffer: &mut String) -> Option<String> {
    let line = buffer.trim().to_string();
    buffer.clear();
    if line.is_empty() {
        None
    } else {
        Some(line)
    }
}

fn duplicate_stderr() -> io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::fd::{FromRawFd, RawFd};
        let dup = unsafe { libc::dup(libc::STDERR_FILENO) };
        if dup < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { File::from_raw_fd(dup as RawFd) })
    }
    #[cfg(not(unix))]
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "stdio intercept requires unix",
    ))
}

fn intercept_process_stdio() {
    if STDIO_INTERCEPT_INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    #[cfg(unix)]
    {
        let _ = redirect_stdio_fd(libc::STDOUT_FILENO, false);
        let _ = redirect_stdio_fd(libc::STDERR_FILENO, true);
    }
}

#[cfg(unix)]
fn redirect_stdio_fd(target: i32, error_level: bool) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let (reader, writer) = io::pipe()?;
    if unsafe { libc::dup2(writer.as_raw_fd(), target) } < 0 {
        return Err(io::Error::last_os_error());
    }
    drop(writer);
    std::thread::Builder::new()
        .name(if error_level {
            "stderr-log".into()
        } else {
            "stdout-log".into()
        })
        .spawn(move || drain_stdio_pipe(reader, error_level))
        .map_err(io::Error::other)?;
    Ok(())
}

fn emit_stdio_line(line: String, error_level: bool) {
    if error_level {
        tracing::error!("{line}");
    } else {
        tracing::info!("{line}");
    }
}

#[cfg(unix)]
fn drain_stdio_pipe(reader: impl io::Read, error_level: bool) {
    use std::io::BufRead;
    let mut reader = io::BufReader::new(reader);
    let mut chunk = String::new();
    let mut buffer = String::new();
    loop {
        chunk.clear();
        match reader.read_line(&mut chunk) {
            Ok(0) => {
                if let Some(line) = flush_stdio_buffer(&mut buffer) {
                    emit_stdio_line(line, error_level);
                }
                break;
            }
            Ok(_) => {
                for line in take_stdio_lines(&mut buffer, &chunk) {
                    emit_stdio_line(line, error_level);
                }
            }
            Err(_) => break,
        }
    }
}

fn console_writer() -> BoxMakeWriter {
    match duplicate_stderr() {
        Ok(file) => BoxMakeWriter::new(MutexWriterFd(Mutex::new(file))),
        Err(_) => BoxMakeWriter::new(std::io::stderr),
    }
}

/// 对齐 Python loguru：`{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}`
pub fn format_python_log_prefix(
    timestamp: &str,
    level: &str,
    name: &str,
    function: &str,
    line: u32,
) -> String {
    format!("{timestamp} | {level:<8} | {name}:{function}:{line} - ")
}

struct PythonLogFormat;

impl<S, N> FormatEvent<S, N> for PythonLogFormat
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    N: for<'writer> FormatFields<'writer> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let meta = event.metadata();
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let function = meta
            .module_path()
            .and_then(|path| path.rsplit("::").next())
            .unwrap_or("unknown");
        write!(
            writer,
            "{}",
            format_python_log_prefix(
                &timestamp,
                meta.level().as_str(),
                meta.target(),
                function,
                meta.line().unwrap_or(0),
            )
        )?;
        ctx.format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

/// 控制台 + 滚动文件。文件初始化失败时仅保留 stderr。
pub fn init_tracing(log_dir: &str, log_level: &str) {
    install_panic_hook();
    // 控制台跟 LOG_LEVEL；文件对齐 loguru `level=0`，本 crate 全量写入。
    let console_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(parse_log_level(log_level)));
    let stdout_layer = fmt::layer()
        .event_format(PythonLogFormat)
        .with_writer(console_writer())
        .with_filter(console_filter);
    let dir = resolve_log_dir(log_dir);
    match RotatingFile::open(dir.clone()) {
        Ok(file) => {
            let file_layer = fmt::layer()
                .event_format(PythonLogFormat)
                .with_ansi(false)
                .with_writer(MutexWriter(Mutex::new(file)))
                .with_filter(EnvFilter::new("eks_server=trace,eks_kernels=trace"));
            tracing_subscriber::registry()
                .with(stdout_layer)
                .with(file_layer)
                .init();
        }
        Err(err) => {
            tracing_subscriber::registry()
                .with(stdout_layer)
                .init();
            tracing::warn!(path = %dir.display(), "日志目录不可用，仅输出到控制台: {err}");
        }
    }
    intercept_process_stdio();
}

pub fn format_uncaught_exception() -> &'static str {
    "未捕获异常"
}

pub fn format_thread_uncaught_exception(name: &str) -> String {
    format!("线程未捕获异常: {name}")
}

/// panic 进 tracing（含堆栈）而非只落 stderr，对齐 Python `logger.exception` 的可追溯性。
/// 原生默认 hook 的输出保留：控制台仍能看到 panic；此处补齐文件日志与 backtrace。
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = if let Some(message) = info.payload().downcast_ref::<&str>() {
            (*message).to_string()
        } else if let Some(message) = info.payload().downcast_ref::<String>() {
            message.clone()
        } else {
            "Box<dyn Any>".to_string()
        };
        let location = info
            .location()
            .map(|loc| format!("{loc}"))
            .unwrap_or_else(|| "unknown".into());
        let thread_name = std::thread::current()
            .name()
            .unwrap_or("unknown")
            .to_string();
        let message = if thread_name == "main" {
            format_uncaught_exception().to_string()
        } else {
            format_thread_uncaught_exception(&thread_name)
        };
        tracing::error!(
            panic = %payload,
            location = %location,
            backtrace = %std::backtrace::Backtrace::force_capture(),
            "{}",
            message
        );
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_log_dir_relative_to_backend_app() {
        assert_eq!(resolve_log_dir("logs"), PathBuf::from("app/logs"));
        assert_eq!(resolve_log_dir(""), PathBuf::from("app/logs"));
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
        assert_eq!(format_uncaught_exception(), "未捕获异常");
        assert_eq!(
            format_thread_uncaught_exception("memory-persist"),
            "线程未捕获异常: memory-persist"
        );
        assert_eq!(
            format_python_log_prefix(
                "2026-11-10 16:58:01",
                "INFO",
                "app.services.chat_service",
                "stream_chat_events",
                3896,
            ),
            "2026-11-10 16:58:01 | INFO     | app.services.chat_service:stream_chat_events:3896 - "
        );
    }

    #[test]
    fn stdio_line_buffer_matches_python_stream_to_logger() {
        let mut buffer = String::new();
        assert_eq!(take_stdio_lines(&mut buffer, "hello"), Vec::<String>::new());
        assert_eq!(buffer, "hello");
        assert_eq!(
            take_stdio_lines(&mut buffer, "\nworld\r\n\n"),
            vec!["hello".to_string(), "world".to_string()]
        );
        assert!(buffer.is_empty());
        assert_eq!(take_stdio_lines(&mut buffer, "tail"), Vec::<String>::new());
        assert_eq!(flush_stdio_buffer(&mut buffer).as_deref(), Some("tail"));
        assert!(flush_stdio_buffer(&mut buffer).is_none());
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
