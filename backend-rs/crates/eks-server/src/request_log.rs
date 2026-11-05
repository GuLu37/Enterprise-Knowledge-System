//! 对齐 Python `log_requests`：debug 记录请求开始与完成，未捕获异常打 error。
//! 续期 JWT 仍由 `access_token_renewal_middleware` 写入 `X-Access-Token`。

use std::any::Any;
use std::net::SocketAddr;
use std::panic::{AssertUnwindSafe, resume_unwind};
use std::time::Instant;

use axum::{
    extract::{ConnectInfo, Request},
    middleware::Next,
    response::Response,
};
use futures_util::FutureExt;

pub fn request_line(method: &str, path: &str, query: &str) -> String {
    if query.is_empty() {
        format!("{method} {path}")
    } else {
        format!("{method} {path}?{query}")
    }
}

pub fn client_host_from_addr(addr: Option<SocketAddr>) -> String {
    addr.map(|value| value.ip().to_string())
        .unwrap_or_else(|| "-".into())
}

pub fn format_request_start(request_line: &str, client_host: &str) -> String {
    format!("请求开始: {request_line} client={client_host}")
}

pub fn format_request_complete(request_line: &str, status: u16, duration_ms: f64) -> String {
    format!("请求完成: {request_line} status={status} duration={duration_ms:.2}ms")
}

pub fn format_request_exception(request_line: &str, duration_ms: f64) -> String {
    format!("请求异常: {request_line} duration={duration_ms:.2}ms")
}

/// Err 型 5xx 的请求异常文案：带状态码与时长，对齐 Python `logger.exception` 的定位作用。
pub fn format_request_exception_status(
    request_line: &str,
    status: u16,
    duration_ms: f64,
) -> String {
    format!("请求异常: {request_line} status={status} duration={duration_ms:.2}ms")
}

fn connect_addr(request: &Request) -> Option<SocketAddr> {
    request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0)
}

/// 对齐 Python `logger.exception`：把 panic payload 收成可读文本。
pub fn panic_payload_text(payload: &Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "Box<dyn Any>".into()
    }
}

fn emit_request_exception(request_line: &str, duration_ms: f64, panic_text: &str) {
    tracing::error!(
        panic = %panic_text,
        "{}",
        format_request_exception(request_line, duration_ms)
    );
}

pub async fn log_requests(request: Request, next: Next) -> Response {
    let started = Instant::now();
    let line = request_line(
        request.method().as_str(),
        request.uri().path(),
        request.uri().query().unwrap_or(""),
    );
    let client = client_host_from_addr(connect_addr(&request));
    tracing::debug!("{}", format_request_start(&line, &client));
    match AssertUnwindSafe(next.run(request)).catch_unwind().await {
        Ok(response) => {
            let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
            let status = response.status().as_u16();
            if status >= 500 {
                // 对齐 Python `logger.exception`：5xx 响应以 error 级记录请求异常，
                // 与 panic 分支共用前缀，便于日志检索定位。
                tracing::error!(
                    "{}",
                    format_request_exception_status(&line, status, duration_ms)
                );
            } else {
                tracing::debug!("{}", format_request_complete(&line, status, duration_ms));
            }
            response
        }
        Err(payload) => {
            let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
            emit_request_exception(&line, duration_ms, &panic_payload_text(&payload));
            resume_unwind(payload);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, http::Request as HttpRequest, routing::get};
    use tower::ServiceExt;

    #[test]
    fn request_log_copy_matches_python() {
        assert_eq!(
            request_line("GET", "/api/v1/chat/runtime-status", ""),
            "GET /api/v1/chat/runtime-status"
        );
        assert_eq!(
            request_line("GET", "/health", "ready=1"),
            "GET /health?ready=1"
        );
        assert_eq!(
            format_request_start("GET /api/v1/chat/runtime-status", "127.0.0.1"),
            "请求开始: GET /api/v1/chat/runtime-status client=127.0.0.1"
        );
        assert_eq!(
            format_request_complete("GET /health", 200, 12.5),
            "请求完成: GET /health status=200 duration=12.50ms"
        );
        assert_eq!(
            format_request_exception("POST /api/v1/chat/generate", 3.2),
            "请求异常: POST /api/v1/chat/generate duration=3.20ms"
        );
        assert_eq!(
            format_request_exception_status("POST /api/v1/chat/generate", 500, 3.2),
            "请求异常: POST /api/v1/chat/generate status=500 duration=3.20ms"
        );
        assert_eq!(client_host_from_addr(None), "-");
        assert_eq!(
            client_host_from_addr(Some("127.0.0.1:8000".parse().unwrap())),
            "127.0.0.1"
        );
    }

    #[tokio::test]
    async fn middleware_passes_through_successful_response() {
        let app = Router::new()
            .route(
                "/api/v1/chat/runtime-status",
                get(|| async { "ok" }),
            )
            .layer(axum::middleware::from_fn(log_requests));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/api/v1/chat/runtime-status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }

    #[test]
    fn panic_payload_text_reads_str_and_string() {
        let str_payload: Box<dyn Any + Send> = Box::new("handler boom");
        assert_eq!(panic_payload_text(&str_payload), "handler boom");
        let string_payload: Box<dyn Any + Send> = Box::new("owned boom".to_string());
        assert_eq!(panic_payload_text(&string_payload), "owned boom");
        let other_payload: Box<dyn Any + Send> = Box::new(42u32);
        assert_eq!(panic_payload_text(&other_payload), "Box<dyn Any>");
    }

    #[tokio::test]
    async fn middleware_logs_uncaught_panic_then_rethrows_like_python() {
        use std::sync::{Arc, Mutex};
        use tracing::Subscriber;
        use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
        use tracing_subscriber::Registry;

        struct Capture(Arc<Mutex<Vec<String>>>);
        impl<S: Subscriber> Layer<S> for Capture {
            fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
                struct Visitor {
                    message: String,
                }
                impl tracing::field::Visit for Visitor {
                    fn record_debug(
                        &mut self,
                        field: &tracing::field::Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        if field.name() == "message" {
                            self.message = format!("{value:?}").trim_matches('"').to_string();
                        }
                    }
                    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                        if field.name() == "message" {
                            self.message = value.to_string();
                        }
                    }
                }
                let mut visitor = Visitor {
                    message: String::new(),
                };
                event.record(&mut visitor);
                self.0.lock().unwrap().push(visitor.message);
            }
        }

        async fn boom() -> &'static str {
            panic!("handler boom")
        }

        let logs = Arc::new(Mutex::new(Vec::new()));
        let subscriber = Registry::default().with(Capture(logs.clone()));
        let _guard = tracing::subscriber::set_default(subscriber);
        let app = Router::new()
            .route("/boom", get(boom))
            .layer(axum::middleware::from_fn(log_requests));

        let panicked = AssertUnwindSafe(
            app.oneshot(
                HttpRequest::builder()
                    .uri("/boom")
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .catch_unwind()
        .await;
        assert!(panicked.is_err(), "uncaught handler panic must be re-raised");
        let captured = logs.lock().unwrap().clone();
        assert!(
            captured
                .iter()
                .any(|line| line.starts_with("请求异常: GET /boom duration=")
                    && line.ends_with("ms")),
            "expected 请求异常 log, got {captured:?}"
        );
        assert!(
            captured.iter().all(|line| !line.contains("请求完成:")),
            "panic path must not emit 请求完成"
        );
    }
}
