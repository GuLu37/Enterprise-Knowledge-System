//! 对齐 Python `log_requests`：debug 记录请求开始与完成，未捕获异常打 error。
//! 续期 JWT 仍由 `access_token_renewal_middleware` 写入 `X-Access-Token`。

use std::net::SocketAddr;
use std::time::Instant;

use axum::{
    extract::{ConnectInfo, Request},
    middleware::Next,
    response::Response,
};

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

fn connect_addr(request: &Request) -> Option<SocketAddr> {
    request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0)
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
    let response = next.run(request).await;
    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    tracing::debug!(
        "{}",
        format_request_complete(&line, response.status().as_u16(), duration_ms)
    );
    response
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
}
