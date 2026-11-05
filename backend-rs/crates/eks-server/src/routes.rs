//! API 路由：/api/v1/auth/*（与 Python FastAPI 契约一致）+ /health。
use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    response::{IntoResponse, Response},
    routing::{any, get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::auth::{
    AuthError, build_access_token, build_refresh_token, decode_token, hash_password,
    hash_token, new_uuid_hex, token_bundle_json, verify_password,
};
use crate::db::{self, now_naive, fmt_dt};
use crate::state::AppState;

/// 与 FastAPI HTTPException 的 {"detail": ...} 响应体一致。
impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let body = Json(json!({ "detail": self.detail }));
        (self.status, body).into_response()
    }
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub password: String,
    pub confirm_password: String,
}

#[derive(Deserialize)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

#[derive(Deserialize)]
pub struct PasswordChangeRequest {
    pub current_password: String,
    pub new_password: String,
}

pub fn health_payload(config: &crate::config::Config) -> Value {
    json!({
        "status": "ok",
        "app": config.app_name,
        "version": config.app_version,
        "environment": config.environment,
    })
}

pub fn root_payload(config: &crate::config::Config) -> Value {
    json!({
        "name": config.app_name,
        "version": config.app_version,
        "docs": "/docs",
    })
}

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(health_payload(&state.config))
}

pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(root_payload(&state.config))
}

pub async fn docs() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/html; charset=utf-8",
        )],
        crate::openapi::docs_html(),
    )
}

pub async fn openapi_json(State(state): State<AppState>) -> impl IntoResponse {
    Json(crate::openapi::spec(&state.config))
}

/// 从请求头解析 Bearer 令牌（与 Python `HTTPBearer(auto_error=False)` + scheme 检查一致）。
fn bearer_token(headers: &HeaderMap) -> Result<String, AuthError> {
    let value = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| AuthError::unauthorized("未登录或登录已失效"))?;
    let mut parts = value.splitn(2, ' ');
    let scheme = parts.next().unwrap_or("");
    let token = parts.next().unwrap_or("").trim();
    if !scheme.eq_ignore_ascii_case("bearer") || token.is_empty() {
        return Err(AuthError::unauthorized("未登录或登录已失效"));
    }
    Ok(token.to_string())
}

/// 与 Python `require_current_user` 等价：验令牌 + 会话有效性 + 滑动续期。
pub async fn require_current_user(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<db::UserRow, AuthError> {
    authenticate_and_renew(&state.pool, &state.config, headers).await
}

async fn authenticate_and_renew(
    pool: &sqlx::SqlitePool,
    config: &crate::config::Config,
    headers: &HeaderMap,
) -> Result<db::UserRow, AuthError> {
    let token = bearer_token(headers)?;
    let claims = decode_token(&token, "access", &config.jwt_secret_key)?;

    let user = db::get_user_by_id(pool, &claims.uid)
        .await?
        .filter(|user| user.is_active)
        .ok_or_else(|| AuthError::unauthorized("未登录或登录已失效"))?;
    let session = db::get_session_by_id(pool, &claims.sid)
        .await?
        .ok_or_else(|| AuthError::unauthorized("未登录或登录已失效"))?;
    let now = now_naive();
    let valid = session.user_id == user.user_id
        && session.revoked_at.is_none()
        && db::parse_dt(&session.expires_at)
            .map(|exp| exp > now)
            .unwrap_or(false);
    if !valid {
        return Err(AuthError::unauthorized("未登录或登录已失效"));
    }
    db::slide_session(pool, &session.session_id, config.session_idle_timeout_minutes).await?;
    match build_access_token(&user, &session.session_id, config) {
        Ok(access_token) => crate::auth::publish_renewed_access_token(access_token),
        Err(err) => tracing::warn!("滑动续期签发访问令牌失败: {err}"),
    }
    Ok(user)
}

async fn issue_token_bundle(
    state: &AppState,
    user: &db::UserRow,
    revoke_existing: bool,
) -> Result<Value, AuthError> {
    if revoke_existing {
        db::revoke_user_sessions(&state.pool, &user.user_id, None).await?;
    }
    let session_id = new_uuid_hex();
    let refresh_token = build_refresh_token(user, &session_id, &state.config)?;
    let access_token = build_access_token(user, &session_id, &state.config)?;
    let expires_at = fmt_dt(&(now_naive() + chrono::Duration::minutes(state.config.refresh_token_expire_minutes)));
    db::insert_session(
        &state.pool,
        &session_id,
        &user.user_id,
        &hash_token(&refresh_token),
        &expires_at,
    )
    .await?;
    Ok(token_bundle_json(
        access_token,
        refresh_token,
        &state.config,
        user,
    ))
}

pub async fn login(
    State(state): State<AppState>,
    Json(request): Json<LoginRequest>,
) -> Result<Json<Value>, AuthError> {
    crate::validate::login_fields(&request.username, &request.password)?;
    let username = request.username.trim().to_lowercase();
    if username.is_empty() || request.password.is_empty() {
        return Err(AuthError::unauthorized("用户名或密码错误"));
    }
    let user = match db::get_user_by_username(&state.pool, &username).await? {
        Some(user) if user.is_active => user,
        _ => return Err(AuthError::unauthorized("用户名或密码错误")),
    };
    if !verify_password(&request.password, &user.password_salt, &user.password_hash) {
        return Err(AuthError::unauthorized("用户名或密码错误"));
    }
    db::touch_last_login(&state.pool, &user.user_id).await?;
    let user = db::get_user_by_id(&state.pool, &user.user_id)
        .await?
        .expect("user just verified");
    Ok(Json(issue_token_bundle(&state, &user, true).await?))
}

pub async fn register(
    State(state): State<AppState>,
    Json(request): Json<RegisterRequest>,
) -> Result<Json<Value>, AuthError> {
    crate::validate::register_fields(
        &request.username,
        &request.password,
        &request.confirm_password,
    )?;
    let username = request.username.trim().to_lowercase();
    if username.is_empty() {
        return Err(AuthError::bad_request("账号不能为空"));
    }
    if request.password.is_empty() {
        return Err(AuthError::bad_request("密码不能为空"));
    }
    if request.password != request.confirm_password {
        return Err(AuthError::bad_request("两次输入的密码不一致"));
    }
    if db::get_user_by_username(&state.pool, &username)
        .await?
        .is_some()
    {
        return Err(AuthError::conflict("账号已存在"));
    }
    let (salt_hex, password_hash) = hash_password(&request.password, None);
    let user_id = new_uuid_hex();
    db::insert_user(&state.pool, &user_id, &username, &password_hash, &salt_hex, "user").await?;
    let user = db::get_user_by_id(&state.pool, &user_id)
        .await?
        .expect("user just inserted");
    Ok(Json(issue_token_bundle(&state, &user, true).await?))
}

pub async fn refresh(
    State(state): State<AppState>,
    Json(request): Json<RefreshTokenRequest>,
) -> Result<Json<Value>, AuthError> {
    crate::validate::refresh_token_field(&request.refresh_token)?;
    let claims = decode_token(&request.refresh_token, "refresh", &state.config.jwt_secret_key)?;
    let user = db::get_user_by_id(&state.pool, &claims.uid)
        .await?
        .filter(|user| user.is_active)
        .ok_or_else(|| AuthError::unauthorized("登录已过期，请重新登录"))?;
    let session = db::get_session_by_id(&state.pool, &claims.sid)
        .await?
        .ok_or_else(|| AuthError::unauthorized("登录已过期，请重新登录"))?;
    let now = now_naive();
    let valid = session.user_id == user.user_id
        && session.revoked_at.is_none()
        && db::parse_dt(&session.expires_at)
            .map(|exp| exp > now)
            .unwrap_or(false)
        && session.refresh_token_hash == hash_token(&request.refresh_token);
    if !valid {
        return Err(AuthError::unauthorized("登录已过期，请重新登录"));
    }
    db::revoke_session(&state.pool, &session.session_id).await?;
    Ok(Json(issue_token_bundle(&state, &user, false).await?))
}

pub async fn change_password(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<PasswordChangeRequest>,
) -> Result<Json<Value>, AuthError> {
    let user = require_current_user(&state, &headers).await?;
    crate::validate::password_change_fields(&request.current_password, &request.new_password)?;
    if !verify_password(&request.current_password, &user.password_salt, &user.password_hash) {
        return Err(AuthError::bad_request("当前密码不正确"));
    }
    let (salt_hex, password_hash) = hash_password(&request.new_password, None);
    db::update_password(&state.pool, &user.user_id, &password_hash, &salt_hex).await?;
    db::revoke_user_sessions(&state.pool, &user.user_id, None).await?;
    let user = db::get_user_by_id(&state.pool, &user.user_id)
        .await?
        .expect("user exists");
    Ok(Json(issue_token_bundle(&state, &user, false).await?))
}

pub async fn me(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AuthError> {
    let user = require_current_user(&state, &headers).await?;
    Ok(Json(user.to_public()))
}

/// 兼容 Python 侧的 404 表现：/api/v1/auth/{unknown} 返回 {"detail":"Not Found"}。
pub async fn auth_fallback(Path(_rest): Path<String>) -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "detail": "Not Found" })),
    )
}

pub fn router(state: AppState) -> axum::Router {
    let prefix = state.config.api_prefix.clone();
    axum::Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/docs", get(docs))
        .route("/redoc", get(docs))
        .route("/openapi.json", get(openapi_json))
        .route(
            &format!("{prefix}/auth/login"),
            post(login),
        )
        .route(
            &format!("{prefix}/auth/register"),
            post(register),
        )
        .route(
            &format!("{prefix}/auth/refresh"),
            post(refresh),
        )
        .route(
            &format!("{prefix}/auth/password"),
            post(change_password),
        )
        .route(&format!("{prefix}/auth/me"), get(me))
        .route(&format!("{prefix}/auth/{{*rest}}"), any(auth_fallback))
        // 检索路由（与 Python 一样需要登录态）
        .route(
            &format!("{prefix}/retrieval/search/hybrid"),
            post(retrieval_hybrid),
        )
        .route(
            &format!("{prefix}/retrieval/search/dense"),
            post(retrieval_dense),
        )
        .route(
            &format!("{prefix}/retrieval/search/sparse"),
            post(retrieval_sparse),
        )
        .route(&format!("{prefix}/retrieval/rag"), post(retrieval_rag))
        .route(
            &format!("{prefix}/retrieval/cache-clear"),
            post(retrieval_cache_clear),
        )
        .route(
            &format!("{prefix}/chat/settings"),
            get(crate::chat_stream::chat_settings),
        )
        .route(
            &format!("{prefix}/chat/runtime-status"),
            get(crate::chat_stream::chat_runtime_status),
        )
        .route(
            &format!("{prefix}/chat/warmup"),
            post(crate::chat_stream::chat_warmup),
        )
        .route(
            &format!("{prefix}/chat/generate"),
            post(crate::chat_stream::generate_chat),
        )
        .route(
            &format!("{prefix}/chat/consolidate"),
            post(crate::chat_stream::consolidate_chat),
        )
        .route(
            &format!("{prefix}/chat/stream"),
            post(crate::chat_stream::stream_chat),
        )
        .route(
            &format!("{prefix}/chat/conversations"),
            get(crate::conversations::list_conversations)
                .delete(crate::conversations::clear_conversations),
        )
        .route(
            &format!("{prefix}/chat/conversations/{{conversation_id}}"),
            axum::routing::put(crate::conversations::upsert_conversation)
                .delete(crate::conversations::delete_conversation),
        )
        .route(
            &format!("{prefix}/documents"),
            get(crate::documents::list_documents),
        )
        .route(
            &format!("{prefix}/documents/list"),
            get(crate::documents::list_documents_compat),
        )
        .route(
            &format!("{prefix}/documents/upload"),
            axum::routing::post(crate::documents::upload_document),
        )
        .route(
            &format!("{prefix}/documents/content/{{document_id}}"),
            get(crate::documents::get_document_content),
        )
        .route(
            &format!("{prefix}/documents/delete/{{document_id}}"),
            axum::routing::delete(crate::documents::delete_document),
        )
        .route(
            &format!("{prefix}/documents/{{document_id}}"),
            axum::routing::delete(crate::documents::delete_document),
        )
        .route(
            &format!("{prefix}/skills"),
            get(list_skills),
        )
        .route(
            &format!("{prefix}/skills/function-calling"),
            get(list_skill_function_calling),
        )
        .route(
            &format!("{prefix}/skills/capabilities"),
            get(skill_capabilities),
        )
        .route(
            &format!("{prefix}/skills/execute"),
            post(execute_skill),
        )
        .with_state(state.clone())
        .layer(axum::middleware::from_fn(crate::auth::access_token_renewal_middleware))
        .layer(crate::auth::cors_layer(&state.config.cors_origins))
        .layer(axum::middleware::from_fn(crate::request_log::log_requests))
}

#[derive(Deserialize)]
pub struct RetrievalRequest {
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
}

fn reject_empty_query(query: &str) -> Result<(), AuthError> {
    crate::validate::string_min_length(query, 1)
}

/// 对齐 Python `_build_response(request.query, …)`：HTTP 回显原始查询串。
fn echo_request_query(query: &str) -> &str {
    query
}

fn retrieval_search_query(query: &str) -> &str {
    query.trim()
}

pub async fn retrieval_dense(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RetrievalRequest>,
) -> Result<Json<Value>, AuthError> {
    retrieval_simple(state, headers, request, "dense").await
}

pub async fn retrieval_sparse(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RetrievalRequest>,
) -> Result<Json<Value>, AuthError> {
    retrieval_simple(state, headers, request, "sparse").await
}

async fn retrieval_simple(
    state: AppState,
    headers: HeaderMap,
    request: RetrievalRequest,
    method: &str,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    reject_empty_query(&request.query)?;
    let top_k = crate::validate::retrieval_top_k(request.top_k)?;
    let query = retrieval_search_query(&request.query);
    let mut results = if method == "dense" {
        state.engine.search_dense(query, top_k).await
    } else {
        state.engine.search_sparse(query, top_k).await
    }
    .map_err(|detail| retrieval_internal_error(method, detail))?;
    if method == "dense" && state.config.similarity_threshold > 0.0 {
        results.retain(|item| item.score >= state.config.similarity_threshold);
    }
    Ok(Json(json!({
        "query": echo_request_query(&request.query),
        "retrieval_method": method,
        "results": results.iter().map(|item| item.to_json()).collect::<Vec<_>>(),
        "top_k": top_k,
    })))
}

#[derive(Deserialize, Default)]
pub struct SkillExecuteContext {
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub trace_id: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Deserialize)]
pub struct SkillExecuteBody {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
    #[serde(default)]
    pub context: SkillExecuteContext,
}

pub async fn list_skills(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    Ok(Json(Value::Array(crate::skills::list_skill_definitions_from(
        &state.config.mcp_remote_tool_manifests,
    ))))
}

pub async fn list_skill_function_calling(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    Ok(Json(json!({
        "tools": crate::skills::function_calling_tools_from(&state.config.mcp_remote_tool_manifests),
    })))
}

pub async fn skill_capabilities(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    Ok(Json(crate::skills::skill_capabilities_payload_from(
        &state.config.mcp_remote_tool_manifests,
    )))
}

pub async fn execute_skill(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<SkillExecuteBody>,
) -> Result<Json<Value>, AuthError> {
    let user = require_current_user(&state, &headers).await?;
    crate::validate::skill_name_field(&request.name)?;
    let name = request.name.trim();
    let manifests = &state.config.mcp_remote_tool_manifests;
    // 未知技能 404；remote_mcp 不在此提前返回，统一走 execute_named_from，
    // 对齐 Python execute_skill 的三段 trace 日志（调用开始/调用异常）。
    if crate::skills::get_skill_definition_from(name, manifests).is_none() {
        return Err(AuthError {
            status: StatusCode::NOT_FOUND,
            detail: crate::skills::skill_not_found_http_detail(name),
        });
    }
    let context = crate::skills::SkillExecutionContext {
        user_id: &user.user_id,
        conversation_id: request.context.conversation_id.as_deref(),
        session_id: request.context.session_id.as_deref(),
        query: request.context.query.as_deref(),
        provider: None,
        model: None,
        trace_id: request.context.trace_id.as_deref(),
        metadata: request.context.metadata,
    };
    let result = crate::skills::execute_named_from(
        &state.pool,
        name,
        &request.arguments,
        &context,
        manifests,
        &[],
        Some(state.engine.embedder.as_ref()),
    )
    .await
    .map_err(|detail| AuthError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        detail,
    })?;
    let payload = crate::skills::wrap_skill_execution(
        name,
        &request.arguments,
        &context,
        result,
    )
    .map_err(|detail| {
        if detail.starts_with("未找到技能:") {
            AuthError {
                status: StatusCode::NOT_FOUND,
                detail: crate::skills::skill_not_found_http_detail(name),
            }
        } else {
            AuthError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                detail,
            }
        }
    })?;
    Ok(Json(crate::skills::skill_execute_http_payload(payload)))
}

pub async fn retrieval_hybrid(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RetrievalRequest>,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    reject_empty_query(&request.query)?;
    let top_k = crate::validate::retrieval_top_k(request.top_k)?;
    let query = retrieval_search_query(&request.query);
    let (recalled, _) = state
        .engine
        .retrieve_documents_multi_query(
            query,
            top_k,
            crate::planner::RETRIEVAL_METHOD_HYBRID,
            None,
        )
        .await
        .map_err(|detail| retrieval_internal_error("hybrid", detail))?;
    let results = state
        .engine
        .post_process(query, recalled, top_k)
        .await
        .map_err(|detail| retrieval_internal_error("hybrid", detail))?;
    Ok(Json(json!({
        "query": echo_request_query(&request.query),
        "retrieval_method": "hybrid",
        "results": results.iter().map(|item| item.to_json()).collect::<Vec<_>>(),
        "top_k": top_k,
    })))
}

pub async fn retrieval_rag(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RetrievalRequest>,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    reject_empty_query(&request.query)?;
    let top_k = crate::validate::retrieval_top_k(request.top_k)?;
    let query = retrieval_search_query(&request.query);
    let planned = state
        .engine
        .run_rag_workflow(query, top_k)
        .await
        .map_err(internal_error)?;
    Ok(Json(json!({
        "query": echo_request_query(&request.query),
        "retrieval_method": "hybrid",
        "expanded_queries": planned.expanded_queries,
        "results": planned.results.iter().map(|item| item.to_json()).collect::<Vec<_>>(),
        "context": planned.context,
        "message": if planned.results.is_empty() { "当前知识库未检索到足够相关资料" } else { "检索完成" },
    })))
}

pub async fn retrieval_cache_clear(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    state.engine.invalidate().await;
    Ok(Json(json!({ "cleared": true })))
}

fn format_retrieval_route_error(method: &str, err: &str) -> String {
    match method {
        "dense" => format!("密集检索失败: {err}"),
        "sparse" => format!("稀疏检索失败: {err}"),
        _ => format!("检索失败: {err}"),
    }
}

fn retrieval_internal_error(method: &str, detail: String) -> AuthError {
    tracing::error!("{}", format_retrieval_route_error(method, &detail));
    internal_error(detail)
}

fn internal_error(detail: String) -> AuthError {
    AuthError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        detail,
    }
}

#[cfg(test)]
mod health_tests {
    use super::*;

    #[test]
    fn health_and_root_payloads_match_python() {
        let config = crate::config::Config::jwt_test();
        let health = health_payload(&config);
        assert_eq!(health["status"], "ok");
        assert_eq!(health["app"], "Enterprise Knowledge System");
        assert_eq!(health["version"], "1.0.0");
        assert_eq!(health["environment"], "开发版");
        assert!(health.get("service").is_none());

        let root = root_payload(&config);
        assert_eq!(root["name"], "Enterprise Knowledge System");
        assert_eq!(root["version"], "1.0.0");
        assert_eq!(root["docs"], "/docs");
        assert!(root.get("service").is_none());
        assert!(root.get("health").is_none());
    }

    #[test]
    fn empty_retrieval_query_is_unprocessable() {
        let err = reject_empty_query("").unwrap_err();
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.detail, "String should have at least 1 character");
        assert!(reject_empty_query("   ").is_ok());
        assert!(reject_empty_query("差旅").is_ok());
        assert_eq!(echo_request_query("  差旅报销  "), "  差旅报销  ");
        assert_eq!(retrieval_search_query("  差旅报销  "), "差旅报销");
    }

    #[test]
    fn retrieval_route_error_logs_match_python() {
        assert_eq!(format_retrieval_route_error("hybrid", "boom"), "检索失败: boom");
        assert_eq!(
            format_retrieval_route_error("dense", "timeout"),
            "密集检索失败: timeout"
        );
        assert_eq!(
            format_retrieval_route_error("sparse", "index missing"),
            "稀疏检索失败: index missing"
        );
    }
}

#[cfg(test)]
mod docs_tests {
    use super::*;
    use axum::{Router, body::Body, http::Request, response::IntoResponse};
    use tower::ServiceExt;

    #[tokio::test]
    async fn docs_and_openapi_routes_are_public() {
        let app = Router::new()
            .route("/docs", get(docs))
            .route("/openapi.json", get(|| async {
                Json(crate::openapi::spec(&crate::config::Config::jwt_test()))
            }));
        let docs_response = app
            .clone()
            .oneshot(Request::builder().uri("/docs").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(docs_response.status(), StatusCode::OK);
        let content_type = docs_response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        assert!(content_type.contains("text/html"), "{content_type}");
        let body = axum::body::to_bytes(docs_response.into_body(), 64 * 1024)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("/openapi.json"));

        let spec_response = app
            .oneshot(
                Request::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(spec_response.status(), StatusCode::OK);
        let spec_body = axum::body::to_bytes(spec_response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let spec: Value = serde_json::from_slice(&spec_body).unwrap();
        assert_eq!(spec["openapi"], "3.1.0");
        assert!(spec["paths"].get("/health").is_some());
    }

    #[tokio::test]
    async fn auth_unknown_path_matches_python_not_found() {
        let response = auth_fallback(Path("nope".into())).await.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let payload: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["detail"], "Not Found");
    }
}

#[cfg(test)]
mod auth_renewal_tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue, header::AUTHORIZATION};

    async fn setup_pool() -> sqlx::SqlitePool {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE users (
                user_id TEXT PRIMARY KEY, username TEXT, password_hash TEXT, password_salt TEXT,
                role TEXT, is_active INTEGER, created_at TEXT, updated_at TEXT, last_login_at TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE auth_sessions (
                session_id TEXT PRIMARY KEY, user_id TEXT, refresh_token_hash TEXT, expires_at TEXT,
                revoked_at TEXT, created_at TEXT, updated_at TEXT, last_used_at TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool
    }

    #[tokio::test]
    async fn renews_access_token_into_slot() {
        let pool = setup_pool().await;
        let config = crate::config::Config::jwt_test();
        db::insert_user(&pool, "u1", "admin", "hash", "salt", "admin")
            .await
            .unwrap();
        let expires = fmt_dt(&(now_naive() + chrono::Duration::minutes(60)));
        db::insert_session(&pool, "s1", "u1", "refresh-hash", &expires)
            .await
            .unwrap();
        let user = db::get_user_by_id(&pool, "u1").await.unwrap().unwrap();
        let old_token = build_access_token(&user, "s1", &config).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {old_token}")).unwrap(),
        );

        let (user, renewed) = crate::auth::with_renewed_slot(|| authenticate_and_renew(&pool, &config, &headers)).await;
        let user = user.unwrap();
        let renewed = renewed.expect("应签发续期访问令牌");
        assert_eq!(user.username, "admin");
        assert_ne!(renewed, old_token);
        let claims = decode_token(&renewed, "access", &config.jwt_secret_key).unwrap();
        assert_eq!(claims.uid, "u1");
        assert_eq!(claims.sid, "s1");
        assert_ne!(claims.jti, decode_token(&old_token, "access", &config.jwt_secret_key).unwrap().jti);
    }
}
