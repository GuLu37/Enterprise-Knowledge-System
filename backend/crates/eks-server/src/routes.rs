//! API 路由：/api/v1/auth/*（与 Python FastAPI 契约一致）+ /health。
use axum::{
    Json,
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode, Uri, header::AUTHORIZATION, header::LOCATION},
    middleware::Next,
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
use crate::extract::ApiJson;
use crate::state::AppState;

/// 与 FastAPI HTTPException / RequestValidationError 的 {"detail": ...} 响应体一致。
impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let status = self.status;
        (status, Json(self.detail_json())).into_response()
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

pub async fn redoc() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/html; charset=utf-8",
        )],
        crate::openapi::redoc_html(),
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
    ApiJson(request): ApiJson<LoginRequest>,
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
    ApiJson(request): ApiJson<RegisterRequest>,
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
    ApiJson(request): ApiJson<RefreshTokenRequest>,
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
    ApiJson(request): ApiJson<PasswordChangeRequest>,
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
        .route("/redoc", get(redoc))
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
        .fallback(json_not_found)
        .method_not_allowed_fallback(json_method_not_allowed)
        .with_state(state.clone())
        .layer(axum::middleware::from_fn_with_state(
            prefix.clone(),
            redirect_trailing_slashes,
        ))
        .layer(axum::middleware::from_fn(crate::auth::access_token_renewal_middleware))
        .layer(crate::auth::cors_layer(&state.config.cors_origins))
        .layer(axum::middleware::from_fn(crate::request_log::log_requests))
}

/// 对齐 Starlette `redirect_slashes=True`：仅当去掉尾斜杠后的路径能匹配已注册路由时 307。
fn slash_redirect_location(uri: &Uri) -> Option<String> {
    let path = uri.path();
    if path.len() <= 1 || !path.ends_with('/') {
        return None;
    }
    let trimmed = path.trim_end_matches('/');
    let trimmed = if trimmed.is_empty() { "/" } else { trimmed };
    Some(match uri.query() {
        Some(query) => format!("{trimmed}?{query}"),
        None => trimmed.to_string(),
    })
}

fn trimmed_path(uri: &Uri) -> Option<&str> {
    let path = uri.path();
    if path.len() <= 1 || !path.ends_with('/') {
        return None;
    }
    let trimmed = path.trim_end_matches('/');
    Some(if trimmed.is_empty() { "/" } else { trimmed })
}

/// 与 `router()` 注册表对齐；auth `{*rest}` 404 兜底不计入，避免未知 `/auth/x/` 多一次 307。
fn registered_path(prefix: &str, path: &str) -> bool {
    if matches!(path, "/" | "/health" | "/docs" | "/redoc" | "/openapi.json") {
        return true;
    }
    let prefix = prefix.trim_end_matches('/');
    let Some(rest) = path.strip_prefix(prefix) else {
        return false;
    };
    if !rest.starts_with('/') {
        return false;
    }
    const EXACT: &[&str] = &[
        "/auth/login",
        "/auth/register",
        "/auth/refresh",
        "/auth/password",
        "/auth/me",
        "/retrieval/search/hybrid",
        "/retrieval/search/dense",
        "/retrieval/search/sparse",
        "/retrieval/rag",
        "/retrieval/cache-clear",
        "/chat/settings",
        "/chat/runtime-status",
        "/chat/warmup",
        "/chat/generate",
        "/chat/consolidate",
        "/chat/stream",
        "/chat/conversations",
        "/documents",
        "/documents/list",
        "/documents/upload",
        "/skills",
        "/skills/function-calling",
        "/skills/capabilities",
        "/skills/execute",
    ];
    if EXACT.contains(&rest) {
        return true;
    }
    one_segment_after(rest, "/chat/conversations/")
        || one_segment_after(rest, "/documents/content/")
        || one_segment_after(rest, "/documents/delete/")
        || one_segment_after(rest, "/documents/")
}

fn one_segment_after(path: &str, stem: &str) -> bool {
    let Some(tail) = path.strip_prefix(stem) else {
        return false;
    };
    !tail.is_empty() && !tail.contains('/')
}

async fn redirect_trailing_slashes(
    State(prefix): State<String>,
    req: Request,
    next: Next,
) -> Response {
    if let Some(trimmed) = trimmed_path(req.uri()) {
        if registered_path(&prefix, trimmed) {
            if let Some(location) = slash_redirect_location(req.uri()) {
                return (StatusCode::TEMPORARY_REDIRECT, [(LOCATION, location)]).into_response();
            }
        }
    }
    next.run(req).await
}

pub async fn json_not_found() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, Json(json!({ "detail": "Not Found" })))
}

pub async fn json_method_not_allowed() -> impl IntoResponse {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({ "detail": "Method Not Allowed" })),
    )
}

fn default_retrieval_top_k() -> usize {
    5
}

#[derive(Deserialize)]
pub struct RetrievalRequest {
    pub query: String,
    #[serde(default = "default_retrieval_top_k", deserialize_with = "crate::extract::coerce_usize")]
    pub top_k: usize,
}

fn reject_empty_query(query: &str) -> Result<(), AuthError> {
    crate::validate::string_min_length_at(query, 1, &["body", "query"])
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
    ApiJson(request): ApiJson<RetrievalRequest>,
) -> Result<Json<Value>, AuthError> {
    retrieval_simple(state, headers, request, "dense").await
}

pub async fn retrieval_sparse(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiJson(request): ApiJson<RetrievalRequest>,
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
    let top_k = crate::validate::retrieval_top_k(Some(request.top_k))?;
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

fn default_json_object() -> Value {
    json!({})
}

fn deserialize_json_object<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // 字段缺省走 `default = "{}"`；本函数只处理显式出现的值（含 JSON `null`）。
    Value::deserialize(deserializer)
}

fn json_opt_string(value: &Value, key: &str) -> Result<Option<String>, AuthError> {
    match value.get(key) {
        None => Ok(None),
        Some(item) => crate::validate::optional_string_value(
            item,
            &["body", "context", key],
        ),
    }
}

fn skill_execute_context(body: &SkillExecuteBody) -> Result<SkillExecuteContext, AuthError> {
    crate::validate::require_dict(&body.arguments, &["body", "arguments"])?;
    crate::validate::require_model(
        &body.context,
        &["body", "context"],
        "SkillExecutionContextRequest",
    )?;
    let metadata = body
        .context
        .get("metadata")
        .cloned()
        .unwrap_or_else(default_json_object);
    crate::validate::require_dict(&metadata, &["body", "context", "metadata"])?;
    Ok(SkillExecuteContext {
        conversation_id: json_opt_string(&body.context, "conversation_id")?,
        session_id: json_opt_string(&body.context, "session_id")?,
        query: json_opt_string(&body.context, "query")?,
        trace_id: json_opt_string(&body.context, "trace_id")?,
        metadata,
    })
}

#[derive(Debug, Deserialize)]
pub struct SkillExecuteContext {
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub trace_id: Option<String>,
    pub metadata: Value,
}

impl Default for SkillExecuteContext {
    fn default() -> Self {
        Self {
            conversation_id: None,
            session_id: None,
            query: None,
            trace_id: None,
            metadata: default_json_object(),
        }
    }
}

#[derive(Deserialize)]
pub struct SkillExecuteBody {
    pub name: String,
    #[serde(default = "default_json_object", deserialize_with = "deserialize_json_object")]
    pub arguments: Value,
    #[serde(default = "default_json_object", deserialize_with = "deserialize_json_object")]
    pub context: Value,
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
    ApiJson(request): ApiJson<SkillExecuteBody>,
) -> Result<Json<Value>, AuthError> {
    let user = require_current_user(&state, &headers).await?;
    crate::validate::skill_name_field(&request.name)?;
    let SkillExecuteContext {
        conversation_id,
        session_id,
        query,
        trace_id,
        metadata,
    } = skill_execute_context(&request)?;
    let name = request.name.trim();
    let manifests = &state.config.mcp_remote_tool_manifests;
    // 未知技能 404；remote_mcp 不在此提前返回，统一走 execute_named_from，
    // 对齐 Python execute_skill 的三段 trace 日志（调用开始/调用异常）。
    if crate::skills::get_skill_definition_from(name, manifests).is_none() {
        return Err(AuthError::http(
            StatusCode::NOT_FOUND,
            crate::skills::skill_not_found_http_detail(name),
        ));
    }
    let context = crate::skills::SkillExecutionContext {
        user_id: &user.user_id,
        conversation_id: conversation_id.as_deref(),
        session_id: session_id.as_deref(),
        query: query.as_deref(),
        provider: None,
        model: None,
        trace_id: trace_id.as_deref(),
        metadata,
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
    .map_err(|detail| AuthError::http(StatusCode::INTERNAL_SERVER_ERROR, detail))?;
    let payload = crate::skills::wrap_skill_execution(
        name,
        &request.arguments,
        &context,
        result,
    )
    .map_err(|detail| {
        if detail.starts_with("未找到技能:") {
            AuthError::http(
                StatusCode::NOT_FOUND,
                crate::skills::skill_not_found_http_detail(name),
            )
        } else {
            AuthError::http(StatusCode::INTERNAL_SERVER_ERROR, detail)
        }
    })?;
    Ok(Json(crate::skills::skill_execute_http_payload(payload)))
}

pub async fn retrieval_hybrid(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiJson(request): ApiJson<RetrievalRequest>,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    reject_empty_query(&request.query)?;
    let top_k = crate::validate::retrieval_top_k(Some(request.top_k))?;
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
    ApiJson(request): ApiJson<RetrievalRequest>,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    reject_empty_query(&request.query)?;
    let top_k = crate::validate::retrieval_top_k(Some(request.top_k))?;
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
    AuthError::http(StatusCode::INTERNAL_SERVER_ERROR, detail)
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
        assert_eq!(err.loc, vec!["body".to_string(), "query".to_string()]);
        assert_eq!(err.error_type, "string_too_short");
        assert_eq!(
            err.detail_json()["detail"][0]["msg"],
            "String should have at least 1 character"
        );
        assert!(reject_empty_query("   ").is_ok());
        assert!(reject_empty_query("差旅").is_ok());
        assert_eq!(echo_request_query("  差旅报销  "), "  差旅报销  ");
        assert_eq!(retrieval_search_query("  差旅报销  "), "差旅报销");
    }

    #[test]
    fn skill_execute_omitted_arguments_default_to_object() {
        let body: SkillExecuteBody = serde_json::from_str(r#"{"name":"memory_query"}"#).unwrap();
        let context = skill_execute_context(&body).unwrap();
        assert_eq!(body.arguments, json!({}));
        assert_eq!(context.metadata, json!({}));
        let explicit_null: SkillExecuteBody =
            serde_json::from_str(r#"{"name":"memory_query","arguments":null}"#).unwrap();
        let err = skill_execute_context(&explicit_null).unwrap_err();
        assert_eq!(err.error_type, "dict_type");
        assert_eq!(err.loc, vec!["body".to_string(), "arguments".to_string()]);
        assert_eq!(err.detail, "Input should be a valid dictionary");
        assert_eq!(err.input, json!(null));
        let array_args: SkillExecuteBody =
            serde_json::from_str(r#"{"name":"memory_query","arguments":[]}"#).unwrap();
        assert_eq!(
            skill_execute_context(&array_args).unwrap_err().error_type,
            "dict_type"
        );
        let null_context: SkillExecuteBody =
            serde_json::from_str(r#"{"name":"memory_query","context":null}"#).unwrap();
        let context_err = skill_execute_context(&null_context).unwrap_err();
        assert_eq!(context_err.error_type, "model_type");
        assert_eq!(context_err.loc, vec!["body".to_string(), "context".to_string()]);
        let null_metadata: SkillExecuteBody = serde_json::from_str(
            r#"{"name":"memory_query","context":{"metadata":null}}"#,
        )
        .unwrap();
        let metadata_err = skill_execute_context(&null_metadata).unwrap_err();
        assert_eq!(metadata_err.error_type, "dict_type");
        assert_eq!(
            metadata_err.loc,
            vec![
                "body".to_string(),
                "context".to_string(),
                "metadata".to_string()
            ]
        );
        let number_id: SkillExecuteBody = serde_json::from_str(
            r#"{"name":"memory_query","context":{"conversation_id":123}}"#,
        )
        .unwrap();
        let id_err = skill_execute_context(&number_id).unwrap_err();
        assert_eq!(id_err.error_type, "string_type");
        assert_eq!(
            id_err.loc,
            vec![
                "body".to_string(),
                "context".to_string(),
                "conversation_id".to_string()
            ]
        );
        assert_eq!(id_err.detail, "Input should be a valid string");
        assert_eq!(id_err.input, json!(123));
        let null_id: SkillExecuteBody = serde_json::from_str(
            r#"{"name":"memory_query","context":{"conversation_id":null}}"#,
        )
        .unwrap();
        assert!(skill_execute_context(&null_id).unwrap().conversation_id.is_none());
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

    #[test]
    fn slash_redirect_location_matches_starlette() {
        assert_eq!(
            slash_redirect_location(&"/health/".parse().unwrap()).as_deref(),
            Some("/health")
        );
        assert_eq!(
            slash_redirect_location(&"/health/?q=1".parse().unwrap()).as_deref(),
            Some("/health?q=1")
        );
        assert!(slash_redirect_location(&"/health".parse().unwrap()).is_none());
        assert!(slash_redirect_location(&"/".parse().unwrap()).is_none());
        assert_eq!(
            slash_redirect_location(&"/api/v1/chat/settings/".parse().unwrap()).as_deref(),
            Some("/api/v1/chat/settings")
        );
        assert!(registered_path("/api/v1", "/health"));
        assert!(registered_path("/api/v1", "/api/v1/chat/settings"));
        assert!(registered_path("/api/v1", "/api/v1/chat/conversations/c1"));
        assert!(!registered_path("/api/v1", "/missing"));
        assert!(!registered_path("/api/v1", "/api/v1/auth/nope"));
    }

    #[tokio::test]
    async fn trailing_slash_returns_307_like_starlette() {
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .fallback(json_not_found)
            .layer(axum::middleware::from_fn_with_state(
                "/api/v1".to_string(),
                redirect_trailing_slashes,
            ));
        let redirected = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(redirected.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            redirected
                .headers()
                .get(LOCATION)
                .and_then(|value| value.to_str().ok()),
            Some("/health")
        );

        let with_query = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health/?x=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            with_query
                .headers()
                .get(LOCATION)
                .and_then(|value| value.to_str().ok()),
            Some("/health?x=1")
        );

        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/missing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);

        let missing_slash = app
            .oneshot(
                Request::builder()
                    .uri("/missing/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_slash.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn global_fallback_returns_fastapi_json() {
        let not_found = json_not_found().await.into_response();
        assert_eq!(not_found.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(not_found.into_body(), 1024).await.unwrap();
        let payload: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["detail"], "Not Found");

        let method = json_method_not_allowed().await.into_response();
        assert_eq!(method.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body = axum::body::to_bytes(method.into_body(), 1024).await.unwrap();
        let payload: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["detail"], "Method Not Allowed");
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
