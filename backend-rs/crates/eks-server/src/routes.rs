//! API 路由：/api/v1/auth/*（与 Python FastAPI 契约一致）+ /health。
use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    response::{IntoResponse, Response},
    routing::{get, post},
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

pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok", "service": "eks-rs" }))
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
    let token = bearer_token(headers)?;
    let claims = decode_token(&token, "access", &state.config.jwt_secret_key)?;

    let user = db::get_user_by_id(&state.pool, &claims.uid)
        .await?
        .filter(|user| user.is_active)
        .ok_or_else(|| AuthError::unauthorized("未登录或登录已失效"))?;
    let session = db::get_session_by_id(&state.pool, &claims.sid)
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
    db::slide_session(&state.pool, &session.session_id, state.config.session_idle_timeout_minutes)
        .await?;
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
        .route("/health", get(health))
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
        // 检索路由（与 Python 一样需要登录态）
        .route(
            &format!("{prefix}/retrieval/search/hybrid"),
            post(retrieval_hybrid),
        )
        .route(&format!("{prefix}/retrieval/rag"), post(retrieval_rag))
        .route(
            &format!("{prefix}/retrieval/cache-clear"),
            post(retrieval_cache_clear),
        )
        .route(
            &format!("{prefix}/chat/stream"),
            post(crate::chat_stream::stream_chat),
        )
        .with_state(state)
}

#[derive(Deserialize)]
pub struct RetrievalRequest {
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
}

pub async fn retrieval_hybrid(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RetrievalRequest>,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    let top_k = request.top_k.unwrap_or(5).clamp(1, 50);
    let query = request.query.trim().to_string();
    if query.is_empty() {
        return Ok(Json(json!({
            "query": query, "retrieval_method": "hybrid", "results": [], "top_k": top_k,
        })));
    }
    let recalled = state
        .engine
        .retrieve_documents_multi_query(&query, top_k)
        .await
        .map_err(internal_error)?;
    let results = state
        .engine
        .post_process(&query, recalled, top_k)
        .await
        .map_err(internal_error)?;
    Ok(Json(json!({
        "query": query,
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
    let top_k = request.top_k.unwrap_or(5).clamp(1, 50);
    let query = request.query.trim().to_string();
    if query.is_empty() {
        return Ok(Json(json!({
            "query": "", "results": [], "context": "", "message": "未检索到相关内容"
        })));
    }
    let (results, context) = state
        .engine
        .run_rag_workflow(&query, top_k)
        .await
        .map_err(internal_error)?;
    Ok(Json(json!({
        "query": query,
        "retrieval_method": "hybrid",
        "expanded_queries": crate::retrieval::expand_multi_query(&query),
        "results": results.iter().map(|item| item.to_json()).collect::<Vec<_>>(),
        "context": context,
        "message": if results.is_empty() { "当前知识库未检索到足够相关资料" } else { "检索完成" },
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

fn internal_error(detail: String) -> AuthError {
    AuthError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        detail,
    }
}
