//! 对齐 FastAPI：JSON / Query 反序列化失败返回 422 `detail: [{loc, msg, type}]`。
use axum::body::Bytes;
use axum::extract::rejection::QueryRejection;
use axum::extract::{FromRequest, FromRequestParts, Query, Request};
use axum::http::header::CONTENT_TYPE;
use axum::http::request::Parts;
use serde::de::{Deserialize, DeserializeOwned, Deserializer, Error as DeError};
use serde_json::Value;

use crate::auth::AuthError;

/// 对齐 Pydantic `Optional[List[...]] = None`：JSON `null` 与缺省字段都回落成 `Default`。
pub fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// 对齐 Pydantic 宽松整数：`"5"` / `5` / `5.0`。
pub fn coerce_usize<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    value_as_usize(Value::deserialize(deserializer)?).ok_or_else(|| DeError::custom("invalid integer"))
}

/// 对齐 Pydantic `Optional[int]`：缺省/null → None，字符串数字也可。
pub fn coerce_opt_usize<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value_as_usize(value)
            .map(Some)
            .ok_or_else(|| DeError::custom("invalid integer")),
    }
}

/// 对齐 Pydantic bool：`"yes"`/`"true"`/`1` 等。
pub fn coerce_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    value_as_bool(Value::deserialize(deserializer)?).ok_or_else(|| DeError::custom("invalid boolean"))
}

/// 对齐 Pydantic `Optional[float]`。
pub fn coerce_opt_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value_as_f64(value)
            .map(Some)
            .ok_or_else(|| DeError::custom("invalid number")),
    }
}

fn value_as_usize(value: Value) -> Option<usize> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .map(|value| value as usize)
            .or_else(|| {
                let float = number.as_f64()?;
                if float >= 0.0 && float.fract() == 0.0 {
                    Some(float as usize)
                } else {
                    None
                }
            }),
        Value::String(raw) => raw.trim().parse().ok(),
        Value::Bool(true) => Some(1),
        Value::Bool(false) => Some(0),
        _ => None,
    }
}

fn value_as_bool(value: Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(flag),
        Value::Number(number) => match number.as_i64() {
            Some(1) => Some(true),
            Some(0) => Some(false),
            _ => None,
        },
        Value::String(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "t" | "yes" | "y" | "on" => Some(true),
            "0" | "false" | "f" | "no" | "n" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn value_as_f64(value: Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(raw) => raw.trim().parse().ok(),
        Value::Bool(true) => Some(1.0),
        Value::Bool(false) => Some(0.0),
        _ => None,
    }
}

pub struct ApiJson<T>(pub T);

pub struct ApiQuery<T>(pub T);

fn json_content_type(req: &Request) -> Option<String> {
    req.headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string())
}

fn is_json_media_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .unwrap_or(content_type)
        .trim()
        .eq_ignore_ascii_case("application/json")
}

impl<S, T> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let content_type = json_content_type(&req).unwrap_or_default();
        // FastAPI 对缺省 Content-Type 仍按 JSON 解析；非 JSON 媒体类型走 model_attributes_type。
        if !content_type.is_empty() && !is_json_media_type(&content_type) {
            return Err(AuthError::unprocessable_at(
                &["body"],
                "Input should be a valid dictionary or object to extract fields from",
                "model_attributes_type",
            ));
        }
        let bytes = match Bytes::from_request(req, state).await {
            Ok(bytes) => bytes,
            Err(_) => {
                return Err(AuthError::unprocessable_at(
                    &["body"],
                    "JSON decode error",
                    "json_invalid",
                ));
            }
        };
        match serde_json::from_slice::<T>(&bytes) {
            Ok(value) => Ok(ApiJson(value)),
            Err(err) if err.is_syntax() || err.is_eof() => {
                Err(AuthError::unprocessable_at(
                    &["body"],
                    "JSON decode error",
                    "json_invalid",
                ))
            }
            Err(err) => Err(map_data_error(&err.to_string())),
        }
    }
}

impl<S, T> FromRequestParts<S> for ApiQuery<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match Query::<T>::from_request_parts(parts, state).await {
            Ok(Query(value)) => Ok(ApiQuery(value)),
            Err(rejection) => Err(map_query_rejection(rejection)),
        }
    }
}

pub fn map_query_rejection(rejection: QueryRejection) -> AuthError {
    map_query_error_text(&rejection.body_text())
}

fn map_query_error_text(text: &str) -> AuthError {
    if let Some(field) = missing_field_name(text) {
        return AuthError::unprocessable_at(&["query", &field], "Field required", "missing");
    }
    if let Some((field, raw)) = query_int_parse_failure(text) {
        return AuthError::unprocessable_at_input(
            &["query", &field],
            "Input should be a valid integer, unable to parse string as an integer",
            "int_parsing",
            serde_json::json!(raw),
        );
    }
    AuthError::unprocessable_at(&["query"], "Input should be a valid value", "value_error")
}

fn query_int_parse_failure(text: &str) -> Option<(String, String)> {
    let looks_int = text.contains("invalid digit")
        || text.contains("expected i64")
        || text.contains("expected u64")
        || text.contains("expected i32")
        || text.contains("expected u32")
        || text.contains("expected isize")
        || text.contains("expected usize");
    if !looks_int {
        return None;
    }
    let field = quoted_ident(text)
        .or_else(|| path_field(text))
        .unwrap_or_else(|| "limit".to_string());
    let raw = quoted_value(text).unwrap_or_default();
    Some((field, raw))
}

fn quoted_ident(text: &str) -> Option<String> {
    const MARKER: &str = "missing field `";
    if text.contains(MARKER) {
        return None;
    }
    let start = text.find('`')?;
    let rest = &text[start + 1..];
    let end = rest.find('`')?;
    let name = rest[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn path_field(text: &str) -> Option<String> {
    for token in ["limit", "skip", "conversation_id"] {
        if text.contains(token) {
            return Some(token.to_string());
        }
    }
    None
}

fn quoted_value(text: &str) -> Option<String> {
    let start = text.find('"')?;
    let rest = &text[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn map_data_error(text: &str) -> AuthError {
    if let Some(field) = missing_field_name(text) {
        return AuthError::unprocessable_at(&["body", &field], "Field required", "missing");
    }
    AuthError::unprocessable_at(&["body"], "Input should be a valid object", "model_type")
}

fn missing_field_name(text: &str) -> Option<String> {
    const MARKER: &str = "missing field `";
    let start = text.find(MARKER)? + MARKER.len();
    let rest = &text[start..];
    let end = rest.find('`')?;
    let name = rest[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use axum::http::StatusCode;

    #[test]
    fn missing_json_field_matches_pydantic() {
        let err = map_data_error(
            "Failed to deserialize the JSON body into the target type: missing field `query`",
        );
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.detail, "Field required");
        assert_eq!(err.loc, vec!["body".to_string(), "query".to_string()]);
        assert_eq!(err.error_type, "missing");
        assert_eq!(
            err.detail_json()["detail"][0]["msg"],
            "Field required"
        );
        assert_eq!(
            err.detail_json()["detail"][0]["input"],
            serde_json::Value::Null
        );
        assert_eq!(
            err.detail_json()["detail"][0]["url"],
            "https://errors.pydantic.dev/2.12/v/missing"
        );
    }

    #[test]
    fn missing_query_field_uses_query_loc() {
        let err = map_query_error_text("missing field `conversation_id`");
        assert_eq!(err.loc, vec!["query".to_string(), "conversation_id".to_string()]);
        assert_eq!(err.detail, "Field required");
    }

    #[test]
    fn query_int_parsing_matches_pydantic() {
        let err = map_query_error_text(
            r#"Failed to deserialize query string: invalid type: string "abc", expected i64"#,
        );
        assert_eq!(err.error_type, "int_parsing");
        assert_eq!(err.loc, vec!["query".to_string(), "limit".to_string()]);
        assert_eq!(
            err.detail,
            "Input should be a valid integer, unable to parse string as an integer"
        );
        assert_eq!(err.input, serde_json::json!("abc"));
    }

    #[test]
    fn null_optional_list_deserializes_as_empty() {
        #[derive(serde::Deserialize)]
        struct Payload {
            #[serde(default, deserialize_with = "null_as_default")]
            history: Vec<String>,
        }
        let omitted: Payload = serde_json::from_str("{}").unwrap();
        assert!(omitted.history.is_empty());
        let explicit_null: Payload = serde_json::from_str(r#"{"history":null}"#).unwrap();
        assert!(explicit_null.history.is_empty());
        let items: Payload = serde_json::from_str(r#"{"history":["user"]}"#).unwrap();
        assert_eq!(items.history, vec!["user"]);
    }

    #[test]
    fn pydantic_coercion_accepts_string_numbers_and_yes() {
        #[derive(serde::Deserialize)]
        struct Payload {
            #[serde(deserialize_with = "coerce_usize")]
            top_k: usize,
            #[serde(deserialize_with = "coerce_bool")]
            use_retrieval: bool,
            #[serde(deserialize_with = "coerce_opt_f64")]
            temperature: Option<f64>,
            #[serde(default, deserialize_with = "coerce_opt_usize")]
            extra: Option<usize>,
        }
        let parsed: Payload = serde_json::from_str(
            r#"{"top_k":"5","use_retrieval":"yes","temperature":"0.7","extra":null}"#,
        )
        .unwrap();
        assert_eq!(parsed.top_k, 5);
        assert!(parsed.use_retrieval);
        assert_eq!(parsed.temperature, Some(0.7));
        assert_eq!(parsed.extra, None);
        let omitted: Payload =
            serde_json::from_str(r#"{"top_k":5,"use_retrieval":true,"temperature":null}"#).unwrap();
        assert_eq!(omitted.temperature, None);
        assert_eq!(omitted.extra, None);
    }

    #[tokio::test]
    async fn missing_content_type_still_parses_json() {
        #[derive(serde::Deserialize)]
        struct Payload {
            query: String,
        }
        let request = Request::builder()
            .method("POST")
            .uri("http://example.com/chat")
            .body(axum::body::Body::from(r#"{"query":"差旅"}"#))
            .unwrap();
        let parsed = match ApiJson::<Payload>::from_request(request, &()).await {
            Ok(parsed) => parsed,
            Err(_) => panic!("expected JSON without Content-Type to parse"),
        };
        assert_eq!(parsed.0.query, "差旅");
    }

    #[tokio::test]
    async fn wrong_content_type_is_model_attributes_type() {
        #[derive(serde::Deserialize)]
        struct Payload {
            query: String,
        }
        let request = Request::builder()
            .method("POST")
            .uri("http://example.com/chat")
            .header(CONTENT_TYPE, "application/octet-stream")
            .body(axum::body::Body::from(r#"{"query":"差旅"}"#))
            .unwrap();
        let err = match ApiJson::<Payload>::from_request(request, &()).await {
            Ok(_) => panic!("expected model_attributes_type"),
            Err(err) => err,
        };
        assert_eq!(err.error_type, "model_attributes_type");
        assert_eq!(
            err.detail,
            "Input should be a valid dictionary or object to extract fields from"
        );
    }

    #[tokio::test]
    async fn query_limit_abc_is_int_parsing() {
        #[derive(serde::Deserialize)]
        struct LimitQuery {
            limit: i64,
        }
        let request = Request::builder()
            .uri("http://example.com/list?limit=abc")
            .body(())
            .unwrap();
        let (mut parts, _) = request.into_parts();
        let err = match ApiQuery::<LimitQuery>::from_request_parts(&mut parts, &()).await {
            Ok(_) => panic!("expected int_parsing for limit=abc"),
            Err(err) => err,
        };
        assert_eq!(err.error_type, "int_parsing");
        assert_eq!(err.loc, vec!["query".to_string(), "limit".to_string()]);
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY);
    }
}
