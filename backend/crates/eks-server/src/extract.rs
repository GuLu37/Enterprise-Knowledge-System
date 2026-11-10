//! 对齐 FastAPI：JSON / Query 反序列化失败返回 422 `detail: [{loc, msg, type}]`。
use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::extract::{FromRequest, FromRequestParts, Json, Query, Request};
use axum::http::request::Parts;
use serde::de::{Deserialize, DeserializeOwned, Deserializer};

use crate::auth::AuthError;

/// 对齐 Pydantic `Optional[List[...]] = None`：JSON `null` 与缺省字段都回落成 `Default`。
pub fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

pub struct ApiJson<T>(pub T);

pub struct ApiQuery<T>(pub T);

impl<S, T> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(ApiJson(value)),
            Err(rejection) => Err(map_json_rejection(rejection)),
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

pub fn map_json_rejection(rejection: JsonRejection) -> AuthError {
    match rejection {
        JsonRejection::JsonDataError(inner) => map_data_error(&inner.body_text()),
        JsonRejection::JsonSyntaxError(_) => {
            AuthError::unprocessable_at(&["body"], "JSON decode error", "json_invalid")
        }
        JsonRejection::MissingJsonContentType(_) => {
            AuthError::unprocessable_at(&["body"], "Field required", "missing")
        }
        other => AuthError::unprocessable_at(&["body"], &other.body_text(), "value_error"),
    }
}

pub fn map_query_rejection(rejection: QueryRejection) -> AuthError {
    let text = rejection.body_text();
    if let Some(field) = missing_field_name(&text) {
        return AuthError::unprocessable_at(&["query", &field], "Field required", "missing");
    }
    AuthError::unprocessable_at(&["query"], "Input should be a valid value", "value_error")
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
    }

    #[test]
    fn missing_query_field_uses_query_loc() {
        let err = map_query_rejection_text("missing field `conversation_id`");
        assert_eq!(err.loc, vec!["query".to_string(), "conversation_id".to_string()]);
        assert_eq!(err.detail, "Field required");
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

    fn map_query_rejection_text(text: &str) -> AuthError {
        if let Some(field) = missing_field_name(text) {
            AuthError::unprocessable_at(&["query", &field], "Field required", "missing")
        } else {
            AuthError::unprocessable_at(&["query"], "Input should be a valid value", "value_error")
        }
    }
}
