//! 对齐 FastAPI/Pydantic v2 的请求字段校验：422 + 英文 `detail` 文案。
use crate::auth::AuthError;

pub fn string_min_length(value: &str, min: usize) -> Result<(), AuthError> {
    if char_len(value) < min {
        Err(AuthError::unprocessable(&format!(
            "String should have at least {min} character{}",
            if min == 1 { "" } else { "s" }
        )))
    } else {
        Ok(())
    }
}

pub fn string_max_length(value: &str, max: usize) -> Result<(), AuthError> {
    if char_len(value) > max {
        Err(AuthError::unprocessable(&format!(
            "String should have at most {max} characters"
        )))
    } else {
        Ok(())
    }
}

pub fn bounded_string(value: &str, min: usize, max: usize) -> Result<(), AuthError> {
    string_min_length(value, min)?;
    string_max_length(value, max)
}

pub fn ge_le_int(value: i64, ge: i64, le: i64) -> Result<(), AuthError> {
    if value < ge {
        Err(AuthError::unprocessable(&format!(
            "Input should be greater than or equal to {ge}"
        )))
    } else if value > le {
        Err(AuthError::unprocessable(&format!(
            "Input should be less than or equal to {le}"
        )))
    } else {
        Ok(())
    }
}

pub fn ge_le_float(value: f64, ge: f64, le: f64) -> Result<(), AuthError> {
    if !value.is_finite() || value < ge {
        Err(AuthError::unprocessable(&format!(
            "Input should be greater than or equal to {}",
            format_pydantic_number(ge)
        )))
    } else if value > le {
        Err(AuthError::unprocessable(&format!(
            "Input should be less than or equal to {}",
            format_pydantic_number(le)
        )))
    } else {
        Ok(())
    }
}

pub fn literal(value: &str, allowed: &[&str]) -> Result<(), AuthError> {
    if allowed.iter().any(|item| *item == value) {
        Ok(())
    } else {
        Err(AuthError::unprocessable(&literal_message(allowed)))
    }
}

fn char_len(value: &str) -> usize {
    value.chars().count()
}

fn format_pydantic_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

fn literal_message(allowed: &[&str]) -> String {
    match allowed {
        [] => "Input should be a valid literal".into(),
        [only] => format!("Input should be '{only}'"),
        [first, second] => format!("Input should be '{first}' or '{second}'"),
        _ => {
            let head = &allowed[..allowed.len() - 1];
            let last = allowed[allowed.len() - 1];
            let quoted: Vec<String> = head.iter().map(|item| format!("'{item}'")).collect();
            format!("Input should be {} or '{last}'", quoted.join(", "))
        }
    }
}

pub fn login_fields(username: &str, password: &str) -> Result<(), AuthError> {
    bounded_string(username, 1, 128)?;
    bounded_string(password, 1, 128)
}

pub fn register_fields(username: &str, password: &str, confirm_password: &str) -> Result<(), AuthError> {
    bounded_string(username, 1, 128)?;
    bounded_string(password, 6, 128)?;
    bounded_string(confirm_password, 6, 128)
}

pub fn password_change_fields(current_password: &str, new_password: &str) -> Result<(), AuthError> {
    bounded_string(current_password, 1, 128)?;
    bounded_string(new_password, 6, 128)
}

pub fn refresh_token_field(refresh_token: &str) -> Result<(), AuthError> {
    string_min_length(refresh_token, 1)
}

pub fn skill_name_field(name: &str) -> Result<(), AuthError> {
    string_min_length(name, 1)
}

pub fn session_id_field(session_id: Option<&str>) -> Result<(), AuthError> {
    if let Some(value) = session_id {
        string_max_length(value, 64)?;
    }
    Ok(())
}

/// 对齐 Pydantic `Optional[datetime]`：缺省/null 通过，非法字符串 422。
pub fn optional_datetime_field(value: Option<&str>) -> Result<(), AuthError> {
    match value {
        None => Ok(()),
        Some(text) if crate::db::parse_client_datetime(text).is_some() => Ok(()),
        Some(_) => Err(AuthError::unprocessable("Input should be a valid datetime")),
    }
}

/// 对齐 `RetrievalRequest.top_k`：缺省 5，越界 422，不夹取。
pub fn retrieval_top_k(top_k: Option<usize>) -> Result<usize, AuthError> {
    let value = top_k.unwrap_or(5);
    ge_le_int(value as i64, 1, 50)?;
    Ok(value)
}

pub fn document_list_skip(skip: i64) -> Result<i64, AuthError> {
    ge_le_int(skip, 0, i64::MAX)?;
    Ok(skip)
}

/// 对齐 `/documents/list`：`limit` 缺省 10，范围 1–100。
pub fn document_list_limit(limit: Option<i64>) -> Result<i64, AuthError> {
    let value = limit.unwrap_or(10);
    ge_le_int(value, 1, 100)?;
    Ok(value)
}

pub struct ChatSchema {
    pub temperature: Option<f64>,
    pub top_k: usize,
    pub retrieval_method: Option<String>,
    pub short_memory_strategy: String,
    pub short_memory_n: usize,
    pub short_memory_m: usize,
    pub history: Vec<(String, String)>,
    pub query: String,
}

pub fn chat_request_fields(schema: &ChatSchema) -> Result<(), AuthError> {
    string_min_length(&schema.query, 1)?;
    ge_le_int(schema.top_k as i64, 1, 50)?;
    if let Some(temperature) = schema.temperature {
        ge_le_float(temperature, 0.0, 2.0)?;
    }
    ge_le_int(schema.short_memory_n as i64, 1, i64::MAX)?;
    ge_le_int(schema.short_memory_m as i64, 1, i64::MAX)?;
    literal(&schema.short_memory_strategy, &["window", "summary"])?;
    if let Some(method) = &schema.retrieval_method {
        literal(method, &["hybrid", "dense", "sparse"])?;
    }
    for (role, content) in &schema.history {
        literal(role, &["system", "user", "assistant"])?;
        string_min_length(content, 1)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    fn detail(err: AuthError) -> (StatusCode, String) {
        (err.status, err.detail)
    }

    #[test]
    fn password_and_username_follow_pydantic_lengths() {
        assert!(login_fields("admin", "1").is_ok());
        assert_eq!(
            detail(login_fields("", "secret").unwrap_err()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "String should have at least 1 character".into()
            )
        );
        assert_eq!(
            detail(register_fields("user", "12345", "12345").unwrap_err()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "String should have at least 6 characters".into()
            )
        );
        assert!(register_fields("user", "123456", "123456").is_ok());
        let long = "a".repeat(129);
        assert_eq!(
            detail(login_fields(&long, "secret").unwrap_err()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "String should have at most 128 characters".into()
            )
        );
        assert_eq!(
            detail(password_change_fields("old", "12345").unwrap_err()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "String should have at least 6 characters".into()
            )
        );
        assert!(refresh_token_field("token").is_ok());
        assert!(refresh_token_field("").is_err());
    }

    #[test]
    fn retrieval_top_k_defaults_to_five_and_rejects_out_of_range() {
        assert_eq!(retrieval_top_k(None).unwrap(), 5);
        assert_eq!(retrieval_top_k(Some(1)).unwrap(), 1);
        assert_eq!(retrieval_top_k(Some(50)).unwrap(), 50);
        assert_eq!(
            detail(retrieval_top_k(Some(0)).unwrap_err()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "Input should be greater than or equal to 1".into()
            )
        );
        assert_eq!(
            detail(retrieval_top_k(Some(51)).unwrap_err()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "Input should be less than or equal to 50".into()
            )
        );
    }

    #[test]
    fn chat_schema_rejects_temperature_role_and_empty_query() {
        let mut schema = ChatSchema {
            temperature: Some(0.7),
            top_k: 5,
            retrieval_method: Some("hybrid".into()),
            short_memory_strategy: "window".into(),
            short_memory_n: 5,
            short_memory_m: 10,
            history: vec![("user".into(), "你好".into())],
            query: "差旅报销".into(),
        };
        assert!(chat_request_fields(&schema).is_ok());

        schema.temperature = Some(-0.1);
        assert_eq!(
            detail(chat_request_fields(&schema).unwrap_err()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "Input should be greater than or equal to 0".into()
            )
        );
        schema.temperature = Some(2.1);
        assert_eq!(
            detail(chat_request_fields(&schema).unwrap_err()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "Input should be less than or equal to 2".into()
            )
        );
        schema.temperature = Some(2.0);
        schema.history = vec![("tool".into(), "结果".into())];
        assert_eq!(
            detail(chat_request_fields(&schema).unwrap_err()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "Input should be 'system', 'user' or 'assistant'".into()
            )
        );
        schema.history = vec![("user".into(), "".into())];
        assert_eq!(
            detail(chat_request_fields(&schema).unwrap_err()).1,
            "String should have at least 1 character"
        );
        schema.history.clear();
        schema.query = "".into();
        assert_eq!(
            detail(chat_request_fields(&schema).unwrap_err()).1,
            "String should have at least 1 character"
        );
        schema.query = "差旅".into();
        schema.retrieval_method = Some("bm25".into());
        assert_eq!(
            detail(chat_request_fields(&schema).unwrap_err()).1,
            "Input should be 'hybrid', 'dense' or 'sparse'"
        );
        schema.retrieval_method = Some("hybrid".into());
        schema.short_memory_strategy = "fifo".into();
        assert_eq!(
            detail(chat_request_fields(&schema).unwrap_err()).1,
            "Input should be 'window' or 'summary'"
        );
    }

    #[test]
    fn session_id_and_document_list_follow_pydantic() {
        assert!(session_id_field(None).is_ok());
        assert!(session_id_field(Some("s1")).is_ok());
        assert_eq!(
            detail(session_id_field(Some(&"x".repeat(65))).unwrap_err()).1,
            "String should have at most 64 characters"
        );
        assert_eq!(document_list_skip(0).unwrap(), 0);
        assert_eq!(
            detail(document_list_skip(-1).unwrap_err()).1,
            "Input should be greater than or equal to 0"
        );
        assert_eq!(document_list_limit(None).unwrap(), 10);
        assert_eq!(
            detail(document_list_limit(Some(0)).unwrap_err()).1,
            "Input should be greater than or equal to 1"
        );
        assert_eq!(
            detail(document_list_limit(Some(101)).unwrap_err()).1,
            "Input should be less than or equal to 100"
        );
        assert_eq!(
            detail(skill_name_field("").unwrap_err()).1,
            "String should have at least 1 character"
        );
        assert!(optional_datetime_field(None).is_ok());
        assert!(optional_datetime_field(Some("2026-03-15T08:30:00.123Z")).is_ok());
        assert!(optional_datetime_field(Some("2026-03-15T16:30:00+08:00")).is_ok());
        assert_eq!(
            detail(optional_datetime_field(Some("")).unwrap_err()).1,
            "Input should be a valid datetime"
        );
        assert_eq!(
            detail(optional_datetime_field(Some("bad")).unwrap_err()).1,
            "Input should be a valid datetime"
        );
    }
}
