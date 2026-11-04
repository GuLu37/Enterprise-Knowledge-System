//! 工具与技能调用轨迹日志（对齐 Python `app.tools.trace`）。
//!
//! 记录 call / result / error 三段，payload 先截断再打成一行 JSON。

use serde_json::{Map, Value, json};

const MAX_LOG_STRING_LENGTH: usize = 300;
const MAX_LOG_LIST_ITEMS: usize = 6;
const MAX_LOG_DICT_ITEMS: usize = 12;

pub fn truncate_text(value: &str) -> String {
    let count = value.chars().count();
    if count <= MAX_LOG_STRING_LENGTH {
        return value.to_string();
    }
    let prefix: String = value.chars().take(MAX_LOG_STRING_LENGTH).collect();
    format!("{prefix}...<len={count}>")
}

pub fn compact_for_log(value: &Value) -> Value {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
        Value::String(text) => Value::String(truncate_text(text)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .take(MAX_LOG_LIST_ITEMS)
                .map(compact_for_log)
                .collect(),
        ),
        Value::Object(object) => {
            let mut compacted = Map::new();
            for (key, item) in object.iter().take(MAX_LOG_DICT_ITEMS) {
                compacted.insert(key.clone(), compact_for_log(item));
            }
            Value::Object(compacted)
        }
    }
}

pub fn format_payload(kind: &str, stage: &str, name: &str, payload: Value) -> String {
    let mut event = json!({
        "kind": kind,
        "stage": stage,
        "name": name,
        "timestamp": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
    });
    if let (Some(object), Value::Object(extra)) = (event.as_object_mut(), payload) {
        for (key, value) in extra {
            object.insert(key, value);
        }
    }
    compact_for_log(&event).to_string()
}

fn object_or_empty(value: &Value) -> Value {
    if value.is_null() {
        json!({})
    } else {
        value.clone()
    }
}

pub fn log_tool_call(kind: &str, name: &str, arguments: &Value, context: &Value) {
    tracing::info!(
        "调用开始: {}",
        format_payload(
            kind,
            "call",
            name,
            json!({
                "arguments": object_or_empty(arguments),
                "context": object_or_empty(context),
            }),
        )
    );
}

pub fn log_tool_result(kind: &str, name: &str, result: &Value, context: &Value) {
    tracing::info!(
        "调用结果: {}",
        format_payload(
            kind,
            "result",
            name,
            json!({
                "result": object_or_empty(result),
                "context": object_or_empty(context),
            }),
        )
    );
}

pub fn log_tool_error(kind: &str, name: &str, error: &str, context: &Value) {
    tracing::error!(
        "调用异常: {}",
        format_payload(
            kind,
            "error",
            name,
            json!({
                "error": error,
                "context": object_or_empty(context),
            }),
        )
    );
}

pub fn skill_result_summary(payload: &Value) -> Value {
    let result_keys = payload
        .as_object()
        .map(|object| {
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort();
            json!(keys)
        })
        .unwrap_or_else(|| json!([]));
    json!({
        "result_keys": result_keys,
        "summary": payload.get("summary").cloned().unwrap_or(json!({})),
    })
}

pub fn rag_result_summary(method: &str, result_count: usize, empty: bool) -> Value {
    json!({
        "retrieval_method": method,
        "result_count": result_count,
        "message": if empty {
            "当前知识库未检索到足够相关资料"
        } else {
            "检索完成"
        },
    })
}

pub fn trace_kind(skill_kind: Option<&str>, name: &str) -> &'static str {
    if skill_kind == Some("remote_mcp") {
        "mcp"
    } else if name == "search_knowledge_base" {
        "tool"
    } else {
        "skill"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_long_strings_with_original_len() {
        let text: String = "汉".repeat(301);
        let truncated = truncate_text(&text);
        assert!(truncated.ends_with("...<len=301>"));
        assert_eq!(truncated.chars().count(), MAX_LOG_STRING_LENGTH + "...<len=301>".chars().count());
        assert_eq!(truncate_text("短文本"), "短文本");
    }

    #[test]
    fn compact_keeps_scalars_and_caps_collections() {
        assert_eq!(compact_for_log(&json!(null)), json!(null));
        assert_eq!(compact_for_log(&json!(true)), json!(true));
        assert_eq!(compact_for_log(&json!(12)), json!(12));
        let list: Vec<u64> = (0..10).collect();
        let compacted = compact_for_log(&json!(list));
        assert_eq!(compacted.as_array().unwrap().len(), MAX_LOG_LIST_ITEMS);
        let mut object = serde_json::Map::new();
        for index in 0..20 {
            object.insert(format!("k{index}"), json!(index));
        }
        let compacted = compact_for_log(&Value::Object(object));
        assert_eq!(compacted.as_object().unwrap().len(), MAX_LOG_DICT_ITEMS);
    }

    #[test]
    fn format_payload_includes_kind_stage_and_nested_fields() {
        let raw = format_payload(
            "skill",
            "call",
            "memory_query",
            json!({
                "arguments": {"query": "记忆查询"},
                "context": {"user_id": "u1"},
            }),
        );
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed["kind"], "skill");
        assert_eq!(parsed["stage"], "call");
        assert_eq!(parsed["name"], "memory_query");
        assert_eq!(parsed["arguments"]["query"], "记忆查询");
        assert_eq!(parsed["context"]["user_id"], "u1");
        assert!(parsed.get("timestamp").and_then(Value::as_str).is_some());
    }

    #[test]
    fn skill_and_rag_summaries_match_python_shape() {
        let summary = skill_result_summary(&json!({
            "summary": {"profile_count": 1},
            "matched_memories": [],
            "profile_memories": [{"k": "v"}],
        }));
        assert_eq!(
            summary["result_keys"],
            json!(["matched_memories", "profile_memories", "summary"])
        );
        assert_eq!(summary["summary"]["profile_count"], 1);
        assert_eq!(
            rag_result_summary("hybrid", 0, true)["message"],
            "当前知识库未检索到足够相关资料"
        );
        assert_eq!(rag_result_summary("sparse", 3, false)["result_count"], 3);
        assert_eq!(trace_kind(Some("remote_mcp"), "crm_lookup"), "mcp");
        assert_eq!(trace_kind(Some("local"), "memory_query"), "skill");
        assert_eq!(trace_kind(None, "search_knowledge_base"), "tool");
    }
}
