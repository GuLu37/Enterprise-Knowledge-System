//! 远程 MCP → Skill 的注册适配（对齐 Python `app.skills.remote_mcp`）。
//!
//! 配置里的 manifest 会变成 `kind=remote_mcp` 的 Skill，出现在 `/skills` 与
//! `/skills/capabilities`。未配置 endpoint 时 `executor_ready=false`，不会进入
//! Function Calling 动作空间。执行器仍是占位：调用一律返回“尚未接入实际执行器”。

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::skills::SkillDefinition;

const DEFAULT_PARAMETERS: &str = r#"{"type":"object","properties":{},"additionalProperties":true}"#;

fn first_non_empty(manifest: &Value, keys: &[&str]) -> Option<String> {
    let object = manifest.as_object()?;
    for key in keys {
        let Some(value) = object.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let text = match value {
            Value::String(text) => text.trim().to_string(),
            other => other.to_string().trim().to_string(),
        };
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

fn normalize_string_list(value: Option<&Value>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    let items: Vec<Value> = match value {
        Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    };
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for item in items {
        let text = match item {
            Value::Null => continue,
            Value::String(text) => text,
            other => other.to_string(),
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if !seen.insert(text.to_lowercase()) {
            continue;
        }
        normalized.push(text.to_string());
    }
    normalized
}

fn coerce_parameters(manifest: &Value) -> Value {
    for key in [
        "parameters",
        "schema",
        "input_schema",
        "inputSchema",
        "arguments_schema",
    ] {
        if let Some(candidate) = manifest.get(key) {
            if candidate.is_object() {
                return candidate.clone();
            }
        }
    }
    serde_json::from_str(DEFAULT_PARAMETERS).unwrap_or_else(|_| json!({}))
}

fn manifest_enabled(manifest: &Value) -> bool {
    match manifest.get("enabled") {
        None => true,
        Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::String(value)) => {
            let normalized = value.trim().to_lowercase();
            !matches!(normalized.as_str(), "false" | "0" | "no" | "off" | "")
        }
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(object)) => !object.is_empty(),
    }
}

pub fn is_remote_mcp_ready(manifest: &Value) -> bool {
    first_non_empty(manifest, &["endpoint", "url", "server_url", "serverUrl"]).is_some()
}

pub fn execution_error_message(definition: &SkillDefinition) -> String {
    let server = definition
        .metadata
        .get("server_name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("remote_mcp");
    let tool = definition
        .metadata
        .get("tool_name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(&definition.name);
    format!(
        "远程 MCP 工具 {server}/{tool} 尚未接入实际执行器，当前不可用。不要重试该工具，请改用其他可用工具或直接回答。"
    )
}

pub fn build_remote_mcp_skill_definition(manifest: &Value) -> Result<SkillDefinition, String> {
    let name = first_non_empty(manifest, &["skill_name", "name", "tool_name"])
        .ok_or_else(|| "远程 MCP manifest 缺少 name/skill_name/tool_name".to_string())?;
    let raw_description = first_non_empty(manifest, &["description", "summary", "title"])
        .unwrap_or_else(|| format!("远程 MCP 工具 {name}"));
    let ready = is_remote_mcp_ready(manifest);
    let description = if ready {
        raw_description
    } else {
        format!("[未接入执行器·当前不可用] {raw_description}")
    };
    let category = first_non_empty(manifest, &["category"]).unwrap_or_else(|| "mcp".into());
    let provider = first_non_empty(manifest, &["provider", "server_name", "server"]);
    let server_name =
        first_non_empty(manifest, &["server_name", "server", "provider"]).unwrap_or_default();
    let tool_name = first_non_empty(manifest, &["tool_name", "remote_tool_name", "name"])
        .unwrap_or_else(|| name.clone());
    let mut metadata = match manifest.get("metadata") {
        Some(Value::Object(object)) => Value::Object(object.clone()),
        _ => json!({}),
    };
    let manifest_snapshot = match manifest {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "executor"
                            | "parameters"
                            | "schema"
                            | "input_schema"
                            | "inputSchema"
                            | "arguments_schema"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ),
        other => other.clone(),
    };
    if let Some(object) = metadata.as_object_mut() {
        object.insert("source".into(), json!("remote_mcp"));
        object.insert("executor_ready".into(), json!(ready));
        object.insert("server_name".into(), json!(server_name));
        object.insert("tool_name".into(), json!(tool_name));
        object.insert("manifest".into(), manifest_snapshot);
    }
    Ok(SkillDefinition {
        name,
        description,
        category,
        kind: "remote_mcp".into(),
        tags: normalize_string_list(manifest.get("tags")),
        provider,
        parameters: coerce_parameters(manifest),
        metadata,
    })
}

/// 对齐 Python `register_skill` 对远程 MCP 的校验失败告警。
pub fn format_register_remote_skill_fail(name: &str, reason: &str) -> String {
    format!("注册远程 MCP skill 失败: {name} reason={reason}")
}

fn accept_remote_skill(definition: &SkillDefinition) -> Result<(), String> {
    if definition.name.trim().is_empty() {
        return Err("skill name 不能为空".into());
    }
    Ok(())
}

/// 把配置中的 manifest 转成可注册 Skill；跳过未启用或缺少名称的条目。
pub fn skills_from_manifests(manifests: &[Value]) -> Vec<SkillDefinition> {
    let mut registered = Vec::new();
    for manifest in manifests {
        if !manifest.is_object() {
            continue;
        }
        if !manifest_enabled(manifest) {
            continue;
        }
        match build_remote_mcp_skill_definition(manifest) {
            Ok(definition) => match accept_remote_skill(&definition) {
                Ok(()) => registered.push(definition),
                Err(exc) => tracing::warn!(
                    "{}",
                    format_register_remote_skill_fail(&definition.name, &exc)
                ),
            },
            Err(err) => tracing::warn!("跳过无效的远程 MCP manifest: {err} ({manifest})"),
        }
    }
    registered
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_unready() -> Value {
        json!({
            "skill_name": "crm_lookup",
            "tool_name": "lookup_account",
            "server_name": "crm",
            "description": "查询客户档案",
            "tags": ["CRM", "crm", "客户"],
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"]
            }
        })
    }

    fn sample_ready() -> Value {
        json!({
            "name": "ticket_create",
            "provider": "itsm",
            "endpoint": "https://mcp.example.internal/ticket",
            "title": "创建工单",
            "schema": {
                "type": "object",
                "properties": {"title": {"type": "string"}}
            }
        })
    }

    #[test]
    fn unready_manifest_is_cataloged_but_not_callable() {
        let definition = build_remote_mcp_skill_definition(&sample_unready()).unwrap();
        assert_eq!(definition.name, "crm_lookup");
        assert_eq!(definition.kind, "remote_mcp");
        assert_eq!(definition.category, "mcp");
        assert_eq!(definition.provider.as_deref(), Some("crm"));
        assert_eq!(definition.tags, vec!["CRM", "客户"]);
        assert!(definition
            .description
            .starts_with("[未接入执行器·当前不可用]"));
        assert_eq!(definition.metadata["executor_ready"], false);
        assert_eq!(definition.metadata["tool_name"], "lookup_account");
        assert_eq!(
            definition.parameters["properties"]["account_id"]["type"],
            "string"
        );
        assert!(!definition.executor_ready());
        assert!(execution_error_message(&definition).contains("crm/lookup_account"));
    }

    #[test]
    fn ready_manifest_enters_function_calling_but_still_placeholder() {
        let definition = build_remote_mcp_skill_definition(&sample_ready()).unwrap();
        assert!(definition.executor_ready());
        assert_eq!(definition.description, "创建工单");
        assert_eq!(definition.provider.as_deref(), Some("itsm"));
        assert_eq!(
            definition.parameters["properties"]["title"]["type"],
            "string"
        );
        assert!(execution_error_message(&definition).contains("itsm/ticket_create"));
    }

    #[test]
    fn register_fail_log_matches_python() {
        assert_eq!(
            format_register_remote_skill_fail("crm_lookup", "skill name 不能为空"),
            "注册远程 MCP skill 失败: crm_lookup reason=skill name 不能为空"
        );
        assert!(accept_remote_skill(
            &build_remote_mcp_skill_definition(&sample_unready()).unwrap()
        )
        .is_ok());
    }

    #[test]
    fn skips_disabled_and_nameless_manifests() {
        let skills = skills_from_manifests(&[
            json!({"name": "off", "enabled": false}),
            json!({"name": "off-str", "enabled": "0"}),
            json!({"description": "missing name"}),
            json!("not-an-object"),
            sample_unready(),
        ]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "crm_lookup");
    }
}
