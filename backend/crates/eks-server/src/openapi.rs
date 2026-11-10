//! 手写 OpenAPI 3.1 与 `/docs` 页面，对齐 FastAPI 默认文档入口。
use serde_json::{json, Value};

use crate::config::Config;

const DOCS_HTML: &str = r#"<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>API Docs</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
    h1 { margin-bottom: 0.25rem; }
    .muted { color: #666; }
    section { border-top: 1px solid #ddd; padding: 0.75rem 0; }
    code { font-size: 0.95em; }
    .method { display: inline-block; min-width: 4.5rem; font-weight: 700; }
  </style>
</head>
<body>
  <h1 id="title">API Docs</h1>
  <p class="muted" id="subtitle">正在加载 OpenAPI…</p>
  <div id="paths"></div>
  <script>
    fetch("/openapi.json").then(function (response) { return response.json(); }).then(function (spec) {
      var info = spec.info || {};
      document.title = (info.title || "API") + " " + (info.version || "");
      document.getElementById("title").textContent = info.title || "API Docs";
      var description = info.description ? info.description + " · " : "";
      document.getElementById("subtitle").textContent = description + "version " + (info.version || "") + " · OpenAPI " + (spec.openapi || "");
      var root = document.getElementById("paths");
      var paths = spec.paths || {};
      Object.keys(paths).sort().forEach(function (path) {
        var methods = paths[path] || {};
        Object.keys(methods).forEach(function (method) {
          var op = methods[method] || {};
          var section = document.createElement("section");
          section.innerHTML = "<div><span class='method'>" + method.toUpperCase() +
            "</span> <code>" + path + "</code></div><p>" + (op.summary || op.operationId || "") + "</p>";
          root.appendChild(section);
        });
      });
    }).catch(function (err) {
      document.getElementById("subtitle").textContent = "无法加载 /openapi.json: " + err;
    });
  </script>
</body>
</html>
"#;

/// 对齐 FastAPI 默认 `/redoc`：加载 ReDoc 并指向 `/openapi.json`。
const REDOC_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
<title>ReDoc</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
<style>body { margin: 0; padding: 0; }</style>
</head>
<body>
<noscript>ReDoc requires Javascript to function. Please enable it to browse the documentation.</noscript>
<redoc spec-url="/openapi.json"></redoc>
<script src="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js"></script>
</body>
</html>
"#;

fn op(summary: &str, operation_id: &str, tag: Option<&str>) -> Value {
    op_full(summary, operation_id, tag, "200", false)
}

fn secured(summary: &str, operation_id: &str, tag: &str) -> Value {
    op_full(summary, operation_id, Some(tag), "200", true)
}

fn accepted(summary: &str, operation_id: &str, tag: &str) -> Value {
    op_full(summary, operation_id, Some(tag), "202", true)
}

fn op_full(
    summary: &str,
    operation_id: &str,
    tag: Option<&str>,
    success: &str,
    bearer: bool,
) -> Value {
    let mut value = json!({
        "summary": summary,
        "operationId": operation_id,
        "responses": { success: { "description": "成功" } }
    });
    if let Some(tag) = tag {
        value["tags"] = json!([tag]);
    }
    if bearer {
        value["security"] = json!([{ "HTTPBearer": [] }]);
    }
    value
}

fn ref_schema(name: &str) -> Value {
    json!({ "$ref": format!("#/components/schemas/{name}") })
}

fn validation_response() -> Value {
    json!({
        "description": "Validation Error",
        "content": {
            "application/json": { "schema": ref_schema("HTTPValidationError") }
        }
    })
}

fn with_query_params(mut op: Value, params: Value) -> Value {
    op["parameters"] = params;
    op["responses"]["422"] = validation_response();
    op
}

fn with_multipart_upload(mut op: Value) -> Value {
    op["requestBody"] = json!({
        "required": true,
        "content": {
            "multipart/form-data": {
                "schema": ref_schema("Body_upload_document_api_v1_documents_upload_post")
            }
        }
    });
    op["responses"]["422"] = validation_response();
    op
}

fn with_array_response(mut op: Value, item: &str) -> Value {
    op["responses"]["200"]["content"] = json!({
        "application/json": {
            "schema": {
                "type": "array",
                "items": ref_schema(item),
                "title": "Response Get Skills Api V1 Skills Get"
            }
        }
    });
    op
}

fn with_json_io(mut op: Value, request: Option<&str>, response: Option<&str>) -> Value {
    if let Some(name) = request {
        op["requestBody"] = json!({
            "required": true,
            "content": {
                "application/json": { "schema": ref_schema(name) }
            }
        });
        op["responses"]["422"] = validation_response();
    }
    if let Some(name) = response {
        let code = if op["responses"].get("202").is_some() {
            "202"
        } else {
            "200"
        };
        op["responses"][code]["content"] = json!({
            "application/json": { "schema": ref_schema(name) }
        });
    }
    op
}

fn components() -> Value {
    json!({
        "securitySchemes": {
            "HTTPBearer": {
                "type": "http",
                "scheme": "bearer"
            }
        },
        "schemas": {
            "HTTPValidationError": {
                "title": "HTTPValidationError",
                "type": "object",
                "properties": {
                    "detail": {
                        "title": "Detail",
                        "type": "array",
                        "items": ref_schema("ValidationError")
                    }
                }
            },
            "ValidationError": {
                "title": "ValidationError",
                "required": ["loc", "msg", "type"],
                "type": "object",
                "properties": {
                    "loc": {
                        "title": "Location",
                        "type": "array",
                        "items": {
                            "anyOf": [{ "type": "string" }, { "type": "integer" }]
                        }
                    },
                    "msg": { "title": "Message", "type": "string" },
                    "type": { "title": "Error Type", "type": "string" }
                }
            },
            "UserResponse": {
                "title": "UserResponse",
                "type": "object",
                "required": ["user_id", "username", "role"],
                "properties": {
                    "user_id": { "type": "string", "title": "User Id" },
                    "username": { "type": "string", "title": "Username" },
                    "role": { "type": "string", "title": "Role" },
                    "is_active": { "type": "boolean", "title": "Is Active", "default": true },
                    "created_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] },
                    "updated_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] },
                    "last_login_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] }
                }
            },
            "TokenResponse": {
                "title": "TokenResponse",
                "type": "object",
                "required": ["access_token", "refresh_token", "user"],
                "properties": {
                    "access_token": { "type": "string", "title": "Access Token" },
                    "refresh_token": { "type": "string", "title": "Refresh Token" },
                    "token_type": { "type": "string", "title": "Token Type", "default": "bearer", "enum": ["bearer"] },
                    "access_expires_in": { "type": "integer", "title": "Access Expires In", "default": 0 },
                    "refresh_expires_in": { "type": "integer", "title": "Refresh Expires In", "default": 0 },
                    "user": ref_schema("UserResponse")
                }
            },
            "LoginRequest": {
                "title": "LoginRequest",
                "type": "object",
                "required": ["username", "password"],
                "properties": {
                    "username": { "type": "string", "title": "Username", "minLength": 1, "maxLength": 128 },
                    "password": { "type": "string", "title": "Password", "minLength": 1, "maxLength": 128 }
                }
            },
            "RegisterRequest": {
                "title": "RegisterRequest",
                "type": "object",
                "required": ["username", "password", "confirm_password"],
                "properties": {
                    "username": { "type": "string", "title": "Username", "minLength": 1, "maxLength": 128 },
                    "password": { "type": "string", "title": "Password", "minLength": 6, "maxLength": 128 },
                    "confirm_password": { "type": "string", "title": "Confirm Password", "minLength": 6, "maxLength": 128 }
                }
            },
            "RefreshTokenRequest": {
                "title": "RefreshTokenRequest",
                "type": "object",
                "required": ["refresh_token"],
                "properties": {
                    "refresh_token": { "type": "string", "title": "Refresh Token", "minLength": 1 }
                }
            },
            "PasswordChangeRequest": {
                "title": "PasswordChangeRequest",
                "type": "object",
                "required": ["current_password", "new_password"],
                "properties": {
                    "current_password": { "type": "string", "title": "Current Password", "minLength": 1, "maxLength": 128 },
                    "new_password": { "type": "string", "title": "New Password", "minLength": 6, "maxLength": 128 }
                }
            },
            "SkillExecuteRequest": {
                "title": "SkillExecuteRequest",
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": { "type": "string", "title": "Name", "minLength": 1 },
                    "arguments": { "type": "object", "title": "Arguments", "additionalProperties": true },
                    "context": ref_schema("SkillExecutionContextRequest")
                }
            },
            "SkillExecutionContextRequest": {
                "title": "SkillExecutionContextRequest",
                "type": "object",
                "properties": {
                    "conversation_id": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "session_id": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "query": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "trace_id": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "metadata": { "type": "object", "title": "Metadata", "additionalProperties": true }
                }
            },
            "SkillExecuteResponse": {
                "title": "SkillExecuteResponse",
                "type": "object",
                "required": ["skill", "category", "kind", "executed_at"],
                "properties": {
                    "skill": { "type": "string" },
                    "category": { "type": "string" },
                    "kind": { "type": "string" },
                    "provider": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "arguments": { "type": "object", "additionalProperties": true },
                    "context": { "type": "object", "additionalProperties": true },
                    "result": { "type": "object", "additionalProperties": true },
                    "executed_at": { "type": "string" }
                }
            },
            "SkillDefinitionResponse": {
                "title": "SkillDefinitionResponse",
                "type": "object",
                "required": ["name", "description", "category", "kind"],
                "properties": {
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "category": { "type": "string" },
                    "kind": { "type": "string" },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "provider": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "metadata": { "type": "object", "additionalProperties": true },
                    "function_calling": { "type": "object", "additionalProperties": true }
                }
            },
            "DocumentResponse": {
                "title": "DocumentResponse",
                "type": "object",
                "required": ["document_id", "original_filename", "stored_filename", "file_path", "file_type", "file_size", "status", "chunk_count"],
                "properties": {
                    "document_id": { "type": "string" },
                    "original_filename": { "type": "string" },
                    "stored_filename": { "type": "string" },
                    "file_path": { "type": "string" },
                    "content_type": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "file_type": { "type": "string" },
                    "file_size": { "type": "integer" },
                    "status": { "type": "string", "enum": ["processing", "ready", "failed", "deleting", "deleted"] },
                    "chunk_count": { "type": "integer", "default": 0 },
                    "error_message": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "created_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] },
                    "updated_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] }
                }
            },
            "DocumentListResponse": {
                "title": "DocumentListResponse",
                "type": "object",
                "required": ["documents", "total", "skip", "limit"],
                "properties": {
                    "documents": { "type": "array", "items": ref_schema("DocumentResponse") },
                    "total": { "type": "integer" },
                    "skip": { "type": "integer" },
                    "limit": { "type": "integer" }
                }
            },
            "DocumentContentResponse": {
                "title": "DocumentContentResponse",
                "type": "object",
                "required": ["document_id", "original_filename", "file_type", "status", "content"],
                "properties": {
                    "document_id": { "type": "string" },
                    "original_filename": { "type": "string" },
                    "file_type": { "type": "string" },
                    "status": { "type": "string" },
                    "content": { "type": "string" },
                    "content_type": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "updated_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] }
                }
            },
            "DocumentDeleteResponse": {
                "title": "DocumentDeleteResponse",
                "type": "object",
                "required": ["document_id", "original_filename", "stored_filename", "file_path", "file_type", "file_size", "status", "chunk_count"],
                "properties": {
                    "document_id": { "type": "string" },
                    "original_filename": { "type": "string" },
                    "stored_filename": { "type": "string" },
                    "file_path": { "type": "string" },
                    "content_type": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "file_type": { "type": "string" },
                    "file_size": { "type": "integer" },
                    "status": { "type": "string", "enum": ["processing", "ready", "failed", "deleting", "deleted"] },
                    "chunk_count": { "type": "integer", "default": 0 },
                    "error_message": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "created_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] },
                    "updated_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] },
                    "milvus_deleted": { "type": "boolean", "default": false },
                    "file_deleted": { "type": "boolean", "default": false }
                }
            },
            "Body_upload_document_api_v1_documents_upload_post": {
                "title": "Body_upload_document_api_v1_documents_upload_post",
                "required": ["file"],
                "type": "object",
                "properties": {
                    "file": { "type": "string", "format": "binary", "title": "File" }
                }
            },
            "RetrievalRequest": {
                "title": "RetrievalRequest",
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": { "type": "string", "minLength": 1 },
                    "top_k": { "type": "integer", "default": 5, "minimum": 1, "maximum": 50 }
                }
            },
            "RetrievalResultResponse": {
                "title": "RetrievalResultResponse",
                "type": "object",
                "required": ["content"],
                "properties": {
                    "content": { "type": "string" },
                    "metadata": { "type": "object", "additionalProperties": true },
                    "score": { "type": "number", "default": 0.0 },
                    "source": { "type": "string", "default": "" }
                }
            },
            "RetrievalResponse": {
                "title": "RetrievalResponse",
                "type": "object",
                "required": ["query", "retrieval_method", "results", "top_k"],
                "properties": {
                    "query": { "type": "string" },
                    "retrieval_method": { "type": "string", "enum": ["hybrid", "dense", "sparse"] },
                    "results": { "type": "array", "items": ref_schema("RetrievalResultResponse") },
                    "top_k": { "type": "integer" }
                }
            },
            "ChatMessage": {
                "title": "ChatMessage",
                "type": "object",
                "required": ["role", "content"],
                "properties": {
                    "role": { "type": "string", "enum": ["system", "user", "assistant"] },
                    "content": { "type": "string", "minLength": 1 }
                }
            },
            "ChatRequest": {
                "title": "ChatRequest",
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": { "type": "string", "minLength": 1 },
                    "history": { "anyOf": [{ "type": "array", "items": ref_schema("ChatMessage") }, { "type": "null" }] },
                    "conversation_id": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "session_id": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "top_k": { "type": "integer", "default": 5, "minimum": 1, "maximum": 50 },
                    "use_retrieval": { "type": "boolean", "default": true },
                    "retrieval_method": { "type": "string", "default": "hybrid", "enum": ["hybrid", "dense", "sparse"] },
                    "short_memory_strategy": { "type": "string", "default": "window", "enum": ["window", "summary"] },
                    "short_memory_n": { "type": "integer", "default": 5, "minimum": 1 },
                    "short_memory_m": { "type": "integer", "default": 10, "minimum": 1 },
                    "temperature": { "anyOf": [{ "type": "number", "minimum": 0, "maximum": 2 }, { "type": "null" }] },
                    "provider": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "model": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
                }
            },
            "ChatResponse": {
                "title": "ChatResponse",
                "type": "object",
                "required": ["query", "response", "model"],
                "properties": {
                    "query": { "type": "string" },
                    "response": { "type": "string" },
                    "sources": { "type": "array", "items": ref_schema("RetrievalResultResponse") },
                    "model": { "type": "string" },
                    "retrieval_method": { "anyOf": [{ "type": "string", "enum": ["hybrid", "dense", "sparse"] }, { "type": "null" }] }
                }
            },
            "ChatSettingsResponse": {
                "title": "ChatSettingsResponse",
                "type": "object",
                "required": ["max_conversations"],
                "properties": { "max_conversations": { "type": "integer" } }
            },
            "ChatRuntimeStatusResponse": {
                "title": "ChatRuntimeStatusResponse",
                "type": "object",
                "properties": {
                    "ready": { "type": "boolean", "default": false },
                    "status": { "type": "string", "default": "initializing" },
                    "llm_warmed": { "type": "boolean", "default": false },
                    "embedding_warmed": { "type": "boolean", "default": false },
                    "cross_encoder_enabled": { "type": "boolean", "default": false },
                    "cross_encoder_warmed": { "type": "boolean", "default": false },
                    "cross_encoder_model": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "cross_encoder_device": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "cross_encoder_error": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "provider": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
                }
            },
            "ChatWarmupResponse": {
                "title": "ChatWarmupResponse",
                "type": "object",
                "properties": {
                    "status": { "type": "string", "default": "initializing" },
                    "llm_warmed": { "type": "boolean", "default": false },
                    "embedding_warmed": { "type": "boolean", "default": false },
                    "cross_encoder_enabled": { "type": "boolean", "default": false },
                    "cross_encoder_warmed": { "type": "boolean", "default": false },
                    "cross_encoder_model": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "cross_encoder_device": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "cross_encoder_error": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "provider": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
                }
            },
            "ConversationRecordResponse": {
                "title": "ConversationRecordResponse",
                "type": "object",
                "required": ["conversation_id"],
                "properties": {
                    "conversation_id": { "type": "string" },
                    "session_id": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                    "created_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] },
                    "updated_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] },
                    "messages": { "type": "array", "items": { "type": "object", "additionalProperties": true } }
                }
            },
            "ConversationListResponse": {
                "title": "ConversationListResponse",
                "type": "object",
                "properties": {
                    "conversations": { "type": "array", "items": ref_schema("ConversationRecordResponse") }
                }
            },
            "ConversationUpsertRequest": {
                "title": "ConversationUpsertRequest",
                "type": "object",
                "properties": {
                    "session_id": { "anyOf": [{ "type": "string", "maxLength": 64 }, { "type": "null" }] },
                    "created_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] },
                    "updated_at": { "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }] },
                    "messages": { "type": "array", "items": { "type": "object", "additionalProperties": true } }
                }
            },
            "ConversationDeleteResponse": {
                "title": "ConversationDeleteResponse",
                "type": "object",
                "required": ["conversation_id"],
                "properties": {
                    "conversation_id": { "type": "string" },
                    "memory_deleted": { "type": "boolean", "default": false }
                }
            },
            "ConversationClearResponse": {
                "title": "ConversationClearResponse",
                "type": "object",
                "properties": {
                    "deleted_count": { "type": "integer", "default": 0 },
                    "memory_deleted_count": { "type": "integer", "default": 0 }
                }
            }
        }
    })
}

pub fn spec(config: &Config) -> Value {
    let prefix = config.api_prefix.trim_end_matches('/');
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": config.app_name,
            "version": config.app_version,
            "description": "基于 LangChain 的 RAG 三重检索系统",
        },
        "tags": [
            { "name": "Auth" },
            { "name": "Documents" },
            { "name": "Retrieval" },
            { "name": "Chat" },
            { "name": "Skills" },
        ],
        "components": components(),
        "paths": {
            "/": { "get": op("根路由", "root", None) },
            "/health": { "get": op("健康检查", "health", None) },
            "/docs": { "get": op("Swagger 风格文档页", "docs", None) },
            "/redoc": { "get": op("ReDoc 风格文档页", "redoc", None) },
            "/openapi.json": { "get": op("OpenAPI Schema", "openapi_json", None) },
            (format!("{prefix}/auth/login")): { "post": with_json_io(op("登录", "login", Some("Auth")), Some("LoginRequest"), Some("TokenResponse")) },
            (format!("{prefix}/auth/register")): { "post": with_json_io(op("注册", "register", Some("Auth")), Some("RegisterRequest"), Some("TokenResponse")) },
            (format!("{prefix}/auth/refresh")): { "post": with_json_io(op("刷新令牌", "refresh", Some("Auth")), Some("RefreshTokenRequest"), Some("TokenResponse")) },
            (format!("{prefix}/auth/password")): { "post": with_json_io(secured("修改密码", "change_password", "Auth"), Some("PasswordChangeRequest"), Some("TokenResponse")) },
            (format!("{prefix}/auth/me")): { "get": with_json_io(secured("当前用户", "me", "Auth"), None, Some("UserResponse")) },
            (format!("{prefix}/retrieval/search/hybrid")): { "post": with_json_io(secured("混合检索", "retrieval_hybrid", "Retrieval"), Some("RetrievalRequest"), Some("RetrievalResponse")) },
            (format!("{prefix}/retrieval/search/dense")): { "post": with_json_io(secured("稠密检索", "retrieval_dense", "Retrieval"), Some("RetrievalRequest"), Some("RetrievalResponse")) },
            (format!("{prefix}/retrieval/search/sparse")): { "post": with_json_io(secured("稀疏检索", "retrieval_sparse", "Retrieval"), Some("RetrievalRequest"), Some("RetrievalResponse")) },
            (format!("{prefix}/retrieval/rag")): { "post": with_json_io(secured("RAG 检索生成", "retrieval_rag", "Retrieval"), Some("ChatRequest"), Some("ChatResponse")) },
            (format!("{prefix}/retrieval/cache-clear")): { "post": secured("清空检索缓存", "retrieval_cache_clear", "Retrieval") },
            (format!("{prefix}/chat/settings")): { "get": with_json_io(secured("对话设置", "chat_settings", "Chat"), None, Some("ChatSettingsResponse")) },
            (format!("{prefix}/chat/runtime-status")): { "get": with_json_io(secured("运行时状态", "chat_runtime_status", "Chat"), None, Some("ChatRuntimeStatusResponse")) },
            (format!("{prefix}/chat/warmup")): { "post": with_json_io(secured("预热", "chat_warmup", "Chat"), None, Some("ChatWarmupResponse")) },
            (format!("{prefix}/chat/generate")): { "post": with_json_io(secured("生成回复", "generate_chat", "Chat"), Some("ChatRequest"), Some("ChatResponse")) },
            (format!("{prefix}/chat/consolidate")): { "post": with_query_params(
                secured("巩固记忆", "consolidate_chat", "Chat"),
                json!([{
                    "name": "conversation_id",
                    "in": "query",
                    "required": true,
                    "schema": { "type": "string", "title": "Conversation Id" }
                }])
            ) },
            (format!("{prefix}/chat/stream")): { "post": with_json_io(secured("流式对话", "stream_chat", "Chat"), Some("ChatRequest"), None) },
            (format!("{prefix}/chat/conversations")): {
                "get": with_json_io(secured("会话列表", "list_conversations", "Chat"), None, Some("ConversationListResponse")),
                "delete": with_json_io(secured("清空会话", "clear_conversations", "Chat"), None, Some("ConversationClearResponse"))
            },
            (format!("{prefix}/chat/conversations/{{conversation_id}}")): {
                "put": with_json_io(secured("更新会话", "upsert_conversation", "Chat"), Some("ConversationUpsertRequest"), Some("ConversationRecordResponse")),
                "delete": with_json_io(secured("删除会话", "delete_conversation", "Chat"), None, Some("ConversationDeleteResponse"))
            },
            (format!("{prefix}/documents")): { "get": with_query_params(
                with_json_io(secured("文档列表", "list_documents", "Documents"), None, Some("DocumentListResponse")),
                json!([
                    { "name": "skip", "in": "query", "required": false, "schema": { "type": "integer", "minimum": 0, "default": 0, "title": "Skip" } },
                    { "name": "limit", "in": "query", "required": false, "schema": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10, "title": "Limit" } }
                ])
            ) },
            (format!("{prefix}/documents/list")): { "get": with_query_params(
                with_json_io(secured("文档列表（兼容）", "list_documents_compat", "Documents"), None, Some("DocumentListResponse")),
                json!([
                    { "name": "skip", "in": "query", "required": false, "schema": { "type": "integer", "minimum": 0, "default": 0, "title": "Skip" } },
                    { "name": "limit", "in": "query", "required": false, "schema": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10, "title": "Limit" } }
                ])
            ) },
            (format!("{prefix}/documents/upload")): { "post": with_multipart_upload(with_json_io(secured("上传文档", "upload_document", "Documents"), None, Some("DocumentResponse"))) },
            (format!("{prefix}/documents/content/{{document_id}}")): { "get": with_json_io(secured("文档内容", "get_document_content", "Documents"), None, Some("DocumentContentResponse")) },
            (format!("{prefix}/documents/delete/{{document_id}}")): { "delete": with_json_io(accepted("删除文档", "delete_document_compat", "Documents"), None, Some("DocumentDeleteResponse")) },
            (format!("{prefix}/documents/{{document_id}}")): { "delete": with_json_io(accepted("删除文档", "delete_document", "Documents"), None, Some("DocumentDeleteResponse")) },
            (format!("{prefix}/skills")): { "get": with_array_response(secured("技能列表", "list_skills", "Skills"), "SkillDefinitionResponse") },
            (format!("{prefix}/skills/function-calling")): { "get": secured("Function Calling 技能", "list_skill_function_calling", "Skills") },
            (format!("{prefix}/skills/capabilities")): { "get": secured("技能能力", "skill_capabilities", "Skills") },
            (format!("{prefix}/skills/execute")): { "post": with_json_io(secured("执行技能", "execute_skill", "Skills"), Some("SkillExecuteRequest"), Some("SkillExecuteResponse")) },
        }
    })
}

pub fn docs_html() -> &'static str {
    DOCS_HTML
}

pub fn redoc_html() -> &'static str {
    REDOC_HTML
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_matches_python_docs_contract() {
        let config = crate::config::Config::jwt_test();
        let spec = spec(&config);
        assert_eq!(spec["openapi"], "3.1.0");
        assert_eq!(spec["info"]["title"], "Enterprise Knowledge System");
        assert_eq!(spec["info"]["version"], "1.0.0");
        assert_eq!(
            spec["info"]["description"],
            "基于 LangChain 的 RAG 三重检索系统"
        );
        assert!(spec["paths"].get("/health").is_some());
        assert!(spec["paths"].get("/docs").is_some());
        assert!(spec["paths"].get("/openapi.json").is_some());
        assert!(spec["paths"]["/api/v1/auth/login"]["post"].is_object());
        assert!(spec["paths"]["/api/v1/documents/upload"]["post"].is_object());
        assert!(spec["paths"]["/api/v1/chat/stream"]["post"].is_object());
        assert_eq!(
            spec["tags"],
            serde_json::json!([
                { "name": "Auth" },
                { "name": "Documents" },
                { "name": "Retrieval" },
                { "name": "Chat" },
                { "name": "Skills" },
            ])
        );
        assert_eq!(
            spec["paths"]["/api/v1/auth/login"]["post"]["tags"],
            serde_json::json!(["Auth"])
        );
        assert_eq!(
            spec["paths"]["/api/v1/documents/upload"]["post"]["tags"],
            serde_json::json!(["Documents"])
        );
        assert_eq!(
            spec["paths"]["/api/v1/retrieval/rag"]["post"]["tags"],
            serde_json::json!(["Retrieval"])
        );
        assert_eq!(
            spec["paths"]["/api/v1/chat/stream"]["post"]["tags"],
            serde_json::json!(["Chat"])
        );
        assert_eq!(
            spec["paths"]["/api/v1/skills"]["get"]["tags"],
            serde_json::json!(["Skills"])
        );
        assert!(spec["paths"]["/health"]["get"].get("tags").is_none());
        assert_eq!(
            spec["components"]["securitySchemes"]["HTTPBearer"],
            serde_json::json!({ "type": "http", "scheme": "bearer" })
        );
        assert_eq!(
            spec["paths"]["/api/v1/auth/me"]["get"]["security"],
            serde_json::json!([{ "HTTPBearer": [] }])
        );
        assert!(spec["paths"]["/api/v1/auth/login"]["post"]
            .get("security")
            .is_none());
        assert_eq!(
            spec["paths"]["/api/v1/documents/{document_id}"]["delete"]["responses"]["202"]
                ["description"],
            "成功"
        );
        assert!(spec["components"]["schemas"]["HTTPValidationError"].is_object());
        assert!(spec["components"]["schemas"]["TokenResponse"].is_object());
        assert!(spec["components"]["schemas"]["SkillExecuteRequest"].is_object());
        assert!(spec["components"]["schemas"]["DocumentResponse"].is_object());
        assert!(spec["components"]["schemas"]["ChatRequest"].is_object());
        assert_eq!(
            spec["paths"]["/api/v1/auth/login"]["post"]["requestBody"]["content"]["application/json"]
                ["schema"]["$ref"],
            "#/components/schemas/LoginRequest"
        );
        assert_eq!(
            spec["paths"]["/api/v1/chat/generate"]["post"]["responses"]["200"]["content"]
                ["application/json"]["schema"]["$ref"],
            "#/components/schemas/ChatResponse"
        );
        assert_eq!(
            spec["paths"]["/api/v1/documents/upload"]["post"]["requestBody"]["content"]
                ["multipart/form-data"]["schema"]["$ref"],
            "#/components/schemas/Body_upload_document_api_v1_documents_upload_post"
        );
        assert_eq!(
            spec["paths"]["/api/v1/documents/list"]["get"]["parameters"][1]["name"],
            "limit"
        );
        assert_eq!(
            spec["paths"]["/api/v1/chat/consolidate"]["post"]["parameters"][0]["name"],
            "conversation_id"
        );
        assert_eq!(
            spec["paths"]["/api/v1/skills"]["get"]["responses"]["200"]["content"]
                ["application/json"]["schema"]["items"]["$ref"],
            "#/components/schemas/SkillDefinitionResponse"
        );
        assert_eq!(
            spec["components"]["schemas"]["SkillExecuteRequest"]["properties"]["arguments"]
                ["additionalProperties"],
            true
        );
    }

    #[test]
    fn docs_page_loads_openapi_json() {
        assert!(docs_html().contains("<!DOCTYPE html>"));
        assert!(docs_html().contains("/openapi.json"));
        assert!(docs_html().contains("info.description"));
        assert!(redoc_html().contains("<redoc spec-url=\"/openapi.json\">"));
        assert!(redoc_html().contains("redoc.standalone.js"));
        assert!(spec(&crate::config::Config::jwt_test())["paths"]
            .get("/redoc")
            .is_some());
    }
}
