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

fn op(summary: &str, operation_id: &str) -> Value {
    json!({
        "summary": summary,
        "operationId": operation_id,
        "responses": { "200": { "description": "成功" } }
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
        "paths": {
            "/": { "get": op("根路由", "root") },
            "/health": { "get": op("健康检查", "health") },
            "/docs": { "get": op("Swagger 风格文档页", "docs") },
            "/redoc": { "get": op("ReDoc 风格文档页", "redoc") },
            "/openapi.json": { "get": op("OpenAPI Schema", "openapi_json") },
            (format!("{prefix}/auth/login")): { "post": op("登录", "login") },
            (format!("{prefix}/auth/register")): { "post": op("注册", "register") },
            (format!("{prefix}/auth/refresh")): { "post": op("刷新令牌", "refresh") },
            (format!("{prefix}/auth/password")): { "post": op("修改密码", "change_password") },
            (format!("{prefix}/auth/me")): { "get": op("当前用户", "me") },
            (format!("{prefix}/retrieval/search/hybrid")): { "post": op("混合检索", "retrieval_hybrid") },
            (format!("{prefix}/retrieval/search/dense")): { "post": op("稠密检索", "retrieval_dense") },
            (format!("{prefix}/retrieval/search/sparse")): { "post": op("稀疏检索", "retrieval_sparse") },
            (format!("{prefix}/retrieval/rag")): { "post": op("RAG 检索生成", "retrieval_rag") },
            (format!("{prefix}/retrieval/cache-clear")): { "post": op("清空检索缓存", "retrieval_cache_clear") },
            (format!("{prefix}/chat/settings")): { "get": op("对话设置", "chat_settings") },
            (format!("{prefix}/chat/runtime-status")): { "get": op("运行时状态", "chat_runtime_status") },
            (format!("{prefix}/chat/warmup")): { "post": op("预热", "chat_warmup") },
            (format!("{prefix}/chat/generate")): { "post": op("生成回复", "generate_chat") },
            (format!("{prefix}/chat/consolidate")): { "post": op("巩固记忆", "consolidate_chat") },
            (format!("{prefix}/chat/stream")): { "post": op("流式对话", "stream_chat") },
            (format!("{prefix}/chat/conversations")): {
                "get": op("会话列表", "list_conversations"),
                "delete": op("清空会话", "clear_conversations")
            },
            (format!("{prefix}/chat/conversations/{{conversation_id}}")): {
                "put": op("更新会话", "upsert_conversation"),
                "delete": op("删除会话", "delete_conversation")
            },
            (format!("{prefix}/documents")): { "get": op("文档列表", "list_documents") },
            (format!("{prefix}/documents/list")): { "get": op("文档列表（兼容）", "list_documents_compat") },
            (format!("{prefix}/documents/upload")): { "post": op("上传文档", "upload_document") },
            (format!("{prefix}/documents/content/{{document_id}}")): { "get": op("文档内容", "get_document_content") },
            (format!("{prefix}/documents/delete/{{document_id}}")): { "delete": op("删除文档", "delete_document_compat") },
            (format!("{prefix}/documents/{{document_id}}")): { "delete": op("删除文档", "delete_document") },
            (format!("{prefix}/skills")): { "get": op("技能列表", "list_skills") },
            (format!("{prefix}/skills/function-calling")): { "get": op("Function Calling 技能", "list_skill_function_calling") },
            (format!("{prefix}/skills/capabilities")): { "get": op("技能能力", "skill_capabilities") },
            (format!("{prefix}/skills/execute")): { "post": op("执行技能", "execute_skill") },
        }
    })
}

pub fn docs_html() -> &'static str {
    DOCS_HTML
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
    }

    #[test]
    fn docs_page_loads_openapi_json() {
        assert!(docs_html().contains("<!DOCTYPE html>"));
        assert!(docs_html().contains("/openapi.json"));
        assert!(docs_html().contains("info.description"));
    }
}
