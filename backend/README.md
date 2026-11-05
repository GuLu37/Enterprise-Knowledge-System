# backend/ —— 运行时资产目录

Python 后端已在本仓库完成向 Rust（`backend-rs`）的全量迁移并移除。本目录不再包含任何后端代码，
仅作为 eks-server 的运行时资产家，路径被 `backend-rs/crates/eks-server/src/config.rs` 的默认值
与 `docker-compose.yml` 的 volume 挂载引用，请保持目录结构不变。

## 目录内容

| 路径 | 说明 | 是否入库 |
| --- | --- | --- |
| `.env.example` | 环境变量模板，复制为 `.env` 后供 eks-server 读取（默认 `../backend/.env`） | 是 |
| `.env` | 实际环境配置（含密钥） | 否 |
| `app/skills/<name>/<name>.md` | 内置技能的详细规则文档，`/api/v1/skills` 会输出其路径，运行时由 Rust 读取 | 是 |
| `app/models/` | BGE embedding 与 bge-reranker-base 的 ONNX 模型（按 `BGE_MODEL_DIR` 等配置定位） | 否 |
| `app/data/` | SQLite（rag_metadata.db）、milvus_lite.db 与 uploads 上传文件 | 否 |
| `app/logs/` | 运行日志（滚动文件） | 否 |

## 历史说明

迁移期（约 150 个「全量Rust化」提交）本目录曾是 Python FastAPI 对照实现；
行为对齐记录见 git 提交历史与 `backend-rs/README.md`。
