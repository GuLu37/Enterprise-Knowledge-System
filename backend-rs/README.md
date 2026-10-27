# Enterprise Knowledge System — Rust 后端

高性能企业知识库 RAG 系统，后端从 Python 全量重写为单一 Rust 二进制（`eks-server`）。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    eks-server (Rust)                     │
├─────────────┬──────────────┬─────────────┬──────────────┤
│  axum HTTP   │  JWT 认证    │  检索引擎    │  LLM 客户端   │
│  + SSE 流式  │  PBKDF2/HS256│  (见下文)   │  多供应商兜底  │
├─────────────┴──────────────┴─────────────┴──────────────┤
│                    eks-kernels (纯 Rust)                  │
│  BM25 Okapi │ CJK n-gram │ 规则重排 │ RRF 融合 │ WordPiece │
├─────────────────────────────────────────────────────────┤
│              ONNX Runtime (ort 2.0-rc.15)                │
│     BGE embedding (fp32)     │  Cross-Encoder (fp32)    │
├─────────────────────────────────────────────────────────┤
│           milvus_lite 直读引擎 (Parquet + Manifest)        │
│              SQLite (WAL) via sqlx                       │
└─────────────────────────────────────────────────────────┘
```

## 检索链路

```
用户查询 → 多查询启发式扩展（含编号保护/结构化别名）
  ├─ Dense: BGE ONNX 向量化 → milvus_lite COSINE 暴力扫描
  ├─ Sparse: CJK 2/3-gram 分词 → BM25Okapi 两段式（全语料选候选→切片重打分）
  └─ 每变体 RRF 融合 (0.5/0.5) → 变体间融合 (原始 1.4 / 变体 0.75)
→ 规则重排（关键词覆盖/焦点/特异度/门控 + adaptive CE 跳过）
  ├─ 高置信/小候选池/编号查询 → 纯规则
  └─ 低置信 → Cross-Encoder 前K精排 → 0.65/0.35 权重混合
→ 结构化同 Sheet 聚合（xlsx 按行展开 + 表头还原）
→ 父块上下文批量回填（SQLite IN 查询）
→ 上下文组装（6.5K/30K 字符限制）
→ SSE 流式推送 (start/sources/token/done)
```

## 快速启动

### 开发环境

```bash
cd backend-rs
cargo run --release
# 默认 127.0.0.1:8001，可通过 EKS_PORT 修改
```

前置条件：
- Rust 1.75+ (rustup + MSVC Build Tools)
- `backend/.env` 存在（共享配置）
- `backend/app/models/bge-base-zh-v1.5/onnx/model.onnx` 存在（见导出流程）
- `backend/app/data/` 存在（SQLite + milvus_lite.db）

### Docker

```bash
docker compose up eks-server
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/auth/login` | 登录（与 Python 互通） |
| POST | `/api/v1/auth/register` | 注册 |
| POST | `/api/v1/auth/refresh` | 刷新令牌 |
| POST | `/api/v1/auth/password` | 修改密码 |
| GET | `/api/v1/auth/me` | 当前用户 |
| POST | `/api/v1/retrieval/search/hybrid` | 混合检索 |
| POST | `/api/v1/retrieval/rag` | 完整 RAG 工作流 |
| POST | `/api/v1/retrieval/cache-clear` | 清空检索缓存 |
| POST | `/api/v1/chat/stream` | SSE 流式聊天（含简化 ReAct） |
| GET | `/api/v1/chat/conversations` | 对话列表 |
| PUT | `/api/v1/chat/conversations/{id}` | 保存对话 |
| DELETE | `/api/v1/chat/conversations/{id}` | 删除对话 |
| DELETE | `/api/v1/chat/conversations` | 清空对话 |
| GET | `/api/v1/documents` | 文档列表 |
| POST | `/api/v1/documents/upload` | 上传文档（multipart） |
| DELETE | `/api/v1/documents/{id}` | 删除文档 |
| GET | `/api/v1/skills` | 技能列表 |

## 支持的文档格式

| 格式 | 解析方式 | 说明 |
|---|---|---|
| .txt / .md | 直接读取 | UTF-8 |
| .docx | zip + XML | 提取 `<w:t>` 文本 |
| .xlsx / .xls | calamine | 按 Sheet 提取，含 `[Sheet]` 标记 |
| .pptx | zip + XML | 提取 `<a:t>` 文本 |
| .pdf | 暂不支持 | 请使用 Python 后端上传 |

## 性能基准（41 条查询，CPU）

| 配置 | mean | P95 | recall@5 | MRR@10 |
|---|---|---|---|---|
| Python 原始基线 | 2.92s | 4.79s | 0.9024 | 0.8618 |
| **Rust 后端** | **0.104s** | **0.120s** | **0.9512** | **0.9024** |

## 与 Python 后端的关系

- **共享同一数据**：SQLite（rag_metadata.db）+ milvus_lite.db 可双向读写
- **令牌互通**：Python 签发的 JWT 在 Rust 可验，反之亦然
- **无需迁移**：两个后端可并行运行（不同端口），随时切换

## 项目结构

```
backend-rs/
├── Cargo.toml                 # workspace
├── crates/
│   ├── eks-kernels/           # 纯 Rust 检索内核（无 FFI）
│   │   └── src/
│   │       ├── text.rs        # CJK 分词 + 归一化
│   │       ├── bm25.rs        # BM25Okapi（rank_bm25 逐位一致）
│   │       ├── lexical.rs     # 词法覆盖打分
│   │       ├── rerank.rs      # 规则重排 + 信号 + 门控
│   │       └── fuse.rs        # RRF 融合
│   └── eks-server/            # HTTP 服务
│       └── src/
│           ├── main.rs        # 入口
│           ├── config.rs      # .env 配置
│           ├── auth.rs        # JWT/PBKDF2/会话
│           ├── db.rs          # SQLite 访问
│           ├── milvus_lite.rs # milvus_lite 直读引擎
│           ├── embedder.rs    # BGE ONNX 推理
│           ├── cross_encoder.rs # CE ONNX 推理
│           ├── retrieval.rs   # 查询扩展 + 稀疏索引
│           ├── retrieval_engine.rs # 检索编排
│           ├── cascade.rs     # adaptive CE 门控
│           ├── llm.rs         # 多供应商 LLM 客户端
│           ├── chat_stream.rs # SSE 聊天（含简化 ReAct）
│           ├── conversations.rs # 对话管理
│           ├── documents.rs   # 文档管理 + 技能列表
│           ├── ingest.rs      # 文档入库（解析/切块/向量化/写入）
│           └── routes.rs      # 路由注册
├── scripts/
│   └── contract_auth.py       # 双端契约测试
└── tests/
    └── fixtures/              # 金向量/tokenizer 对拍数据
```

## 测试

```bash
# Rust 单元 + 集成测试
cd backend-rs && cargo test

# Python parity 测试（需编译 pyo3 扩展）
cd backend && python -m pytest tests/test_rust_kernels_parity.py

# 性能基准（进程内）
cd backend && PYTHONPATH=. python scripts/bench_retrieval.py

# 性能基准（HTTP → Rust 后端）
cd backend && python scripts/bench_retrieval.py --base-url http://127.0.0.1:8001
```

## 部署

### 单二进制

```bash
cargo build --release
# 产物: target/release/eks-server（约 30MB，不含模型）
# 运行时需: .env + models/ + data/
./target/release/eks-server
```

### Docker Compose

```bash
docker compose up eks-server
# 首次构建约 10 分钟（Rust 编译）
# 模型文件通过 volume 挂载，不入镜像
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| EKS_HOST | 127.0.0.1 | 监听地址 |
| EKS_PORT | 8001 | 监听端口 |
| EKS_ENV_FILE | ../backend/.env | .env 路径 |
| EKS_SQLITE_PATH | ../backend/app/data/rag_metadata.db | SQLite 路径 |
| EKS_MILVUS_LITE_PATH | ../backend/app/data/milvus_lite.db | milvus_lite 路径 |
| EKS_BGE_MODEL_DIR | ../backend/app/models/bge-base-zh-v1.5 | BGE 模型目录 |
| EKS_CROSS_ENCODER_ONNX | （见 .env） | CE ONNX 路径 |
| CROSS_ENCODER_ENABLED | true | 是否启用 CE |
| CROSS_ENCODER_TRIGGER_MODE | adaptive | 精排触发模式 |
