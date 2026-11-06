# Enterprise Knowledge System — Rust 后端

高性能企业知识库 RAG 系统的**单体 Rust 后端**（单一二进制 `eks-server`）。
项目最初以 Python 实现，后全量重写为 Rust 并移除 Python 代码；行为对齐记录见
「全量Rust化」系列提交与源码内注释，`backend/` 目录现仅保留运行时资产。

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
用户查询 → 意图路由 intent.rs（规则快判 skill/direct/rag，冲突或模糊时 LLM 消歧）
用户查询 → 多查询启发式扩展 retrieval.rs（含编号保护/结构化别名）
  ├─ Dense: BGE ONNX 向量化 → milvus_lite COSINE 暴力扫描
  ├─ Sparse: CJK 2/3-gram 分词 → BM25Okapi 两段式（全语料选候选→切片重打分）
  └─ 每变体 RRF 融合 (0.5/0.5，产物 source=hybrid 携带 matched_sources/fusion_score)
     → 变体间融合 (原始 1.4 / 变体 0.75)，候选池 candidate_k = max(8·top_k, top_k+20, 60)
→ 规则重排 eks-kernels/rerank（关键词覆盖/焦点/特异度/门控 + adaptive CE 跳过 cascade.rs）
  ├─ 高置信/小候选池/编号查询 → 纯规则
  └─ 低置信 → Cross-Encoder 前K精排 → 0.65/0.35 权重混合
→ 结构化同 Sheet 聚合（xlsx 按行展开 + 表头还原）
→ 父块上下文批量回填（SQLite IN 查询，失败逐条回退）
→ 上下文组装（6.5K/30K 字符限制）
→ 证据不足时统一 ReAct react.rs（全量本地工具，最多 3 轮；失败回退 rag 专用循环）
→ 长期记忆消息按请求缓存（memory.rs MemoryContext：画像 + 按 route 消息复用）
→ 短期记忆（默认最近 5 轮窗口；summary 模式对更早轮次递进摘要并缓存）
→ SSE 流式推送 (start/intent/tool_call/tool_result/sources/message/token/done)
→ 回合写入 SQLite `memory_records`（事件记忆 + 触发词画像抽取）
```

## 快速启动

### 开发环境

```bash
cd backend-rs
cargo run --release
# 默认 127.0.0.1:8000，与前端 Vite 代理一致，可通过 EKS_PORT 修改
```

前置条件：
- Rust 1.92+（Dockerfile 使用 `rust:1.92`）
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
| GET | `/` | 服务标识 |
| POST | `/api/v1/auth/login` | 登录 |
| POST | `/api/v1/auth/register` | 注册 |
| POST | `/api/v1/auth/refresh` | 刷新令牌 |
| POST | `/api/v1/auth/password` | 修改密码 |
| GET | `/api/v1/auth/me` | 当前用户 |
| POST | `/api/v1/retrieval/search/hybrid` | 混合检索 |
| POST | `/api/v1/retrieval/search/dense` | Dense 检索 |
| POST | `/api/v1/retrieval/search/sparse` | Sparse 检索 |
| POST | `/api/v1/retrieval/rag` | 完整 RAG 工作流 |
| POST | `/api/v1/retrieval/cache-clear` | 清空检索缓存 |
| GET | `/api/v1/chat/settings` | 前端对话上限 |
| GET | `/api/v1/chat/runtime-status` | 模型预热状态 |
| POST | `/api/v1/chat/warmup` | 返回当前运行时状态 |
| POST | `/api/v1/chat/generate` | 同步生成回答 |
| POST | `/api/v1/chat/stream` | SSE 流式聊天（`message`/`done` 与前端一致，并保留 `token`） |
| POST | `/api/v1/chat/consolidate` | 会话摘要写入长期记忆 |
| GET | `/api/v1/chat/conversations` | 对话列表 |
| PUT | `/api/v1/chat/conversations/{id}` | 保存对话 |
| DELETE | `/api/v1/chat/conversations/{id}` | 删除对话 |
| DELETE | `/api/v1/chat/conversations` | 清空对话 |
| GET | `/api/v1/documents` | 文档列表 |
| GET | `/api/v1/documents/list` | 分页文档列表（前端契约） |
| POST | `/api/v1/documents/upload` | 上传文档（multipart，含 PDF 文本层） |
| GET | `/api/v1/documents/content/{id}` | 文档正文预览 |
| DELETE | `/api/v1/documents/delete/{id}` | 删除文档（前端契约） |
| DELETE | `/api/v1/documents/{id}` | 删除文档 |
| GET | `/api/v1/skills` | 技能列表 |
| GET | `/api/v1/skills/function-calling` | Function Calling 工具 schema |
| GET | `/api/v1/skills/capabilities` | 统一能力目录（RAG + Skill + MCP） |
| POST | `/api/v1/skills/execute` | 执行本地技能 |
| GET | `/health` / `/docs` / `/openapi.json` | 健康检查 / 自绘 API 文档 |

## 支持的文档格式

| 格式 | 解析方式 | 说明 |
|---|---|---|
| .txt / .md / .csv / .tsv / .html | 直接读取 | UTF-8，自动回退 BOM/GBK/GB18030 |
| .docx | zip + XML | 提取 `<w:t>` 文本 |
| .xlsx / .xls | calamine | 按 Sheet 提取，含 `[Sheet]` 标记 |
| .pptx | zip + XML | 提取 `<a:t>` 文本 |
| .pdf | pdf-extract | 提取文本层；扫描件会失败 |

## 性能基准（41 条查询，CPU）

| 配置 | mean | P95 | recall@5 | MRR@10 |
|---|---|---|---|---|
| 迁移前 Python 基线（历史数据） | 2.92s | 4.79s | 0.9024 | 0.8618 |
| **Rust 后端** | **0.104s** | **0.120s** | **0.9512** | **0.9024** |

> 指标对模型导出物与运行环境敏感：迁移期读数基于当时开发机的原始 ONNX 导出。
> 2026-11 验收复测（hf-mirror Xenova fp32 ONNX、多查询候选池修复后）：mean 0.11s /
> P95 0.14s / recall@5 0.878 / MRR@10 0.7276，与 Python 基线同量级；Cross-Encoder
> 开启且工件就绪时 adaptive 门控在本数据集上均跳过精排（rule_only_small_pool /
> rule_only_identifier），延迟与纯规则一致。复测命令见 `scripts/bench_retrieval.py`。

### 文档批量入库提速（GPU / int8 量化两种方案）

BGE embedding 是入库链路的绝对瓶颈（解析/切块/写入仅毫秒级）。实测三种模式
（7 份演示文档、i5-14400F + RTX 3050 6GB、41 查询基准）：

| 模式 | 配置 | 全量入库 | 检索 mean | recall@5 | MRR@10 |
|---|---|---|---|---|---|
| fp32 CPU（默认） | `BGE_DEVICE=cpu` | 24.5s | 0.11s | 0.878 | 0.7276 |
| int8 CPU | `EKS_BGE_MODEL_DIR`→量化模型 | 12.7s | 0.18s | 0.854 | 0.7683 |
| **fp32 GPU (DirectML)** | `BGE_DEVICE=directml` | **9.6s** | 0.29s | **0.902** | **0.793** |

说明：
- `BGE_BATCH_SIZE` 16→64 在 CPU 与 GPU 下均无额外增益（瓶颈已不在批大小）。
- **GPU 走 DirectML**（Windows DX12，免 CUDA 工具链）：`ort` 启用 `directml`
  feature，并将 [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML)
  包内 `bin/x64-win/DirectML.dll` 放到 `eks-server.exe` 同目录；设置
  `BGE_DEVICE=directml`（`CROSS_ENCODER_DEVICE` 同理）。初始化失败自动回退 CPU 并告警。
- **int8 量化**：Xenova 导出的 `model_quantized.onnx`（102MB）重命名为
  `onnx/model.onnx` 放入新模型目录，`EKS_BGE_MODEL_DIR` 指向即可。
  ⚠️ int8 模型与 DirectML 不兼容（量化算子回退 CPU 反而更慢），GPU 场景请配 fp32。
- ⚠️ 查询与库向量必须来自同一编码器——**切换模型/设备后需删除并重新上传全部文档**。

## 与 Python 基线的关系

后端已从 Python 全量重写为 Rust 并移除 Python 实现。迁移期以 Python 行为为基线逐项对齐
（约 150 个「全量Rust化」提交，含检索打分逐位一致、日志文案、SSE 协议与错误语义），
对齐记录见提交历史。`backend/` 目录保留为运行时资产家（.env 模板、技能文档、模型与数据目录）。

## 项目结构

```
backend-rs/
├── Cargo.toml                     # workspace（eks-kernels + eks-server）
├── rust-toolchain.toml            # Rust 1.92.0
├── crates/
│   ├── eks-kernels/               # 纯 Rust 检索内核（零依赖，无 FFI）
│   │   └── src/
│   │       ├── text.rs            # CJK 2/3-gram 分词 + 文本归一化
│   │       ├── bm25.rs            # BM25Okapi（与 rank_bm25 逐位一致）
│   │       ├── lexical.rs         # 词法覆盖打分
│   │       ├── rerank.rs          # 规则重排信号/打分/门控 + 关键词相关性过滤
│   │       └── fuse.rs            # RRF 融合（来源标记 + 融合分）
│   └── eks-server/                # HTTP 服务（单一二进制）
│       ├── src/
│       │   ├── main.rs            # 入口：启动横幅/模型加载/优雅关停
│       │   ├── state.rs           # 共享应用状态（AppState/预热快照）
│       │   ├── config.rs          # backend/.env 加载（含逐行容错）+ 路径解析
│       │   ├── routes.rs          # 路由注册 + OpenAPI 文档端点
│       │   ├── validate.rs        # 请求字段校验（422 语义对齐）
│       │   ├── auth.rs            # PBKDF2/HS256 JWT/会话/令牌续期中间件/CORS
│       │   ├── db.rs              # SQLite 访问层（schema/时间戳/用户表）
│       │   ├── logging.rs         # tracing 控制台+滚动文件/panic 堆栈 hook
│       │   ├── request_log.rs     # 请求开始/完成/异常日志中间件
│       │   ├── openapi.rs         # 手绘 OpenAPI 3.1 + /docs 页面
│       │   ├── llm.rs             # 多供应商 LLM 客户端（流式/工具调用/兜底链）
│       │   ├── intent.rs          # 意图路由（规则快判 + LLM 动作选择）
│       │   ├── skills.rs          # 本地 Skill：目录/参数归一化/SQLite 执行
│       │   ├── remote_mcp.rs      # 远程 MCP manifest → Skill 注册适配
│       │   ├── trace.rs           # 工具与技能调用轨迹日志
│       │   ├── embedder.rs        # BGE ONNX 推理（HF tokenizer 逐位对齐）
│       │   ├── cross_encoder.rs   # Cross-Encoder ONNX 精排推理
│       │   ├── milvus_lite.rs     # milvus_lite 直读引擎（Parquet+Manifest，无 gRPC）
│       │   ├── retrieval.rs       # 查询扩展/稀疏索引/上下文组装/引用去重
│       │   ├── retrieval_engine.rs# 检索编排：多查询融合/重排/父块回填/结构化聚合
│       │   ├── cascade.rs         # adaptive Cross-Encoder 门控
│       │   ├── planner.rs         # 常规检索规划器（首轮策略/证据评估）
│       │   ├── rag_cache.rs       # 检索结果内存缓存（TTL/LRU/失效）
│       │   ├── react.rs           # ReAct 升级：统一编排与 rag 专用循环
│       │   ├── chat_stream.rs     # 聊天端点：generate/SSE 流式/skill 执行
│       │   ├── conversations.rs   # 对话 CRUD
│       │   ├── memory.rs          # 长期记忆：事件/画像/层级召回/consolidate/MemoryContext
│       │   ├── memory_chunk.rs    # 长期记忆切块（对话/语义双粒度）
│       │   ├── short_memory.rs    # 短期记忆（窗口/递进摘要）
│       │   ├── documents.rs       # 文档路由：列表/正文/上传/删除
│       │   ├── ingest.rs          # 文档入库：解析→切块→向量化→milvus/SQLite 写入
│       │   └── tests.rs           # 跨实现互通单测（PBKDF2/JWT 等）
│       └── tests/
│           ├── embed_golden.rs    # BGE 金向量对拍
│           ├── cross_encoder_golden.rs # CE 金分对拍
│           ├── tokenizer_ids.rs   # tokenizer id 对拍
│           └── fixtures/          # 金向量/tokenizer 对拍数据
└── scripts/
    └── bench_retrieval.py         # 纯 HTTP 检索基准（登录→清缓存→/retrieval/rag）
```

## 测试

```bash
# Rust 单元 + 集成测试
cd backend-rs && cargo test

# 性能基准（HTTP → eks-server，需先启动服务并配置好模型/数据）
cd backend-rs/scripts && python bench_retrieval.py --base-url http://127.0.0.1:8000
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

除下表 `EKS_*` 专属变量外，其余配置复用 `backend/.env` 的同名键（完整清单见
[backend/.env.example](../backend/.env.example)）；`EKS_*` 与进程环境变量优先于 .env。

| 变量 | 默认值 | 说明 |
|---|---|---|
| EKS_HOST | 127.0.0.1 | 监听地址 |
| EKS_PORT | 8000 | 监听端口 |
| EKS_ENV_FILE | ../backend/.env | .env 路径 |
| EKS_SQLITE_PATH | ../backend/app/data/rag_metadata.db | SQLite 路径 |
| EKS_MILVUS_LITE_PATH | ../backend/app/data/milvus_lite.db | milvus_lite 路径 |
| EKS_BGE_MODEL_DIR | ../backend/app/models/bge-base-zh-v1.5 | BGE 模型目录（需含 onnx/model.onnx + tokenizer.json） |
| EKS_CROSS_ENCODER_ONNX | 按 CROSS_ENCODER_MODEL_NAME/CACHE_DIR 推导 | CE ONNX 路径（缺省时按 HF snapshot 布局查找） |
| EKS_CROSS_ENCODER_TOKENIZER | 同上 | CE tokenizer.json 路径 |
| EMBED_SESSION_COUNT | 2 | BGE 并发会话数（多用户并发检索/入库并行度；GPU 建议 4-6，每会话一份权重注意显存） |
| CROSS_ENCODER_SESSION_COUNT | 1 | CE 并发会话数（模型较大，多用户精排并发时调 2-4） |
| CROSS_ENCODER_ENABLED | true | 是否启用 CE（工件缺失时启动降级为纯规则，根因经 runtime-status 透出） |
| CROSS_ENCODER_TRIGGER_MODE | adaptive | 精排触发模式 |

## 多用户并发

单进程内的并发能力由以下机制保证（8 并发用户压测，RTX 3050 + fp32/DirectML）：

- **ONNX 会话池**：ort 的 Session 不支持并发推理，`EMBED_SESSION_COUNT` 个独立会话轮询借用，
  消除全局互斥。冷查询（缓存未命中）p50 767→108ms、p95 1007→162ms、吞吐 9.5→62.7 req/s。
- **热路径非阻塞**：检索引擎的查询向量化移入 `spawn_blocking`，不再占住 tokio worker。
- **SQLite 池**：连接池上限 16（WAL 多读单写），多用户读写不串队。
- **稀疏索引 single-flight**：并发首查只构建一次 BM25 索引，其余请求等待复用。
- **milvus_lite manifest 写互斥 + 原子替换**：并发上传/删除不损坏索引文件（见性能与可靠性提交）。
- 热查询（缓存命中）p50 ~30ms 与会话数无关；SSE 流式首事件 ~50-120ms；
  并发上传 4 文档全部入库 ~3s；流式完整时长由 LLM 供应商网络主导。
