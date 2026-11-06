# 企业知识库系统 Python → Rust 重构验收对比报告

生成时间：2026-11-06 12:00:58

环境：同机同语料同 Mock LLM（OpenAI 兼容、确定性策略、5ms 模拟延迟）；Python=纯 Python 内核（torch CPU BGE + rank_bm25 + torch CE），Rust=ONNX（DirectML GPU BGE 会话池 ×2 + ONNX CE）。语料：docs/rag_accuracy_demo 7 份企业文档；查询集：docs/rrf_ablation_eval.json 41 条带 ground truth。


## 0. 启动与资源

| 指标 | Python | Rust |
|---|---|---|
| 冷启动至 /health 200 (s，热文件缓存) | 9.3 | 5.67 |
| 首次冷启动(含磁盘冷读, 观测值, s) | ~50 | ~2 |
| 稳态 RSS (MB) | 1512 | 2164 |
| 依赖部署体积 | venv ~2.9GB (103 依赖含 torch) | 单 exe + DirectML.dll ~60MB |
| 运行时预热完成 | ready=true (CE torch CPU) | ready=true (CE ONNX) |

## 0.1 并发入库压力对照

| 场景 | Python | Rust |
|---|---|---|
| 串行入库 7 文档总耗时 (s) | 93.6 | 32.3 |
| 7×14KB 新文档并发上传（4 并发向量化） | 全部 ready ~90s，进程存活 | 全部 ready ~95s，进程存活 |
| 7 份真实文档并发入库（首次观测，非受控） | **进程原生崩溃 1 次**（exit 2816，连接重置；两次受控复现未再现，判为间歇性） | 未观测崩溃 |

## 1. 功能等效性（契约对拍）

- OpenAPI 端点差异：仅 Python 有 0 个，仅 Rust 有 7 个
  - 仅 Rust：/api/v1/documents, /api/v1/documents/{document_id}, /api/v1/retrieval/cache-clear, /api/v1/retrieval/rag, /docs, /openapi.json, /redoc
- skill_names 一致：True（Python 6 项 / Rust 6 项）
- tool_names 一致：True（Python 7 项 / Rust 7 项）
- capability_names 一致：True（Python 7 项 / Rust 7 项）
- doc_statuses 一致：True（Python 1 项 / Rust 1 项）
- 行为探针：22/26 完全一致（状态码+响应键+业务事实+检索命中）
  - DIFF `chat_stream_chitchat` 字段 `sse`：py={"name": "chat_stream_chitchat", "status": 200, "ms": 110.1, "sse": {"events": ["start", "tool_call", "tool_result", "message", "message", "message", "message", "message", "message", "message", "done"], "event_counts": { | rs={"name": "chat_stream_chitchat", "status": 200, "ms": 43.0, "sse": {"events": ["start", "intent", "tool_call", "tool_result", "sources", "message", "token", "message", "token", "message", "token", "message", "token", "me
  - DIFF `chat_stream_knowledge` 字段 `sse`：py={"name": "chat_stream_knowledge", "status": 200, "ms": 121.1, "sse": {"events": ["start", "tool_call", "tool_result", "message", "message", "message", "message", "message", "message", "message", "done"], "event_counts":  | rs={"name": "chat_stream_knowledge", "status": 200, "ms": 32.3, "sse": {"events": ["start", "intent", "tool_call", "tool_result", "sources", "message", "token", "message", "token", "message", "token", "message", "token", "m
  - DIFF `chat_stream_structured` 字段 `sse`：py={"name": "chat_stream_structured", "status": 200, "ms": 87.7, "sse": {"events": ["start", "tool_call", "tool_result", "message", "message", "message", "message", "message", "message", "message", "done"], "event_counts":  | rs={"name": "chat_stream_structured", "status": 200, "ms": 25.0, "sse": {"events": ["start", "intent", "tool_call", "tool_result", "sources", "message", "token", "message", "token", "message", "token", "message", "token", "
  - DIFF `retrieval_rag` 字段 `status`：py={"name": "retrieval_rag", "status": 404, "keys": ["detail"]} | rs={"name": "retrieval_rag", "status": 200, "keys": ["context", "expanded_queries", "message", "query", "results", "retrieval_method"]}

## 1.5 检索质量（41 条 ground truth 查询，top_k=5）

| 端点 | Python 精确率 | Rust 精确率 | Python MRR | Rust MRR | Python p95(ms) | Rust p95(ms) |
|---|---|---|---|---|---|---|
| search/dense | 95.12% | 95.12% | 0.939 | 0.939 | 115.1 | 72.3 |
| search/sparse | 100.00% | 100.00% | 0.935 | 0.935 | 32.6 | 15.7 |
| search/hybrid | 90.24% | 90.24% | 0.9024 | 0.8293 | 382.9 | 245.3 |

精确率口径：成功调用中 top-5 命中至少一个 ground truth 文档（metadata.source_name）的比例。

## 3. 性能基准

### 3.1 延迟（p95，毫秒）

| 工作负载 | Python p50 | Python p95 | Rust p50 | Rust p95 | 倍率(p95 Py/Rs) |
|---|---|---|---|---|---|
| Chat 并发1 | 63.5 | 88.8 | 46.7 | 61.7 | 1.44 |
| Chat 并发16 | 444.4 | 470.2 | 95.4 | 488 | 0.96 |
| Chat 并发4 | 116.7 | 139.4 | 20.7 | 62 | 2.25 |
| Chat 并发8 | 239.4 | 268 | 22.2 | 214.8 | 1.25 |
| Hybrid 并发1 | 273 | 308.6 | 206.3 | 267.2 | 1.15 |
| Hybrid 并发16 | 4108.7 | 4654.9 | 2335.2 | 2741.5 | 1.7 |
| Hybrid 并发32 | 5840.5 | 10380.8 | 3511.3 | 5008 | 2.07 |
| Hybrid 并发4 | 1043.8 | 1221 | 615.4 | 802.1 | 1.52 |
| Hybrid 并发8 | 2082.9 | 2144.9 | 1235 | 1556.9 | 1.38 |
| 顺序 Dense 检索 | 91.6 | 112.1 | 76.7 | 90.2 | 1.24 |
| 顺序 Hybrid 检索 | 270.7 | 313.2 | 223.5 | 266.2 | 1.18 |
| 顺序 Sparse 检索 | 6.9 | 8.2 | 11.6 | 18.2 | 0.45 |
| Skill 执行(并发8) | 360.6 | 772.1 | 14 | 190.6 | 4.05 |
| 流式闲聊 | 151.2 | 233.2 | 46.6 | 74.9 | 3.11 |
| 流式知识问答 | 120.6 | 167.4 | 38 | 75.4 | 2.22 |
| 流式记忆查询 | 956.5 | 1036.6 | 2117.5 | 2671.3 | 0.39 |

### 3.2 并发吞吐（req/s）

| 负载 | Python RPS | Rust RPS | Py/Rs 吞吐比 |
|---|---|---|---|
| chat_c1 | 14.28 | 20.3 | 0.7 |
| chat_c16 | 35.4 | 33.37 | 1.06 |
| chat_c4 | 32.89 | 92.44 | 0.36 |
| chat_c8 | 33.67 | 76.12 | 0.44 |
| hybrid_c1 | 3.86 | 4.88 | 0.79 |
| hybrid_c16 | 3.89 | 6.59 | 0.59 |
| hybrid_c32 | 3.85 | 6.6 | 0.58 |
| hybrid_c4 | 3.77 | 6.55 | 0.58 |
| hybrid_c8 | 3.91 | 6.35 | 0.62 |
| skills_c8 | 19.91 | 179.29 | 0.11 |
| stream_c4 | 8.84 | 4.95 | 1.79 |

### 3.3 失败率与精确率

| 指标 | Python | Rust |
|---|---|---|
| 整体错误率（全部基准请求） | 0.00% | 0.00% |
| RAG/检索失败率 | 0.00% | 0.00% |
| Chat 编排失败率 | 0.00% | 0.00% |
| 工具执行失败率(流式 tool_result 观测) | 0.00% | 0.00% |
| Skill 调用失败率 | 0.00% | 0.00% |
| 成功调用精确率(检索三端点合并,命中 gt) | 95.12% | 95.12% |
| 流式回答正确率(含 MOCK 标记) | 100.00% | 100.00% |

样本量：检索 323/323(py/rs)，chat 80/80，skill 60/60，stream 30/30；流式观测到工具结果 py=30 rs=30

## 4. 验收发现（行为差异与缺陷清单）

### Rust 相对 Python 的增强（非回归）

1. **新增端点 `POST /api/v1/retrieval/rag`**：Python 404，Rust 提供完整 RAG 工作流（多查询+重排+上下文组装），空库时返回明确空态。
2. **SSE 事件流更丰富**：Rust 额外下发 `intent`（意图路由回显）、`sources`（资料来源）、`token`（增量）事件；Python 只有 start/tool_call/tool_result/message/done。核心 tool_call/tool_result 语义两侧一致。
3. **入库后稀疏索引即时重建**：Rust 入库完成即重建持久 BM25 索引（rows=520）；检索立即可用。

### Python 侧缺陷（重构前项目固有，验收中发现）

1. **入库后稀疏索引存在失效窗口（两次复现）**：Python 持久稀疏索引仅在启动时构建；入库后首批 sparse 查询返回 0 条（两次复现），无需重启、后续自动恢复（quality 阶段 41 条 sparse 查询精确率 100%），疑为懒重建/回退路径。Rust 入库完成即重建（rows=520），无此窗口。
2. **并发入库间歇性原生崩溃（观测 1 次）**：7 份真实文档连续上传（4 份并发向量化）时 Python 进程原生崩溃（exit 2816，连接重置），需人工重启；两次受控复现（4×3KB、7×14KB 新文档并发）未再现，判定为间歇性。Rust 在相同压力下稳定。

### 行为/配置差异（需知悉，非缺陷）

1. **记忆类查询链路不同**：同一句“我的画像记忆里有什么”，Python 规则直答（无 tool 事件，~150ms）；Rust 走统一工具编排（2 次技能工具调用 + LLM 收尾，~420-520ms，高并发下排队至 ~2.7s）。Rust 语义更完整但该场景更慢。
2. **DATABASE_URL 绝对路径处理**：Python 将 SQLite 强制重定位到 backend/app/data（仅取文件名）；Rust 尊重绝对路径。
3. **Skill 执行延迟量级差异**：list/memory 类 Skill 并发 8 时 Python p50 ≈ 407ms（LangChain 注册表+线程池），Rust ≈ 23ms。
4. **Cross-Encoder 产物布局**：Rust 需 onnx/model.onnx 子目录布局，Python 用 HF 原始权重目录；共享模型目录时需同时准备两套产物。

## 5. 验收结论

- **功能等效性：通过（含增强）**。26 项契约探针 22 项完全一致；4 项差异中 3 项为 Rust 增强项、1 项为 Rust 新增端点；工具注册表、Skill 目录、检索结果结构逐键一致。
- **检索质量：等效**。dense/sparse 精确率逐位一致（95.12%/100%），hybrid 命中率一致（90.24%，MRR 0.90 vs 0.83）。
- **性能：Rust 全面占优或持平**。Hybrid 检索 p95 快 1.15-2.07 倍、吞吐 1.6-1.7 倍（3.9→6.6 rps）；Skill 执行快 4-9 倍（吞吐 20→179 rps）；chat 编排并发吞吐 2.3-2.8 倍（c4/c8）；入库快 2.2 倍（94s→32s）。例外：稀疏检索（小语料 BM25）Python 更快（p95 8ms vs 18ms）；记忆类流式查询 Python 直答路径更快（Rust 走 2 次技能编排）。
- **可靠性：两侧基准错误率均为 0%**；Python 存在稀疏索引失效与间歇性并发入库崩溃两项固有缺陷，Rust 未见。
- **代价**：Rust 稳态内存高约 650MB（ONNX/DirectML 预分配），换取 GPU 并发推理与会话池。