# RAG 验收评测集

用 **RAG 常见任务类型**（不是搬运受版权的 MS MARCO / NQ 原文）衡量 Python 与 Rust 重构前后的 **检索准确度** 和 **延迟**。两端打同一套 HTTP 契约：`POST /api/v1/retrieval/search/{dense,sparse,hybrid}`。

任务类型对齐公开基准，语料全部自建、可随仓库分发：

| 切片 | 任务 | 对标 | 规模 |
| --- | --- | --- | --- |
| `enterprise` | 制度 / FAQ / 表格 / 工号 | 领域 RAG（FiQA 类） | 7 文档，41 条 GT（`docs/rag_accuracy_demo` + `docs/rrf_ablation_eval.json`） |
| `standard` | 单跳、领域问答、多跳、多文档综合、长上下文、噪声、知识冲突、跨语言、时效性、模糊查询、拒答、表格图表、声明核验 | BEIR / NQ / HotpotQA / SciFact / RGB | 25 篇 Markdown，100 条 GT |

`standard` 100 题是场景评测口径，原 33 题全部保留。`enterprise` 41 题继续作为历史回归，不并进这 100 题，避免改掉旧的 41 条对拍口径。两套一起跑是 141 题。

每题都有：

- `query`：问题
- `answers`：标准答案（可答题必须能在 gold 原文中逐字找到；拒答题固定为「资料不足，应拒答」）
- `evidence`：支撑答案的参考上下文
- `relevant_sources`：应命中的文档；拒答题为空，并用 `bait_sources` 标出容易误召的文档
- `difficulty`：简单 / 中等 / 困难
- `scenario`：场景分类

`standard` 场景分布：单跳事实 10、领域问答 10、拒答边界 10、表格图表 9、模糊查询 8、多跳推理 8、时效性 8、长上下文 7、知识冲突 7、跨语言 7、噪声干扰 7、多文档综合 6、声明核验 3。新增事实使用砚台打印、鹤鸣 SDK、青岚涂料、折纸项目、泊岸仓班车，不改写原来 33 题的数字和工号。

`standard` 难度是简单 33、中等 39、困难 28。全量 141 题是简单 45、中等 63、困难 33。极端题（冲突、跨语言、长上下文里的易混数字、多跳）标成困难；单跳查表和现行口径标成简单。

指标（文档级，按 `metadata.source_name` 对齐 `relevant_sources`）：

- **Hit@k**：top-k 是否命中至少一个相关文档（Success@k）
- **Recall@k**：相关文档被召回的比例（多跳题 gold 常为 2 篇）
- **MRR@k / nDCG@k**：排序质量
- **Reject@k**：不可回答题的 `bait_sources` 未进入 top-k
- **p50 / p95 / QPS**：逐查询清缓存后的端到端延迟

三轮结果的口径和对照写在 `docs/rag_acceptance_history.md`。分轮原文：

- 2026-11-06，enterprise 41 题：`docs/acceptance_py_vs_rust_report.md`（dense/sparse Hit@5 两侧 95.12% / 100%，hybrid 90.24%）
- 2026-11-12，74 题 Python / Rust 对拍：`docs/rag_acceptance_eval_report.md`，明细 `eval/rag_acceptance/runs/py_vs_rust.json`
- 2026-11-13，141 题 Rust 全量：明细 `eval/rag_acceptance/runs/full_141.json`

Python FastAPI 后台入库不会让 cached BM25 失效。语料 `ready` 后若 sparse 空结果，重启 Python 再跑（本轮重启后索引 559 行）。

## 离线自检（不启服务）

```bash
python eval/rag_acceptance/run.py check
```

检查：query id 唯一、`standard` 恰好 100 题、难度和场景合法、gold 文件存在、可答题与 gold 有词法重叠、标准答案和 `evidence` 能在 gold 原文中逐字对上（含 docx/xlsx）、拒答题带诱饵文档、nDCG 公式钉扎。

## 对拍 Python / Rust

两端都要能登录（`BOOTSTRAP_ADMIN_PASSWORD`）。先各自入库同一语料，或让跑手 `--ingest`：

```bash
# 仅 Rust
python eval/rag_acceptance/run.py compare \
  --system rust=http://127.0.0.1:8000 \
  --ingest --out eval/rag_acceptance/runs/rust.json

# Python（历史 FastAPI，commit 3900d36）与 Rust 同机对拍
python eval/rag_acceptance/run.py compare \
  --system python=http://127.0.0.1:8001 \
  --system rust=http://127.0.0.1:8000 \
  --slices enterprise,standard \
  --methods dense,sparse,hybrid \
  --ingest --out eval/rag_acceptance/runs/py_vs_rust.json
```

- Python 无 `/retrieval/rag` 时自动跳过该方法。
- `--limit N` 冒烟；`--concurrency 8` 测吞吐（默认 1，保证延迟口径与验收报告一致：逐条清缓存）。
- `--slices enterprise` 只跑原 41 条，便于对照旧报告。

## 目录

```
eval/rag_acceptance/
  suite.json                          # 切片、指标、端点
  datasets/standard/corpus/*.md       # 自建标准切片语料（含长文、冲突、时效、双语、图表）
  datasets/standard/queries.jsonl     # 100 题：query、答案、证据、难度、场景
  run.py                              # check / compare
  README.md
docs/rag_accuracy_demo/               # enterprise 语料
docs/rrf_ablation_eval.json           # enterprise 41 条 GT（含答案、证据、难度、场景）
```

## 与旧 bench 的关系

`backend/scripts/bench_retrieval.py` 仍可单独打 Rust `/retrieval/rag`。本评测集覆盖 dense/sparse/hybrid 三端点与双系统对拍，是验收口径。
