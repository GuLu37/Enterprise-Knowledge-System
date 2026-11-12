# RAG 验收评测集

用 **RAG 常见任务类型**（不是搬运受版权的 MS MARCO / NQ 原文）衡量 Python 与 Rust 重构前后的 **检索准确度** 和 **延迟**。两端打同一套 HTTP 契约：`POST /api/v1/retrieval/search/{dense,sparse,hybrid}`。

任务类型对齐公开基准，语料全部自建、可随仓库分发：

| 切片 | 任务 | 对标 | 规模 |
| --- | --- | --- | --- |
| `enterprise` | 制度 / FAQ / 表格 / 工号 | 领域 RAG（FiQA 类） | 7 文档，41 条 GT（`docs/rag_accuracy_demo` + `docs/rrf_ablation_eval.json`） |
| `standard` | 单跳事实、多跳、领域 FAQ、声明核验、结构化查找、改写、拒识、噪声 | BEIR / NQ / HotpotQA / SciFact / RGB | 9 篇 Markdown，33 条 GT |

指标（文档级，按 `metadata.source_name` 对齐 `relevant_sources`）：

- **Hit@k**：top-k 是否命中至少一个相关文档（Success@k）
- **Recall@k**：相关文档被召回的比例（多跳题 gold 常为 2 篇）
- **MRR@k / nDCG@k**：排序质量
- **Reject@k**：不可回答题的 `bait_sources` 未进入 top-k
- **p50 / p95 / QPS**：逐查询清缓存后的端到端延迟

历史 41 条 enterprise 基线见 `docs/acceptance_py_vs_rust_report.md`（2026-11-06，dense/sparse Hit@5 两侧 95.12% / 100%，hybrid 90.24%）。

## 离线自检（不启服务）

```bash
python eval/rag_acceptance/run.py check
```

检查：query id 唯一、gold 文件存在、可答题与 Markdown gold 有词法重叠、nDCG 公式钉扎。

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
  datasets/standard/corpus/*.md       # 自建标准切片语料
  datasets/standard/queries.jsonl     # BEIR 风格 query + qrels
  run.py                              # check / compare
  README.md
docs/rag_accuracy_demo/               # enterprise 语料
docs/rrf_ablation_eval.json           # enterprise 41 条 GT
```

## 与旧 bench 的关系

`backend/scripts/bench_retrieval.py` 仍可单独打 Rust `/retrieval/rag`。本评测集覆盖 dense/sparse/hybrid 三端点与双系统对拍，是验收口径。
