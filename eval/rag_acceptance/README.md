# RAG 验收评测集

用 **RAG 常见任务类型**（不是搬运受版权的 MS MARCO / NQ 原文）衡量 Python 与 Rust 重构前后的 **检索准确度** 和 **延迟**。两端打同一套 HTTP 契约：`POST /api/v1/retrieval/search/{dense,sparse,hybrid}`。

任务类型对齐公开基准，语料全部自建、可随仓库分发：

| 切片 | 任务 | 对标 | 规模 |
| --- | --- | --- | --- |
| `enterprise` | 制度 / FAQ / 表格 / 工号 | 领域 RAG（FiQA 类） | `docs/rag_accuracy_demo` 的 01–07，41 条 GT |
| `standard` | 单跳、领域问答、多跳、多文档综合、长上下文、噪声、知识冲突、跨语言、时效性、模糊查询、拒答、表格图表、声明核验 | BEIR / NQ / HotpotQA / SciFact / RGB | 原 25 篇 Markdown + 19 篇澄川合订卷，1000 条 GT |

`standard` 1000 题是场景评测口径。原 100 题（含更早的 33 题）全部保留，仍指向 `datasets/standard/corpus` 的 25 篇。新增 900 题的语料合并成 19 篇澄川实验室合订卷，文件名按 `08` 到 `26` 排序，放在 `docs/rag_accuracy_demo`，可直接整份入库。合订后文档级命中以整份卷宗为单位。`enterprise` 41 题继续作为历史回归，只入库同目录的 `01`–`07`，不并进这 1000 题。两套一起跑是 1041 题。2026-11-13 14:36 的 141 题跑分只覆盖当时的 100 题。合订卷入库后的 1041 题结果在 `eval/rag_acceptance/runs/full_1041.json`，口径写在 `docs/rag_acceptance_history.md`。

每题都有：

- `query`：问题
- `answers`：标准答案（可答题必须能在 gold 原文中逐字找到；拒答题固定为「资料不足，应拒答」）
- `evidence`：支撑答案的参考上下文
- `relevant_sources`：应命中的文档；拒答题为空，并用 `bait_sources` 标出容易误召的文档
- `difficulty`：简单 / 中等 / 困难
- `scenario`：场景分类

`standard` 场景尽量拉平：单跳事实、领域问答、表格图表、模糊查询、多跳推理、拒答边界、噪声干扰、长上下文、知识冲突、时效性、跨语言、多文档综合各 77 题，声明核验 76 题。难度是简单 300、中等 400、困难 300。

新增题用课题代号 `CX-E001` 起编，和原来的砚台、鹤鸣、青岚、折纸、泊岸数字分开。每道可答题的答案和证据句都能在 gold 原文里逐字找到；综合题的差值直接写在现行卷里，并与两年电价一致。多跳里有 15 道要串三篇文档。长文把事故针点埋在巡检垫文里，困难题另放带 `X` 后缀的邻近针点。冲突、时效题同时给出现行卷和已废止卷。跨语言困难题用英文提问，数字只写在中文句。拒答题的档案写明预算、证件或手机号没有记载。

重新生成新增卷册和 `ex_` 题目：

```bash
python eval/rag_acceptance/expand_to_1000.py
```

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

检查：query id 唯一、`standard` 恰好 1000 题、13 类场景各 76 或 77 题、难度为简单 300 / 中等 400 / 困难 300、gold 文件存在、可答题与 gold 有词法重叠、标准答案和 `evidence` 能在 gold 原文中逐字对上（含 docx/xlsx）、拒答题带诱饵文档、nDCG 公式钉扎。

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
  datasets/standard/corpus/*.md       # 原 100 题使用的 25 篇语料
  datasets/standard/queries.jsonl     # 1000 题：query、答案、证据、难度、场景
  expand_to_1000.py                   # 按同一规则重生澄川合订卷与 ex_ 题
  run.py                              # check / compare
  README.md
docs/rag_accuracy_demo/               # 01–07 为企业 41 题语料；08–26 为澄川合订卷
docs/rrf_ablation_eval.json           # enterprise 41 条 GT（含答案、证据、难度、场景）
```

## 与旧 bench 的关系

`backend/scripts/bench_retrieval.py` 仍可单独打 Rust `/retrieval/rag`。本评测集覆盖 dense/sparse/hybrid 三端点与双系统对拍，是验收口径。
