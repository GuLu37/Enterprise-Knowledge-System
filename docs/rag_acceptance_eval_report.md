# RAG 验收评测：Python vs Rust（2026-11-12）

本轮是评测历史的第二轮（74 题）。三轮对照见 `docs/rag_acceptance_history.md`。

生成时间：2026-11-12 11:33（Asia/Shanghai）

口径：`eval/rag_acceptance` 74 条 GT（enterprise 41 + standard 33），16 篇语料，top_k=5。两端打同一套 `POST /api/v1/retrieval/search/{dense,sparse,hybrid}`。Cross-Encoder 关闭。逐查询尝试清缓存。

明细 JSON：`eval/rag_acceptance/runs/py_vs_rust.json`。

## 环境

| 项 | Python | Rust |
|---|---|---|
| 实现 | FastAPI SoT `3900d36` | `eks-server` release（本分支） |
| Embedding | torch CPU BGE `bge-base-zh-v1.5` | ONNX CPU 同模型，会话池 ×2 |
| 稀疏 | rank_bm25 0.2.2，cached 索引 559 行 | 纯 Rust BM25，入库后重建 |
| 向量库 | milvus-lite | milvus_lite parquet |
| 设备 | 同机 CPU，无 GPU | 同机 CPU，无 GPU |
| 精排 | `CROSS_ENCODER_ENABLED=false` | 同左 |

Python 后台入库路径不会让 cached BM25 失效（启动时 rows=0）。本轮在语料 ready 后**重启 Python 进程**重建稀疏索引（559 行），再评测；否则 sparse 会空返回、hybrid 退化成 dense。

## 准确度（74 查询）

Hit@5 / MRR@10 / nDCG@10 在 dense、sparse 上 **Python 与 Rust 完全一致**；hybrid 的 Hit@5 一致，Rust 排序略差（MRR/nDCG 低约 0.05）。

| 方法 | 指标 | Python | Rust |
|---|---|---|---|
| dense | Hit@5 | 90.14% | 90.14% |
| dense | MRR@10 | 0.8873 | 0.8873 |
| dense | nDCG@10 | 0.8908 | 0.8908 |
| sparse | Hit@5 | 98.59% | 98.59% |
| sparse | MRR@10 | 0.9359 | 0.9359 |
| sparse | nDCG@10 | 0.9494 | 0.9494 |
| hybrid | Hit@5 | 88.73% | 88.73% |
| hybrid | MRR@10 | 0.8803 | 0.8310 |
| hybrid | nDCG@10 | 0.8698 | 0.8260 |

按切片（命中率，含拒识题按 bait 未入 top-k 计成功）：

| 方法 | enterprise 41 | standard 33 |
|---|---|---|
| dense | 95.12% / 95.12% | 84.85% / 84.85% |
| sparse | 100% / 100% | 93.94% / 93.94% |
| hybrid | 90.24% / 90.24% | 78.79% / 78.79% |

enterprise 三路 Hit 与 2026-11-06 旧报告（41 条）相同：dense 95.12%、sparse 100%、hybrid 90.24%。

拒识 Reject@5：dense 100%；sparse 33%（3 题中 2 题 bait 进了 top-5）；hybrid 0%。两侧相同。

## 延迟（清缓存失败则 Python 无 cache-clear 端点）

| 方法 | Python p50 | Python p95 | Rust p50 | Rust p95 | QPS Py / Rs |
|---|---|---|---|---|---|
| dense | 703.4 ms | 1054.3 ms | 35.6 ms | 40.9 ms | 1.32 / 27.57 |
| sparse | 11.3 ms | 18.6 ms | 24.7 ms | 33.8 ms | 83.15 / 39.26 |
| hybrid | 989.5 ms | 1285.9 ms | 111.5 ms | 138.8 ms | 1.01 / 8.94 |

- Dense / Hybrid：Rust 明显更快（本机 CPU 上大约 20× / 9× p95）。Python 走 torch 暴力扫 559 个 child。
- Sparse：Python 略快（18.6 vs 33.8 ms p95），与旧报告「顺序 Sparse Python 略快」同向；本 Rust 分支尚未叠 PR #3 的倒排优化。

## 任务类型（两侧 Hit 一致）

| 任务 | n | dense | sparse | hybrid |
|---|---|---|---|---|
| 差旅制度 | 8 | 1.00 | 1.00 | 1.00 |
| 采购流程 | 7 | 1.00 | 1.00 | 0.86 |
| 项目上线 | 6 | 1.00 | 1.00 | 1.00 |
| 产品账号 FAQ | 6 | 1.00 | 1.00 | 0.83 |
| 费用标准表 | 5 | 1.00 | 1.00 | 0.60 |
| 员工信息表 | 5 | 0.60 | 1.00 | 1.00 |
| 工资表 | 4 | 1.00 | 1.00 | 1.00 |
| 单跳事实 | 5 | 1.00 | 1.00 | 1.00 |
| 领域 FAQ | 9 | 0.78 | 1.00 | 0.89 |
| 工号/标识 | 1 | 1.00 | 1.00 | 1.00 |
| 声明核验 | 3 | 1.00 | 1.00 | 1.00 |
| 结构化 | 5 | 0.60 | 1.00 | 0.60 |
| 改写 | 3 | 0.67 | 1.00 | 0.67 |
| 多跳 | 3 | 1.00 | 1.00 | 1.00 |
| 拒识 | 3 | 1.00 | 0.33 | 0.00 |
| 噪声 | 1 | 1.00 | 1.00 | 1.00 |

结论：检索准确度已经对齐；性能差距主要在 dense/hybrid 的向量推理。Sparse 仍是 Python 微领先的唯一路径。
