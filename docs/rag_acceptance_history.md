# RAG 验收评测历史

本文把已经跑过的评测收在一处。各轮的原始数字仍在各自的报告和 JSON 里，这里只记录口径、结果和前后能对上的变化。

检索指标都按文档名对齐 `relevant_sources`，`top_k=5`：

- **Hit@5**：可答题的 top-5 是否命中至少一篇 gold
- **Recall@5 / MRR@10 / nDCG@10**：可答题的召回和排序
- **Reject@5**：拒答题的 `bait_sources` 没有进入 top-5
- **切片综合成功**：可答命中，或拒答时诱饵未进 top-5，分母含拒答题。2026-11-12 报告里的切片命中率用的是这个口径

延迟都是逐条清缓存后的端到端时间。三轮的设备和精排开关不一样，延迟不能横比。

| 轮次 | 时间（北京时间） | 题集 | 系统 | 精排 | 设备 | 原始记录 |
| --- | --- | --- | --- | --- | --- | --- |
| 企业基线 | 2026-11-06 12:47 | enterprise 41 | Python 与 Rust | 开启 | Rust 为 DirectML GPU | [`docs/acceptance_py_vs_rust_report.md`](acceptance_py_vs_rust_report.md) |
| 74 题对拍 | 2026-11-12 11:33 | enterprise 41 + standard 33 | Python 与 Rust | 关闭 | 同机 CPU | [`docs/rag_acceptance_eval_report.md`](rag_acceptance_eval_report.md)，[`eval/rag_acceptance/runs/py_vs_rust.json`](../eval/rag_acceptance/runs/py_vs_rust.json) |
| 141 题全量 | 2026-11-13 14:36 | enterprise 41 + standard 100 | 仅 Rust（Python 实现已移除） | 关闭 | CPU，debug 构建 | [`eval/rag_acceptance/runs/full_141.json`](../eval/rag_acceptance/runs/full_141.json) |

## 2026-11-06：企业 41 题基线

语料是 `docs/rag_accuracy_demo` 的 7 份文档，查询是当时的 `docs/rrf_ablation_eval.json`。Rust 使用 ONNX BGE，DirectML GPU，会话池 4，Cross-Encoder 开启。报告里的「精确率」就是 Hit@5。该报告的 MRR 没有写成 MRR@10，不和后面两轮的 MRR@10 并列。

| 方法 | Python Hit@5 | Rust Hit@5 | Python MRR | Rust MRR | Rust p95 |
| --- | --- | --- | --- | --- | --- |
| dense | 95.12% | 95.12% | 0.939 | 0.939 | 61.7 ms |
| sparse | 100% | 100% | 0.935 | 0.935 | 9 ms |
| hybrid | 90.24% | 90.24% | 0.9024 | 0.8293 | 105.2 ms |

dense、sparse 两侧逐位一致。hybrid 命中率一致，Rust 的排序分更低。

## 2026-11-12：74 题 Python / Rust 对拍

在 41 题之外加上当时的 standard 33 题，共 16 篇语料。两端都关 Cross-Encoder，CPU 推理。Python 用 torch BGE 和 rank_bm25；Rust 用 ONNX BGE，会话池 2。Python 入库后重启过一次，稀疏索引 559 行。

可答题上的 dense、sparse，Hit@5 / MRR@10 / nDCG@10 两侧相同。hybrid 的 Hit@5 也相同，Rust 的 MRR@10、nDCG@10 大约低 0.05。

| 方法 | Hit@5 | MRR@10 | nDCG@10 | Reject@5 | Rust p50 | Rust p95 |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 90.14% | 0.8873 | 0.8908 | 100% | 35.6 ms | 40.9 ms |
| sparse | 98.59% | 0.9359 | 0.9494 | 33.33% | 24.7 ms | 33.8 ms |
| hybrid | 88.73% | Py 0.8803 / Rs 0.8310 | Py 0.8698 / Rs 0.8260 | 0% | 111.5 ms | 138.8 ms |

切片综合成功：

| 方法 | enterprise 41 | standard 33 |
| --- | --- | --- |
| dense | 95.12% | 84.85% |
| sparse | 100% | 93.94% |
| hybrid | 90.24% | 78.79% |

enterprise 三路 Hit 与 11 月 6 日相同。拒识 3 题：dense 全部拦住诱饵，sparse 拦住 1 题，hybrid 3 题的诱饵都进了 top-5。

## 2026-11-13：141 题 Rust 全量

standard 扩到 100 题（保留原 33 题，新增 67 题和 16 篇语料），加上 enterprise 41 题。只跑 Rust：Python 服务已经不在仓库里。Cross-Encoder 关闭。模型是 Xenova `bge-base-zh-v1.5` fp32 ONNX，CPU，1 个会话。二进制是 debug 构建，延迟只代表这次机器，不能和前两轮的 release 数字比较。

可答题 131，拒答题 10。

| 方法 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 | Reject@5 | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 91.6% | 90.8% | 0.897 | 0.894 | 40% | 39 ms | 46 ms |
| sparse | 99.2% | 99.2% | 0.933 | 0.947 | 10% | 114 ms | 147 ms |
| hybrid | 91.6% | 90.1% | 0.889 | 0.884 | 30% | 254 ms | 555 ms |

hybrid 再按切片拆开：enterprise 41 题 Hit@5 仍是 90.2%（37/41）。standard 100 题里，可答 90 题 Hit@5 为 92.2%，10 道拒答的 Reject@5 为 30%。

standard 上 hybrid 的 Hit@5：

| 场景 | 可答 | Hit@5 | 拒答 Reject@5 |
| --- | --- | --- | --- |
| 单跳事实 | 10 | 80% | |
| 领域问答 | 10 | 90% | |
| 表格图表 | 9 | 77.8% | |
| 模糊查询 | 8 | 87.5% | |
| 多跳推理 | 8 | 100%（Recall@5 81.2%） | |
| 时效性 | 8 | 87.5% | |
| 长上下文 | 7 | 100% | |
| 噪声干扰 | 7 | 100% | |
| 知识冲突 | 7 | 100% | |
| 跨语言 | 7 | 100% | |
| 多文档综合 | 6 | 100%（Recall@5 91.7%） | |
| 声明核验 | 3 | 100% | |
| 拒答边界 | 0 | | 30%（10 题） |

sparse 在 standard 的可答题上 Hit@5 和 Recall@5 都是 100%。dense 的薄弱处是领域问答 80%、表格图表 77.8%、噪声干扰 71.4%。

### 和 11 月 12 日同一批 74 题

用 hybrid、同一套成功定义（可答则 top-5 命中，拒答则诱饵不进 top-5）：

| 范围 | 11 月 12 日 | 11 月 13 日 |
| --- | --- | --- |
| enterprise 41 | 37/41（90.24%） | 37/41（90.24%） |
| 当时的 standard 33 | 26/33（78.79%） | 28/33（84.85%） |
| 合计 74 | 63/74（85.14%） | 65/74（87.84%） |

变好的两题是 `para_03`（可答，现在能命中）和 `reject_03`（三峡大坝，诱饵不再进入 top-5）。没有旧题从成功变成失败。新语料和原来 33 题的事实是分开写的，这 74 题的 hybrid 命中没有被冲掉。

新增 67 题的综合成功是 58/67。没成功的 9 题是 4 道可答（`sh_01`、`sh_03`、`fz_03`、`tm_07`）和 5 道拒答（`rj_04`、`rj_06`、`rj_08`、`rj_09`、`rj_10`）。

### 这一轮 hybrid 没进 top-5 的可答题

企业集里原来就有的 4 道：`procurement_01`、`faq_04`、`expense_01`、`expense_02`。

原 standard 里原来就有的 3 道：`struct_02`、`struct_03`、`hr_01`。

新题 4 道：

- `sh_01` 折纸项目值班邮箱
- `sh_03` 青岚现行规程编号
- `fz_03` 折纸牵头人工号
- `tm_07` 泊岸仓首班是否仍是 07:40

`struct_03` 和 `sh_01` 在 dense、sparse 单独检索时都是第 1 名，hybrid 的响应是空列表。`tm_07` 的 sparse 把现行 2026 时刻表排在第 1，hybrid 把已废止的 2024 年表排到了前面。

拒答失败的共同情况是：诱饵文档就是题目提到的那份材料，文中写明没有预算、手机号、售价或司机姓名。词面检索会把这份材料召回，Reject@5 于是记为失败。
