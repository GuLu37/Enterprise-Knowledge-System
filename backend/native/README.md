# rag_kernels — 检索链路 Rust 加速内核

以 PyO3 进程内扩展（`.pyd`）的形式，把检索链路中纯 Python 的 CPU 热点下放到 Rust，
不引入任何外部进程或服务。**未编译时系统自动回退纯 Python 实现，功能与正确性不受影响。**

## 覆盖范围

| 内核 | 替代的 Python 热点 | 说明 |
|---|---|---|
| `SparseIndex` | `sparse_index.py` 的 rank_bm25 全语料打分/切片重建 + `_tokenize` + `_lexical_match_score` | 常驻索引（token/df/avgdl 均在 Rust 侧），两段式打分语义与 Python 逐位一致 |
| `rerank` / `fuse_ranked` | `reranker.py` 的关键词/焦点词提取、相关性信号、特异度、门控、综合分排序、RRF 融合 | pydantic 重建与 metadata 合并留在 Python |
| `RustEmbedder` | torch BGE 前向（查询 batch(3) ~82ms → ONNX Runtime fp32 ~55ms） | **分词留在 Python（HF 权威实现，id 逐位一致），推理在 Rust**；CLS 池化 + L2 归一化；向量余弦 vs torch = 1.000000 |
| `tokenize` / `normalize` / `bm25_scores` / `lexical_score` / `query_keywords` 等纯函数 | — | 主要供 parity 测试对拍 |

不在此内核中：cross-encoder 推理（torch 保留，adaptive 触发已挡在请求路径外）、LLM 调用、
Milvus/SQLite IO、编排逻辑。

## BGE ONNX 模型产物

`RustEmbedder` 需要 `app/models/bge-base-zh-v1.5/onnx/model.onnx`（fp32，~400MB，随模型目录
gitignore）。一次性导出（在 backend/ 下）：

```python
# 1) 导出 fp32 ONNX（包装模块规避 transformers 5.x 与 torch 经典导出器的签名冲突）
#    torch.onnx.export(BGEForExport(model), (input_ids, attention_mask, token_type_ids), ...)
# 2) 量化已验证不可用：int8 动态量化在本图上输出崩塌（余弦 0.32），仅保留 fp32。
```

> 历史坑：模型目录曾因下载中断缺失 vocab.txt，transformers 静默回退成 5-token 空壳分词器，
> 所有中文被编码成 [UNK]、dense 通道实际失效。已补齐 vocab.txt/tokenizer.json 并全量重算
> Milvus 向量；新环境需确保模型目录完整。

## 构建（开发机）

前置：rustup（stable-msvc）+ VS Build Tools（C++ 工作负载）+ `pip install maturin`。

```bash
cd backend/native/rag_kernels
# conda base 环境需要显式指定；venv 激活后可省略
export CONDA_PREFIX="F:/miniconda3"        # 按实际环境修改
maturin develop --release                  # 构建并安装到当前 Python 环境
```

国内网络建议配置镜像（否则组件下载可能极慢）：

```bash
export RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"
export RUSTUP_UPDATE_ROOT="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"
```

注意（Windows）：运行中的 Python 进程会锁定 `.pyd`，重新编译前需先停掉 uvicorn/测试进程。

## 验证

```bash
cd backend
python -m pytest tests/test_rust_kernels_parity.py -q   # Rust vs Python 逐位对拍（113 项）
PYTHONPATH=. python scripts/bench_retrieval.py          # 41 条查询延迟/质量基准
```

parity 套件通过 `pytest.importorskip` 在无扩展环境自动跳过；
有扩展时**必须全绿**——任何不一致都代表排序语义漂移，禁止发布。

## 效果（41 条查询基准，CPU）

| 配置 | mean | P95 | recall@5 | MRR@10 |
|---|---|---|---|---|
| 优化前原始基线（精排开启） | 2.92s | 4.79s | 0.9024 | 0.8618 |
| Python 优化轮 + Rust 双检索内核 | 0.1844s | 0.2172s | 0.9024 | 0.8659 |
| + Rust ONNX embedding 内核 | **0.1084s** | **0.1427s** | 0.9024 | 0.8618 |

embedding 内核选择：`.env` 的 `EMBEDDING_KERNEL=auto|rust|torch`（auto=产物与扩展可用时
选 rust，异常自动回退 torch）。**大语料下稀疏内核收益随行数放大**：纯 Python BM25 全语料
打分在数万行时达百毫秒级/查询，Rust 侧为个位数毫秒。

## Docker（多阶段构建示意，未启用）

```dockerfile
FROM rust:1 AS builder
WORKDIR /build
COPY native/rag_kernels .
RUN cargo build --release
# 将产物以 maturin 打成 wheel 后在 python 阶段 pip install
```
