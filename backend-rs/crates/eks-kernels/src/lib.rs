//! eks-kernels: 检索链路纯 Rust 内核（BM25/CJK 分词/词法打分/规则重排）。
//!
//! 本 crate 不含任何 FFI 依赖；`backend/native/rag_kernels`（pyo3 扩展）与
//! `eks-server`（Rust 后端）共用同一份实现，由 Python 侧 parity 测试钉死语义。

pub mod bm25;
pub mod lexical;
pub mod rerank;
pub mod text;

pub use bm25::BM25Okapi;
pub use text::{normalize_text, tokenize_cjk};
