//! eks-kernels: 检索链路纯 Rust 内核（BM25/CJK 分词/词法打分/规则重排）。
//!
//! 本 crate 不含任何 FFI 依赖，由 `eks-server`（Rust 后端）消费；
//! 语义由 crates/eks-server/tests 下的金向量对拍测试与 rank_bm25 基线钉死。

pub mod bm25;
pub mod fuse;
pub mod lexical;
pub mod rerank;
pub mod text;

pub use bm25::BM25Okapi;
pub use lexical::{build_filter_terms, lexical_match_score};
pub use text::{normalize_text, structured_sheet_name, tokenize_cjk};
