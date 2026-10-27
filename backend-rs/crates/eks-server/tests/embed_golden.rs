//! 原生 embedder 与 Python 路径的金向量对拍（tests/fixtures/embed_golden.json
//! 由 Python RustBGEEmbeddings 生成）。

use std::path::Path;

use eks_server::embedder::embedder_for;
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    query_instruction: String,
    max_length: usize,
    queries: Vec<String>,
    query_vectors: Vec<Vec<f32>>,
    docs: Vec<String>,
    doc_vectors: Vec<Vec<f32>>,
}

fn model_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../backend/app/models/bge-base-zh-v1.5")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    // 双方已 L2 归一化
    dot
}

#[test]
fn matches_python_golden_vectors() {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embed_golden.json");
    if !fixture_path.exists() {
        eprintln!("跳过: 金向量夹具不存在");
        return;
    }
    let golden: Golden =
        serde_json::from_reader(std::fs::File::open(&fixture_path).unwrap()).unwrap();
    if !model_dir().join("onnx/model.onnx").exists() {
        eprintln!("跳过: ONNX 产物不存在");
        return;
    }

    let embedder = embedder_for(&model_dir(), &golden.query_instruction, golden.max_length)
        .expect("embedder 初始化失败");

    let query_vectors = embedder.embed_queries(&golden.queries).unwrap();
    for (actual, expected) in query_vectors.iter().zip(&golden.query_vectors) {
        assert_eq!(actual.len(), expected.len());
        assert!(
            (cosine(actual, expected) - 1.0).abs() < 1e-4,
            "查询向量与金向量不一致: cos={}",
            cosine(actual, expected)
        );
    }

    let doc_vectors = embedder.embed_documents(&golden.docs).unwrap();
    for (actual, expected) in doc_vectors.iter().zip(&golden.doc_vectors) {
        assert!(
            (cosine(actual, expected) - 1.0).abs() < 1e-4,
            "文档向量与金向量不一致: cos={}",
            cosine(actual, expected)
        );
    }
}
