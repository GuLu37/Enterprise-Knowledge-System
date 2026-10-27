//! Cross-Encoder 金分对拍：Rust ONNX 分数 vs torch sigmoid 分数（逐条 |Δ| < 0.01）。

use std::path::Path;

use eks_server::cross_encoder::{CrossEncoder, CrossEncoderConfig};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    query: String,
    texts: Vec<String>,
    scores: Vec<f64>,
}

#[test]
fn cross_encoder_matches_torch_golden() {
    let fixture_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cross_encoder_golden.json");
    if !fixture_path.exists() {
        eprintln!("跳过: 金分夹具不存在");
        return;
    }
    let golden: Golden =
        serde_json::from_reader(std::fs::File::open(&fixture_path).unwrap()).unwrap();

    // 默认模型目录（HF snapshot）
    let model_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../backend/app/models/models--BAAI--bge-reranker-base/snapshots");
    let snapshot = match std::fs::read_dir(&model_root) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .find(|p| p.is_dir())
        ,
        Err(_) => None,
    };
    let Some(snapshot) = snapshot else {
        eprintln!("跳过: 未找到 reranker snapshot");
        return;
    };
    let onnx = snapshot.join("onnx").join("model.onnx");
    if !onnx.exists() {
        eprintln!("跳过: ONNX 不存在 {}", onnx.display());
        return;
    }
    let config = CrossEncoderConfig {
        onnx_path: onnx,
        tokenizer_path: snapshot.join("tokenizer.json"),
        max_length: 512,
        batch_size: 8,
    };
    let encoder = CrossEncoder::load(&config).expect("cross-encoder 初始化失败");
    let actual = encoder
        .score(&golden.query, &golden.texts)
        .expect("打分失败");
    assert_eq!(actual.len(), golden.scores.len());
    for (got, expected) in actual.iter().zip(&golden.scores) {
        assert!(
            (got - expected).abs() < 0.01,
            "分数偏差过大: got={got} expected={expected}"
        );
    }
    // 排序一致性：top-1 必须一致；整体 Kendall τ ≥ 0.9（|Δ|<0.02 的近平局视为一致）
    let top1_rust = actual
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let top1_torch = golden
        .scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(top1_rust, top1_torch, "top-1 不一致");

    let mut concordant = 0usize;
    let mut comparable = 0usize;
    for i in 0..actual.len() {
        for j in (i + 1)..actual.len() {
            let d_rust = actual[i] - actual[j];
            let d_torch = golden.scores[i] - golden.scores[j];
            if d_rust.abs() < 0.02 && d_torch.abs() < 0.02 {
                continue; // 双侧近平局：不计入
            }
            comparable += 1;
            if (d_rust > 0.0) == (d_torch > 0.0) {
                concordant += 1;
            }
        }
    }
    if comparable > 0 {
        let tau = concordant as f64 / comparable as f64;
        assert!(tau >= 0.9, "Kendall 一致率过低: {tau} ({concordant}/{comparable})");
    }
}
