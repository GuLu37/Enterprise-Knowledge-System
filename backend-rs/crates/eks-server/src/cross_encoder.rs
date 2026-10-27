//! Cross-Encoder 精排（bge-reranker-base, XLM-R 单 logit）的 ONNX 原生推理。
//!
//! 分词用 HF tokenizers crate 的 pair 编码（与 Python `tokenizer(query, text)` 一致），
//! 前向输出单 logit → sigmoid（对齐 cross_encoder._normalize_logits 的单 logit 分支）。
//! 由 tests/fixtures/cross_encoder_golden.json 金分对拍。

use std::sync::Mutex;
use std::sync::Arc;

use tokenizers::Tokenizer;

pub struct CrossEncoder {
    session: Mutex<ort::session::Session>,
    tokenizer: Tokenizer,
    pad_id: i64,
    max_length: usize,
    batch_size: usize,
}

pub struct CrossEncoderConfig {
    pub onnx_path: std::path::PathBuf,
    pub tokenizer_path: std::path::PathBuf,
    pub max_length: usize,
    pub batch_size: usize,
}

impl CrossEncoder {
    pub fn load(config: &CrossEncoderConfig) -> Result<Self, String> {
        if !config.onnx_path.exists() {
            return Err(format!(
                "Cross-Encoder ONNX 不存在: {}（先运行导出流程）",
                config.onnx_path.display()
            ));
        }
        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| format!("cross-encoder tokenizer 加载失败: {}", e))?;
        tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(|e| format!("tokenizer 配置失败: {}", e))?;
        let pad_id = tokenizer
            .token_to_id("<pad>")
            .ok_or("cross-encoder vocab 缺少 <pad>")? as i64;

        let environment = crate::embedder::ort_environment_shared()?;
        let session = ort::session::Session::builder(environment)
            .and_then(|b| {
                b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
                    .with_intra_threads(
                        std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(4),
                    )?
                    .commit_from_file(&config.onnx_path)
            })
            .map_err(|e| format!("Cross-Encoder session 初始化失败: {}", e))?;
        Ok(CrossEncoder {
            session: Mutex::new(session),
            tokenizer,
            pad_id,
            max_length: config.max_length.max(16),
            batch_size: config.batch_size.max(1),
        })
    }

    /// 批量 query-text 打分（sigmoid 单 logit），结果顺序与 texts 一致。
    pub fn score(&self, query: &str, texts: &[String]) -> Result<Vec<f64>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let query = query.trim();
        if query.is_empty() {
            return Ok(vec![0.0; texts.len()]);
        }
        let mut all_scores: Vec<f64> = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size) {
            // pair 编码（与 Python tokenizer(query, text) 一致）
            let mut encodings = Vec::with_capacity(batch.len());
            for text in batch {
                let encoding = self
                    .tokenizer
                    .encode((query, text.as_str()), true)
                    .map_err(|e| format!("pair 分词失败: {}", e))?;
                let mut ids: Vec<i64> = encoding
                    .get_ids()
                    .iter()
                    .map(|id| *id as i64)
                    .collect();
                if ids.len() > self.max_length {
                    ids.truncate(self.max_length);
                }
                encodings.push(ids);
            }
            let seq_len = encodings.iter().map(|ids| ids.len()).max().unwrap_or(1).max(1);
            let batch_size = encodings.len();
            let mut input_ids: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
            let mut attention_mask: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
            for ids in &encodings {
                for position in 0..seq_len {
                    match ids.get(position) {
                        Some(id) => {
                            input_ids.push(*id);
                            attention_mask.push(1);
                        }
                        None => {
                            input_ids.push(self.pad_id);
                            attention_mask.push(0);
                        }
                    }
                }
            }
            let mut session = self
                .session
                .lock()
                .map_err(|_| "session 锁中毒".to_string())?;
            let outputs = session
                .run(ort::inputs! {
                    "input_ids" => ort::value::Tensor::from_array(([batch_size, seq_len], input_ids))
                        .map_err(|e| e.to_string())?,
                    "attention_mask" => ort::value::Tensor::from_array(([batch_size, seq_len], attention_mask))
                        .map_err(|e| e.to_string())?,
                })
                .map_err(|e| format!("Cross-Encoder 推理失败: {}", e))?;
            let logits = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("输出提取失败: {}", e))?;
            let (_shape, data) = logits;
            // 单 logit: sigmoid；兼容偶发 [b, 1] 布局
            for row in 0..batch_size {
                let logit = data.get(row).copied().unwrap_or(0.0);
                let score = 1.0 / (1.0 + (-logit as f64).exp());
                all_scores.push(if score.is_finite() { score } else { 0.0 });
            }
        }
        Ok(all_scores)
    }
}

pub type SharedCrossEncoder = Arc<CrossEncoder>;
