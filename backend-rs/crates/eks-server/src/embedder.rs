//! BGE embedding 原生推理（ONNX Runtime fp32 + HF tokenizers）。
//!
//! 与 Python 侧 RustBGEEmbeddings 完全一致：HF tokenizer.json 分词（id 逐位
//! 一致）→ 动态 batch 前向 → CLS 池化 → L2 归一化。由 tests/fixtures/embed_golden.json
//! 金向量钉死（余弦 >0.9999）。

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use tokenizers::Tokenizer;

static ORT_ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

/// 按设备偏好构建 ONNX session：
/// - cpu（含空/auto）：纯 CPU（保持与金向量对拍可复现）
/// - directml / dml / gpu / cuda：尝试 DirectML（Windows DX12 GPU，免 CUDA 工具链，
///   需在 exe 旁放置 DirectML.dll），初始化失败自动回退 CPU 并告警。
/// 供 embedder 与 cross_encoder 共用。
pub fn build_ort_session(
    environment: &ort::environment::Environment,
    device: &str,
    onnx_path: &std::path::Path,
) -> Result<ort::session::Session, String> {
    let normalized = device.trim().to_lowercase();
    let wants_gpu = matches!(
        normalized.as_str(),
        "directml" | "dml" | "gpu" | "cuda"
    );
    if wants_gpu {
        let attempt = ort::session::Session::builder(environment).and_then(|b| {
            b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
                .with_execution_providers([ort::ep::DirectML::default().build()])?
                .commit_from_file(onnx_path)
        });
        match attempt {
            Ok(session) => {
                tracing::info!("ONNX session 已启用 DirectML (GPU): {}", onnx_path.display());
                return Ok(session);
            }
            Err(err) => {
                tracing::warn!(
                    "DirectML 初始化失败，回退 CPU（需在 exe 旁放置 DirectML.dll 且系统支持 DX12）: {err}"
                );
            }
        }
    }
    ort::session::Session::builder(environment)
        .and_then(|b| {
            b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
                .with_intra_threads(
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4),
                )?
                .commit_from_file(onnx_path)
        })
        .map_err(|e| format!("ONNX session 初始化失败: {e}"))
}

pub fn ort_environment_shared() -> Result<&'static ort::environment::Environment, String> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }
    let env = ort::environment::init()
        .with_name("eks-server")
        .with_telemetry(false)
        .build()
        .map_err(|e| e.to_string())?;
    let _ = ORT_ENV.set(env);
    ORT_ENV.get().ok_or_else(|| "ORT 环境初始化失败".to_string())
}

pub struct Embedder {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: Tokenizer,
    pad_id: i64,
    pub query_instruction: String,
    max_length: usize,
    batch_size: usize,
    normalize_embeddings: bool,
}

#[derive(Clone)]
pub struct EmbedderConfig {
    pub model_dir: PathBuf,
    pub query_instruction: String,
    pub max_length: usize,
    pub batch_size: usize,
    pub normalize_embeddings: bool,
    /// 推理设备：cpu（默认）/ directml|dml|gpu（DirectML，失败回退 CPU）。
    pub device: String,
}

impl EmbedderConfig {
    pub fn from_env(config: &crate::config::Config) -> Self {
        EmbedderConfig {
            model_dir: PathBuf::from(&config.bge_model_dir),
            query_instruction: config.bge_query_instruction.clone(),
            max_length: config.bge_max_length as usize,
            batch_size: config.bge_batch_size.max(1),
            normalize_embeddings: config.bge_normalize_embeddings,
            device: config.bge_device.clone(),
        }
    }

    fn onnx_path(&self) -> PathBuf {
        self.model_dir.join("onnx").join("model.onnx")
    }

    fn tokenizer_path(&self) -> PathBuf {
        self.model_dir.join("tokenizer.json")
    }
}

pub type SharedEmbedder = Arc<Embedder>;

impl Embedder {
    pub fn load(config: &EmbedderConfig) -> Result<Self, String> {
        let onnx_path = config.onnx_path();
        if !onnx_path.exists() {
            return Err(format!("ONNX 模型不存在: {}", onnx_path.display()));
        }
        let mut tokenizer = Tokenizer::from_file(config.tokenizer_path())
            .map_err(|e| format!("tokenizer.json 加载失败: {}", e))?;
        tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(|e| format!("tokenizer 配置失败: {}", e))?;
        let pad_id = tokenizer
            .token_to_id("[PAD]")
            .ok_or("vocab 缺少 [PAD]")? as i64;

        let environment = ort_environment_shared()?;
        let session = build_ort_session(environment, &config.device, &onnx_path)?;
        Ok(Embedder {
            session: std::sync::Mutex::new(session),
            tokenizer,
            pad_id,
            query_instruction: config.query_instruction.clone(),
            max_length: config.max_length.max(3),
            batch_size: config.batch_size.max(1),
            normalize_embeddings: config.normalize_embeddings,
        })
    }

    /// 批量向量化（分词 + 前向 + CLS 池化 + L2）。is_query 时前置查询指令。
    /// 按 `BGE_BATCH_SIZE` 切批，批内补齐到该批最长序列（对齐 Python）。
    pub fn embed(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prepared: Vec<String> = texts
            .iter()
            .map(|text| {
                if is_query && !self.query_instruction.is_empty() {
                    format!("{}{}", self.query_instruction, text)
                } else {
                    text.clone()
                }
            })
            .collect();

        let mut result = Vec::with_capacity(prepared.len());
        for chunk in prepared.chunks(self.batch_size) {
            result.extend(self.embed_prepared_batch(chunk)?);
        }
        Ok(result)
    }

    fn embed_prepared_batch(&self, prepared: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut encodings = Vec::with_capacity(prepared.len());
        for text in prepared {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| format!("分词失败: {}", e))?;
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
        let batch = encodings.len();

        let mut input_ids: Vec<i64> = Vec::with_capacity(batch * seq_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch * seq_len);
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
        let token_type_ids: Vec<i64> = vec![0; batch * seq_len];

        let mut session = self
            .session
            .lock()
            .map_err(|_| "ONNX session 锁中毒".to_string())?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => ort::value::Tensor::from_array(([batch, seq_len], input_ids))
                    .map_err(|e| e.to_string())?,
                "attention_mask" => ort::value::Tensor::from_array(([batch, seq_len], attention_mask))
                    .map_err(|e| e.to_string())?,
                "token_type_ids" => ort::value::Tensor::from_array(([batch, seq_len], token_type_ids))
                    .map_err(|e| e.to_string())?,
            })
            .map_err(|e| format!("ONNX 推理失败: {}", e))?;
        let hidden = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("输出提取失败: {}", e))?;
        let (_shape, data) = hidden;
        let total = batch * seq_len;
        if total == 0 || data.len() < total {
            return Err("ONNX 输出维度异常".into());
        }
        let dim = data.len() / total;

        let mut result = Vec::with_capacity(batch);
        for row in 0..batch {
            let start = row * seq_len * dim;
            let vector: Vec<f32> = data[start..start + dim].to_vec();
            result.push(maybe_l2_normalize(vector, self.normalize_embeddings));
        }
        Ok(result)
    }

    pub fn embed_queries(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        self.embed(texts, true)
    }

    pub fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        self.embed(texts, false)
    }
}

/// 进程级懒加载单例。
pub fn shared_embedder(config: &crate::config::Config) -> Result<SharedEmbedder, String> {
    static EMBEDDER: OnceLock<Result<SharedEmbedder, String>> = OnceLock::new();
    if let Some(cached) = EMBEDDER.get() {
        return cached.clone();
    }
    let embedder_config = EmbedderConfig::from_env(config);
    let result = Embedder::load(&embedder_config).map(Arc::new);
    let _ = EMBEDDER.set(result.clone());
    result
}

/// 供测试直接构造。
pub fn embedder_for(model_dir: &Path, instruction: &str, max_length: usize) -> Result<Embedder, String> {
    Embedder::load(&EmbedderConfig {
        model_dir: model_dir.to_path_buf(),
        query_instruction: instruction.to_string(),
        max_length,
        batch_size: 16,
        normalize_embeddings: true,
        device: "cpu".into(),
    })
}

/// 对齐 Python `BGE_NORMALIZE_EMBEDDINGS`：默认 L2，关闭时返回 CLS 原向量。
fn maybe_l2_normalize(vector: Vec<f32>, enabled: bool) -> Vec<f32> {
    if !enabled {
        return vector;
    }
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.into_iter().map(|v| v / norm).collect()
    } else {
        vector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_reads_bge_batch_size() {
        let config = EmbedderConfig::from_env(&crate::config::Config::jwt_test());
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.max_length, 512);
        assert!(config.normalize_embeddings);
    }

    #[test]
    fn l2_normalize_can_be_disabled() {
        let raw = vec![3.0_f32, 4.0];
        let unit = maybe_l2_normalize(raw.clone(), true);
        assert!((unit[0] - 0.6).abs() < 1e-6);
        assert!((unit[1] - 0.8).abs() < 1e-6);
        assert_eq!(maybe_l2_normalize(raw.clone(), false), raw);
        assert_eq!(maybe_l2_normalize(vec![0.0, 0.0], true), vec![0.0, 0.0]);
    }
}
