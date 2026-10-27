//! BGE embedding 的 ONNX Runtime 推理内核（fp32）。
//!
//! 职责划分: 分词留在 Python 侧（HF BertTokenizer 权威实现，保证 id 逐位
//! 一致），本内核只负责张量组装 + ONNX 前向 + CLS 池化 + L2 归一化。
//! torch 前向（查询 batch(3) 实测 ~82ms）→ ONNX Runtime fp32（~52ms），
//! 向量余弦 1.000000，由 parity 测试钉死。
//! int8 动态量化在本模型导出图上质量崩塌（余弦 0.32），已弃用，仅保留 fp32。

use std::path::Path;
use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// ort 每进程只允许一个 Environment，进程级单例共享给所有 session。
static ORT_ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

fn ort_environment() -> Result<&'static ort::environment::Environment, String> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }
    let env = ort::environment::init()
        .with_name("rag_kernels")
        .with_telemetry(false)
        .build()
        .map_err(|e| format!("{}", e))?;
    let _ = ORT_ENV.set(env);
    ORT_ENV.get().ok_or_else(|| "ORT 环境初始化失败".to_string())
}

/// pyo3 暴露的 ONNX embedding 推理器。
#[pyclass(module = "rag_kernels")]
pub struct RustEmbedder {
    session: std::sync::Mutex<ort::session::Session>,
    pad_id: i64,
}

#[pymethods]
impl RustEmbedder {
    /// onnx_path: fp32 ONNX 模型路径; pad_id: HF tokenizer 的 [PAD] id。
    #[new]
    #[pyo3(signature = (onnx_path, pad_id=0))]
    fn new(onnx_path: String, pad_id: i64) -> PyResult<Self> {
        if !Path::new(&onnx_path).exists() {
            return Err(PyValueError::new_err(format!(
                "ONNX 模型不存在: {}",
                onnx_path
            )));
        }
        let environment = ort_environment().map_err(PyValueError::new_err)?;
        let session = ort::session::Session::builder(&environment)
            .and_then(|b| {
                b.with_optimization_level(
                    ort::session::builder::GraphOptimizationLevel::Level3,
                )?
                .with_intra_threads(
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4),
                )?
                .commit_from_file(&onnx_path)
            })
            .map_err(|e| PyValueError::new_err(format!("ONNX session 初始化失败: {}", e)))?;
        Ok(RustEmbedder {
            session: std::sync::Mutex::new(session),
            pad_id,
        })
    }

    /// 批量推理: 输入 HF tokenizer 的 id 序列（含 CLS/SEP），右侧补 pad 到批内最长，
    /// 返回 CLS 池化 + L2 归一化后的向量。id 与 Python 侧完全一致，分词语义零漂移。
    fn embed_ids(&self, ids: Vec<Vec<i64>>) -> PyResult<Vec<Vec<f32>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let seq_len = ids.iter().map(|row| row.len()).max().unwrap_or(1).max(1);
        let batch = ids.len();

        let mut input_ids: Vec<i64> = Vec::with_capacity(batch * seq_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch * seq_len);
        for row in &ids {
            if row.len() > seq_len {
                return Err(PyValueError::new_err("输入序列长度异常"));
            }
            for position in 0..seq_len {
                match row.get(position) {
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
            .map_err(|_| PyValueError::new_err("ONNX session 锁中毒"))?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => ort::value::Tensor::from_array((
                    [batch, seq_len],
                    input_ids,
                ))
                .map_err(|e| PyValueError::new_err(format!("input_ids 张量失败: {}", e)))?,
                "attention_mask" => ort::value::Tensor::from_array((
                    [batch, seq_len],
                    attention_mask,
                ))
                .map_err(|e| PyValueError::new_err(format!("mask 张量失败: {}", e)))?,
                "token_type_ids" => ort::value::Tensor::from_array((
                    [batch, seq_len],
                    token_type_ids,
                ))
                .map_err(|e| PyValueError::new_err(format!("token_type_ids 张量失败: {}", e)))?,
            })
            .map_err(|e| PyValueError::new_err(format!("ONNX 推理失败: {}", e)))?;
        let hidden = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| PyValueError::new_err(format!("输出提取失败: {}", e)))?;
        let (_shape, data) = hidden;
        // 输出布局 [batch, seq, hidden]; 输出 seq 与输入一致，hidden 由长度反推。
        let total = batch * seq_len;
        if total == 0 || data.len() < total {
            return Err(PyValueError::new_err("ONNX 输出维度异常"));
        }
        let dim = data.len() / total;

        let mut result = Vec::with_capacity(batch);
        for row in 0..batch {
            let start = row * seq_len * dim;
            let vector: Vec<f32> = data[start..start + dim].to_vec();
            let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            let normalized: Vec<f32> = if norm > 0.0 {
                vector.into_iter().map(|v| v / norm).collect()
            } else {
                vector
            };
            result.push(normalized);
        }
        Ok(result)
    }
}
