//! Milvus Lite 直读引擎：不经 gRPC/Python，直接读取 milvus_lite.db 的
//! manifest.json + Parquet 数据/删除文件。
//!
//! 读取语义（已与 pymilvus 对拍验证，520/19 行一致）：
//! 1. manifest.json 声明 data_files / delta_files（快照，WAL 在干净停机后为空）
//! 2. 数据行按 pk 去重，保留 _seq 最大的版本（LSM 语义）
//! 3. delta 文件中的 pk 全部剔除
//! 4. 索引为 FLAT + COSINE → 暴力余弦扫描
//!
//! 过渡期与 Python 端共存：manifest mtime 变化时自动重载（读路径无锁，
//! 整体替换 Arc 快照）。写入路径在 Phase E 之前仍由 Python 负责。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int64Array, LargeStringArray, ListArray, RecordBatch,
    StringArray,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub enum FieldValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl FieldValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            FieldValue::Str(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            FieldValue::Int(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Row {
    pub pk: i64,
    pub vector: Vec<f32>,
    pub fields: HashMap<String, FieldValue>,
}

#[derive(Debug)]
pub struct CollectionSnapshot {
    pub rows: Vec<Row>,
    pub loaded_at: SystemTime,
}

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    partitions: HashMap<String, PartitionFiles>,
}

#[derive(Deserialize)]
struct PartitionFiles {
    #[serde(default)]
    data_files: Vec<String>,
    #[serde(default)]
    delta_files: Vec<String>,
}

#[derive(Clone)]
pub struct MilvusLiteStore {
    pub root: PathBuf,
    cache: Arc<RwLock<HashMap<String, (SystemTime, Arc<CollectionSnapshot>)>>>,
}

fn read_manifest_mtime(collection_dir: &Path) -> std::io::Result<SystemTime> {
    std::fs::metadata(collection_dir.join("manifest.json"))?.modified()
}

fn column_value(array: &ArrayRef, index: usize) -> FieldValue {
    if array.is_null(index) {
        return FieldValue::Null;
    }
    let any = array.as_any();
    if let Some(col) = any.downcast_ref::<Int64Array>() {
        FieldValue::Int(col.value(index))
    } else if let Some(col) = any.downcast_ref::<StringArray>() {
        FieldValue::Str(col.value(index).to_string())
    } else if let Some(col) = any.downcast_ref::<LargeStringArray>() {
        FieldValue::Str(col.value(index).to_string())
    } else if let Some(col) = any.downcast_ref::<arrow_array::Float64Array>() {
        FieldValue::Float(col.value(index))
    } else if let Some(col) = any.downcast_ref::<arrow_array::Float32Array>() {
        FieldValue::Float(col.value(index) as f64)
    } else if let Some(col) = any.downcast_ref::<arrow_array::BooleanArray>() {
        FieldValue::Bool(col.value(index))
    } else if let Some(col) = any.downcast_ref::<arrow_array::Int32Array>() {
        FieldValue::Int(col.value(index) as i64)
    } else if let Some(col) = any.downcast_ref::<arrow_array::UInt64Array>() {
        FieldValue::Int(col.value(index) as i64)
    } else {
        FieldValue::Null
    }
}

fn vector_from(array: &ArrayRef, index: usize) -> Vec<f32> {
    let any = array.as_any();
    if let Some(list) = any.downcast_ref::<FixedSizeListArray>() {
        let values = list.value(index);
        if let Some(floats) = values.as_any().downcast_ref::<arrow_array::Float32Array>() {
            return floats.values().to_vec();
        }
    }
    if let Some(list) = any.downcast_ref::<ListArray>() {
        let slice = list.value(index);
        if let Some(floats) = slice.as_any().downcast_ref::<arrow_array::Float32Array>() {
            return floats.values().to_vec();
        }
    }
    Vec::new()
}

impl MilvusLiteStore {
    pub fn open(root: impl Into<PathBuf>) -> Self {
        MilvusLiteStore {
            root: root.into(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 获取集合快照；manifest mtime 未变时命中缓存，变化时整体重载。
    pub async fn collection(&self, name: &str) -> std::io::Result<Arc<CollectionSnapshot>> {
        let collection_dir = self.root.join("collections").join(name);
        let mtime = read_manifest_mtime(&collection_dir)?;
        {
            let cache = self.cache.read().await;
            if let Some((cached_mtime, snapshot)) = cache.get(name) {
                if *cached_mtime == mtime {
                    return Ok(snapshot.clone());
                }
            }
        }
        let snapshot = Arc::new(self.load_collection(&collection_dir)?);
        let mut cache = self.cache.write().await;
        cache.insert(name.to_string(), (mtime, snapshot.clone()));
        Ok(snapshot)
    }

    /// 强制失效（文档写入后由外部调用）。
    pub async fn invalidate(&self, name: &str) {
        self.cache.write().await.remove(name);
    }

    fn load_collection(&self, collection_dir: &Path) -> std::io::Result<CollectionSnapshot> {
        let manifest: Manifest = serde_json::from_reader(std::fs::File::open(
            collection_dir.join("manifest.json"),
        )?)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut rows: Vec<Row> = Vec::new();
        let mut deleted: HashSet<i64> = HashSet::new();
        let mut latest_by_pk: HashMap<i64, usize> = HashMap::new();

        for partition in manifest.partitions.values() {
            let partition_dir = collection_dir.join("partitions").join("_default");
            for file in &partition.data_files {
                for row in read_parquet_rows(&partition_dir.join(file))? {
                    let seq = row
                        .fields
                        .get("_seq")
                        .and_then(|v| v.as_int())
                        .unwrap_or(0);
                    if let Some(&existing) = latest_by_pk.get(&row.pk) {
                        let existing_seq = rows[existing]
                            .fields
                            .get("_seq")
                            .and_then(|v| v.as_int())
                            .unwrap_or(0);
                        if seq > existing_seq {
                            rows[existing] = row;
                        }
                    } else {
                        latest_by_pk.insert(row.pk, rows.len());
                        rows.push(row);
                    }
                }
            }
            for file in &partition.delta_files {
                let path = partition_dir.join(file);
                if !path.exists() {
                    continue;
                }
                for pk in read_delta_pks(&path)? {
                    deleted.insert(pk);
                }
            }
        }

        rows.retain(|row| !deleted.contains(&row.pk));
        // 保留 _seq 供调试，但不影响语义
        Ok(CollectionSnapshot {
            rows,
            loaded_at: SystemTime::now(),
        })
    }

    /// FLAT + COSINE 暴力扫描，返回 (行下标, 余弦分) 降序 top_k。
    pub async fn search_cosine(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
    ) -> std::io::Result<Vec<(usize, f64)>> {
        let snapshot = self.collection(collection).await?;
        let mut scored: Vec<(usize, f64)> = snapshot
            .rows
            .iter()
            .enumerate()
            .map(|(index, row)| (index, cosine(query, &row.vector)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn read_parquet_rows(path: &Path) -> std::io::Result<Vec<Row>> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(io_err)?
        .build()
        .map_err(io_err)?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(io_err)?;
        let schema = batch.schema();
        let fields = schema.fields();
        let row_count = batch.num_rows();

        for index in 0..row_count {
            let mut row = Row {
                pk: 0,
                vector: Vec::new(),
                fields: HashMap::new(),
            };
            for (position, field) in fields.iter().enumerate() {
                let name = field.name();
                let column = batch.column(position);
                match name.as_str() {
                    "pk" => {
                        if let Some(col) = column.as_any().downcast_ref::<Int64Array>() {
                            row.pk = col.value(index);
                        }
                    }
                    "vector" => {
                        row.vector = vector_from(column, index);
                    }
                    _ => {
                        row.fields
                            .insert(name.clone(), column_value(column, index));
                    }
                }
            }
            rows.push(row);
        }
    }
    Ok(rows)
}

fn read_delta_pks(path: &Path) -> std::io::Result<Vec<i64>> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(io_err)?
        .build()
        .map_err(io_err)?;
    let mut pks = Vec::new();
    for batch in reader {
        let batch = batch.map_err(io_err)?;
        if let Some(col) = batch.column_by_name("pk") {
            if let Some(pk) = col.as_any().downcast_ref::<Int64Array>() {
                pks.extend(pk.iter().flatten());
            }
        }
    }
    Ok(pks)
}

fn io_err(e: impl std::error::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev_db_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../backend/app/data/milvus_lite.db")
    }

    #[tokio::test]
    async fn reads_doc_chunk_with_expected_rows() {
        let path = dev_db_path();
        if !path.exists() {
            eprintln!("跳过: 未找到开发库 {path:?}");
            return;
        }
        let store = MilvusLiteStore::open(&path);
        let snapshot = store.collection("doc_chunk").await.expect("加载失败");
        assert_eq!(snapshot.rows.len(), 520, "活跃行数应为 520");
        let sample = &snapshot.rows[0];
        assert_eq!(sample.vector.len(), 768);
        assert!(sample
            .fields
            .get("source_name")
            .and_then(|v| v.as_str())
            .is_some());
    }

    #[tokio::test]
    async fn cosine_self_search_top1_is_self() {
        let path = dev_db_path();
        if !path.exists() {
            return;
        }
        let store = MilvusLiteStore::open(&path);
        let snapshot = store.collection("doc_chunk").await.unwrap();
        let probe = snapshot.rows[7].vector.clone();
        let hits = store.search_cosine("doc_chunk", &probe, 1).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 7);
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
    }
}
