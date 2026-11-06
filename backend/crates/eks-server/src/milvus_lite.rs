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
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
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

    pub async fn has_collection(&self, name: &str) -> bool {
        self.root.join("collections").join(name).join("manifest.json").is_file()
    }

    /// 按字段等值删除，对齐 Python `client.delete(filter=memory_id in [...])`。
    pub async fn delete_by_field_values(
        &self,
        collection: &str,
        field: &str,
        values: &[String],
    ) -> Result<bool, String> {
        let wanted: HashSet<&str> = values
            .iter()
            .map(|item| item.trim())
            .filter(|item| !item.is_empty())
            .collect();
        if wanted.is_empty() {
            return Ok(self.has_collection(collection).await);
        }
        let snapshot = match self.collection(collection).await {
            Ok(snapshot) => snapshot,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(err) => return Err(err.to_string()),
        };
        let pks: Vec<i64> = snapshot
            .rows
            .iter()
            .filter(|row| {
                row.fields
                    .get(field)
                    .and_then(FieldValue::as_str)
                    .map(str::trim)
                    .is_some_and(|value| wanted.contains(value))
            })
            .map(|row| row.pk)
            .collect();
        if !pks.is_empty() {
            append_delta_pks(&self.root, collection, &pks)?;
        }
        self.invalidate(collection).await;
        Ok(true)
    }

    /// 按多字段等值过滤删除，对齐 Python `client.delete(filter='user_id == ".." and conversation_id == ".."')`。
    /// 集合不存在时返回 false；存在则返回 true（即使没有匹配行）。
    pub async fn delete_by_matching_fields(
        &self,
        collection: &str,
        matches: &[(&str, &str)],
    ) -> Result<bool, String> {
        let snapshot = match self.collection(collection).await {
            Ok(snapshot) => snapshot,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(err) => return Err(err.to_string()),
        };
        let pks: Vec<i64> = snapshot
            .rows
            .iter()
            .filter(|row| {
                matches.iter().all(|(field, expected)| {
                    row.fields
                        .get(*field)
                        .and_then(FieldValue::as_str)
                        .map(str::trim)
                        == Some(expected.trim())
                })
            })
            .map(|row| row.pk)
            .collect();
        if !pks.is_empty() {
            append_delta_pks(&self.root, collection, &pks)?;
        }
        self.invalidate(collection).await;
        Ok(true)
    }

    /// 按 document_id 写入 delta PK，对齐 Python `client.delete(filter=document_id == ...)`。
    /// 集合不存在时返回 false；存在则返回 true（即使没有匹配行）。
    pub async fn delete_by_document_id(
        &self,
        collection: &str,
        document_id: &str,
    ) -> Result<bool, String> {
        let snapshot = match self.collection(collection).await {
            Ok(snapshot) => snapshot,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(err) => return Err(err.to_string()),
        };
        let pks = pks_for_document(&snapshot.rows, document_id);
        if !pks.is_empty() {
            append_delta_pks(&self.root, collection, &pks)?;
        }
        self.invalidate(collection).await;
        Ok(true)
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
        Ok(sort_top_k_desc(scored, top_k))
    }

    /// COSINE 检索并按行字段过滤（长期记忆 user/conversation/session）。
    pub async fn search_cosine_filtered(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        predicate: impl Fn(&Row) -> bool,
    ) -> std::io::Result<Vec<(usize, f64)>> {
        let snapshot = match self.collection(collection).await {
            Ok(snapshot) => snapshot,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut scored: Vec<(usize, f64)> = snapshot
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| predicate(row))
            .map(|(index, row)| (index, cosine(query, &row.vector)))
            .collect();
        Ok(sort_top_k_desc(scored, top_k))
    }
}

/// 降序取 top_k：先 O(n) 部分选择再对前 k 排序，避免大库下的全量 O(n log n) 排序；
/// 输出与「全量排序后 truncate」完全一致。
fn sort_top_k_desc(mut scored: Vec<(usize, f64)>, top_k: usize) -> Vec<(usize, f64)> {
    let by_score_desc = |a: &(usize, f64), b: &(usize, f64)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    if top_k == 0 {
        return Vec::new();
    }
    if scored.len() > top_k {
        scored.select_nth_unstable_by(top_k - 1, by_score_desc);
        scored.truncate(top_k);
    }
    scored.sort_by(by_score_desc);
    scored
}

pub const DEFAULT_EMBEDDING_DIMENSION: usize = 768;

/// 对齐 Python `_validate_vector_dimension` 的错误文案。
pub fn vector_dimension_mismatch(collection: &str, actual: usize, expected: usize) -> String {
    format!("Milvus collection 维度不匹配: {collection} 当前为 {actual}, 配置为 {expected}")
}

pub fn validate_vector_dimension(
    collection: &str,
    actual: usize,
    expected: usize,
) -> Result<(), String> {
    if actual != expected {
        return Err(vector_dimension_mismatch(collection, actual, expected));
    }
    Ok(())
}

/// 读取已有 parquet 的 FixedSizeList 维度；空集合返回 None。
pub fn collection_stored_vector_dimension(root: &Path, collection: &str) -> Option<usize> {
    let collection_dir = root.join("collections").join(collection);
    let manifest_str = std::fs::read_to_string(collection_dir.join("manifest.json")).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_str).ok()?;
    let files = manifest
        .pointer("/partitions/_default/data_files")?
        .as_array()?;
    let partition_dir = collection_dir.join("partitions").join("_default");
    for file in files {
        let rel = file.as_str()?;
        let path = partition_dir.join(rel);
        if !path.exists() {
            continue;
        }
        if let Some(dim) = parquet_vector_list_size(&path) {
            return Some(dim);
        }
    }
    None
}

fn parquet_vector_list_size(path: &Path) -> Option<usize> {
    let file = std::fs::File::open(path).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    for field in builder.schema().fields() {
        if field.name() == "vector" {
            if let DataType::FixedSizeList(_, size) = field.data_type() {
                return Some(*size as usize);
            }
        }
    }
    None
}

pub fn ensure_collection_dimension(
    root: &Path,
    collection: &str,
    expected: usize,
) -> Result<(), String> {
    if let Some(actual) = collection_stored_vector_dimension(root, collection) {
        validate_vector_dimension(collection, actual, expected)?;
    }
    Ok(())
}

fn flatten_fixed_vectors<'a>(
    collection: &str,
    expected: usize,
    vectors: impl IntoIterator<Item = &'a [f32]>,
) -> Result<Vec<f32>, String> {
    let mut flat = Vec::new();
    for vector in vectors {
        validate_vector_dimension(collection, vector.len(), expected)?;
        flat.extend_from_slice(vector);
    }
    Ok(flat)
}

#[derive(Debug, Clone)]
pub struct MemoryChunkRow {
    pub memory_id: String,
    pub chunk_index: i64,
    pub source_name: String,
    pub chunk_text: String,
    pub user_id: String,
    pub conversation_id: String,
    pub session_id: String,
    pub chunk_type: String,
    pub topic: String,
    pub turn_start: i64,
    pub turn_end: i64,
    pub created_at: String,
    pub vector: Vec<f32>,
}

/// 创建空的 milvus_lite collection 目录与 manifest。
/// manifest.json 的读-改-写互斥锁：并发上传（append 数据文件）与并发删除（append delta）
/// 同时改写 manifest 时会互相读到半写状态（EOF while parsing），批量入库/删除必现。
/// 所有 manifest 写路径（append_memory_chunks / append_delta_pks / ingest append）经此串行化；
/// 锁内只有毫秒级文件 IO，embedding 等重活都在锁外，吞吐不受影响。
static MANIFEST_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn manifest_write_guard() -> std::sync::MutexGuard<'static, ()> {
    MANIFEST_WRITE_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

/// manifest 原子替换写入：先写临时文件再 rename，读侧永远只会看到
/// 完整的旧版本或完整的新版本，杜绝并发读到半写 JSON。
pub fn write_manifest_atomic(path: &Path, content: &str) -> Result<(), String> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, content).map_err(|err| err.to_string())?;
    std::fs::rename(&tmp, path).map_err(|err| err.to_string())?;
    Ok(())
}

pub fn ensure_collection(root: &Path, collection: &str) -> Result<(), String> {
    let collection_dir = root.join("collections").join(collection);
    let manifest_path = collection_dir.join("manifest.json");
    if manifest_path.is_file() {
        return Ok(());
    }
    std::fs::create_dir_all(collection_dir.join("partitions").join("_default").join("data"))
        .map_err(|err| err.to_string())?;
    std::fs::create_dir_all(collection_dir.join("partitions").join("_default").join("delta"))
        .map_err(|err| err.to_string())?;
    let manifest = serde_json::json!({
        "version": 1,
        "current_seq": 0,
        "partitions": {
            "_default": {
                "data_files": [],
                "delta_files": []
            }
        }
    });
    write_manifest_atomic(
        &manifest_path,
        &serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?,
    )?;
    Ok(())
}

pub fn append_memory_chunks(
    root: &Path,
    collection: &str,
    rows: &[MemoryChunkRow],
    embedding_dimension: usize,
) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }
    let _guard = manifest_write_guard();
    let embedding_dimension = embedding_dimension.max(1);
    ensure_collection(root, collection)?;
    ensure_collection_dimension(root, collection, embedding_dimension)?;
    let collection_dir = root.join("collections").join(collection);
    let manifest_path = collection_dir.join("manifest.json");
    let manifest_str =
        std::fs::read_to_string(&manifest_path).map_err(|err| format!("manifest 读取失败: {err}"))?;
    let mut manifest: serde_json::Value =
        serde_json::from_str(&manifest_str).map_err(|err| format!("manifest 解析失败: {err}"))?;

    let current_seq = manifest["current_seq"].as_i64().unwrap_or(0);
    let data_dir = collection_dir.join("partitions").join("_default").join("data");
    std::fs::create_dir_all(&data_dir).map_err(|err| err.to_string())?;
    let start_seq = current_seq + 1;
    let end_seq = current_seq + rows.len() as i64;
    let new_file = format!("data_{:06}_{:06}.parquet", start_seq, end_seq);
    let new_path = data_dir.join(&new_file);

    let n = rows.len();
    let pk_start = current_seq + 100;
    let pk_values: Vec<i64> = (0..n).map(|i| pk_start + i as i64).collect();
    let seq_values: Vec<i64> = (0..n).map(|i| start_seq + i as i64).collect();
    let flat_vectors = flatten_fixed_vectors(
        collection,
        embedding_dimension,
        rows.iter().map(|row| row.vector.as_slice()),
    )?;
    let vector_array = FixedSizeListArray::new(
        std::sync::Arc::new(Field::new("item", DataType::Float32, true)),
        embedding_dimension as i32,
        std::sync::Arc::new(arrow_array::Float32Array::from(flat_vectors)) as ArrayRef,
        None,
    );
    let fields = vec![
        Field::new("_seq", DataType::Int64, false),
        Field::new("pk", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                std::sync::Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dimension as i32,
            ),
            false,
        ),
        Field::new("memory_id", DataType::Utf8, false),
        Field::new("chunk_index", DataType::Int64, false),
        Field::new("source_name", DataType::Utf8, false),
        Field::new("chunk_text", DataType::Utf8, false),
        Field::new("user_id", DataType::Utf8, false),
        Field::new("conversation_id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("chunk_type", DataType::Utf8, false),
        Field::new("topic", DataType::Utf8, false),
        Field::new("turn_start", DataType::Int64, false),
        Field::new("turn_end", DataType::Int64, false),
        Field::new("created_at", DataType::Utf8, false),
    ];
    let columns: Vec<ArrayRef> = vec![
        std::sync::Arc::new(Int64Array::from(seq_values)),
        std::sync::Arc::new(Int64Array::from(pk_values)),
        std::sync::Arc::new(StringArray::from(
            rows.iter().map(|row| row.chunk_text.clone()).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(vector_array) as ArrayRef,
        std::sync::Arc::new(StringArray::from(
            rows.iter().map(|row| row.memory_id.clone()).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(Int64Array::from(
            rows.iter().map(|row| row.chunk_index).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(StringArray::from(
            rows.iter().map(|row| row.source_name.clone()).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(StringArray::from(
            rows.iter().map(|row| row.chunk_text.clone()).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(StringArray::from(
            rows.iter().map(|row| row.user_id.clone()).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.conversation_id.clone())
                .collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(StringArray::from(
            rows.iter().map(|row| row.session_id.clone()).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(StringArray::from(
            rows.iter().map(|row| row.chunk_type.clone()).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(StringArray::from(
            rows.iter().map(|row| row.topic.clone()).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(Int64Array::from(
            rows.iter().map(|row| row.turn_start).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(Int64Array::from(
            rows.iter().map(|row| row.turn_end).collect::<Vec<_>>(),
        )),
        std::sync::Arc::new(StringArray::from(
            rows.iter().map(|row| row.created_at.clone()).collect::<Vec<_>>(),
        )),
    ];
    let schema = Schema::new(fields);
    let batch = RecordBatch::try_new(std::sync::Arc::new(schema), columns)
        .map_err(|err| format!("RecordBatch 构建失败: {err}"))?;
    let file = std::fs::File::create(&new_path).map_err(|err| err.to_string())?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)
        .map_err(|err| format!("Parquet writer 创建失败: {err}"))?;
    writer
        .write(&batch)
        .map_err(|err| format!("Parquet 写入失败: {err}"))?;
    writer
        .close()
        .map_err(|err| format!("Parquet 关闭失败: {err}"))?;

    manifest["current_seq"] = serde_json::json!(end_seq);
    let version = manifest.get("version").and_then(|value| value.as_i64()).unwrap_or(0) + 1;
    manifest["version"] = serde_json::json!(version);
    if let Some(files) = manifest
        .pointer_mut("/partitions/_default/data_files")
        .and_then(serde_json::Value::as_array_mut)
    {
        files.push(serde_json::json!(format!("data/{new_file}")));
    } else {
        manifest["partitions"]["_default"]["data_files"] = serde_json::json!([format!("data/{new_file}")]);
    }
    write_manifest_atomic(
        &manifest_path,
        &serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?,
    )?;
    Ok(())
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

pub fn pks_for_document(rows: &[Row], document_id: &str) -> Vec<i64> {
    let document_id = document_id.trim();
    if document_id.is_empty() {
        return Vec::new();
    }
    rows.iter()
        .filter(|row| {
            row.fields
                .get("document_id")
                .and_then(FieldValue::as_str)
                .map(str::trim)
                == Some(document_id)
        })
        .map(|row| row.pk)
        .collect()
}

/// 对齐 Python `_get_document_content_from_milvus`：按 chunk_index 重组 child 正文。
pub fn document_content_from_rows(rows: &[Row], document_id: &str) -> String {
    let document_id = document_id.trim();
    if document_id.is_empty() {
        return String::new();
    }
    let mut items: Vec<(i64, String)> = rows
        .iter()
        .filter_map(|row| {
            let id = row
                .fields
                .get("document_id")
                .and_then(FieldValue::as_str)
                .map(str::trim)?;
            if id != document_id {
                return None;
            }
            let index = row
                .fields
                .get("chunk_index")
                .and_then(FieldValue::as_int)
                .unwrap_or(0);
            let text = ["child_text", "chunk_text", "text"]
                .iter()
                .find_map(|name| {
                    row.fields
                        .get(*name)
                        .and_then(FieldValue::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                })?;
            Some((index, text))
        })
        .collect();
    items.sort_by_key(|(index, _)| *index);
    items
        .into_iter()
        .map(|(_, text)| text)
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn append_delta_pks(root: &Path, collection: &str, pks: &[i64]) -> Result<(), String> {
    if pks.is_empty() {
        return Ok(());
    }
    let _guard = manifest_write_guard();
    let collection_dir = root.join("collections").join(collection);
    let manifest_path = collection_dir.join("manifest.json");
    let manifest_str = std::fs::read_to_string(&manifest_path)
        .map_err(|err| format!("manifest 读取失败: {err}"))?;
    let mut manifest: serde_json::Value =
        serde_json::from_str(&manifest_str).map_err(|err| format!("manifest 解析失败: {err}"))?;

    let delta_dir = collection_dir.join("partitions").join("_default").join("delta");
    std::fs::create_dir_all(&delta_dir).map_err(|err| err.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or(0);
    let file_name = format!("delta_{stamp}_{}.parquet", pks.len());
    let relative = format!("delta/{file_name}");
    let file_path = delta_dir.join(&file_name);

    let schema = Schema::new(vec![Field::new("pk", DataType::Int64, false)]);
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![std::sync::Arc::new(Int64Array::from(pks.to_vec())) as ArrayRef],
    )
    .map_err(|err| format!("delta RecordBatch 构建失败: {err}"))?;
    let file = std::fs::File::create(&file_path).map_err(|err| err.to_string())?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)
        .map_err(|err| format!("delta Parquet writer 创建失败: {err}"))?;
    writer
        .write(&batch)
        .map_err(|err| format!("delta Parquet 写入失败: {err}"))?;
    writer
        .close()
        .map_err(|err| format!("delta Parquet 关闭失败: {err}"))?;

    let version = manifest.get("version").and_then(|value| value.as_i64()).unwrap_or(0) + 1;
    manifest["version"] = serde_json::json!(version);
    let delta_files = manifest
        .pointer_mut("/partitions/_default/delta_files")
        .and_then(serde_json::Value::as_array_mut);
    if let Some(files) = delta_files {
        files.push(serde_json::json!(relative));
    } else {
        manifest["partitions"]["_default"]["delta_files"] = serde_json::json!([relative]);
    }
    write_manifest_atomic(
        &manifest_path,
        &serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_top_k_desc_matches_full_sort() {
        let mut rng = 12345u64;
        let mut next = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f64) / (u32::MAX as f64)
        };
        for (len, top_k) in [(0, 3), (1, 3), (5, 3), (5, 5), (5, 0), (200, 7), (200, 200)] {
            let scored: Vec<(usize, f64)> = (0..len).map(|i| (i, next())).collect();
            let mut expected = scored.clone();
            expected.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            expected.truncate(top_k);
            assert_eq!(sort_top_k_desc(scored, top_k), expected, "len={len} top_k={top_k}");
        }
    }

    fn dev_db_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../app/data/milvus_lite.db")
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
        // 断言绑定原开发数据集；全新/少量数据的本地库直接跳过。
        if snapshot.rows.len() < 520 {
            eprintln!("跳过: 开发库行数 {} < 520", snapshot.rows.len());
            return;
        }
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
        if snapshot.rows.len() < 8 {
            eprintln!("跳过: 开发库行数不足 8");
            return;
        }
        let probe = snapshot.rows[7].vector.clone();
        let hits = store.search_cosine("doc_chunk", &probe, 1).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 7);
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pks_for_document_filters_by_id() {
        let row = |pk, document_id: &str| Row {
            pk,
            vector: Vec::new(),
            fields: HashMap::from([(
                "document_id".into(),
                FieldValue::Str(document_id.into()),
            )]),
        };
        let rows = vec![row(1, "doc-a"), row(2, "doc-b"), row(3, "doc-a")];
        assert_eq!(pks_for_document(&rows, "doc-a"), vec![1, 3]);
        assert!(pks_for_document(&rows, "missing").is_empty());
        assert!(pks_for_document(&rows, "  ").is_empty());
    }

    #[test]
    fn document_content_joins_child_chunks_in_index_order() {
        let row = |index, document_id: &str, text: &str| Row {
            pk: index,
            vector: Vec::new(),
            fields: HashMap::from([
                ("document_id".into(), FieldValue::Str(document_id.into())),
                ("chunk_index".into(), FieldValue::Int(index)),
                ("child_text".into(), FieldValue::Str(text.into())),
            ]),
        };
        let rows = vec![
            row(1, "doc-a", "第二段"),
            row(0, "doc-a", "第一段"),
            row(0, "doc-b", "其他"),
        ];
        assert_eq!(document_content_from_rows(&rows, "doc-a"), "第一段\n\n第二段");
        assert!(document_content_from_rows(&rows, "missing").is_empty());
    }

    #[tokio::test]
    async fn delete_by_document_id_writes_delta_and_drops_rows() {
        let root = std::env::temp_dir().join(format!(
            "eks-milvus-delete-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        let collection_dir = root.join("collections").join("doc_chunk");
        let data_dir = collection_dir.join("partitions").join("_default").join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let schema = Schema::new(vec![
            Field::new("pk", DataType::Int64, false),
            Field::new("document_id", DataType::Utf8, false),
        ]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![
                std::sync::Arc::new(Int64Array::from(vec![11, 12, 13])) as ArrayRef,
                std::sync::Arc::new(StringArray::from(vec!["keep", "drop-me", "keep"])) as ArrayRef,
            ],
        )
        .unwrap();
        let data_path = data_dir.join("data_000001_000003.parquet");
        let file = std::fs::File::create(&data_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let manifest = serde_json::json!({
            "version": 1,
            "current_seq": 3,
            "partitions": {
                "_default": {
                    "data_files": ["data/data_000001_000003.parquet"],
                    "delta_files": []
                }
            }
        });
        std::fs::write(
            collection_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let store = MilvusLiteStore::open(&root);
        assert!(store.delete_by_document_id("doc_chunk", "drop-me").await.unwrap());
        let snapshot = store.collection("doc_chunk").await.unwrap();
        let remaining: Vec<i64> = snapshot.rows.iter().map(|row| row.pk).collect();
        assert_eq!(remaining, vec![11, 13]);
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(collection_dir.join("manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["version"], 2);
        let delta_files = manifest["partitions"]["_default"]["delta_files"]
            .as_array()
            .unwrap();
        assert_eq!(delta_files.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn delete_missing_collection_returns_false() {
        let root = std::env::temp_dir().join(format!(
            "eks-milvus-missing-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let store = MilvusLiteStore::open(&root);
        assert!(!store
            .delete_by_document_id("doc_chunk", "any")
            .await
            .unwrap());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn memory_chunks_write_filter_search_and_delete() {
        let root = std::env::temp_dir().join(format!(
            "eks-memory-chunks-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        let mut keep = vec![0.0f32; DEFAULT_EMBEDDING_DIMENSION];
        keep[0] = 1.0;
        let mut other = vec![0.0f32; DEFAULT_EMBEDDING_DIMENSION];
        other[1] = 1.0;
        append_memory_chunks(
            &root,
            "memory_chunks",
            &[
                MemoryChunkRow {
                    memory_id: "u1:c1:0:aaa".into(),
                    chunk_index: 0,
                    source_name: "conversation".into(),
                    chunk_text: "差旅报销时限三天".into(),
                    user_id: "u1".into(),
                    conversation_id: "c1".into(),
                    session_id: "s1".into(),
                    chunk_type: "dialogue".into(),
                    topic: String::new(),
                    turn_start: 0,
                    turn_end: 0,
                    created_at: "2026-01-01T00:00:00Z".into(),
                    vector: keep.clone(),
                },
                MemoryChunkRow {
                    memory_id: "u1:c2:0:bbb".into(),
                    chunk_index: 0,
                    source_name: "conversation".into(),
                    chunk_text: "另一会话".into(),
                    user_id: "u1".into(),
                    conversation_id: "c2".into(),
                    session_id: "s2".into(),
                    chunk_type: "dialogue".into(),
                    topic: String::new(),
                    turn_start: 0,
                    turn_end: 0,
                    created_at: "2026-01-01T00:00:00Z".into(),
                    vector: other,
                },
            ],
            DEFAULT_EMBEDDING_DIMENSION,
        )
        .unwrap();
        let store = MilvusLiteStore::open(&root);
        let hits = store
            .search_cosine_filtered("memory_chunks", &keep, 4, |row| {
                row.fields.get("user_id").and_then(FieldValue::as_str) == Some("u1")
                    && row.fields.get("conversation_id").and_then(FieldValue::as_str) == Some("c1")
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        let snapshot = store.collection("memory_chunks").await.unwrap();
        let memory_id = snapshot.rows[hits[0].0]
            .fields
            .get("memory_id")
            .and_then(FieldValue::as_str);
        assert_eq!(memory_id, Some("u1:c1:0:aaa"));
        assert!(store
            .delete_by_field_values(
                "memory_chunks",
                "memory_id",
                &["u1:c1:0:aaa".into()]
            )
            .await
            .unwrap());
        let remaining = store.collection("memory_chunks").await.unwrap();
        assert_eq!(remaining.rows.len(), 1);
        assert_eq!(
            remaining.rows[0]
                .fields
                .get("memory_id")
                .and_then(FieldValue::as_str),
            Some("u1:c2:0:bbb")
        );
        assert!(store
            .delete_by_matching_fields(
                "memory_chunks",
                &[("user_id", "u1"), ("conversation_id", "c2")]
            )
            .await
            .unwrap());
        let emptied = store.collection("memory_chunks").await.unwrap();
        assert!(emptied.rows.is_empty());
        assert!(!store
            .delete_by_matching_fields("missing_collection", &[("user_id", "u1")])
            .await
            .unwrap());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn vector_dimension_mismatch_matches_python() {
        assert_eq!(
            vector_dimension_mismatch("memory_chunks", 1024, 768),
            "Milvus collection 维度不匹配: memory_chunks 当前为 1024, 配置为 768"
        );
        assert!(validate_vector_dimension("doc_chunk", 768, 768).is_ok());
        assert!(validate_vector_dimension("doc_chunk", 512, 768)
            .unwrap_err()
            .contains("当前为 512, 配置为 768"));
    }

    #[test]
    fn concurrent_appends_keep_manifest_parseable() {
        // 回归：并发 append（数据文件）与并发 delta 删除同时读改写 manifest，
        // 修复前会互相读到半写状态（EOF while parsing）。锁后 manifest 必须始终可解析。
        let root = std::env::temp_dir().join(format!(
            "eks-milvus-lock-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        ensure_collection(&root, "doc_chunk").unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let mut handles = Vec::new();
        for t in 0..8usize {
            let root = root.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for k in 0..20usize {
                    if t % 2 == 0 {
                        let rows = vec![MemoryChunkRow {
                            memory_id: format!("m{t}_{k}"),
                            chunk_index: 0,
                            source_name: "s".into(),
                            chunk_text: "文本".into(),
                            user_id: "u".into(),
                            conversation_id: "c".into(),
                            session_id: "s".into(),
                            chunk_type: "dialogue".into(),
                            topic: "".into(),
                            turn_start: 0,
                            turn_end: 0,
                            created_at: "2026-01-01".into(),
                            vector: vec![0.1f32; 4],
                        }];
                        append_memory_chunks(&root, "memory_chunks", &rows, 4).unwrap();
                    } else {
                        append_delta_pks(&root, "doc_chunk", &[k as i64]).unwrap();
                    }
                    // 任意时刻 manifest 必须完整可解析
                    let text =
                        std::fs::read_to_string(root.join("collections").join(if t % 2 == 0 {
                            "memory_chunks"
                        } else {
                            "doc_chunk"
                        }).join("manifest.json")).unwrap();
                    serde_json::from_str::<serde_json::Value>(&text).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn append_memory_chunks_rejects_wrong_vector_dim() {
        let root = std::env::temp_dir().join(format!(
            "eks-memory-dim-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        let err = append_memory_chunks(
            &root,
            "memory_chunks",
            &[MemoryChunkRow {
                memory_id: "u1:c1:0:aaa".into(),
                chunk_index: 0,
                source_name: "conversation".into(),
                chunk_text: "差旅".into(),
                user_id: "u1".into(),
                conversation_id: "c1".into(),
                session_id: "s1".into(),
                chunk_type: "dialogue".into(),
                topic: String::new(),
                turn_start: 0,
                turn_end: 0,
                created_at: "2026-01-01T00:00:00Z".into(),
                vector: vec![0.1, 0.2],
            }],
            DEFAULT_EMBEDDING_DIMENSION,
        )
        .unwrap_err();
        assert_eq!(
            err,
            "Milvus collection 维度不匹配: memory_chunks 当前为 2, 配置为 768"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn existing_collection_rejects_config_dim_change() {
        let root = std::env::temp_dir().join(format!(
            "eks-memory-dim-change-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        let vector = vec![0.0f32; DEFAULT_EMBEDDING_DIMENSION];
        append_memory_chunks(
            &root,
            "memory_chunks",
            &[MemoryChunkRow {
                memory_id: "u1:c1:0:aaa".into(),
                chunk_index: 0,
                source_name: "conversation".into(),
                chunk_text: "差旅".into(),
                user_id: "u1".into(),
                conversation_id: "c1".into(),
                session_id: "s1".into(),
                chunk_type: "dialogue".into(),
                topic: String::new(),
                turn_start: 0,
                turn_end: 0,
                created_at: "2026-01-01T00:00:00Z".into(),
                vector: vector.clone(),
            }],
            DEFAULT_EMBEDDING_DIMENSION,
        )
        .unwrap();
        assert_eq!(
            collection_stored_vector_dimension(&root, "memory_chunks"),
            Some(DEFAULT_EMBEDDING_DIMENSION)
        );
        let err = append_memory_chunks(
            &root,
            "memory_chunks",
            &[MemoryChunkRow {
                memory_id: "u1:c1:1:bbb".into(),
                chunk_index: 1,
                source_name: "conversation".into(),
                chunk_text: "下一句".into(),
                user_id: "u1".into(),
                conversation_id: "c1".into(),
                session_id: "s1".into(),
                chunk_type: "dialogue".into(),
                topic: String::new(),
                turn_start: 1,
                turn_end: 1,
                created_at: "2026-01-01T00:00:01Z".into(),
                vector: vec![0.0f32; 4],
            }],
            4,
        )
        .unwrap_err();
        assert_eq!(
            err,
            "Milvus collection 维度不匹配: memory_chunks 当前为 768, 配置为 4"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
