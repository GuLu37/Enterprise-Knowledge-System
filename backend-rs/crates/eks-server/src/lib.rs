//! eks-server 库层：供二进制入口与集成测试共用。
pub mod auth;
pub mod chat_stream;
pub mod cascade;
pub mod conversations;
pub mod cross_encoder;
pub mod config;
pub mod db;
pub mod documents;
pub mod embedder;
pub mod ingest;
pub mod intent;
pub mod llm;
pub mod logging;
pub mod memory;
pub mod memory_chunk;
pub mod openapi;
pub mod milvus_lite;
pub mod planner;
pub mod rag_cache;
pub mod react;
pub mod remote_mcp;
pub mod request_log;
pub mod retrieval;
pub mod retrieval_engine;
pub mod routes;
pub mod short_memory;
pub mod skills;
pub mod state;
pub mod trace;

#[cfg(test)]
mod tests;
