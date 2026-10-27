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
pub mod llm;
pub mod milvus_lite;
pub mod retrieval;
pub mod retrieval_engine;
pub mod routes;
pub mod state;

#[cfg(test)]
mod tests;
