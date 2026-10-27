//! eks-server 库层：供二进制入口与集成测试共用。
pub mod auth;
pub mod config;
pub mod db;
pub mod embedder;
pub mod milvus_lite;
pub mod routes;
pub mod state;

#[cfg(test)]
mod tests;
