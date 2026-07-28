"""常量定义"""

# ==================== RAG 相关常量 ====================

# 检索方法
RETRIEVAL_METHOD_DENSE = "dense"
RETRIEVAL_METHOD_SPARSE = "sparse"
RETRIEVAL_METHOD_HYBRID = "hybrid"

# 文本分割器类型
SPLITTER_RECURSIVE = "recursive"
SPLITTER_TOKEN = "token"
SPLITTER_SEMANTIC = "semantic"

# ==================== 支持的文件类型 ====================
SUPPORTED_FILE_TYPES = {
    "pdf": "application/pdf",
    "txt": "text/plain",
    "md": "text/markdown",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
}

# ==================== 默认参数 ====================
#文档上传-固定大小切块规则
DOCUMENT_TOP_K = 5
DOCUMENT_TEMPERATURE = 0.7
DOCUMENT_MAX_TOKENS = 2000
DOCUMENT_CHUNK_SIZE = 600
DOCUMENT_CHUNK_OVERLAP = 100

#对话历史-固定大小切块规则
CHAT_TOP_K = 5
CHAT_TEMPERATURE = 0.3
CHAT_MAX_TOKENS = 2000
CHAT_CHUNK_SIZE = 500
CHAT_CHUNK_OVERLAP = 80

# ==================== LLM 模型列表 ====================
OLLAMA_MODELS = ["mistral", "llama2", "neural-chat", "qwen", "zephyr"]
OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]

# ==================== Embedding 模型列表 ====================
EMBEDDING_MODELS = [
    "nomic-embed-text",
    "all-minilm",
    "mxbai-embed-large",
]
