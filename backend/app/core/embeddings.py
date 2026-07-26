"""向量化模型配置"""
from typing import Optional
import requests
from langchain_community.embeddings import OllamaEmbeddings
from app.config import settings
from app.utils.logger import setup_logger
from app.utils.exceptions import ConfigException

logger = setup_logger(__name__)


def check_ollama_connection():
    """检查 Ollama 连接"""
    try:
        response = requests.get(f"{settings.EMBEDDING_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama 服务连接成功")
            return True
    except Exception as e:
        logger.warning(f"Ollama 连接失败: {str(e)}")
        return False


def get_ollama_embeddings():
    """获取 Ollama Embedding 实例"""
    try:
        if not check_ollama_connection():
            raise ConfigException("无法连接到 Ollama 服务")

        logger.info(f"初始化 Ollama Embedding: {settings.EMBEDDING_MODEL}")
        embeddings = OllamaEmbeddings(
            base_url=settings.EMBEDDING_BASE_URL,
            model=settings.EMBEDDING_MODEL,
        )
        return embeddings
    except Exception as e:
        logger.error(f"Ollama Embedding 初始化失败: {str(e)}")
        raise ConfigException(f"Ollama Embedding 初始化失败: {str(e)}")


# 全局 Embedding 实例
_embeddings_instance: Optional[OllamaEmbeddings] = None


def init_embeddings():
    """初始化 Embedding 实例"""
    global _embeddings_instance
    _embeddings_instance = get_ollama_embeddings()
    logger.info(f"✓ Embedding 初始化完成 (model: {settings.EMBEDDING_MODEL})")
    return _embeddings_instance


def get_default_embeddings():
    """获取默认 Embedding 实例"""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = init_embeddings()
    return _embeddings_instance
