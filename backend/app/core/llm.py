"""LLM 配置和初始化"""
from typing import Optional
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from app.config import settings
from app.utils.logger import setup_logger
from app.utils.exceptions import ConfigException

logger = setup_logger(__name__)


def get_ollama_llm():
    """获取 Ollama LLM 实例"""
    try:
        logger.info(f"初始化 Ollama LLM: {settings.OLLAMA_MODEL}")
        llm = Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )
        return llm
    except Exception as e:
        logger.error(f"Ollama LLM 初始化失败: {str(e)}")
        raise ConfigException(f"Ollama LLM 初始化失败: {str(e)}")


def get_openai_llm():
    """获取 OpenAI LLM 实例"""
    try:
        if not settings.OPENAI_API_KEY:
            raise ConfigException("OPENAI_API_KEY 未配置")

        logger.info(f"初始化 OpenAI LLM: {settings.OPENAI_MODEL}")
        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            base_url=settings.OPENAI_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return llm
    except Exception as e:
        logger.error(f"OpenAI LLM 初始化失败: {str(e)}")
        raise ConfigException(f"OpenAI LLM 初始化失败: {str(e)}")


def get_llm(provider: str = "ollama"):
    """
    获取 LLM 实例

    Args:
        provider: LLM 提供商 (ollama, openai, deepseek, anthropic)
    """
    if provider == "ollama":
        return get_ollama_llm()
    elif provider == "openai":
        return get_openai_llm()
    else:
        raise ConfigException(f"不支持的 LLM 提供商: {provider}")


# 全局 LLM 实例（默认使用 Ollama）
_llm_instance: Optional[object] = None


def init_llm(provider: str = "ollama"):
    """初始化 LLM 实例"""
    global _llm_instance
    _llm_instance = get_llm(provider)
    logger.info(f"✓ LLM 初始化完成 (provider: {provider})")
    return _llm_instance


def get_default_llm():
    """获取默认 LLM 实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = init_llm()
    return _llm_instance
