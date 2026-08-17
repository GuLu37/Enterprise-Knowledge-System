"""LLM 配置和初始化"""
import threading
import time
from typing import Any, Optional

from langchain.chat_models import init_chat_model

from app.config import settings
from app.utils.exceptions import LLMException
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

_PROVIDER_FACTORIES = {
    "ollama": lambda model=None, temperature=None, timeout=None: get_ollama_llm(
        model=model, temperature=temperature, timeout=timeout
    ),
    "openai": lambda model=None, temperature=None, timeout=None: get_openai_llm(
        model=model, temperature=temperature, timeout=timeout
    ),
    "deepseek": lambda model=None, temperature=None, timeout=None: get_deepseek_llm(
        model=model, temperature=temperature, timeout=timeout
    ),
    "openrouter": lambda model=None, temperature=None, timeout=None: get_openrouter_llm(
        model=model, temperature=temperature, timeout=timeout
    ),
    "anthropic": lambda model=None, temperature=None, timeout=None: get_anthropic_llm(
        model=model, temperature=temperature, timeout=timeout
    ),
}

_FALLBACK_ORDER = ("deepseek", "openrouter", "openai", "anthropic", "ollama")
_FAILURE_TTL_SECONDS = 60


def _resolve_temperature(temperature: Optional[float]) -> float:
    return settings.LLM_TEMPERATURE if temperature is None else temperature


def _resolve_timeout(timeout: Optional[int]) -> int:
    return settings.LLM_TIMEOUT if timeout is None else timeout


def _normalize_provider(provider: Optional[str]) -> Optional[str]:
    return provider.lower().strip() if provider else None


def _provider_is_configured(provider: str, model: Optional[str] = None) -> bool:
    """检查 provider 是否具备最基本的配置，避免明知不可用还反复尝试。"""
    if provider == "deepseek":
        return bool(settings.DEEPSEEK_API_KEY and settings.DEEPSEEK_BASE_URL and (model or settings.DEEPSEEK_MODEL))
    if provider == "openai":
        return bool(settings.OPENAI_API_KEY and (model or settings.OPENAI_MODEL))
    if provider == "anthropic":
        return bool(settings.ANTHROPIC_API_KEY)
    if provider == "openrouter":
        return bool(settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_BASE and (model or settings.OPENROUTER_MODEL))
    if provider == "ollama":
        return bool(settings.OLLAMA_BASE_URL and (model or settings.OLLAMA_MODEL))
    return False


def _build_provider_order(preferred_provider: Optional[str] = None) -> list[str]:
    ordered: list[str] = []
    preferred = _normalize_provider(preferred_provider)

    for candidate in (preferred, *_FALLBACK_ORDER):
        if candidate and candidate in _PROVIDER_FACTORIES and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _create_llm(
    provider: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
):
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        raise LLMException(f"不支持的 LLM 提供商: {provider}")
    return factory(model=model, temperature=temperature, timeout=timeout)


class FallbackLLM:
    """支持多 provider 自动兜底的 LLM 代理。"""

    def __init__(
        self,
        preferred_provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        self.preferred_provider = _normalize_provider(preferred_provider)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.active_provider: Optional[str] = None
        self._active_llm: Optional[Any] = None
        self._failure_at: dict[str, float] = {}

    def _candidate_providers(self) -> list[str]:
        providers = _build_provider_order(self.preferred_provider)
        now = time.monotonic()

        # 过滤未配置和近期刚失败过的 provider，避免无效初始化与重复等待。
        candidates = []
        for provider in providers:
            failed_at = self._failure_at.get(provider)
            if (
                provider != self.preferred_provider
                and failed_at is not None
                and (now - failed_at) < _FAILURE_TTL_SECONDS
            ):
                continue
            if _provider_is_configured(provider, self.model):
                candidates.append(provider)

        # 如果所有 provider 都被暂时跳过了，就退回到完整顺序，保证不会永久饿死
        return candidates or [provider for provider in providers if _provider_is_configured(provider, self.model)]

    def _instantiate_provider(self, provider: str):
        logger.info(f"尝试初始化 LLM provider: {provider}")
        return _create_llm(
            provider=provider,
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
        )

    def _attempt_providers(self) -> list[str]:
        """限制单次调用的候选数量，避免故障时轮询全部 provider。"""
        return self._candidate_providers()[:settings.LLM_MAX_FALLBACK_ATTEMPTS]

    def prime(self) -> str:
        """预先挑选一个当前可用的 provider。"""
        errors: list[str] = []
        for provider in self._attempt_providers():
            try:
                self._active_llm = self._instantiate_provider(provider)
                self.active_provider = provider
                return provider
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                logger.warning(f"LLM provider 初始化失败，继续兜底: {provider} -> {exc}")

        raise LLMException(
            "没有可用的 LLM 提供商，请检查配置: " + (" | ".join(errors) if errors else "无可用 provider")
        )

    def _ensure_active_llm(self):
        if self._active_llm is None:
            self.prime()
        return self._active_llm

    def _fallback_call(self, method_name: str, *args, **kwargs):
        errors: list[str] = []
        providers = self._attempt_providers()
        if self.active_provider in providers:
            providers.remove(self.active_provider)
            providers.insert(0, self.active_provider)

        for provider in providers:
            try:
                llm = self._active_llm if provider == self.active_provider and self._active_llm else self._instantiate_provider(provider)
                result = getattr(llm, method_name)(*args, **kwargs)
                self._active_llm = llm
                self.active_provider = provider
                return result
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                logger.warning(f"LLM 调用失败，切换到下一个 provider: {provider} -> {exc}")
                self._failure_at[provider] = time.monotonic()
                if provider == self.active_provider:
                    self._active_llm = None
                    self.active_provider = None

        raise LLMException(
            "所有 LLM 提供商均不可用: " + (" | ".join(errors) if errors else "无可用 provider")
        )

    def invoke(self, *args, **kwargs):
        return self._fallback_call("invoke", *args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self._fallback_call("ainvoke", *args, **kwargs)

    def stream(self, *args, **kwargs):
        errors: list[str] = []
        providers = self._attempt_providers()
        if self.active_provider in providers:
            providers.remove(self.active_provider)
            providers.insert(0, self.active_provider)

        for provider in providers:
            try:
                llm = self._active_llm if provider == self.active_provider and self._active_llm else self._instantiate_provider(provider)
                self._active_llm = llm
                self.active_provider = provider
                for chunk in llm.stream(*args, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                logger.warning(f"LLM 流式调用失败，切换到下一个 provider: {provider} -> {exc}")
                self._failure_at[provider] = time.monotonic()
                if provider == self.active_provider:
                    self._active_llm = None
                    self.active_provider = None

        raise LLMException(
            "所有 LLM 提供商均不可用: " + (" | ".join(errors) if errors else "无可用 provider")
        )

    async def astream(self, *args, **kwargs):
        errors: list[str] = []
        providers = self._attempt_providers()
        if self.active_provider in providers:
            providers.remove(self.active_provider)
            providers.insert(0, self.active_provider)

        for provider in providers:
            try:
                llm = self._active_llm if provider == self.active_provider and self._active_llm else self._instantiate_provider(provider)
                self._active_llm = llm
                self.active_provider = provider
                async for chunk in llm.astream(*args, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                logger.warning(f"LLM 异步流式调用失败，切换到下一个 provider: {provider} -> {exc}")
                self._failure_at[provider] = time.monotonic()
                if provider == self.active_provider:
                    self._active_llm = None
                    self.active_provider = None

        raise LLMException(
            "所有 LLM 提供商均不可用: " + (" | ".join(errors) if errors else "无可用 provider")
        )

    def __getattr__(self, item):
        llm = self._ensure_active_llm()
        return getattr(llm, item)


def get_ollama_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
):
    """获取 Ollama LLM 实例"""
    try:
        target_model = model or settings.OLLAMA_MODEL
        logger.info(f"初始化 Ollama LLM: {target_model}")
        return init_chat_model(
            model=target_model,
            model_provider="ollama",
            base_url=settings.OLLAMA_BASE_URL,
            temperature=_resolve_temperature(temperature),
            timeout=_resolve_timeout(timeout),
        )
    except Exception as e:
        logger.error(f"Ollama LLM 初始化失败: {str(e)}")
        raise LLMException(f"Ollama LLM 初始化失败: {str(e)}")


def get_openai_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
):
    """获取 OpenAI LLM 实例"""
    try:
        if not settings.OPENAI_API_KEY:
            raise LLMException("OPENAI_API_KEY 未配置")

        target_model = model or settings.OPENAI_MODEL
        logger.info(f"初始化 OpenAI LLM: {target_model}")
        return init_chat_model(
            model=target_model,
            model_provider="openai",
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            temperature=_resolve_temperature(temperature),
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=_resolve_timeout(timeout),
        )
    except Exception as e:
        logger.error(f"OpenAI LLM 初始化失败: {str(e)}")
        raise LLMException(f"OpenAI LLM 初始化失败: {str(e)}")


def get_deepseek_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
):
    """获取 DeepSeek LLM 实例（通过 OpenAI 兼容接口）"""
    try:
        if not settings.DEEPSEEK_API_KEY:
            raise LLMException("DEEPSEEK_API_KEY 未配置")
        if not settings.DEEPSEEK_BASE_URL:
            raise LLMException("DEEPSEEK_BASE_URL 未配置")

        target_model = model or settings.DEEPSEEK_MODEL
        if not target_model:
            raise LLMException("DEEPSEEK_MODEL 未配置")

        logger.info(f"初始化 DeepSeek LLM: {target_model}")
        return init_chat_model(
            model=target_model,
            model_provider="openai",
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=_resolve_temperature(temperature),
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=_resolve_timeout(timeout),
        )
    except Exception as e:
        logger.error(f"DeepSeek LLM 初始化失败: {str(e)}")
        raise LLMException(f"DeepSeek LLM 初始化失败: {str(e)}")


def get_openrouter_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
):
    """获取 OpenRouter / OpenAI 兼容中转 LLM 实例"""
    try:
        if not settings.OPENROUTER_API_KEY:
            raise LLMException("OPENROUTER_API_KEY 未配置")
        if not settings.OPENROUTER_API_BASE:
            raise LLMException("OPENROUTER_API_BASE 未配置")

        target_model = model or settings.OPENROUTER_MODEL
        if not target_model:
            raise LLMException("OPENROUTER_MODEL 未配置")

        logger.info(f"初始化 OpenRouter LLM: {target_model}")
        return init_chat_model(
            model=target_model,
            model_provider="openai",
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_API_BASE,
            temperature=_resolve_temperature(temperature),
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=_resolve_timeout(timeout),
        )
    except Exception as e:
        logger.error(f"OpenRouter LLM 初始化失败: {str(e)}")
        raise LLMException(f"OpenRouter LLM 初始化失败: {str(e)}")


def get_anthropic_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
):
    """获取 Anthropic Claude LLM 实例"""
    try:
        if not settings.ANTHROPIC_API_KEY:
            raise LLMException("ANTHROPIC_API_KEY 未配置")

        target_model = model or settings.ANTHROPIC_MODEL or "claude-sonnet-4-6"
        logger.info(f"初始化 Anthropic LLM: {target_model}")
        return init_chat_model(
            model=target_model,
            model_provider="anthropic",
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=_resolve_temperature(temperature),
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=_resolve_timeout(timeout),
        )
    except Exception as e:
        logger.error(f"Anthropic LLM 初始化失败: {str(e)}")
        raise LLMException(f"Anthropic LLM 初始化失败: {str(e)}")


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
):
    """
    获取 LLM 实例

    Args:
        provider: LLM 提供商。若不传，优先使用 LLM_PROVIDER 配置，再按兜底顺序尝试。
        model: 可选，覆盖默认模型名称
        temperature: 可选，覆盖默认温度
        timeout: 可选，覆盖默认超时
    """
    return FallbackLLM(
        preferred_provider=_normalize_provider(provider) or _normalize_provider(settings.LLM_PROVIDER),
        model=model,
        temperature=temperature,
        timeout=timeout,
    )


# 全局 LLM 实例
_llm_instance: Optional[object] = None
_llm_init_lock = threading.Lock()


def init_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
):
    """初始化 LLM 实例"""
    global _llm_instance
    _llm_instance = get_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
    selected_provider = _llm_instance.prime()
    logger.info(
        f"✓ LLM 兜底初始化完成 (preferred: {provider or 'default-order'}, active: {selected_provider})"
    )
    return _llm_instance


def get_default_llm():
    """获取默认 LLM 实例"""
    global _llm_instance
    if _llm_instance is None:
        with _llm_init_lock:
            if _llm_instance is None:
                _llm_instance = init_llm()
    return _llm_instance
