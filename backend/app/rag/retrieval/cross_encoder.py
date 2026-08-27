"""Cross-Encoder 精排模型封装。

模型实例按需创建，但在启用时会跟随服务启动预热提前加载权重并跑一次探测推理。
这样 Cross-Encoder 仍然可以作为可选增强能力，同时避免首个检索请求才触发模型装载。
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Sequence

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class CrossEncoderScorer:
    """使用 HuggingFace SequenceClassification 模型对 query-document 打分。"""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.device_name = device
        self.max_length = max(1, max_length)
        self.batch_size = max(1, batch_size)
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _resolve_device(self, torch: Any) -> Any:
        requested = (self.device_name or "auto").strip().lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            # 延迟导入，保持未启用 Cross-Encoder 时的启动成本和依赖风险最低。
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            load_kwargs = {
                "cache_dir": self.cache_dir,
                "local_files_only": self.local_files_only,
            }
            load_kwargs = {
                key: value
                for key, value in load_kwargs.items()
                if value is not None
            }
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
            device = self._resolve_device(torch)
            model.to(device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            logger.info(
                "Cross-Encoder 初始化完成 (model=%s, device=%s, batch_size=%s)",
                self.model_name,
                device,
                self.batch_size,
            )

    def warmup(self) -> Dict[str, Any]:
        """提前加载模型并跑一次轻量探测，供启动预热使用。"""
        started_at = time.perf_counter()
        result: Dict[str, Any] = {
            "enabled": True,
            "loaded": False,
            "warmed": False,
            "model_name": self.model_name,
            "device_name": self.device_name,
            "device": None,
            "error": None,
        }
        try:
            self._ensure_loaded()
            result["loaded"] = True
            result["device"] = str(self._device) if self._device is not None else None
            probe_scores = self.score(
                "__cross_encoder_warmup__",
                ["__cross_encoder_warmup__"],
            )
            result["warmed"] = bool(probe_scores)
            if probe_scores:
                result["probe_score"] = float(probe_scores[0])
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            result["duration_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        return result

    @staticmethod
    def _normalize_logits(logits: Any, torch: Any) -> Any:
        """兼容单 logit、二分类 logit 和少数多分类模型输出。"""
        if logits.ndim == 1:
            values = logits
        elif logits.shape[-1] == 1:
            values = logits[:, 0]
            return torch.sigmoid(values)
        elif logits.shape[-1] == 2:
            return torch.softmax(logits, dim=-1)[:, 1]
        else:
            values = logits.max(dim=-1).values

        # 对单 logit/多分类异常输出做 sigmoid，保证分数可解释且落在 [0, 1]。
        return torch.sigmoid(values)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """批量返回 query-document 相关性分数，结果顺序与 texts 一致。"""
        if not texts:
            return []
        normalized_query = (query or "").strip()
        if not normalized_query:
            return [0.0 for _ in texts]

        self._ensure_loaded()
        import torch

        scores: List[float] = []
        # 同一个模型实例可能被多个检索请求共享，推理阶段加锁避免设备侧竞态。
        with self._inference_lock:
            for offset in range(0, len(texts), self.batch_size):
                batch_texts = [
                    str(text or "")[: self.max_length * 8]
                    for text in texts[offset : offset + self.batch_size]
                ]
                encoded = self._tokenizer(
                    [normalized_query] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self._device)
                    for key, value in encoded.items()
                }
                with torch.inference_mode():
                    output = self._model(**encoded)
                    batch_scores = self._normalize_logits(output.logits, torch)

                for value in batch_scores.detach().cpu().tolist():
                    numeric_value = float(value)
                    scores.append(
                        numeric_value
                        if math.isfinite(numeric_value)
                        else 0.0
                    )

        return scores


_SCORER: CrossEncoderScorer | None = None
_SCORER_LOCK = threading.Lock()


def get_cross_encoder_scorer() -> CrossEncoderScorer:
    """获取全局 Cross-Encoder 打分器实例。"""
    global _SCORER
    if _SCORER is not None:
        return _SCORER

    with _SCORER_LOCK:
        if _SCORER is None:
            _SCORER = CrossEncoderScorer(
                model_name=settings.CROSS_ENCODER_MODEL_NAME,
                device=settings.CROSS_ENCODER_DEVICE,
                max_length=settings.CROSS_ENCODER_MAX_LENGTH,
                batch_size=settings.CROSS_ENCODER_BATCH_SIZE,
                cache_dir=settings.CROSS_ENCODER_CACHE_DIR,
                local_files_only=settings.CROSS_ENCODER_LOCAL_FILES_ONLY,
            )
        return _SCORER


def warmup_cross_encoder() -> Dict[str, Any]:
    """在启动阶段提前加载 Cross-Encoder。"""
    result: Dict[str, Any] = {
        "enabled": settings.CROSS_ENCODER_ENABLED,
        "loaded": False,
        "warmed": False,
        "model_name": settings.CROSS_ENCODER_MODEL_NAME,
        "device_name": settings.CROSS_ENCODER_DEVICE,
        "device": None,
        "error": None,
        "status": "disabled" if not settings.CROSS_ENCODER_ENABLED else "initializing",
    }
    if not settings.CROSS_ENCODER_ENABLED:
        return result

    try:
        scorer = get_cross_encoder_scorer()
        warmup_result = scorer.warmup()
        result.update(warmup_result)
        result["status"] = "ready" if result.get("warmed") else "failed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
    return result
