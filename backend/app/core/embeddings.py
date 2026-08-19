"""BGE 向量化模型配置"""
import importlib
import sys
import threading
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.utils.exceptions import EmbeddingException
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def _load_transformers_classes():
    """加载当前解释器环境中的 Transformers，绕过错误的用户级包遮蔽。"""
    def import_classes():
        module = importlib.import_module("transformers")
        auto_model = getattr(module, "AutoModel", None)
        auto_tokenizer = getattr(module, "AutoTokenizer", None)
        if auto_model is None or auto_tokenizer is None:
            raise ImportError(
                f"当前 transformers 缺少 AutoModel/AutoTokenizer: "
                f"{getattr(module, '__file__', 'unknown')}"
            )
        return auto_model, auto_tokenizer, getattr(module, "__version__", "unknown")

    try:
        return import_classes()
    except (ImportError, AttributeError) as first_error:
        loaded_module = sys.modules.get("transformers")
        loaded_origin = str(getattr(loaded_module, "__file__", "") or "")
        environment_roots = [
            Path(sys.prefix) / "Lib" / "site-packages",
            Path(sys.prefix) / "lib" / "python" / "site-packages",
            *(
                Path(item)
                for item in sys.path
                if item and "site-packages" in item.lower()
            ),
        ]
        candidate_roots = []
        seen_roots = set()
        loaded_root = (
            Path(loaded_origin).resolve().parent
            if loaded_origin
            else None
        )
        for root in environment_roots:
            try:
                resolved_root = root.resolve()
            except OSError:
                continue
            root_key = str(resolved_root).lower()
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)
            if (
                resolved_root != loaded_root
                and (resolved_root / "transformers").exists()
            ):
                candidate_roots.append(resolved_root)
        if not candidate_roots:
            raise ImportError(
                "Transformers 导入失败。当前 Python 环境中没有可用的 "
                "AutoModel/AutoTokenizer，请检查 transformers 安装和启动解释器。"
            ) from first_error

        original_path = list(sys.path)
        try:
            for module_name in list(sys.modules):
                if module_name == "transformers" or module_name.startswith("transformers."):
                    del sys.modules[module_name]
            sys.path[:] = [
                str(root)
                for root in candidate_roots
            ] + [
                item
                for item in original_path
                if item not in {str(root) for root in candidate_roots}
            ]
            auto_model, auto_tokenizer, version = import_classes()
            logger.warning(
                "检测到错误的 Transformers 导入路径，已切换到当前环境包: "
                "bad_path=%s version=%s",
                loaded_origin or "unknown",
                version,
            )
            return auto_model, auto_tokenizer, version
        except Exception as second_error:
            raise ImportError(
                "Transformers 导入失败，且无法切换到当前 Python 环境中的可用版本。"
                f"当前解释器: {sys.executable}; 原始错误: {first_error}"
            ) from second_error
        finally:
            sys.path[:] = original_path


class BGEEmbeddings(Embeddings):
    """基于 BGE 模型的本地向量化实现。"""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        query_instruction: str = "",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction = query_instruction
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only

        logger.info(f"初始化 BGE Embedding: {self.model_name} ({self.device})")
        try:
            AutoModel, AutoTokenizer, transformers_version = _load_transformers_classes()
            logger.info("Transformers 就绪: version=%s", transformers_version)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"BGE Embedding 初始化失败: {str(e)}")
            raise EmbeddingException(
                "BGE Embedding 初始化失败。"
                "如果这是第一次使用 BGE，需要先能连接 HuggingFace 下载模型，"
                "或把 BGE_MODEL_NAME 改成本地模型目录。"
                "也可以配置 BGE_CACHE_DIR 指向已下载缓存目录，"
                "并在模型已存在本地时设置 BGE_LOCAL_FILES_ONLY=true。"
                f"原始错误: {str(e)}"
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0]
                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().tolist()
        except Exception as e:
            logger.error(f"BGE 文本向量化失败: {str(e)}")
            raise EmbeddingException(f"BGE 文本向量化失败: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """向量化文档文本列表。"""
        cleaned_texts = [text or "" for text in texts]
        embeddings: List[List[float]] = []

        for start in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[start:start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """向量化查询文本。"""
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """批量向量化检索查询，复用一次模型前向计算。"""
        queries = [text or "" for text in texts]
        if self.query_instruction:
            queries = [f"{self.query_instruction}{query}" for query in queries]
        return self._embed_batch(queries)


_embeddings_instance: Optional[BGEEmbeddings] = None
_embeddings_init_lock = threading.Lock()


def get_bge_embeddings(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    normalize_embeddings: Optional[bool] = None,
    query_instruction: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: Optional[bool] = None,
) -> BGEEmbeddings:
    """获取 BGE Embedding 实例。"""
    return BGEEmbeddings(
        model_name=model_name or settings.BGE_MODEL_NAME,
        device=device or settings.BGE_DEVICE,
        max_length=max_length or settings.BGE_MAX_LENGTH,
        batch_size=batch_size or settings.BGE_BATCH_SIZE,
        normalize_embeddings=(
            settings.BGE_NORMALIZE_EMBEDDINGS
            if normalize_embeddings is None
            else normalize_embeddings
        ),
        query_instruction=(
            settings.BGE_QUERY_INSTRUCTION
            if query_instruction is None
            else query_instruction
        ),
        cache_dir=cache_dir or settings.BGE_CACHE_DIR,
        local_files_only=(
            settings.BGE_LOCAL_FILES_ONLY
            if local_files_only is None
            else local_files_only
        ),
    )


def init_embeddings() -> BGEEmbeddings:
    """初始化默认 Embedding 实例。"""
    global _embeddings_instance
    _embeddings_instance = get_bge_embeddings()
    logger.info(f"✓ Embedding 初始化完成 (model: {settings.BGE_MODEL_NAME})")
    return _embeddings_instance


def get_default_embeddings() -> BGEEmbeddings:
    """获取默认 Embedding 实例。"""
    global _embeddings_instance
    if _embeddings_instance is None:
        with _embeddings_init_lock:
            if _embeddings_instance is None:
                _embeddings_instance = init_embeddings()
    return _embeddings_instance
