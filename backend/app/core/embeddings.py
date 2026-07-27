"""BGE 向量化模型配置"""
from typing import List, Optional

import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from app.config import settings
from app.utils.exceptions import EmbeddingException
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


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
        query = text or ""
        if self.query_instruction:
            query = f"{self.query_instruction}{query}"
        return self._embed_batch([query])[0]


_embeddings_instance: Optional[BGEEmbeddings] = None


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
        _embeddings_instance = init_embeddings()
    return _embeddings_instance
