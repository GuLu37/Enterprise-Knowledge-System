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

try:  # Rust ONNX 推理内核可选；缺失时回退 torch 实现
    from rag_kernels import RustEmbedder as _RustEmbedder

    _RUST_EMBEDDER_AVAILABLE = True
except Exception:  # pragma: no cover - 无编译扩展的环境
    _RustEmbedder = None
    _RUST_EMBEDDER_AVAILABLE = False


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


class RustBGEEmbeddings(Embeddings):
    """Rust ONNX 推理内核包装，接口与 BGEEmbeddings 完全一致。

    分词留在 Python（HF BertTokenizer 权威实现，id 逐位一致），
    张量组装 + ONNX 前向 + CLS 池化在 Rust 内核完成。
    torch 前向（查询 batch(3) 实测 ~82ms）→ ONNX Runtime fp32（~52ms），
    向量余弦 1.000000，由 parity 测试钉死。
    """

    def __init__(
        self,
        model_dir: str,
        max_length: int = 512,
        batch_size: int = 16,
        query_instruction: str = "",
    ):
        base = Path(model_dir)
        onnx_path = base / "onnx" / "model.onnx"
        if not onnx_path.exists():
            raise EmbeddingException(
                f"Rust embedding 内核缺少 ONNX 模型: {onnx_path}。"
                "请先运行导出流程（见 backend/native/README.md），或设置 EMBEDDING_KERNEL=torch"
            )
        self.model_name = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        logger.info(f"初始化 Rust BGE Embedding (ONNX fp32): {onnx_path}")
        try:
            _, AutoTokenizer, _ = _load_transformers_classes()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True
            )
            pad_token_id = int(
                getattr(self.tokenizer, "pad_token_id", None)
                or self.tokenizer.convert_tokens_to_ids("[PAD]")
            )
            self._inner = _RustEmbedder(str(onnx_path), pad_token_id)
        except EmbeddingException:
            raise
        except Exception as e:
            raise EmbeddingException(f"Rust BGE Embedding 初始化失败: {e}") from e
        logger.info("✓ Rust BGE Embedding 就绪 (vocab=%d)", self.tokenizer.vocab_size)

    def _embed_id_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        return self._inner.embed_ids(
            [list(map(int, ids)) for ids in encoded["input_ids"]]
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """向量化文档文本列表（分批，避免超长批内存尖峰）。"""
        cleaned = [text or "" for text in texts]
        embeddings: List[List[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start:start + self.batch_size]
            embeddings.extend(self._embed_id_batch(batch))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """向量化查询文本。"""
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """批量向量化检索查询（查询指令前缀在内核外处理，复用一次前向）。"""
        queries = [text or "" for text in texts]
        if self.query_instruction:
            queries = [f"{self.query_instruction}{query}" for query in queries]
        return self._embed_id_batch(queries)


def _resolve_embedding_kernel() -> str:
    """按配置与产物可用性选择推理内核: rust | torch。"""
    mode = (settings.EMBEDDING_KERNEL or "auto").strip().lower()
    if mode == "torch":
        return "torch"
    model_dir = Path(settings.BGE_MODEL_NAME)
    artifacts_ready = (
        model_dir / "onnx" / "model.onnx"
    ).exists() and (model_dir / "vocab.txt").exists()
    if _RUST_EMBEDDER_AVAILABLE and artifacts_ready:
        return "rust"
    if mode == "rust":
        logger.warning(
            "EMBEDDING_KERNEL=rust 但内核或 ONNX 产物不可用，回退 torch"
        )
    return "torch"


_embeddings_instance: Optional[Embeddings] = None
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


def init_embeddings() -> Embeddings:
    """初始化默认 Embedding 实例（Rust ONNX 内核优先，torch 回退）。"""
    global _embeddings_instance
    kernel = _resolve_embedding_kernel()
    if kernel == "rust":
        try:
            _embeddings_instance = RustBGEEmbeddings(
                model_dir=settings.BGE_MODEL_NAME,
                max_length=settings.BGE_MAX_LENGTH,
                batch_size=settings.BGE_BATCH_SIZE,
                query_instruction=settings.BGE_QUERY_INSTRUCTION,
            )
            logger.info(
                f"✓ Embedding 初始化完成 (kernel=rust, model: {settings.BGE_MODEL_NAME})"
            )
            return _embeddings_instance
        except Exception as exc:
            logger.warning("Rust embedding 内核初始化失败，回退 torch: %s", exc)
    _embeddings_instance = get_bge_embeddings()
    logger.info(
        f"✓ Embedding 初始化完成 (kernel=torch, model: {settings.BGE_MODEL_NAME})"
    )
    return _embeddings_instance


def get_default_embeddings() -> Embeddings:
    """获取默认 Embedding 实例。"""
    global _embeddings_instance
    if _embeddings_instance is None:
        with _embeddings_init_lock:
            if _embeddings_instance is None:
                _embeddings_instance = init_embeddings()
    return _embeddings_instance
