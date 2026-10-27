"""Rust rag_kernels 与纯 Python 实现的逐位一致性（parity）守门测试。

无编译扩展时整体跳过（回退路径另有覆盖）；扩展存在时必须全绿，
任何一项不一致都意味着排序语义漂移，禁止发布。
"""
import random

import pytest

rp = pytest.importorskip("rag_kernels", reason="Rust 扩展未编译，跳过 parity 测试")

from app.rag.retrieval.reranker import normalize_text  # noqa: E402
from app.rag.retrieval.sparse_retriever import (  # noqa: E402
    _lexical_match_score,
    _tokenize,
)

TEXT_SAMPLES = [
    "出差结束后最晚多久要提交报销单？",
    "员工号 E0011 的姓名和所属部门是什么？",
    "01_差旅报销制度_V2.0.docx",
    "高铁单程超过六个小时，可以买一等座吗？",
    "一线城市 P5 员工每晚住宿费上限是多少？",
    "全角空格　与半角 space 混合、标点！？～…·《》[]{}()",
    "员工工资明细表里有哪些字段？",
    "ABC-1234_rev2 与 P90003、E0024 混排",
    "。，！？；：（）【】仅标点",
    "",
]


def _random_cjk_texts(count: int, seed: int = 42):
    rng = random.Random(seed)
    words = ["报销", "住宿费", "餐补", "审批", "工号", "权限", "回滚", "灰度", "验收", "合同"]
    texts = []
    for _ in range(count):
        length = rng.randint(5, 60)
        text = "".join(rng.choice(words) for _ in range(length))
        if rng.random() < 0.3:
            text += f" E{rng.randint(1000, 9999)}"
        texts.append(text)
    return texts


class TestTokenizerParity:
    @pytest.mark.parametrize("text", TEXT_SAMPLES)
    def test_tokenizer_matches_python(self, text):
        assert rp.tokenize(text) == _tokenize(text)

    def test_tokenizer_random_texts(self):
        for text in _random_cjk_texts(200, seed=7):
            assert rp.tokenize(text) == _tokenize(text), text


class TestNormalizeParity:
    @pytest.mark.parametrize("text", TEXT_SAMPLES)
    def test_normalize_matches_python(self, text):
        assert rp.normalize(text) == normalize_text(text)


class TestBM25Parity:
    def test_scores_match_rank_bm25(self):
        from rank_bm25 import BM25Okapi

        rng = random.Random(1234)
        vocab = ["报销", "住宿", "餐补", "审批", "工号", "权限", "回滚", "unused"]
        corpus = [
            [rng.choice(vocab) for _ in range(rng.randint(1, 40))]
            for _ in range(120)
        ]
        # 刻意构造重复 token 的高频词（触发负 idf 地板路径）
        corpus.append(["报销"] * 70)

        query = [
            "报销", "餐补", "报销", "工号", "not-in-corpus", "审批",
        ]
        py_bm25 = BM25Okapi([list(doc) for doc in corpus])
        py_scores = py_bm25.get_scores(query)
        rs_scores = rp.bm25_scores([list(doc) for doc in corpus], list(query))
        assert len(py_scores) == len(rs_scores)
        for a, b in zip(py_scores, rs_scores):
            assert abs(a - b) < 1e-9, (a, b)

    def test_slice_order_matches_python_two_stage(self):
        """SparseIndex.search 的两段式结果须与 Python 复刻流程行序一致。"""
        from rank_bm25 import BM25Okapi

        texts = _random_cjk_texts(80, seed=99)
        texts = [t for t in texts if t]
        query = "报销住宿费餐补审批"
        limit = 40

        tokens = [_tokenize(t) for t in texts]
        query_tokens = _tokenize(query)
        global_scores = BM25Okapi(tokens).get_scores(query_tokens)
        order = sorted(range(len(global_scores)), key=lambda i: -global_scores[i])
        slice_indexes = []
        for idx in order[: max(1, limit)]:
            if global_scores[idx] <= 0.0:
                break
            slice_indexes.append(idx)
        expected_rows = list(slice_indexes)

        index = rp.SparseIndex(texts)
        hits = index.search(query, limit)
        assert [row_index for row_index, _, _ in hits] == expected_rows

        # 分数逐位对齐（切片内 BM25 + 词法分）
        slice_bm25 = BM25Okapi([tokens[i] for i in slice_indexes])
        slice_scores = slice_bm25.get_scores(query_tokens)
        for position, (row_index, bm25_score, lexical_score) in enumerate(hits):
            assert abs(bm25_score - slice_scores[position]) < 1e-9
            expected_lexical = _lexical_match_score(
                query_tokens, tokens[row_index], query, texts[row_index]
            )
            assert abs(lexical_score - expected_lexical) < 1e-9


class TestLexicalParity:
    def test_lexical_matches_python(self):
        pairs = [
            ("报销时限是什么", "报销时限说明文档"),
            ("E0011 是谁", "员工表 E0011 张三 运维部"),
            ("完全不相关", "南辕北辙的内容"),
            ("住宿费上限", "费用标准表 住宿费"),
        ]
        for query, row_text in pairs:
            assert abs(rp.lexical_score(query, row_text) -
                       _lexical_match_score(_tokenize(query), _tokenize(row_text), query, row_text)) < 1e-9

    def test_index_row_count(self):
        index = rp.SparseIndex(["a b", "c"])
        assert index.row_count == 2


class TestEmbedderParity:
    """Rust ONNX 内核 vs torch 的向量一致性（>0.999）与内核选择。"""

    def _torch_embedder(self):
        from app.core.embeddings import get_bge_embeddings

        return get_bge_embeddings()

    def _rust_embedder(self):
        from app.core.embeddings import RustBGEEmbeddings
        from app.config import settings

        return RustBGEEmbeddings(
            model_dir=settings.BGE_MODEL_NAME,
            max_length=settings.BGE_MAX_LENGTH,
            batch_size=settings.BGE_BATCH_SIZE,
            query_instruction=settings.BGE_QUERY_INSTRUCTION,
        )

    def test_query_cosine_parity(self):
        rust = self._rust_embedder()
        torch_emb = self._torch_embedder()
        queries = [
            "出差结束后最晚多久要提交报销单？",
            "员工号 E0011 的姓名和所属部门是什么？",
            "一线城市 P5 员工每晚住宿费上限是多少？",
            "哪些情况发生时应该立即停止扩容并启动回滚？",
            "ABC-1234_rev2 与 2024年01月、full-width　空格测试",
        ]
        tv = torch_emb.embed_queries(queries)
        rv = rust.embed_queries(queries)
        for r, t in zip(rv, tv):
            assert abs(sum(a * b for a, b in zip(r, t)) - 1.0) < 0.001

    def test_document_cosine_parity(self):
        rust = self._rust_embedder()
        torch_emb = self._torch_embedder()
        docs = [
            "差旅报销制度第3条：报销时限为出差结束后 30 日内。",
            "住宿费标准 P5 一线城市 600 元/晚，[Sheet] 费用标准表——第3列",
            "回滚条件: 数据库迁移不可逆时先回滚应用层。",
        ]
        td = torch_emb.embed_documents(docs)
        rd = rust.embed_documents(docs)
        for r, t in zip(rd, td):
            assert abs(sum(a * b for a, b in zip(r, t)) - 1.0) < 0.001

    def test_kernel_resolution_prefers_rust_when_available(self):
        from app.core.embeddings import _resolve_embedding_kernel
        from app.config import settings

        if settings.EMBEDDING_KERNEL == "torch":
            import pytest

            pytest.skip("EMBEDDING_KERNEL=torch 显式指定，跳过")
        assert _resolve_embedding_kernel() == "rust"

    def test_empty_inputs(self):
        rust = self._rust_embedder()
        assert rust.embed_documents([]) == []
        assert rust.embed_queries([]) == []


class TestFallbackMode:
    def test_python_fallback_still_importable(self):
        """Rust 存在时，纯 Python 实现仍可作为回退路径正常工作。"""
        from app.rag.retrieval import sparse_index as si

        assert si._RUST_SPARSE_AVAILABLE is True
        # 纯 Python 分词与词法打分仍可独立运行
        assert _tokenize("报销时限") == ["报销时限", "报销", "销时", "时限", "报销时", "销时限"]


# ==================== 重排内核 parity ====================

from app.rag.retrieval import reranker as rk  # noqa: E402
from app.rag.retrieval.base import RetrievalResult  # noqa: E402

RERANK_QUERIES = [
    "出差结束后最晚多久要提交报销单？",
    "员工号 E0011 的姓名和所属部门是什么？",
    "晚上十点以后从机场打车回去，这笔费用能报销吗？",
    "一线城市 P5 员工每晚住宿费上限是多少？",
    "哪些情况发生时应该立即停止扩容并启动回滚？",
    "连续输错几次密码会锁定账号，锁多久？",
    "帮我写一首诗",
    "采购金额在5万到20万之间，需要哪些审批人和材料？",
]


def _make_candidates(seed: int = 5, count: int = 60):
    rng = random.Random(seed)
    bodies = [
        "报销时限说明", "住宿费标准与上限", "餐补扣减规则", "审批流程与材料",
        "回滚触发条件", "账号锁定策略", "权限申请时长", "供应商准入核验",
    ]
    sources = [
        "01_差旅报销制度_V2.0.docx",
        "05_费用标准表.xlsx",
        "04_产品与账号常见问题.md",
    ]
    candidates = []
    for index in range(count):
        body = "。".join(rng.choice(bodies) for _ in range(rng.randint(2, 8)))
        if rng.random() < 0.2:
            body = f"[Sheet] 费用标准\n{body}"
        candidates.append(
            RetrievalResult(
                content=body,
                metadata={"source_name": rng.choice(sources), "document_id": f"doc{index % 3}"},
                score=round(rng.uniform(0.0, 1.5), 4),
                source=rng.choice(["dense", "sparse", "hybrid"]),
            )
        )
    return candidates


class TestKeywordExtractionParity:
    @pytest.mark.parametrize("query", RERANK_QUERIES + TEXT_SAMPLES)
    def test_query_keywords(self, query):
        assert rp.query_keywords(query) == rk.build_query_keywords(query)

    @pytest.mark.parametrize("query", RERANK_QUERIES + TEXT_SAMPLES)
    def test_query_focus_terms(self, query):
        assert rp.query_focus_terms(query) == rk.build_query_focus_terms(query)

    @pytest.mark.parametrize("query", RERANK_QUERIES + TEXT_SAMPLES)
    def test_extract_identifiers(self, query):
        assert rp.extract_identifiers(query) == rk.extract_identifier_terms(query)

    @pytest.mark.parametrize("query", RERANK_QUERIES + TEXT_SAMPLES)
    def test_identifier_query(self, query):
        assert rp.identifier_query(query) == rk.is_identifier_query(query)

    def test_random_identifier_strings(self):
        rng = random.Random(3)
        alphabet = "ABCXY0123456789-_ .张三"
        for _ in range(300):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 24)))
            assert rp.extract_identifiers(text) == rk.extract_identifier_terms(text), text
            assert rp.identifier_query(text) == rk.is_identifier_query(text), text


class TestRerankParity:
    @pytest.mark.parametrize("query", RERANK_QUERIES)
    def test_rerank_matches_python(self, monkeypatch, query):
        candidates = _make_candidates()
        monkeypatch.setattr(rk, "_RUST_RERANK_AVAILABLE", False)
        expected = rk.rerank_results(query, list(candidates), top_k=10)
        monkeypatch.setattr(rk, "_RUST_RERANK_AVAILABLE", True)
        actual = rk.rerank_results(query, list(candidates), top_k=10)

        assert len(expected) == len(actual)
        for exp, act in zip(expected, actual):
            assert exp.content == act.content
            assert abs(exp.score - act.score) < 1e-9, (exp.score, act.score)
            exp_signals = exp.metadata["_relevance_signals"]
            act_signals = act.metadata["_relevance_signals"]
            for key in (
                "keyword_coverage",
                "focus_coverage",
                "heading_coverage",
                "heading_focus_coverage",
            ):
                assert abs(exp_signals[key] - act_signals[key]) < 1e-9, key
            assert exp_signals["exact_match"] == act_signals["exact_match"]
            assert exp_signals["keyword_hits"] == act_signals["keyword_hits"]
            assert exp_signals["query_keywords"] == act_signals["query_keywords"]

    def test_rerank_random_candidates(self, monkeypatch):
        for seed in (11, 22, 33):
            query = RERANK_QUERIES[seed % len(RERANK_QUERIES)]
            candidates = _make_candidates(seed=seed)
            monkeypatch.setattr(rk, "_RUST_RERANK_AVAILABLE", False)
            expected = rk.rerank_results(query, list(candidates))
            monkeypatch.setattr(rk, "_RUST_RERANK_AVAILABLE", True)
            actual = rk.rerank_results(query, list(candidates))
            assert [c.content for c in expected] == [c.content for c in actual]
            for exp, act in zip(expected, actual):
                assert abs(exp.score - act.score) < 1e-9


class TestFuseParity:
    def _build_sets(self, seed: int = 9):
        rng = random.Random(seed)
        docs = ["docA", "docB", "docC"]
        sets = []
        for _ in range(3):
            items = []
            for _ in range(rng.randint(6, 14)):
                doc = rng.choice(docs)
                chunk = rng.randint(0, 6)
                items.append(
                    RetrievalResult(
                        content=f"内容{doc}-{chunk}" + ("X" if rng.random() < 0.1 else ""),
                        metadata={"document_id": doc, "chunk_index": chunk},
                        score=round(rng.uniform(0, 1), 4),
                        source="dense",
                    )
                )
            sets.append(items)
        return sets

    def test_fuse_matches_python(self, monkeypatch):
        result_sets = self._build_sets()
        weights = [0.5, 0.3, 0.2]
        labels = ["dense", "sparse", "route3"]

        monkeypatch.setattr(rk, "_RUST_RERANK_AVAILABLE", False)
        expected = rk.fuse_ranked_results(
            [list(s) for s in result_sets], weights=weights, source_labels=labels
        )
        monkeypatch.setattr(rk, "_RUST_RERANK_AVAILABLE", True)
        actual = rk.fuse_ranked_results(
            [list(s) for s in result_sets], weights=weights, source_labels=labels
        )

        assert len(expected) == len(actual)
        for exp, act in zip(expected, actual):
            assert exp.content == act.content
            assert abs(exp.score - act.score) < 1e-12
            assert exp.metadata["matched_sources"] == act.metadata["matched_sources"]
