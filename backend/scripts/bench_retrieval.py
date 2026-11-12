"""RAG 检索延迟基准（纯 HTTP 版，仅依赖 eks-server）。

用法（先启动 Rust 后端，如 cargo run --release）:
    python scripts/bench_retrieval.py
    python scripts/bench_retrieval.py --limit 10
    python scripts/bench_retrieval.py --base-url http://127.0.0.1:8000
    python scripts/bench_retrieval.py | tee bench_out.json   # 需留档时重定向

原 Python 进程内基准已随 Python 后端退役；本脚本打 /api/v1/retrieval/rag，
逐查询清缓存计时。查询集读取 docs/rrf_ablation_eval.json（与 eval/rag_acceptance 企业切片同源）。

完整 Python/Rust 对拍请用仓库根目录：
    python eval/rag_acceptance/run.py compare --system rust=http://127.0.0.1:8000 --ingest

指标: 每条查询端到端耗时、P50/P95/均值、recall@k / MRR@10（依据 relevant_sources 判断命中）。
"""
import argparse
import http.client
import json
import os
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse


def _eval_queries() -> list[tuple[str, str, list[str]]]:
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[2] / "docs" / "rrf_ablation_eval.json",
        here.parents[1] / "docs" / "rrf_ablation_eval.json",
    ):
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return [
                (item["id"], item["query"], list(item.get("relevant_sources") or []))
                for item in payload
            ]
    raise FileNotFoundError("找不到 docs/rrf_ablation_eval.json")


EVAL_QUERIES = _eval_queries()


def _source_stem(name: str) -> str:
    return str(name or "").rsplit(".", 1)[0].strip().lower()


def _is_hit(result_meta: dict, relevant_sources: list) -> bool:
    source_name = _source_stem(result_meta.get("source_name"))
    if not source_name:
        return False
    return any(
        relevant and (relevant in source_name or source_name in relevant)
        for relevant in (_source_stem(item) for item in relevant_sources)
    )


def _load_admin_password() -> str:
    """优先环境变量，其次 backend/.env（eks-server 与其共享同一配置文件）。"""
    from_env = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "").strip()
    if from_env:
        return from_env
    for env_path in ("../../backend/.env", "../backend/.env", ".env"):
        try:
            with open(env_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("BOOTSTRAP_ADMIN_PASSWORD="):
                        return line.split("=", 1)[1].strip()
        except OSError:
            continue
    return ""


def run_bench(args: argparse.Namespace) -> dict:
    parsed = urlparse(args.base_url)
    host, port = parsed.hostname, parsed.port or 80
    conn = http.client.HTTPConnection(host, port, timeout=120)

    def call(method, path, body=None, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        conn.request(method, args.prefix + path, json.dumps(body) if body else None, headers)
        resp = conn.getresponse()
        payload = resp.read().decode()
        return resp.status, json.loads(payload) if payload else {}

    pw = args.password or _load_admin_password()
    assert pw, "未找到 BOOTSTRAP_ADMIN_PASSWORD（环境变量或 backend/.env）"
    status, login = call("POST", "/auth/login", {"username": args.username, "password": pw})
    assert status == 200, f"登录失败: {status} {login}"
    token = login["access_token"]
    print(f"[bench] 目标 {host}:{port}{args.prefix} 登录成功，预热一次请求...")
    _, runtime = call("GET", "/chat/runtime-status", token=token)
    call("POST", "/retrieval/cache-clear", {}, token)
    call("POST", "/retrieval/rag", {"query": "预热", "top_k": args.top_k}, token)

    dataset = EVAL_QUERIES[: args.limit or None]
    records, rerank_stage_counter = [], {}
    for index, (item_id, query, relevant_sources) in enumerate(dataset, start=1):
        call("POST", "/retrieval/cache-clear", {}, token)
        t0 = time.perf_counter()
        status, rag = call("POST", "/retrieval/rag", {"query": query, "top_k": args.top_k}, token)
        elapsed = time.perf_counter() - t0
        assert status == 200, f"查询失败: {status} {rag}"
        results = rag.get("results") or []
        hit_rank = None
        for rank, entry in enumerate(results, start=1):
            if _is_hit(entry.get("metadata") or {}, relevant_sources):
                hit_rank = rank
                break
            if rank >= 10:
                break
        for entry in results:
            stage = (entry.get("metadata") or {}).get("rerank_stage")
            if stage:
                rerank_stage_counter[stage] = rerank_stage_counter.get(stage, 0) + 1
        records.append({
            "id": item_id, "query": query, "seconds": round(elapsed, 4),
            "result_count": len(results), "hit_rank": hit_rank,
            "top_sources": [(e.get("metadata") or {}).get("source_name") for e in results[:3]],
        })
        print(f"[bench] {index}/{len(dataset)} {elapsed*1000:8.1f}ms "
              f"hit@{args.top_k}={'Y' if hit_rank else 'N'} q={query[:24]}")

    seconds = [item["seconds"] for item in records]
    sorted_seconds = sorted(seconds)
    total = len(records)
    hits = sum(1 for item in records if item["hit_rank"] is not None)

    def percentile(fraction: float) -> float:
        if not sorted_seconds:
            return 0.0
        position = min(
            len(sorted_seconds) - 1,
            max(0, round(fraction * (len(sorted_seconds) - 1))),
        )
        return sorted_seconds[position]

    summary = {
        "config": {
            "cross_encoder_enabled": runtime.get("cross_encoder_enabled"),
            "cross_encoder_device": runtime.get("cross_encoder_device"),
            "query_count": total,
            "top_k": args.top_k,
            "method": args.method,
        },
        "latency": {
            "mean_seconds": round(statistics.mean(seconds), 4) if seconds else 0.0,
            "p50_seconds": round(percentile(0.50), 4),
            "p95_seconds": round(percentile(0.95), 4),
            "total_seconds": round(sum(seconds), 3),
        },
        "quality": {
            f"recall@{args.top_k}": round(hits / total, 4) if total else 0.0,
            "mrr@10": round(
                sum(1.0 / item["hit_rank"] for item in records if item["hit_rank"]) / total,
                4,
            ) if total else 0.0,
        },
        "rerank_stages": rerank_stage_counter,
        "records": records,
    }

    print("\n[bench] ===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索延迟基准（纯 HTTP，打 eks-server）")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--prefix", type=str, default="/api/v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--method", type=str, default="hybrid")
    parser.add_argument("--username", type=str, default="admin")
    parser.add_argument("--password", type=str, default="")
    run_bench(parser.parse_args())


if __name__ == "__main__":
    main()
