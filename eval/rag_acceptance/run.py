#!/usr/bin/env python3
"""EKS RAG 验收评测跑手（标准库，无第三方依赖）。

用法：
  python eval/rag_acceptance/run.py check
  python eval/rag_acceptance/run.py compare \\
      --system rust=http://127.0.0.1:8000 \\
      --system python=http://127.0.0.1:8001 \\
      --methods dense,sparse,hybrid --ingest --out eval/rag_acceptance/runs/latest.json

对拍两端共用 /api/v1/retrieval/search/{dense,sparse,hybrid}。
Rust 独有 /retrieval/rag 在方法列表含 rag 且接口 404 时自动跳过。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = Path(__file__).resolve().parent / "suite.json"
STOPWORDS = {
    "的", "了", "是", "在", "吗", "呢", "吧", "啊", "和", "与", "及", "或",
    "什么", "怎么", "多少", "哪些", "哪个", "是否", "能否", "可以", "请问",
    "根据", "基于", "查询", "检索",
}


def load_suite() -> dict[str, Any]:
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def repo_path(rel: str) -> Path:
    return (REPO_ROOT / rel).resolve()


def stem(name: str) -> str:
    text = str(name or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[0]
    return text.lower()


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    buf: list[str] = []
    cjk: list[str] = []

    def flush_latin() -> None:
        if buf:
            token = "".join(buf)
            if token not in STOPWORDS:
                tokens.append(token)
            buf.clear()

    def flush_cjk() -> None:
        if not cjk:
            return
        word = "".join(cjk)
        if word not in STOPWORDS:
            tokens.append(word)
        if len(cjk) > 1:
            for i in range(len(cjk) - 1):
                gram = cjk[i] + cjk[i + 1]
                if gram not in STOPWORDS:
                    tokens.append(gram)
        cjk.clear()

    for ch in lowered:
        if ch.isascii() and ch.isalnum():
            flush_cjk()
            buf.append(ch)
        elif "\u4e00" <= ch <= "\u9fff":
            flush_latin()
            cjk.append(ch)
        else:
            flush_latin()
            flush_cjk()
    flush_latin()
    flush_cjk()
    return tokens


def load_queries(slice_name: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = repo_path(spec["queries"])
    fmt = spec.get("query_format", "jsonl")
    if fmt == "json_array":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    queries = []
    for item in raw:
        queries.append(
            {
                "id": item["id"],
                "slice": slice_name,
                "task": item.get("task") or item.get("category") or "enterprise",
                "inspired_by": item.get("inspired_by", "enterprise domain RAG"),
                "query": item["query"],
                "relevant_sources": list(item.get("relevant_sources") or []),
                "bait_sources": list(item.get("bait_sources") or []),
                "answers": list(item.get("answers") or []),
            }
        )
    return queries


def corpus_files(spec: dict[str, Any]) -> list[Path]:
    pattern = spec["corpus_glob"]
    files = sorted(REPO_ROOT.glob(pattern))
    return [path for path in files if path.is_file() and not path.name.startswith(".")]


def source_hit(result: dict[str, Any], relevant: list[str]) -> bool:
    meta = result.get("metadata") or {}
    name = stem(meta.get("source_name") or result.get("source") or "")
    if not name:
        return False
    return any(rel and (rel in name or name in rel) for rel in (stem(item) for item in relevant))


def match_sources(
    results: list[dict[str, Any]], relevant: list[str], cap: int
) -> tuple[list[int], list[str]]:
    ranks: list[int] = []
    names: list[str] = []
    seen: set[str] = set()
    for rank, item in enumerate(results[:cap], start=1):
        if not source_hit(item, relevant):
            continue
        meta = item.get("metadata") or {}
        name = stem(meta.get("source_name") or item.get("source") or "")
        if name in seen:
            continue
        seen.add(name)
        ranks.append(rank)
        names.append(name)
    return ranks, names


def dcg(ranks: list[int], k: int) -> float:
    score = 0.0
    for rank in ranks:
        if rank <= k:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(ranks: list[int], relevant_count: int, k: int) -> float:
    if relevant_count <= 0:
        return 0.0
    ideal = dcg(list(range(1, min(relevant_count, k) + 1)), k)
    if ideal <= 0:
        return 0.0
    return dcg(ranks, k) / ideal


def summarize_records(records: list[dict[str, Any]], k_values: list[int], top_k: int) -> dict[str, Any]:
    total = len(records)
    answerable = [item for item in records if item["task"] != "unanswerable"]
    unanswerable = [item for item in records if item["task"] == "unanswerable"]
    quality: dict[str, float] = {}
    for k in k_values:
        if answerable:
            hits = sum(1 for item in answerable if any(rank <= k for rank in item["hit_ranks"]))
            recalls = []
            ndcgs = []
            for item in answerable:
                gold = max(1, item["relevant_count"])
                found = len([rank for rank in item["hit_ranks"] if rank <= k])
                recalls.append(min(1.0, found / gold))
                ndcgs.append(ndcg_at_k(item["hit_ranks"], item["relevant_count"], k))
            mrr = sum(
                (1.0 / min(item["hit_ranks"]) if item["hit_ranks"] and min(item["hit_ranks"]) <= k else 0.0)
                for item in answerable
            ) / len(answerable)
            quality[f"hit@{k}"] = round(hits / len(answerable), 4)
            quality[f"recall@{k}"] = round(sum(recalls) / len(recalls), 4)
            quality[f"mrr@{k}"] = round(mrr, 4)
            quality[f"ndcg@{k}"] = round(sum(ndcgs) / len(ndcgs), 4)
        if unanswerable:
            rejected = sum(1 for item in unanswerable if item.get("bait_hit_rank") is None or item["bait_hit_rank"] > k)
            quality[f"reject@{k}"] = round(rejected / len(unanswerable), 4)

    latencies = [item["seconds"] for item in records]
    latencies_sorted = sorted(latencies)

    def pct(fraction: float) -> float:
        if not latencies_sorted:
            return 0.0
        idx = min(len(latencies_sorted) - 1, max(0, round(fraction * (len(latencies_sorted) - 1))))
        return latencies_sorted[idx]

    by_task: dict[str, dict[str, Any]] = {}
    for item in records:
        bucket = by_task.setdefault(item["task"], {"n": 0, "hits": 0})
        bucket["n"] += 1
        if item["task"] == "unanswerable":
            bucket["hits"] += int(item.get("bait_hit_rank") is None)
        else:
            bucket["hits"] += int(bool(item["hit_ranks"]))
    for bucket in by_task.values():
        bucket["hit_rate"] = round(bucket["hits"] / bucket["n"], 4) if bucket["n"] else 0.0

    return {
        "n": total,
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "quality": quality,
        "latency": {
            "mean_ms": round(1000 * statistics.mean(latencies), 1) if latencies else 0.0,
            "p50_ms": round(1000 * pct(0.50), 1),
            "p95_ms": round(1000 * pct(0.95), 1),
            "total_s": round(sum(latencies), 3),
        },
        "by_task": by_task,
        "top_k": top_k,
    }


def _admin_password(explicit: str) -> str:
    if explicit:
        return explicit
    env = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "").strip()
    if env:
        return env
    for rel in ("backend/.env", ".env"):
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("BOOTSTRAP_ADMIN_PASSWORD="):
                return line.split("=", 1)[1].strip()
    return ""


class ApiClient:
    def __init__(self, base_url: str, prefix: str = "/api/v1") -> None:
        parsed = urlparse(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.tls = parsed.scheme == "https"
        self.prefix = prefix.rstrip("/")
        self.token = ""
        self.username = ""
        self.password = ""

    def _conn(self, timeout: int = 300):
        if self.tls:
            return HTTPSConnection(self.host, self.port, timeout=timeout)
        return HTTPConnection(self.host, self.port, timeout=timeout)

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        raw: bytes | None = None,
        headers: dict | None = None,
        retry_auth: bool = True,
    ) -> tuple[int, Any]:
        hdrs = {"Accept": "application/json"}
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
        payload = raw
        if body is not None:
            hdrs["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        if headers:
            hdrs.update(headers)
        conn = self._conn()
        try:
            conn.request(method, self.prefix + path, payload, hdrs)
            resp = conn.getresponse()
            renewed = resp.getheader("X-Access-Token") or resp.getheader("x-access-token")
            if renewed:
                self.token = renewed
            text = resp.read().decode("utf-8")
            status = resp.status
        finally:
            conn.close()
        data: Any = {}
        if text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"raw": text}
        if status == 401 and retry_auth and path != "/auth/login" and self.username:
            self.login(self.username, self.password)
            return self.request(method, path, body=body, raw=raw, headers=headers, retry_auth=False)
        return status, data

    def login(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        status, payload = self.request(
            "POST", "/auth/login", {"username": username, "password": password}, retry_auth=False
        )
        if status != 200 or not payload.get("access_token"):
            raise SystemExit(f"登录失败 {status}: {payload}")
        self.token = payload["access_token"]

    def search(self, method: str, query: str, top_k: int) -> tuple[int, dict[str, Any], float]:
        path = "/retrieval/rag" if method == "rag" else f"/retrieval/search/{method}"
        t0 = time.perf_counter()
        status, payload = self.request("POST", path, {"query": query, "top_k": top_k})
        return status, payload, time.perf_counter() - t0

    def list_documents(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        skip = 0
        while True:
            status, payload = self.request("GET", f"/documents/list?skip={skip}&limit=100")
            if status == 404:
                status, payload = self.request("GET", f"/documents?skip={skip}&limit=100")
            if status != 200:
                raise RuntimeError(f"文档列表失败 {status}: {payload}")
            batch = payload.get("documents") or []
            docs.extend(batch)
            if len(batch) < 100:
                break
            skip += 100
        return docs

    def upload(self, path: Path) -> dict[str, Any]:
        boundary = f"----eks{uuid.uuid4().hex}"
        filename = path.name
        data = path.read_bytes()
        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
        tail = f"\r\n--{boundary}--\r\n".encode("ascii")
        status, payload = self.request(
            "POST",
            "/documents/upload",
            raw=head + data + tail,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        if status not in (200, 201, 202):
            raise RuntimeError(f"上传失败 {filename} {status}: {payload}")
        return payload

    def wait_ready(self, filenames: set[str], timeout_s: int = 1200) -> None:
        wanted = {path.name for path in (Path(name) for name in filenames)}
        deadline = time.time() + timeout_s
        last_note = ""
        while time.time() < deadline:
            listed = self.list_documents()
            by_name = {item.get("original_filename"): item for item in listed}
            pending = []
            failed = []
            for name in wanted:
                row = by_name.get(name)
                status = (row or {}).get("status")
                if status == "ready":
                    continue
                if status in {"failed", "error"}:
                    failed.append(f"{name}:{status}")
                    continue
                pending.append(f"{name}:{status or 'missing'}")
            if failed:
                raise RuntimeError(f"文档入库失败: {failed}")
            if not pending:
                return
            note = f"等待 ready: {', '.join(pending)}"
            if note != last_note:
                print(f"[ingest] {note}")
                last_note = note
            time.sleep(2.0)
        raise TimeoutError(f"文档未在 {timeout_s}s 内变为 ready: {sorted(pending)}")


def ingest_slice(client: ApiClient, spec: dict[str, Any]) -> None:
    files = corpus_files(spec)
    if not files:
        raise SystemExit(f"语料为空: {spec['corpus_glob']}")
    listed = {item.get("original_filename"): item for item in client.list_documents()}
    uploaded = []
    for path in files:
        row = listed.get(path.name)
        if row and row.get("status") == "ready":
            continue
        if not row:
            print(f"[ingest] 上传 {path.name}")
            client.upload(path)
            uploaded.append(path.name)
        client.wait_ready({path.name})
        listed[path.name] = {"original_filename": path.name, "status": "ready"}
    if uploaded:
        print(f"[ingest] 新上传 {len(uploaded)} 份，已全部 ready")
    else:
        print(f"[ingest] {len(files)} 份语料已在库中")


def evaluate_system(
    name: str,
    client: ApiClient,
    queries: list[dict[str, Any]],
    methods: list[str],
    top_k: int,
    k_values: list[int],
    concurrency: int,
) -> dict[str, Any]:
    status, runtime = client.request("GET", "/chat/runtime-status")
    runtime = runtime if status == 200 else {}
    client.request("POST", "/retrieval/cache-clear", {})

    available_methods = []
    for method in methods:
        status, payload, _ = client.search(method, "预热", top_k)
        if status == 404 and method == "rag":
            print(f"[{name}] 跳过 rag（该端点不存在，通常是 Python）")
            continue
        if status != 200:
            raise SystemExit(f"[{name}] {method} 预热失败 {status}: {payload}")
        available_methods.append(method)

    out_methods: dict[str, Any] = {}
    for method in available_methods:
        records = []

        def run_one(item: dict[str, Any]) -> dict[str, Any]:
            client.request("POST", "/retrieval/cache-clear", {})
            status, payload, elapsed = client.search(method, item["query"], max(top_k, 10))
            if status != 200:
                raise RuntimeError(f"{item['id']} {method} 失败 {status}: {payload}")
            results = payload.get("results") or []
            relevant = item["relevant_sources"]
            bait = item["bait_sources"]
            hit_ranks, _ = match_sources(results, relevant, 10) if relevant else ([], [])
            bait_ranks, _ = match_sources(results, bait, 10) if bait else ([], [])
            return {
                "id": item["id"],
                "slice": item["slice"],
                "task": item["task"],
                "query": item["query"],
                "seconds": round(elapsed, 4),
                "result_count": len(results),
                "relevant_count": len(relevant),
                "hit_ranks": hit_ranks,
                "bait_hit_rank": bait_ranks[0] if bait_ranks else None,
                "top_sources": [
                    (entry.get("metadata") or {}).get("source_name") for entry in results[:3]
                ],
            }

        if concurrency <= 1:
            for index, item in enumerate(queries, start=1):
                record = run_one(item)
                records.append(record)
                mark = "Y" if (record["hit_ranks"] or item["task"] == "unanswerable") else "N"
                print(f"[{name}/{method}] {index}/{len(queries)} {record['seconds']*1000:7.1f}ms hit={mark} {item['id']}")
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = {pool.submit(run_one, item): item for item in queries}
                for index, fut in enumerate(as_completed(futs), start=1):
                    record = fut.result()
                    records.append(record)
                    print(f"[{name}/{method}] {index}/{len(queries)} done {record['id']}")
            records.sort(key=lambda item: item["id"])

        summary = summarize_records(records, k_values, top_k)
        qps = (len(records) / summary["latency"]["total_s"]) if summary["latency"]["total_s"] else 0.0
        summary["qps"] = round(qps, 3)
        out_methods[method] = {"summary": summary, "records": records}
        print(
            f"[{name}/{method}] hit@{top_k}={summary['quality'].get(f'hit@{top_k}')} "
            f"mrr@10={summary['quality'].get('mrr@10')} "
            f"p95={summary['latency']['p95_ms']}ms"
        )
    return {
        "name": name,
        "runtime": {
            "cross_encoder_enabled": runtime.get("cross_encoder_enabled"),
            "cross_encoder_device": runtime.get("cross_encoder_device"),
        },
        "methods": out_methods,
    }


def compare_table(results: list[dict[str, Any]], method: str, metric_keys: list[str]) -> str:
    header = ["系统", *metric_keys, "p50(ms)", "p95(ms)", "QPS"]
    lines = [
        f"### {method}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for item in results:
        block = item["methods"].get(method)
        if not block:
            continue
        summary = block["summary"]
        row = [item["name"]]
        for key in metric_keys:
            row.append(str(summary["quality"].get(key, "—")))
        row.extend(
            [
                str(summary["latency"]["p50_ms"]),
                str(summary["latency"]["p95_ms"]),
                str(summary.get("qps", "—")),
            ]
        )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def cmd_check(args: argparse.Namespace) -> int:
    suite = load_suite()
    errors: list[str] = []
    seen_ids: set[str] = set()
    counts = {"queries": 0, "docs": 0, "slices": {}}
    for slice_name, spec in suite["slices"].items():
        files = corpus_files(spec)
        if not files:
            errors.append(f"{slice_name}: 语料不存在 {spec['corpus_glob']}")
        names = {path.name for path in files}
        counts["docs"] += len(files)
        queries = load_queries(slice_name, spec)
        counts["slices"][slice_name] = {"docs": len(files), "queries": len(queries)}
        counts["queries"] += len(queries)
        texts = {}
        for path in files:
            if path.suffix.lower() in {".md", ".txt"}:
                texts[path.name] = path.read_text(encoding="utf-8")
        for item in queries:
            qid = item["id"]
            if qid in seen_ids:
                errors.append(f"重复 query id: {qid}")
            seen_ids.add(qid)
            if not item["query"].strip():
                errors.append(f"{qid}: 空 query")
            for src in item["relevant_sources"] + item["bait_sources"]:
                if src not in names:
                    errors.append(f"{qid}: 找不到语料 {src}")
            if item["task"] == "unanswerable":
                if item["relevant_sources"]:
                    errors.append(f"{qid}: unanswerable 不应带 relevant_sources")
                continue
            if not item["relevant_sources"]:
                errors.append(f"{qid}: 可答题缺少 relevant_sources")
                continue
            gold_text = "\n".join(texts[src] for src in item["relevant_sources"] if src in texts)
            if not gold_text:
                continue  # binary enterprise docs
            q_tokens = {tok for tok in tokenize(item["query"]) if len(tok) >= 2}
            g_tokens = set(tokenize(gold_text))
            if q_tokens and q_tokens.isdisjoint(g_tokens):
                errors.append(f"{qid}: 与 gold 文档无词法重叠，GT 可能标错")

    # 指标函数钉扎
    fake_ranks = [1]
    if abs(ndcg_at_k(fake_ranks, 1, 5) - 1.0) > 1e-9:
        errors.append("nDCG 自检失败")
    if abs(ndcg_at_k([2], 1, 5) - (1 / math.log2(3))) > 1e-9:
        errors.append("nDCG@2 自检失败")

    if errors:
        print("数据集检查失败：")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("数据集检查通过")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    if cmd_check(args) != 0:
        return 1
    suite = load_suite()
    slices = [item.strip() for item in args.slices.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    queries: list[dict[str, Any]] = []
    specs = []
    for slice_name in slices:
        spec = suite["slices"][slice_name]
        specs.append(spec)
        queries.extend(load_queries(slice_name, spec))
    if args.limit:
        queries = queries[: args.limit]

    systems = []
    for item in args.system:
        if "=" not in item:
            raise SystemExit("--system 格式为 name=url，例如 rust=http://127.0.0.1:8000")
        name, url = item.split("=", 1)
        systems.append((name, url.rstrip("/")))

    password = _admin_password(args.password)
    if not password:
        raise SystemExit("未找到 BOOTSTRAP_ADMIN_PASSWORD")

    results = []
    for name, url in systems:
        client = ApiClient(url, args.prefix)
        client.login(args.username, password)
        if args.ingest:
            for spec in specs:
                ingest_slice(client, spec)
        print(f"\n===== {name} {url} n={len(queries)} =====")
        results.append(
            evaluate_system(
                name,
                client,
                queries,
                methods,
                args.top_k,
                suite["k_values"],
                args.concurrency,
            )
        )

    report = {
        "suite": suite["name"],
        "version": suite["version"],
        "n_queries": len(queries),
        "slices": slices,
        "systems": results,
    }
    metric_keys = [f"hit@{args.top_k}", f"recall@{args.top_k}", "mrr@10", "ndcg@10", "reject@5"]
    markdown = [f"# {suite['name']} 对拍报告", "", f"查询数 {len(queries)}；切片 {', '.join(slices)}", ""]
    for method in methods:
        markdown.append(compare_table(results, method, metric_keys))
    text = "\n".join(markdown)
    print("\n" + text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = out.with_suffix(".md")
        md_path.write_text(text, encoding="utf-8")
        print(f"已写入 {out} 与 {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EKS RAG 验收评测")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="离线检查语料、GT 与指标函数")
    cmp = sub.add_parser("compare", help="对一个或多个后端跑准确度 + 延迟")
    cmp.add_argument("--system", action="append", required=True, help="name=url，可重复")
    cmp.add_argument("--slices", default="enterprise,standard")
    cmp.add_argument("--methods", default="dense,sparse,hybrid")
    cmp.add_argument("--top-k", type=int, default=5)
    cmp.add_argument("--limit", type=int, default=0)
    cmp.add_argument("--concurrency", type=int, default=1)
    cmp.add_argument("--ingest", action="store_true")
    cmp.add_argument("--prefix", default="/api/v1")
    cmp.add_argument("--username", default="admin")
    cmp.add_argument("--password", default="")
    cmp.add_argument("--out", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
