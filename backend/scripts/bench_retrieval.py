"""检索链路延迟基准脚本（零文件 I/O 版）。

用法（在 backend/ 目录下执行）:
    PYTHONPATH=. python scripts/bench_retrieval.py
    PYTHONPATH=. python scripts/bench_retrieval.py --limit 10
    PYTHONPATH=. python scripts/bench_retrieval.py --no-cross-encoder   # 关精排对照
    PYTHONPATH=. python scripts/bench_retrieval.py | tee bench_out.json # 需留档时重定向

本脚本不读写任何文件、不接受路径参数：41 条查询内联自
docs/rrf_ablation_eval.json（更新数据集时同步此常量），结果仅打印 stdout。

指标: 每条查询端到端耗时（清缓存后计时）、P50/P95/均值、
      recall@k / MRR@10（依据 relevant_sources 判断命中）。
"""
import argparse
import json
import statistics
import time

# (id, query, relevant_sources) — 源: docs/rrf_ablation_eval.json
EVAL_QUERIES = [
    ("travel_01", "出差结束后最晚多久要提交报销单？", ["01_差旅报销制度_V2.0.docx"]),
    ("travel_02", "晚上十点以后从机场打车回去，这笔费用能报销吗？", ["01_差旅报销制度_V2.0.docx"]),
    ("travel_03", "高铁单程超过六个小时，可以买一等座吗？", ["01_差旅报销制度_V2.0.docx"]),
    ("travel_04", "电子发票重复打印后，能不能再报销一次？", ["01_差旅报销制度_V2.0.docx"]),
    ("travel_05", "酒店已经提供早餐，当天餐补应该扣多少钱？", ["01_差旅报销制度_V2.0.docx"]),
    ("travel_06", "几个人一起住酒店，住宿费怎么分摊报销？", ["01_差旅报销制度_V2.0.docx"]),
    ("travel_07", "出差期间因为个人原因多住几天，公司还承担这几天的住宿和餐补吗？", ["01_差旅报销制度_V2.0.docx"]),
    ("travel_08", "因为客户临时取消行程产生的退票费可以报吗，需要留什么材料？", ["01_差旅报销制度_V2.0.docx"]),
    ("procurement_01", "含税金额不超过5000元的采购由谁审批？", ["02_采购审批流程.docx"]),
    ("procurement_02", "采购金额在5万到20万之间，需要哪些审批人和材料？", ["02_采购审批流程.docx"]),
    ("procurement_03", "紧急采购可以先做，之后多长时间内补录申请？", ["02_采购审批流程.docx"]),
    ("procurement_04", "采购申请审批通过后，能不能先给供应商下订单？", ["02_采购审批流程.docx"]),
    ("procurement_05", "单一来源采购需要提供什么价格合理性证明？", ["02_采购审批流程.docx"]),
    ("procurement_06", "第一次合作的供应商在下单前要完成哪些准入核验？", ["02_采购审批流程.docx"]),
    ("procurement_07", "同一个供应商在30天内多次购买同类服务，金额怎么计算？", ["02_采购审批流程.docx"]),
    ("alpha_01", "项目 Alpha 常规生产发布窗口是每周什么时候？", ["03_项目Alpha上线手册.docx"]),
    ("alpha_02", "哪些情况发生时应该立即停止扩容并启动回滚？", ["03_项目Alpha上线手册.docx"]),
    ("alpha_03", "Alpha 生产回滚要求在多长时间内恢复？", ["03_项目Alpha上线手册.docx"]),
    ("alpha_04", "测试环境验证通过后，可以直接把版本发到生产吗？", ["03_项目Alpha上线手册.docx"]),
    ("alpha_05", "如果数据库迁移不可逆，发生故障时应该先回滚什么？", ["03_项目Alpha上线手册.docx"]),
    ("alpha_06", "生产全量发布后至少要观察多久才能结束首轮观察？", ["03_项目Alpha上线手册.docx"]),
    ("faq_01", "还能正常登录时，修改密码应该怎么操作？", ["04_产品与账号常见问题.md"]),
    ("faq_02", "管理员和系统管理员的权限边界有什么不同？", ["04_产品与账号常见问题.md"]),
    ("faq_03", "项目 Alpha 的生产环境访问权限最长可以申请多久？", ["04_产品与账号常见问题.md"]),
    ("faq_04", "连续输错几次密码会锁定账号，锁多久？", ["04_产品与账号常见问题.md"]),
    ("faq_05", "文档上传后为什么不能马上搜索到？", ["04_产品与账号常见问题.md"]),
    ("faq_06", "什么情况下回答不会显示知识库引用资料？", ["04_产品与账号常见问题.md"]),
    ("expense_01", "一线城市 P5 员工每晚住宿费上限是多少？", ["05_费用标准表.xlsx"]),
    ("expense_02", "新一线或省会城市的 P4 员工每天餐补标准是多少？", ["05_费用标准表.xlsx"]),
    ("expense_03", "酒店含早餐时，餐补按什么规则扣减？", ["05_费用标准表.xlsx"]),
    ("expense_04", "客户宴请可以算在差旅餐补里吗？", ["05_费用标准表.xlsx"]),
    ("expense_05", "出差地的公交、地铁和机场快线能否据实报销？", ["05_费用标准表.xlsx"]),
    ("employee_01", "员工号 E0011 的姓名和所属部门是什么？", ["06_企业内部信息总表.xlsx"]),
    ("employee_02", "采购部的部门邮箱是什么？", ["06_企业内部信息总表.xlsx"]),
    ("employee_03", "哪些产品适合运维部采购使用？", ["06_企业内部信息总表.xlsx"]),
    ("employee_04", "产品号 P90003 主要适合什么业务场景？", ["06_企业内部信息总表.xlsx"]),
    ("employee_05", "财务部负责费用审核和付款的员工是谁？", ["06_企业内部信息总表.xlsx"]),
    ("salary_01", "员工号 E0008 的实发工资是多少？", ["07_本月工资表.xlsx"]),
    ("salary_02", "员工工资明细表里有哪些字段？", ["07_本月工资表.xlsx"]),
    ("salary_03", "运维部员工 E0013 当前是什么状态，实发工资多少？", ["07_本月工资表.xlsx"]),
    ("salary_04", "员工号 E0024 的岗位和实发工资是什么？", ["07_本月工资表.xlsx"]),
]


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


def run_bench_http(args: argparse.Namespace) -> dict:
    """HTTP 模式：打 /api/v1/retrieval/rag（Rust 或 Python 后端），逐查询清缓存计时。"""
    import http.client
    from urllib.parse import urlparse

    parsed = urlparse(args.base_url)
    host, port = parsed.hostname, parsed.port or 80
    prefix = args.base_url.rstrip("/").split(host)[-1].lstrip(":0123456789") or ""
    base_prefix = prefix or "/api/v1"

    conn = http.client.HTTPConnection(host, port, timeout=120)

    def call(method, path, body=None, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        conn.request(method, base_prefix + path, json.dumps(body) if body else None, headers)
        resp = conn.getresponse()
        payload = resp.read().decode()
        return resp.status, json.loads(payload) if payload else {}

    pw = ""
    for env_path in (".env", "../backend/.env"):
        try:
            for line in open(env_path, encoding="utf-8"):
                if line.startswith("BOOTSTRAP_ADMIN_PASSWORD="):
                    pw = line.split("=", 1)[1].strip()
                    break
        except OSError:
            continue
        if pw:
            break
    assert pw, "未找到 BOOTSTRAP_ADMIN_PASSWORD"
    status, login = call("POST", "/auth/login", {"username": "admin", "password": pw})
    assert status == 200, f"登录失败: {status} {login}"
    token = login["access_token"]
    print(f"[bench-http] 目标 {host}:{port}{base_prefix} 登录成功，预热一次请求...")
    call("POST", "/retrieval/cache-clear", {}, token)
    call("POST", "/retrieval/rag", {"query": "预热", "top_k": args.top_k}, token)

    dataset = EVAL_QUERIES[: args.limit or None]
    records, rerank_stage_counter = [], {}
    for index, (item_id, query, relevant_sources) in enumerate(dataset, start=1):
        call("POST", "/retrieval/cache-clear", {}, token)
        t0 = time.perf_counter()
        status, rag = call("POST", "/retrieval/rag", {"query": query, "top_k": args.top_k}, token)
        elapsed = time.perf_counter() - t0
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
        print(f"[bench-http] {index}/{len(dataset)} {elapsed*1000:8.1f}ms "
              f"hit@{args.top_k}={'Y' if hit_rank else 'N'} q={query[:24]}")

    summary = _summarize(records, rerank_stage_counter, args)
    print("\n[bench-http] ===== 汇总 =====")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"},
                     ensure_ascii=False, indent=2))
    return summary


def _summarize(records, rerank_stage_counter, args) -> dict:
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

    return {
        "config": {
            "cross_encoder_enabled": False,
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


def run_bench(args: argparse.Namespace) -> dict:
    if args.base_url:
        return run_bench_http(args)
    from app.config import settings
    from app.storage.sqlite_metadata import init_metadata_db
    from app.storage.milvus_store import initialize_collections, warmup_collections
    from app.core.embeddings import get_default_embeddings
    from app.services.retrieval_service import (
        clear_rag_result_cache,
        run_rag_workflow,
    )

    dataset = EVAL_QUERIES[: args.limit or None]
    if args.no_cross_encoder:
        settings.CROSS_ENCODER_ENABLED = False

    print("[bench] 初始化元数据库与 Milvus collection...")
    init_metadata_db()
    if not initialize_collections():
        raise RuntimeError("Milvus collection 初始化失败，无法运行基准")
    warmup_collections()
    print("[bench] 预热 embedding 模型...")
    get_default_embeddings()
    print(
        f"[bench] 配置: cross_encoder={settings.CROSS_ENCODER_ENABLED}, "
        f"queries={len(dataset)}, top_k={args.top_k}, method={args.method}"
    )

    records = []
    rerank_stage_counter: dict = {}
    for index, (item_id, query, relevant_sources) in enumerate(dataset, start=1):
        clear_rag_result_cache()
        started_at = time.perf_counter()
        result = run_rag_workflow(
            query=query,
            top_k=args.top_k,
            retrieval_method=args.method,
            llm=None,
        )
        elapsed = time.perf_counter() - started_at

        results = result.results or []
        hit_rank = None
        for rank, entry in enumerate(results, start=1):
            if _is_hit(entry.metadata or {}, relevant_sources):
                hit_rank = rank
                break
            if rank >= 10:
                break
        for entry in results:
            stage = (entry.metadata or {}).get("rerank_stage")
            if stage:
                rerank_stage_counter[stage] = rerank_stage_counter.get(stage, 0) + 1

        records.append(
            {
                "id": item_id,
                "query": query,
                "seconds": round(elapsed, 4),
                "result_count": len(results),
                "hit_rank": hit_rank,
                "top_sources": [
                    (entry.metadata or {}).get("source_name") for entry in results[:3]
                ],
            }
        )
        print(
            f"[bench] {index}/{len(dataset)} {elapsed*1000:8.1f}ms "
            f"hit@{args.top_k}={'Y' if hit_rank else 'N'} q={query[:24]}"
        )

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
            "cross_encoder_enabled": settings.CROSS_ENCODER_ENABLED,
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
                sum(1.0 / item["hit_rank"] for item in records if item["hit_rank"])
                / total,
                4,
            )
            if total
            else 0.0,
        },
        "rerank_stages": rerank_stage_counter,
        "records": records,
    }

    print("\n[bench] ===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索延迟基准")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--method", type=str, default="hybrid")
    parser.add_argument("--no-cross-encoder", action="store_true")
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="HTTP 模式：如 http://127.0.0.1:8001（打 /api/v1/retrieval/rag）",
    )
    run_bench(parser.parse_args())


if __name__ == "__main__":
    main()
