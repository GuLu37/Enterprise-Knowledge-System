#!/usr/bin/env python3
"""生成 500 道多跳检索题和 4 份朔方精密台账。

每条事实只写在一份台账里，跳数等于必须连读的文档数。
简单 200 题为 2 跳，中等 200 题为 3 跳，困难 100 题为 4 跳。
答案句和证据句都是台账原文，可逐字核对。
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC_DIR = REPO / "docs" / "rag_accuracy_demo"
QUERY_PATH = REPO / "eval" / "rag_acceptance" / "datasets" / "multihop" / "queries.jsonl"

DOC_PERSON = "27_朔方精密_人员台账.md"
DOC_SYSTEM = "28_朔方精密_系统台账.md"
DOC_RUNBOOK = "29_朔方精密_规程台账.md"
DOC_PHONE = "30_朔方精密_升级组台账.md"

SURNAMES = (
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳唐罗薛伍"
    "余米贝姚孟顾黄萧尹汪祁毛禹狄明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
    "程蔡田樊"
)
GIVEN_A = "清宁安和景行修远承德"
GIVEN_B = "川衡石泽林桐柏楠杉槐"


def people(n: int = 100) -> list[dict[str, str]]:
    surnames = list(SURNAMES)
    if len(surnames) != 100 or len(set(surnames)) != 100:
        raise SystemExit(f"姓氏应为 100 个互异字，当前 {len(surnames)}")
    if len(GIVEN_A) != 10 or len(GIVEN_B) != 10:
        raise SystemExit("名字用字应为各 10 个")
    rows = []
    for i in range(n):
        number = i + 1
        code = f"{number:03d}"
        name = f"{surnames[i]}{GIVEN_A[i // 10]}{GIVEN_B[i % 10]}"
        rows.append(
            {
                "name": name,
                "emp": f"EMP-S{code}",
                "sys": f"SYS-S{code}",
                "rb": f"RB-S{code}",
                "eg": f"EG-S{code}",
                "phone": f"0571-8801{number:04d}",
            }
        )
    names = [row["name"] for row in rows]
    if len(set(names)) != n:
        raise SystemExit("姓名出现重复")
    return rows


def line_person(row: dict[str, str]) -> str:
    return f"{row['name']}（{row['emp']}）负责的生产系统是 {row['sys']}。"


def line_system(row: dict[str, str]) -> str:
    return f"{row['sys']} 的值班规程是 {row['rb']}。"


def line_runbook(row: dict[str, str]) -> str:
    return f"{row['rb']} 的升级组是 {row['eg']}。"


def line_phone(row: dict[str, str]) -> str:
    return f"{row['eg']} 的夜间升级电话是 {row['phone']}。"


def write_docs(rows: list[dict[str, str]]) -> None:
    docs = {
        DOC_PERSON: (
            "# 朔方精密人员台账\n\n"
            "本台账只登记员工与其负责的生产系统。\n\n"
            + "\n".join(line_person(row) for row in rows)
            + "\n"
        ),
        DOC_SYSTEM: (
            "# 朔方精密系统台账\n\n"
            "本台账只登记生产系统与值班规程的对应关系。\n\n"
            + "\n".join(line_system(row) for row in rows)
            + "\n"
        ),
        DOC_RUNBOOK: (
            "# 朔方精密规程台账\n\n"
            "本台账只登记值班规程与升级组的对应关系。\n\n"
            + "\n".join(line_runbook(row) for row in rows)
            + "\n"
        ),
        DOC_PHONE: (
            "# 朔方精密升级组台账\n\n"
            "本台账只登记升级组的夜间升级电话。\n\n"
            + "\n".join(line_phone(row) for row in rows)
            + "\n"
        ),
    }
    for name, text in docs.items():
        (DOC_DIR / name).write_text(text, encoding="utf-8")


def question(row: dict[str, str], kind: str) -> dict[str, object]:
    number = int(row["emp"].rsplit("S", 1)[1])
    if kind == "2a":
        return {
            "id": f"mh2a_{number:03d}",
            "task": "multi_hop",
            "scenario": "多跳推理",
            "difficulty": "简单",
            "inspired_by": "HotpotQA",
            "query": f"{row['name']}负责的生产系统，值班规程编号是哪一份？",
            "relevant_sources": [DOC_PERSON, DOC_SYSTEM],
            "distractor_sources": [],
            "bait_sources": [],
            "answers": [line_system(row).rstrip("。")],
            "evidence": [line_person(row), line_system(row)],
        }
    if kind == "2b":
        return {
            "id": f"mh2b_{number:03d}",
            "task": "multi_hop",
            "scenario": "多跳推理",
            "difficulty": "简单",
            "inspired_by": "HotpotQA",
            "query": f"规程 {row['rb']} 夜间升级要拨打的电话是多少？",
            "relevant_sources": [DOC_RUNBOOK, DOC_PHONE],
            "distractor_sources": [],
            "bait_sources": [],
            "answers": [line_phone(row).rstrip("。")],
            "evidence": [line_runbook(row), line_phone(row)],
        }
    if kind == "3a":
        return {
            "id": f"mh3a_{number:03d}",
            "task": "multi_hop",
            "scenario": "多跳推理",
            "difficulty": "中等",
            "inspired_by": "HotpotQA",
            "query": f"{row['name']}负责的生产系统，升级组代号是什么？",
            "relevant_sources": [DOC_PERSON, DOC_SYSTEM, DOC_RUNBOOK],
            "distractor_sources": [],
            "bait_sources": [],
            "answers": [line_runbook(row).rstrip("。")],
            "evidence": [line_person(row), line_system(row), line_runbook(row)],
        }
    if kind == "3b":
        return {
            "id": f"mh3b_{number:03d}",
            "task": "multi_hop",
            "scenario": "多跳推理",
            "difficulty": "中等",
            "inspired_by": "HotpotQA",
            "query": f"系统 {row['sys']} 夜间升级要拨打的电话是多少？",
            "relevant_sources": [DOC_SYSTEM, DOC_RUNBOOK, DOC_PHONE],
            "distractor_sources": [],
            "bait_sources": [],
            "answers": [line_phone(row).rstrip("。")],
            "evidence": [line_system(row), line_runbook(row), line_phone(row)],
        }
    if kind == "4":
        return {
            "id": f"mh4_{number:03d}",
            "task": "multi_hop",
            "scenario": "多跳推理",
            "difficulty": "困难",
            "inspired_by": "HotpotQA",
            "query": f"{row['name']}负责的生产系统，夜间升级电话是多少？",
            "relevant_sources": [DOC_PERSON, DOC_SYSTEM, DOC_RUNBOOK, DOC_PHONE],
            "distractor_sources": [],
            "bait_sources": [],
            "answers": [line_phone(row).rstrip("。")],
            "evidence": [line_person(row), line_system(row), line_runbook(row), line_phone(row)],
        }
    raise SystemExit(kind)


def write_queries(rows: list[dict[str, str]]) -> None:
    items = []
    for kind in ("2a", "2b", "3a", "3b", "4"):
        items.extend(question(row, kind) for row in rows)
    if len(items) != 500:
        raise SystemExit(f"题量应为 500，当前 {len(items)}")
    QUERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    QUERY_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    rows = people()
    write_docs(rows)
    write_queries(rows)
    print(f"已写 4 份台账和 {QUERY_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
