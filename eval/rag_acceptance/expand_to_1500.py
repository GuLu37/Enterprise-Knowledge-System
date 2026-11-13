#!/usr/bin/env python3
"""把评测补到 1500 题。原 1041 题不动。

新增 459 题，使 13 个场景里 12 个落到 100 题，多跳推理落到 300 题。
朔方精密语料仍是短台账：多跳 4 份，其余场景再加 9 份。
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC_DIR = REPO / "docs" / "rag_accuracy_demo"
QUERY_PATH = REPO / "eval" / "rag_acceptance" / "datasets" / "supplement" / "queries.jsonl"

DOC_PERSON = "27_朔方精密_人员台账.md"
DOC_SYSTEM = "28_朔方精密_系统台账.md"
DOC_RUNBOOK = "29_朔方精密_规程台账.md"
DOC_PHONE = "30_朔方精密_升级组台账.md"
DOC_FACT = "31_朔方精密_单跳口径.md"
DOC_DOMAIN = "32_朔方精密_领域与声明.md"
DOC_TABLE = "33_朔方精密_备件价格表.md"
DOC_REFUSE = "34_朔方精密_未记载清单.md"
DOC_LONG = "35_朔方精密_巡检长文.md"
DOC_NOISE = "36_朔方精密_噪声夹注.md"
DOC_CURRENT = "37_朔方精密_现行汇刊.md"
DOC_OBSOLETE = "38_朔方精密_废止汇刊.md"
DOC_BI = "39_朔方精密_双语条款.md"

SURNAMES = (
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳唐罗薛伍"
    "余米贝姚孟顾黄萧尹汪祁毛禹狄明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
    "程蔡田樊"
)
GIVEN_A = "清宁安和景行修远承德"
GIVEN_B = "川衡石泽林桐柏楠杉槐"
PLACES = [
    "门禁", "食堂", "仓库", "码头", "实验室", "更衣室", "装卸台", "消防栓",
    "电梯", "停车场", "会议室", "工具间", "冷却塔", "空压站", "配电房", "污水站",
    "危化柜", "传达室", "医务室", "档案室", "收发室", "车棚", "门岗",
]
FILLER = (
    "本段只记当班是否到岗、通道是否畅通、照明是否完好。"
    "未写明针点编号的句子不能当作正式耗时。"
    "口头交代、便利贴和上一班闲谈都不改写正式句子。"
)


def people(n: int = 100) -> list[dict[str, str]]:
    surnames = list(SURNAMES)
    if len(surnames) != 100 or len(set(surnames)) != 100:
        raise SystemExit(f"姓氏应为 100 个互异字，当前 {len(surnames)}")
    rows = []
    for i in range(n):
        number = i + 1
        code = f"{number:03d}"
        rows.append(
            {
                "name": f"{surnames[i]}{GIVEN_A[i // 10]}{GIVEN_B[i % 10]}",
                "emp": f"EMP-S{code}",
                "sys": f"SYS-S{code}",
                "rb": f"RB-S{code}",
                "eg": f"EG-S{code}",
                "phone": f"0571-8801{number:04d}",
            }
        )
    return rows


def line_person(row: dict[str, str]) -> str:
    return f"{row['name']}（{row['emp']}）负责的生产系统是 {row['sys']}。"


def line_system(row: dict[str, str]) -> str:
    return f"{row['sys']} 的值班规程是 {row['rb']}。"


def line_runbook(row: dict[str, str]) -> str:
    return f"{row['rb']} 的升级组是 {row['eg']}。"


def line_phone(row: dict[str, str]) -> str:
    return f"{row['eg']} 的夜间升级电话是 {row['phone']}。"


def write_text(name: str, text: str) -> None:
    (DOC_DIR / name).write_text(text, encoding="utf-8")


def write_ledgers(rows: list[dict[str, str]]) -> None:
    write_text(
        DOC_PERSON,
        "# 朔方精密人员台账\n\n本台账只登记员工与其负责的生产系统。\n\n"
        + "\n".join(line_person(row) for row in rows)
        + "\n",
    )
    write_text(
        DOC_SYSTEM,
        "# 朔方精密系统台账\n\n本台账只登记生产系统与值班规程的对应关系。\n\n"
        + "\n".join(line_system(row) for row in rows)
        + "\n",
    )
    write_text(
        DOC_RUNBOOK,
        "# 朔方精密规程台账\n\n本台账只登记值班规程与升级组的对应关系。\n\n"
        + "\n".join(line_runbook(row) for row in rows)
        + "\n",
    )
    write_text(
        DOC_PHONE,
        "# 朔方精密升级组台账\n\n本台账只登记升级组的夜间升级电话。\n\n"
        + "\n".join(line_phone(row) for row in rows)
        + "\n",
    )


def hop_question(row: dict[str, str], kind: str) -> dict[str, object]:
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


def base_question(**kwargs: object) -> dict[str, object]:
    item = {
        "distractor_sources": [],
        "bait_sources": [],
        "inspired_by": "enterprise ledger",
    }
    item.update(kwargs)
    return item


def build_rest(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    fact_lines = ["# 朔方精密单跳口径\n", "本文件只登记各点位的通行核验耗时。\n"]
    for i, place in enumerate(PLACES):
        seconds = 12 + i
        code = f"{i + 1:02d}"
        sentence = f"朔方{place}的通行核验耗时是 {seconds} 秒。"
        fact_lines.append(sentence)
        items.append(
            base_question(
                id=f"sf_sh_{code}",
                task="single_hop",
                scenario="单跳事实",
                difficulty="简单",
                query=f"朔方{place}的通行核验耗时是多少秒？",
                relevant_sources=[DOC_FACT],
                answers=[f"{seconds} 秒"],
                evidence=[sentence],
            )
        )
        items.append(
            base_question(
                id=f"sf_fz_{code}",
                task="paraphrase",
                scenario="模糊查询",
                difficulty="中等",
                query=f"进朔方{place}，核验大概要花多少时间？",
                relevant_sources=[DOC_FACT],
                answers=[f"{seconds} 秒"],
                evidence=[sentence],
            )
        )
    write_text(DOC_FACT, "\n".join(fact_lines) + "\n")

    domain_lines = [
        "# 朔方精密领域与声明\n",
        "本文件只登记两条领域口径和焊丝熔点声明。\n",
        "朔方精密进项专票须在开具后 150 日内完成勾选。",
        "朔方精密 SEV-1 要在 8 分钟内拉起战室。",
    ]
    items.append(
        base_question(
            id="sf_dm_01",
            task="domain_faq",
            scenario="领域问答",
            difficulty="简单",
            query="朔方精密的进项专票要在开具后多少日内勾选？",
            relevant_sources=[DOC_DOMAIN],
            answers=["150 日内"],
            evidence=["朔方精密进项专票须在开具后 150 日内完成勾选。"],
        )
    )
    items.append(
        base_question(
            id="sf_dm_02",
            task="domain_faq",
            scenario="领域问答",
            difficulty="简单",
            query="朔方精密 SEV-1 要在多长时间内拉起战室？",
            relevant_sources=[DOC_DOMAIN],
            answers=["8 分钟内拉起战室"],
            evidence=["朔方精密 SEV-1 要在 8 分钟内拉起战室。"],
        )
    )
    for i in range(24):
        code = f"SF-WLD-{i + 1:02d}"
        degree = 610 + i * 3
        sentence = f"霜白焊丝 {code} 的熔点是 {degree} 摄氏度。"
        domain_lines.append(sentence)
        items.append(
            base_question(
                id=f"sf_cv_{i + 1:02d}",
                task="claim",
                scenario="声明核验",
                difficulty="中等",
                query=f"霜白焊丝 {code} 的熔点是 {degree} 摄氏度，这一说法有依据吗？",
                relevant_sources=[DOC_DOMAIN],
                answers=[f"{degree} 摄氏度"],
                evidence=[sentence],
            )
        )
    write_text(DOC_DOMAIN, "\n".join(domain_lines) + "\n")

    table_lines = ["# 朔方精密备件价格表\n", "| 备件号 | 名称 | 单价 |", "| --- | --- | --- |"]
    part_names = ["密封圈", "滤芯", "轴承", "喷嘴", "皮带", "接头", "垫片", "阀芯", "探头", "卡箍", "油封", "弹簧"]
    for i, name in enumerate(part_names):
        code = f"SF-P{i + 1:02d}"
        price = 18 + i
        row = f"| {code} | {name} | {price} 元/只 |"
        table_lines.append(row)
        items.append(
            base_question(
                id=f"sf_tb_{i + 1:02d}",
                task="structured",
                scenario="表格图表",
                difficulty="简单",
                query=f"备件 {code} 的单价是多少？",
                relevant_sources=[DOC_TABLE],
                answers=[f"{price} 元/只"],
                evidence=[row],
            )
        )
    write_text(DOC_TABLE, "\n".join(table_lines) + "\n")

    topics = [
        ("董事长姓名", "朔方精密现任董事长是谁？"),
        ("注册资本", "朔方精密的注册资本是多少？"),
        ("股票代码", "朔方精密的股票代码是什么？"),
        ("厂房产权证号", "朔方精密厂房的产权证号是多少？"),
        ("食堂承包商", "朔方精密食堂现在的承包商是哪家？"),
        ("年会预算", "朔方精密今年的年会预算是多少？"),
        ("班车司机姓名", "朔方精密夜班班车的司机叫什么？"),
        ("门禁供应商", "朔方精密门禁系统的供应商是谁？"),
        ("商标注册号", "朔方精密主商标的注册号是多少？"),
        ("开户行账号", "朔方精密基本户的银行账号是多少？"),
        ("法人身份证号", "朔方精密法定代表人的身份证号是多少？"),
        ("土地出让金", "朔方精密厂区的土地出让金是多少？"),
        ("专利号", "朔方精密那项焊接专利的专利号是什么？"),
        ("母公司名称", "朔方精密的母公司叫什么？"),
        ("审计师事务所", "朔方精密今年的审计师事务所是哪家？"),
        ("工会主席", "朔方精密的工会主席是谁？"),
        ("宿舍地址", "朔方精密员工宿舍的地址是哪里？"),
        ("保险经纪", "朔方精密团体保险的经纪人是谁？"),
        ("域名注册商", "朔方精密官网域名的注册商是谁？"),
        ("消防验收文号", "朔方精密厂房的消防验收文号是多少？"),
        ("环评批复号", "朔方精密厂区的环评批复号是多少？"),
        ("主设备原值", "朔方精密主生产线的设备原值是多少？"),
        ("外部董事名单", "朔方精密的外部董事有哪些人？"),
    ]
    refuse_lines = ["# 朔方精密未记载清单\n", "本清单只写明档案里没有的字段，不补写该字段的值。\n"]
    for i, (topic, query) in enumerate(topics):
        sentence = f"朔方档案不记载{topic}。"
        refuse_lines.append(sentence)
        items.append(
            base_question(
                id=f"sf_rj_{i + 1:02d}",
                task="unanswerable",
                scenario="拒答边界",
                difficulty="困难",
                query=query,
                relevant_sources=[],
                bait_sources=[DOC_REFUSE],
                answers=["资料不足，应拒答"],
                evidence=[sentence],
            )
        )
    write_text(DOC_REFUSE, "\n".join(refuse_lines) + "\n")

    long_parts = [
        "# 朔方精密巡检长文\n",
        "本文件是朔方精密夜班巡检纪要。正式耗时只以带针点编号的句子为准。\n",
    ]
    for i in range(23):
        code = f"SF-ND-{i + 1:02d}"
        minutes = 41 + i
        long_parts.append(f"## 巡段 {code}\n")
        long_parts.extend([FILLER] * 6)
        sentence = f"巡检针点 {code} 的清洁耗时是 {minutes} 分钟。"
        long_parts.append(sentence)
        items.append(
            base_question(
                id=f"sf_lc_{i + 1:02d}",
                task="long_context",
                scenario="长上下文",
                difficulty="困难",
                query=f"巡检针点 {code} 的清洁耗时是多少？",
                relevant_sources=[DOC_LONG],
                answers=[f"{minutes} 分钟"],
                evidence=[sentence],
            )
        )
    write_text(DOC_LONG, "\n".join(long_parts) + "\n")

    chatter = "闲谈：食堂的汤偏咸，有人约在晚饭后打球，这些话不能改写正式补贴。"
    noise_lines = ["# 朔方精密噪声夹注\n", "正式补贴只看带编号的口径句，闲谈不能当作补贴标准。\n"]
    for i in range(23):
        code = f"SF-NS-{i + 1:02d}"
        amount = 60 + i
        sentence = f"正式口径：朔方夜班加餐补贴 {code} 是 {amount} 元。"
        noise_lines.extend([chatter, chatter, sentence])
        items.append(
            base_question(
                id=f"sf_ns_{i + 1:02d}",
                task="noise",
                scenario="噪声干扰",
                difficulty="中等",
                query=f"朔方夜班加餐补贴 {code} 是多少元？",
                relevant_sources=[DOC_NOISE],
                answers=[f"{amount} 元"],
                evidence=[sentence],
            )
        )
    write_text(DOC_NOISE, "\n".join(noise_lines) + "\n")

    current_lines = ["# 朔方精密现行汇刊\n", "本汇刊只保留仍生效的涂料限值、班车首班和产线差额。\n", "## 现行涂料\n"]
    obsolete_lines = ["# 朔方精密废止汇刊\n", "本汇刊只保留已废止的海报限值、旧班车和旧合格数。\n", "## 已废止涂料\n"]
    for i in range(23):
        code = f"SF-CT-{i + 1:02d}"
        current = 21 + i
        old = 61 + i
        current_sentence = f"朔方涂料 {code} 的现行 VOC 限值是 {current} g/L。"
        old_sentence = f"朔方涂料 {code} 的已废止海报把 VOC 限值写成 {old} g/L。"
        current_lines.append(current_sentence)
        obsolete_lines.append(old_sentence)
        items.append(
            base_question(
                id=f"sf_cf_{i + 1:02d}",
                task="conflict",
                scenario="知识冲突",
                difficulty="中等",
                query=f"朔方涂料 {code} 的现行 VOC 限值是多少？",
                relevant_sources=[DOC_CURRENT],
                distractor_sources=[DOC_OBSOLETE],
                answers=[f"{current} g/L"],
                evidence=[current_sentence],
            )
        )
    current_lines.extend(["", "## 现行班车", ""])
    obsolete_lines.extend(["", "## 已废止班车", ""])
    for i in range(23):
        code = f"SF-BUS-{i + 1:02d}"
        now = f"06:{10 + i:02d}"
        old = f"07:{10 + i:02d}"
        current_sentence = f"朔方班车 {code} 的现行首班是 {now}。"
        old_sentence = f"朔方班车 {code} 在 2024 年已废止时刻表中的首班是 {old}。"
        current_lines.append(current_sentence)
        obsolete_lines.append(old_sentence)
        items.append(
            base_question(
                id=f"sf_tm_{i + 1:02d}",
                task="temporal",
                scenario="时效性",
                difficulty="中等",
                query=f"朔方班车 {code} 现在的首班是几点？",
                relevant_sources=[DOC_CURRENT],
                distractor_sources=[DOC_OBSOLETE],
                answers=[now],
                evidence=[current_sentence],
            )
        )
    current_lines.extend(["", "## 产线差额", ""])
    obsolete_lines.extend(["", "## 旧合格数", ""])
    for i in range(14):
        code = f"SF-LN-{i + 1:02d}"
        old_count = 800 + i * 10
        delta = 150 + i
        new_count = old_count + delta
        current_count = f"朔方产线 {code} 的 2026 年合格数是 {new_count} 件。"
        current_delta = f"朔方产线 {code} 与 2024 年相差 {delta} 件。"
        old_sentence = f"朔方产线 {code} 的 2024 年合格数是 {old_count} 件。"
        current_lines.extend([current_count, current_delta])
        obsolete_lines.append(old_sentence)
        items.append(
            base_question(
                id=f"sf_md_{i + 1:02d}",
                task="multi_doc",
                scenario="多文档综合",
                difficulty="困难",
                query=f"朔方产线 {code} 的 2026 年合格数与 2024 年相差多少件？",
                relevant_sources=[DOC_CURRENT, DOC_OBSOLETE],
                answers=[f"相差 {delta} 件"],
                evidence=[current_delta, old_sentence],
            )
        )
    write_text(DOC_CURRENT, "\n".join(current_lines) + "\n")
    write_text(DOC_OBSOLETE, "\n".join(obsolete_lines) + "\n")

    bi_lines = ["# 朔方精密双语条款\n", "数量只写在中文句子里。英文只保留津贴代号，不能另造金额。\n"]
    for i in range(23):
        code = f"SF-ALW-{i + 1:02d}"
        amount = 36 + i
        sentence = f"{code} 的夜班津贴是 {amount} 元。"
        bi_lines.append(sentence)
        items.append(
            base_question(
                id=f"sf_xl_{i + 1:02d}",
                task="cross_lingual",
                scenario="跨语言",
                difficulty="中等",
                query=f"What is the night-shift allowance for {code}?",
                relevant_sources=[DOC_BI],
                answers=[f"{amount} 元"],
                evidence=[sentence],
            )
        )
    write_text(DOC_BI, "\n".join(bi_lines) + "\n")
    return items


def main() -> None:
    rows = people()
    write_ledgers(rows)
    items = [hop_question(row, "2a") for row in rows[:23]]
    items.extend(hop_question(row, "3a") for row in rows)
    items.extend(hop_question(row, "4") for row in rows)
    items.extend(build_rest(rows))
    if len(items) != 459:
        raise SystemExit(f"新增题应为 459，当前 {len(items)}")
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("题目 id 重复")
    QUERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUERY_PATH.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
    print(f"已写朔方精密台账和 {len(items)} 题到 {QUERY_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
