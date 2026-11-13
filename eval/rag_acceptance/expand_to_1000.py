#!/usr/bin/env python3
"""把 standard 场景评测从既有 100 题扩到 1000 题。

保留 queries.jsonl 里原有题目，按澄川实验室合订卷新增语料和 900 题。
每道可答题的标准答案、证据句都能在 gold 原文中逐字找到；拒答题的
标准答案固定为「资料不足，应拒答」，诱饵文档写明该字段未记载。

新增语料按角色合成 19 份较长的合订卷，文件名从 08 排到 26，写入
``docs/rag_accuracy_demo``。重复运行会删掉旧的 ``60_扩_*.md`` 和
已生成的合订卷，再按同一规则写回。
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CORPUS = HERE / "datasets" / "standard" / "corpus"
DEMO = ROOT / "docs" / "rag_accuracy_demo"
QUERIES = HERE / "datasets" / "standard" / "queries.jsonl"
CODE_RE = re.compile(r"CX-E\d{3}")

PLAN = {
    # 场景: (新增题数, 简单, 中等, 困难)
    "单跳事实": (67, 60, 7, 0),
    "领域问答": (67, 40, 22, 5),
    "拒答边界": (67, 8, 24, 35),
    "表格图表": (68, 36, 24, 8),
    "模糊查询": (69, 6, 35, 28),
    "多跳推理": (69, 0, 20, 49),
    "时效性": (69, 28, 30, 11),
    "长上下文": (70, 12, 36, 22),
    "噪声干扰": (70, 20, 36, 14),
    "知识冲突": (70, 12, 32, 26),
    "跨语言": (70, 18, 36, 16),
    "多文档综合": (71, 0, 28, 43),
    "声明核验": (73, 27, 31, 15),
}
FINAL_PER_SCENARIO = {name: (76 if name == "声明核验" else 77) for name in PLAN}
PREFIX = {
    "单跳事实": "sh",
    "领域问答": "dm",
    "拒答边界": "rj",
    "表格图表": "tb",
    "模糊查询": "fz",
    "多跳推理": "hp",
    "时效性": "tm",
    "长上下文": "lc",
    "噪声干扰": "ns",
    "知识冲突": "cf",
    "跨语言": "xl",
    "多文档综合": "md",
    "声明核验": "cl",
}
TASK = {
    "单跳事实": "single_hop",
    "领域问答": "domain_faq",
    "拒答边界": "unanswerable",
    "表格图表": "structured",
    "模糊查询": "fuzzy",
    "多跳推理": "multi_hop",
    "时效性": "temporal",
    "长上下文": "long_context",
    "噪声干扰": "noise",
    "知识冲突": "conflict",
    "跨语言": "cross_lingual",
    "多文档综合": "multi_doc",
    "声明核验": "claim",
}
INSPIRED = {
    "单跳事实": "single-hop fact",
    "领域问答": "FiQA / policy QA",
    "拒答边界": "RGB refusal",
    "表格图表": "table QA / chart QA",
    "模糊查询": "fuzzy query",
    "多跳推理": "HotpotQA",
    "时效性": "temporal QA",
    "长上下文": "long-context needle",
    "噪声干扰": "RGB noise",
    "知识冲突": "knowledge conflict",
    "跨语言": "cross-lingual retrieval",
    "多文档综合": "multi-document synthesis",
    "声明核验": "SciFact",
}
KEY_ORDER = (
    "id",
    "task",
    "scenario",
    "difficulty",
    "inspired_by",
    "query",
    "relevant_sources",
    "distractor_sources",
    "bait_sources",
    "answers",
    "evidence",
)
GIVENS = [
    "照野", "晚晴", "行止", "挽月", "听潮", "知夏", "兰舟", "晚宁", "疏影", "见鹿",
    "南枝", "清越", "晚照", "望舒", "澄川", "疏桐", "拾光", "南风", "见山", "晚枫",
    "澄明", "听雨", "南乔", "书衡", "既望", "怀瑾", "景行", "予安", "从舟", "清和",
    "麦冬", "青渚", "白蘋", "沧浪", "星渚", "霜桥", "露荷", "烟屿", "潮音", "汐灯",
    "浦月", "汀兰", "渚烟", "湾柳", "矶石", "滩芦", "洲荻", "浜竹", "泾荷", "溪苔",
    "泉松", "涧柏", "岭梅", "谷兰", "峰桐", "崖柏", "坞竹", "陂柳", "津苇", "渡枫",
    "浦荻", "汀蓼", "洲萍", "湾棠", "矶藤", "滩芹", "浜菊", "泾薇", "溪芷", "泉蕙",
    "涧萱", "岭蓉", "谷芷",
]
SURNAMES = list("沈顾裴江陆宋贺叶程周许曹冯丁罗梁谢唐潘魏崔邹秦尹卢白金方任姚")
BANNED_CLOCKS = {"07:10", "07:40", "19:10", "20:30", "22:15", "08:00", "19:00"}
DIGIT_BAN = ("7702", "2048", "4401", "1088", "6400")


def load_runner():
    spec = importlib.util.spec_from_file_location("rag_acceptance_run", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact(text: str) -> str:
    return "".join(text.split())


def sent(text: str) -> tuple[str, str]:
    body = " ".join(text.split())
    if not body:
        raise SystemExit("空句子")
    if body.endswith("。") or body.endswith("."):
        full = body
    elif any("\u4e00" <= ch <= "\u9fff" for ch in body):
        full = body + "。"
    else:
        full = body + "."
    return full, full[:-1]


DOC_FILES = {
    "sh": "08_澄川实验室_通讯录.md",
    "dm": "09_澄川实验室_值班与备件制度.md",
    "cl": "10_澄川实验室_纯度实验记录.md",
    "tb": "11_澄川实验室_合格比季报.md",
    "fz": "12_澄川实验室_口头口径对照.md",
    "hpA": "13_澄川实验室_维护人索引.md",
    "hpB": "14_澄川实验室_人员名册.md",
    "hpC": "15_澄川实验室_房间挂牌.md",
    "tmO": "16_澄川实验室_班车时刻表_2024.md",
    "tmN": "17_澄川实验室_班车时刻表_2026.md",
    "lc": "18_澄川实验室_夜班事故纪要.md",
    "ns": "19_澄川实验室_杂录夹注.md",
    "nz": "20_澄川实验室_俱乐部接龙.md",
    "cfO": "21_澄川实验室_挥发物海报.md",
    "cfN": "22_澄川实验室_挥发物规程.md",
    "xl": "23_澄川实验室_双语条款.md",
    "mdO": "24_澄川实验室_冷链电价_2024.md",
    "mdN": "25_澄川实验室_冷链电价_2026.md",
    "rj": "26_澄川实验室_档案摘录.md",
}
DOC_TITLES = {
    "sh": "澄川实验室课题通讯录",
    "dm": "澄川实验室值班与备件制度",
    "cl": "澄川实验室纯度实验记录",
    "tb": "澄川实验室合格比季报",
    "fz": "澄川实验室口头口径对照",
    "hpA": "澄川实验室维护人索引",
    "hpB": "澄川实验室人员名册",
    "hpC": "澄川实验室房间挂牌",
    "tmO": "澄川实验室班车时刻表 2024",
    "tmN": "澄川实验室班车时刻表 2026",
    "lc": "澄川实验室夜班事故纪要",
    "ns": "澄川实验室杂录夹注",
    "nz": "澄川俱乐部接龙",
    "cfO": "澄川实验室挥发物海报",
    "cfN": "澄川实验室挥发物规程",
    "xl": "澄川实验室双语条款",
    "mdO": "澄川实验室冷链电价 2024",
    "mdN": "澄川实验室冷链电价 2026",
    "rj": "澄川实验室档案摘录",
}
READING_GUIDE = [
    "阅读说明：本文件按整份上传。检索命中的是这一份合订卷，回答时仍要落到带课题代号的正式句子。",
    "导读、分节说明和文末代号速查只帮助定位，不构成口径。",
    "已废止海报、二零二四年档案和俱乐部接龙即使用词接近，也不能改写现行句子。",
    "人员姓名、房间和挂牌色号分散在不同卷宗。只看到姓名时，不要在本文件里补写没有出现的员工号或色号。",
    "英文问题如果写明数量只出现在中文句，就不要从英文提示句里另造数量。",
    "表格和季报以核对句为准。井号和文末口述不能覆盖核对句。",
    "事故纪要里的清洁耗时和邻近针点，都不能改写正式事故针点。",
    "同一代号可以出现在多份卷宗。先看问题要求的字段，再读对应卷宗的正文，不要把另一份卷宗的数字抄过来。",
]


def chunk_of(index: int) -> int:
    return (index - 1) // 8 + 1


def fname(kind: str, chunk: int | None = None) -> str:
    return DOC_FILES[kind]


def dossier(entity, topic: str) -> list[str]:
    """课题定位段。只说明卷宗和联系人，不复述正式口径里的数字。"""
    return [
        "",
        f"### {entity.code} · {entity.project}课题",
        (
            f"{entity.project}课题以 {entity.code} 归档到「{topic}」。"
            "本小节先交代卷宗位置，正式口径写在随后的正文句子里，不要用本小节的交代代替正文。"
        ),
        (
            f"联系人栏登记为{entity.owner}，候补栏登记为{entity.alt}。"
            "这两栏只说明进度找谁问，不能拿去填写员工号、房间、挂牌色号、班车时刻、电价或挥发物限值。"
        ),
        (
            "通讯录、值班与备件、纯度记录、合格比、口头津贴、维护人索引、人员名册、房间挂牌、"
            "班车时刻、事故纪要、杂录夹注、挥发物海报、挥发物规程、双语条款、冷链电价和档案摘录各自单独成卷。"
            "俱乐部接龙、已废止印刷品和二零二四年档案即使用到同一代号，也不改写本卷正文。"
        ),
        "",
    ]


def section_block(index: int) -> list[str]:
    label = f"{index:02d}"
    return [
        f"## 分节 {label}",
        "",
        f"分节 {label} 起笔。同一课题代号的正式句子只在本分节正文里生效。",
        "先读本分节的正式句子，再决定要不要翻到其他卷宗。概括句、图注、接龙和已废止印刷品都不能改写正式句子。",
        "本分节没有写出的字段，留在写出该字段的那一份卷宗里，不要在这里补写。",
        "",
    ]


def code_index(lines: list[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for line in lines:
        for code in CODE_RE.findall(line):
            if code not in seen:
                seen.add(code)
                found.append(code)
    if not found:
        return [
            "## 代号速查",
            "",
            "本卷不收录课题代号。俱乐部接龙不能回答课题的正式口径。",
            "",
        ]
    rows = [
        "## 代号速查",
        "",
        "下列代号只说明本卷收录了哪些课题，具体口径仍以正文句子为准。",
        "",
    ]
    for start in range(0, len(found), 8):
        rows.append("、".join(found[start : start + 8]) + "。")
    rows.append("")
    return rows


def render(title: str, status: str, lines: list[str]) -> str:
    body = [
        f"# {title}",
        "",
        f"文档状态：{status}",
        "适用范围：澄川实验室知识库评测语料",
        "更新时间：2026-11-01",
        "装订：同一角色的分节已合并为本文件，按文件名序号与 01 至 07 号企业样例一起存放。",
        "",
        *READING_GUIDE,
        "",
        *lines,
        "",
        *code_index(lines),
        "本文件是 RAG 评测语料，口径只在本文件列出的课题代号上有效。",
        "",
    ]
    return "\n".join(body)


class Nums:
    def __init__(self) -> None:
        self.cur = 25000
        self.used: set[int] = set()

    def bad(self, value: int) -> bool:
        text = str(value)
        return any(token in text for token in DIGIT_BAN)

    def take(self) -> int:
        while self.cur in self.used or self.bad(self.cur):
            self.cur += 1
            if self.cur >= 49000:
                raise SystemExit("数字段用尽")
        value = self.cur
        self.used.add(value)
        self.cur += 1
        return value

    def reserve_ahead(self, start: int) -> int:
        value = start
        while value in self.used or self.bad(value):
            value += 1
        if value >= 49000:
            raise SystemExit("预留数字越界")
        self.used.add(value)
        return value


class Bundle:
    def __init__(self) -> None:
        self.parts: dict[str, dict] = {}
        self.docs: dict[str, str] = {}
        self.queries: list[dict] = []
        self.nums = Nums()
        self.clocks: set[str] = set()

    def add_doc(self, name: str, title: str, status: str, lines: list[str]) -> None:
        del title
        if not lines:
            raise SystemExit(f"空文档 {name}")
        slot = self.parts.setdefault(name, {"status": status, "blocks": []})
        if slot["status"] != status:
            raise SystemExit(f"状态不一致 {name}: {slot['status']} / {status}")
        slot["blocks"].append(list(lines))

    def finalize(self) -> None:
        expected = set(DOC_FILES.values())
        if set(self.parts) != expected:
            missing = sorted(expected - set(self.parts))
            extra = sorted(set(self.parts) - expected)
            raise SystemExit(f"合订卷数量不符，缺少 {missing}，多出 {extra}")
        kind_of = {filename: kind for kind, filename in DOC_FILES.items()}
        self.docs = {}
        for name in sorted(self.parts):
            slot = self.parts[name]
            merged: list[str] = []
            for index, block in enumerate(slot["blocks"], start=1):
                merged.extend(section_block(index))
                merged.extend(block)
                merged.append("")
            self.docs[name] = render(DOC_TITLES[kind_of[name]], slot["status"], merged)

    def add_q(
        self,
        *,
        scenario: str,
        entity,
        diff: str,
        query: str,
        relevant: list[str],
        answers: list[str],
        evidence: list[str],
        distractor: list[str] | None = None,
        bait: list[str] | None = None,
    ) -> None:
        self.queries.append(
            {
                "id": f"ex_{PREFIX[scenario]}_{entity.i:03d}",
                "task": TASK[scenario],
                "scenario": scenario,
                "difficulty": diff,
                "inspired_by": INSPIRED[scenario],
                "query": query,
                "relevant_sources": relevant,
                "distractor_sources": distractor or [],
                "bait_sources": bait or [],
                "answers": answers,
                "evidence": evidence,
            }
        )

    def clock(self, index: int, salt: int) -> str:
        for bump in range(400):
            total = (index * 37 + salt * 91 + bump * 13) % (13 * 60)
            hour = 6 + total // 60
            minute = total % 60
            text = f"{hour:02d}:{minute:02d}"
            if text in self.clocks or text in BANNED_CLOCKS:
                continue
            self.clocks.add(text)
            return text
        raise SystemExit(f"无法分配时刻 {index}")


def build_entities() -> list:
    if len(GIVENS) != 73 or len(set(GIVENS)) != 73 or any(len(item) != 2 for item in GIVENS):
        raise SystemExit("人名用字表必须是 73 个互异的双字名")
    owners = [SURNAMES[i % len(SURNAMES)] + GIVENS[i] for i in range(73)]
    if len(set(owners)) != 73:
        raise SystemExit("维护人重名")
    used_names = set(owners)
    alts = []
    for i in range(73):
        found = None
        for shift in range(1, 73):
            for surname_shift in range(1, len(SURNAMES)):
                name = SURNAMES[(i + surname_shift) % len(SURNAMES)] + GIVENS[(i + shift) % 73]
                if name not in used_names:
                    found = name
                    break
            if found:
                break
        if not found:
            raise SystemExit(f"无法分配候补人 {i}")
        alts.append(found)
        used_names.add(found)
    banned = set(GIVENS) | {"青岚", "北辰", "云栖", "泊岸", "鹤鸣", "折纸", "临江", "砚台", "星澜", "青萝"}
    projects = []
    for left in "绛疏望霜露烟潮星沧月雪雾汐":
        for right in "浦澜川桥汀湾屿岸港台南西":
            name = left + right
            if name in banned or name in projects:
                continue
            projects.append(name)
            if len(projects) == 73:
                break
        if len(projects) == 73:
            break
    if len(projects) != 73:
        raise SystemExit("课题名不足")
    entities = []
    base = date(2023, 5, 1)
    for i in range(1, 74):
        entities.append(
            SimpleNamespace(
                i=i,
                code=f"CX-E{i:03d}",
                project=projects[i - 1],
                owner=owners[i - 1],
                alt=alts[i - 1],
                email=f"cxe{i:03d}@example.internal",
                founded=(base + timedelta(days=i)).isoformat(),
            )
        )
    return entities


def plan_rows(entities: list, scenario: str) -> tuple[list, dict]:
    count, easy, mid, hard = PLAN[scenario]
    diffs = ["简单"] * easy + ["中等"] * mid + ["困难"] * hard
    if len(diffs) != count or easy + mid + hard != count:
        raise SystemExit(f"{scenario} 难度分配不正确")
    rows = [(entities[i], diffs[i]) for i in range(count)]
    groups: dict[int, list] = defaultdict(list)
    for entity, diff in rows:
        groups[chunk_of(entity.i)].append((entity, diff))
    return rows, groups


def neighbor(kind: str, chunk: int, count: int) -> str | None:
    present = {chunk_of(i) for i in range(1, count + 1)}
    for other in (chunk + 1, chunk - 1):
        if other in present:
            return fname(kind, other)
    return None


def extras(*names: str | None, relevant: list[str]) -> list[str]:
    out = []
    for name in names:
        if name and name not in relevant and name not in out:
            out.append(name)
    return out


def room_manual(chunk: int) -> list[str]:
    steps = [
        "先在维护人索引确认现行维护人，不要填写已经卸任的姓名。",
        "再到人员名册把现行维护人对应到房间。",
        "最后回到本卷读取该房间的访客挂牌色号。",
        "离开时把挂牌交回值班台。本卷不记载押金，也不记载私人联系方式。",
        "索引、名册和本卷要一起读。少一份就不要填写色号。",
        "已废止海报和俱乐部接龙不出现挂牌色号，不能拿来补本卷没有写出的字段。",
    ]
    lines = [
        "访客挂牌按下面的顺序领取。色号只以本卷随后写出的挂牌句为准，备忘里没有色号。",
    ]
    for n in range(1, 41):
        lines.append(f"挂牌流程备忘 {chunk:02d}-{n:02d}：{steps[n % len(steps)]}")
    return lines


def club(tag: str, count: int) -> list[str]:
    lines = []
    notes = [
        "观星小组只记录英仙座，建议选无月夜，不进入课题口径。",
        "面点卡只记录醒发，不记录课题津贴或电价。",
        "陶艺接龙写在活动室，不能改正式口径。",
        "合唱排练只排时段，不记载课题分机或纯度。",
    ]
    for offset in range(count):
        lines.append(f"{tag} 杂录 {offset + 1}：{notes[offset % len(notes)]}")
    return lines


def add_single(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "单跳事实")
    count = PLAN["单跳事实"][0]
    for chunk, members in groups.items():
        lines = ["本卷是课题通讯录。每条只回答对应课题代号，不要串到其他卷宗。"]
        for entity, diff in members:
            lines.extend(dossier(entity, "通讯录"))
            kind = entity.i % 4
            if kind == 0:
                line, answer = sent(f"{entity.code} 的值班邮箱是 {entity.email}")
                query = (
                    f"{entity.code} 的值班邮箱是什么？"
                    if diff == "简单"
                    else f"要给{entity.project}课题值班写信，{entity.code} 的邮箱地址是哪一个？"
                )
            elif kind == 1:
                number = bundle.nums.take()
                line, answer = sent(f"{entity.code} 的紧急分机是 {number}")
                query = (
                    f"{entity.code} 的紧急分机是多少？"
                    if diff == "简单"
                    else f"{entity.project}课题夜里找人，{entity.code} 的紧急分机报多少？"
                )
            elif kind == 2:
                line, answer = sent(f"{entity.code} 的立项日期是 {entity.founded}")
                query = (
                    f"{entity.code} 是哪一天立项的？"
                    if diff == "简单"
                    else f"{entity.project}课题立项落在哪一天？代号是 {entity.code}。"
                )
            else:
                number = bundle.nums.take()
                line, answer = sent(f"{entity.code} 的库房湿度读数是 {number}")
                query = (
                    f"{entity.code} 的库房湿度读数是多少？"
                    if diff == "简单"
                    else f"查 {entity.project}课题库房，{entity.code} 的湿度读数记的是多少？"
                )
            lines.append(line)
            relevant = [fname("sh", chunk)]
            bundle.add_q(
                scenario="单跳事实",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=[answer],
                evidence=[line],
                distractor=extras(neighbor("sh", chunk, count), relevant=relevant),
            )
        bundle.add_doc(fname("sh", chunk), f"澄川实验室课题通讯录（卷 {chunk:02d}）", "当前有效", lines)


def add_domain(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "领域问答")
    count = PLAN["领域问答"][0]
    for chunk, members in groups.items():
        lines = ["本卷是值班与备件制度。加档条件写明之前，不得套用加档金额。"]
        for entity, diff in members:
            lines.extend(dossier(entity, "值班与备件"))
            if diff == "简单":
                number = bundle.nums.take()
                line, answer = sent(f"{entity.code} 普通值班津贴是 {number} 元每班")
                query = f"{entity.code} 的普通值班津贴是多少元每班？"
                evidence = [line]
                answers = [answer]
                lines.append(line)
            elif diff == "中等":
                number = bundle.nums.take()
                line, answer = sent(
                    f"{entity.code} 在连续值班未满 4 个班次时，备件申领时限是 {number} 分整"
                )
                query = f"{entity.code} 连续值班未满 4 个班次时，备件申领时限是多少分整？"
                evidence = [line]
                answers = [answer]
                lines.append(line)
            else:
                base = bundle.nums.take()
                high = bundle.nums.take()
                base_line, _ = sent(f"{entity.code} 普通值班津贴是 {base} 元每班")
                high_line, answer = sent(
                    f"{entity.code} 在连续值班达到 4 个班次及以上时，值班津贴改按 {high} 元每班发放"
                )
                query = f"{entity.code} 连续值班达到 4 个班次及以上时，值班津贴改按多少元每班发放？"
                evidence = [high_line]
                answers = [answer]
                lines.extend([base_line, high_line, f"不足 4 个班次时，{entity.code} 不得套用加档金额。"])
            relevant = [fname("dm", chunk)]
            bundle.add_q(
                scenario="领域问答",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=answers,
                evidence=evidence,
                distractor=extras(neighbor("dm", chunk, count), relevant=relevant),
            )
        bundle.add_doc(fname("dm", chunk), f"澄川实验室值班与备件制度（卷 {chunk:02d}）", "当前有效", lines)


def add_claim(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "声明核验")
    count = PLAN["声明核验"][0]
    for chunk, members in groups.items():
        lines = ["本卷是实验记录。外部声明与记录冲突时，以记录句为准。"]
        for entity, diff in members:
            lines.extend(dossier(entity, "纯度实验记录"))
            purity = bundle.nums.take()
            if diff == "简单":
                line, answer = sent(f"实验记录载明 {entity.code} 的标称纯度是 {purity} ppm")
                query = f"{entity.code} 的标称纯度是多少 ppm？"
                answers = [answer]
                evidence = [line]
                lines.append(line)
            elif diff == "中等":
                claim, _ = sent(f"外部声明称 {entity.code} 的标称纯度为 {purity} ppm")
                line, answer = sent(f"该声明与实验记录一致，{entity.code} 的标称纯度是 {purity} ppm")
                query = f"有人说 {entity.code} 的标称纯度是 {purity} ppm，实验记录是否支持？"
                answers = [answer]
                evidence = [line]
                lines.extend([claim, line])
            else:
                wrong = bundle.nums.take()
                deny, _ = sent(f"外部声明把 {entity.code} 的标称纯度写成 {wrong} ppm，该声明不成立")
                line, answer = sent(f"实验记录载明 {entity.code} 的标称纯度是 {purity} ppm，不是 {wrong} ppm")
                query = f"外部把 {entity.code} 的标称纯度说成 {wrong} ppm。实验记录里的标称纯度是多少 ppm？"
                answers = [answer]
                evidence = [line]
                lines.extend([deny, line])
            relevant = [fname("cl", chunk)]
            bundle.add_q(
                scenario="声明核验",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=answers,
                evidence=evidence,
                distractor=extras(neighbor("cl", chunk, count), relevant=relevant),
            )
        bundle.add_doc(fname("cl", chunk), f"澄川实验室纯度实验记录（卷 {chunk:02d}）", "当前有效", lines)


def add_table(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "表格图表")
    count = PLAN["表格图表"][0]
    for chunk, members in groups.items():
        lines = [
            "下表是合格比。图 1 的井号只是示意，读数以单元格和核对句为准。",
            "",
            "| 代号 | 2026Q1 | 2026Q2 | 2026Q3 |",
            "| --- | --- | --- | --- |",
        ]
        notes = []
        queries = []
        for entity, diff in members:
            notes.extend(dossier(entity, "合格比季报"))
            first = bundle.nums.take()
            second = bundle.nums.reserve_ahead(first + 500 + entity.i)
            gap = second - first
            third = bundle.nums.take()
            lines.append(f"| {entity.code} | {first} | {second} | {third} |")
            quarter, quarter_answer = sent(f"核对句：{entity.code} 在 2026Q2 的合格比是 {second}")
            delta, delta_answer = sent(f"核对句：{entity.code} 的 2026Q2 合格比比 2026Q1 高 {gap}")
            notes.extend([quarter, delta])
            if diff == "困难":
                query = (
                    f"图注说本卷合格比都超过 90000，图 1 的井号也不能当读数。"
                    f"{entity.code} 在 2026Q2 的合格比以核对句为准是多少？"
                )
                answer, evidence = quarter_answer, quarter
            elif diff == "中等":
                query = f"按核对句，{entity.code} 的 2026Q2 合格比比 2026Q1 高多少？"
                answer, evidence = delta_answer, delta
            else:
                query = f"表中 {entity.code} 在 2026Q2 的合格比是多少？"
                answer, evidence = quarter_answer, quarter
            queries.append((entity, diff, query, answer, evidence))
        lines.extend(["", "```", "图1 井号示意，不是读数"])
        for entity, _diff in members:
            lines.append(f"{entity.code} " + ("#" * ((entity.i % 8) + 3)))
        lines.extend(["```", "图注口述本卷合格比都超过 90000，与表不符，不得采用。", ""])
        lines.extend(notes)
        relevant = [fname("tb", chunk)]
        for entity, diff, query, answer, evidence in queries:
            bundle.add_q(
                scenario="表格图表",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=[answer],
                evidence=[evidence],
                distractor=extras(neighbor("tb", chunk, count), relevant=relevant),
            )
        bundle.add_doc(fname("tb", chunk), f"澄川实验室合格比季报（卷 {chunk:02d}）", "当前有效", lines)


def add_fuzzy(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "模糊查询")
    count = PLAN["模糊查询"][0]
    for chunk, members in groups.items():
        lines = ["本卷把口头叫法和正式津贴写在一起。餐补不是夜补。"]
        for entity, diff in members:
            lines.extend(dossier(entity, "口头口径"))
            number = bundle.nums.take()
            line, answer = sent(
                f"{entity.project}课题（{entity.code}，同事口头把值班津贴叫成「{entity.project}夜补」）的值班津贴是 {number} 元每班"
            )
            lines.append(line)
            if diff == "困难":
                meal = bundle.nums.take()
                meal_line, _ = sent(f"{entity.project}课题的餐补是 {meal} 元每班，和{entity.project}夜补不是同一项")
                lines.append(meal_line)
                query = f"别把餐补算进去，{entity.project}夜补实际是多少元每班？"
            elif diff == "中等":
                query = f"{entity.project}夜补现在发多少元每班？"
            else:
                query = f"{entity.code}，也就是{entity.project}夜补，现在是多少元每班？"
            relevant = [fname("fz", chunk)]
            bundle.add_q(
                scenario="模糊查询",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=[answer],
                evidence=[line],
                distractor=extras(neighbor("fz", chunk, count), fname("nz", 1), relevant=relevant),
            )
        bundle.add_doc(fname("fz", chunk), f"澄川实验室口头口径对照（卷 {chunk:02d}）", "当前有效", lines)


def add_hop(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "多跳推理")
    count = PLAN["多跳推理"][0]
    for chunk, members in groups.items():
        index_lines = ["本卷只记载维护人姓名，不记载员工号、房间和挂牌色号。"]
        roster_lines = ["本卷按姓名记载员工号或房间。历史维护人没有员工号条目。"]
        room_lines = []
        for entity, diff in members:
            index_lines.extend(dossier(entity, "维护人索引"))
            roster_lines.extend(dossier(entity, "人员名册"))
            emp = f"EMP-C9{entity.i:03d}"
            room = f"R9{entity.i:03d}"
            badge = f"H9{entity.i:03d}"
            three = diff == "困难" and entity.i >= 55
            file_a = fname("hpA", chunk)
            file_b = fname("hpB", chunk)
            file_c = fname("hpC", chunk)
            if three:
                room_lines.extend(dossier(entity, "房间挂牌"))
                line_a, _ = sent(f"{entity.code} 的现行维护人是 {entity.owner}")
                line_b, _ = sent(f"{entity.owner}的工位房间是 {room}")
                line_c, answer = sent(f"{room} 的访客挂牌色号是 {badge}")
                index_lines.append(line_a)
                roster_lines.append(line_b)
                room_lines.append(line_c)
                relevant = [file_a, file_b, file_c]
                query = f"去找 {entity.code} 的现行维护人，进门要换的访客挂牌色号是什么？"
                answers = [answer]
                evidence = [line_a, line_b, line_c]
            elif diff == "困难":
                line_a, _ = sent(f"{entity.code} 的历史维护人是 {entity.alt}，现行维护人是 {entity.owner}")
                line_b, answer = sent(f"{entity.owner}的员工号是 {emp}")
                index_lines.append(line_a)
                roster_lines.append(line_b)
                relevant = [file_a, file_b]
                query = f"{entity.code} 现在的维护人员工号是哪一个？不要填已经卸任的人。"
                answers = [answer]
                evidence = [line_a, line_b]
            else:
                line_a, _ = sent(f"{entity.code} 的现行维护人是 {entity.owner}")
                line_b, answer = sent(f"{entity.owner}的员工号是 {emp}")
                index_lines.append(line_a)
                roster_lines.append(line_b)
                relevant = [file_a, file_b]
                query = f"{entity.code} 现行维护人的员工号是什么？"
                answers = [answer]
                evidence = [line_a, line_b]
            bundle.add_q(
                scenario="多跳推理",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=answers,
                evidence=evidence,
                distractor=extras(neighbor("hpA", chunk, count), relevant=relevant),
            )
        bundle.add_doc(fname("hpA", chunk), f"澄川实验室维护人索引（卷 {chunk:02d}）", "当前有效", index_lines)
        bundle.add_doc(fname("hpB", chunk), f"澄川实验室人员名册（卷 {chunk:02d}）", "当前有效", roster_lines)
        if room_lines:
            room_lines[0:0] = room_manual(chunk)
            bundle.add_doc(fname("hpC", chunk), f"澄川实验室房间挂牌（卷 {chunk:02d}）", "当前有效", room_lines)


def add_temporal(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "时效性")
    count = PLAN["时效性"][0]
    for chunk, members in groups.items():
        old_lines = ["本卷是 2024 年时刻表，停用日是 2024-12-31，不能回答现行首班。"]
        new_lines = ["本卷自 2026-01-01 起生效。现行首班以本卷句子为准，不以已废止时刻表为准。"]
        for entity, diff in members:
            old_lines.extend(dossier(entity, "二零二四年班车时刻"))
            new_lines.extend(dossier(entity, "现行班车时刻"))
            old_clock = bundle.clock(entity.i, 1)
            new_clock = bundle.clock(entity.i, 2)
            old_line, old_answer = sent(f"{entity.code} 在 2024 年已废止时刻表中的班车首班是 {old_clock}")
            new_line, new_answer = sent(f"{entity.code} 的现行班车首班是 {new_clock}")
            hard_line, hard_answer = sent(f"{entity.code} 今天的首班不是已废止时刻，现行班车首班是 {new_clock}")
            old_lines.append(old_line)
            file_old = fname("tmO", chunk)
            file_new = fname("tmN", chunk)
            if diff == "简单":
                new_lines.append(new_line)
                query = f"{entity.code} 已废止的 2024 年时刻表里，班车首班是几点？"
                relevant = [file_old]
                answers = [old_answer]
                evidence = [old_line]
                distractor = extras(file_new, neighbor("tmO", chunk, count), relevant=relevant)
            elif diff == "中等":
                new_lines.append(new_line)
                query = f"按 2026 年仍生效的时刻表，{entity.code} 的班车首班是几点？"
                relevant = [file_new]
                answers = [new_answer]
                evidence = [new_line]
                distractor = extras(file_old, neighbor("tmN", chunk, count), relevant=relevant)
            else:
                new_lines.append(hard_line)
                query = f"还有人按已废止的 2024 年时刻表去等 {entity.code} 的首班。现在首班还是废止表上的时间吗？"
                relevant = [file_new]
                answers = [hard_answer]
                evidence = [hard_line]
                distractor = extras(file_old, neighbor("tmN", chunk, count), relevant=relevant)
            bundle.add_q(
                scenario="时效性",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=answers,
                evidence=evidence,
                distractor=distractor,
            )
        bundle.add_doc(fname("tmO", chunk), f"澄川实验室班车时刻表 2024（卷 {chunk:02d}）", "已废止", old_lines)
        bundle.add_doc(fname("tmN", chunk), f"澄川实验室班车时刻表 2026（卷 {chunk:02d}）", "当前有效", new_lines)


def filler(chunk: int, seq: int, count: int) -> list[str]:
    verbs = ["清洁纸路", "校准色带", "复核温湿度计", "盘点硒鼓", "检查传送带", "清空废纸"]
    lines = []
    for offset in range(count):
        number = 51000 + chunk * 300 + seq * 5 + offset
        verb = verbs[(chunk + offset) % len(verbs)]
        lines.append(
            f"巡检垫文 {chunk:02d}-{seq:02d}-{offset:02d}：队列 LQ{chunk:02d}{offset:02d} 完成{verb}，"
            f"过程耗时 {number} 分整，未更换证书，未做赔付。"
        )
    return lines


def add_long(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "长上下文")
    count = PLAN["长上下文"][0]
    width = {"简单": 12, "中等": 24, "困难": 36}
    for chunk, members in groups.items():
        lines = [
            "下文垫文只记录清洁耗时，不能用来回答事故针点。",
            "邻近针点带有代号后缀 X，不能改写正式事故针点。",
        ]
        lines.extend(filler(chunk, 0, 36))
        for entity, diff in members:
            lines.extend(dossier(entity, "夜班事故纪要"))
            lines.extend(filler(chunk, entity.i, width[diff]))
            duration = bundle.nums.take()
            pay = bundle.nums.take()
            alias = f"cert-{entity.code}-retired"
            duration_line, duration_answer = sent(f"事故针点 {entity.code} 的中断时长是 {duration} 分整")
            alias_line, alias_answer = sent(f"事故针点 {entity.code} 的误用证书别名是 {alias}")
            pay_line, pay_answer = sent(f"事故针点 {entity.code} 的赔付券是 {pay} 枚")
            if diff == "困难":
                other = bundle.nums.take()
                other_pay = bundle.nums.take()
                lines.append(f"邻近针点 {entity.code}X 的中断时长是 {other} 分整，该邻近针点作废。")
                lines.append(
                    f"邻近针点 {entity.code}X 的误用证书别名是 cert-{entity.code}X-retired，该邻近针点作废。"
                )
                lines.append(f"邻近针点 {entity.code}X 的赔付券是 {other_pay} 枚，该邻近针点作废。")
            lines.extend([duration_line, alias_line, pay_line])
            field = entity.i % 3
            if field == 0:
                query = f"{entity.code} 事故针点的中断时长是多少分整？"
                answer, evidence = duration_answer, duration_line
            elif field == 1:
                query = f"{entity.code} 事故针点误用的证书别名是什么？"
                answer, evidence = alias_answer, alias_line
            else:
                query = f"{entity.code} 事故针点赔付了多少枚券？"
                answer, evidence = pay_answer, pay_line
            if diff == "困难":
                query += "邻近针点的记录作废。"
            relevant = [fname("lc", chunk)]
            bundle.add_q(
                scenario="长上下文",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=[answer],
                evidence=[evidence],
                distractor=extras(neighbor("lc", chunk, count), relevant=relevant),
            )
        lines.extend(filler(chunk, 90, 28))
        bundle.add_doc(fname("lc", chunk), f"澄川实验室夜班事故纪要（卷 {chunk:02d}）", "当前有效", lines)


def add_noise_books(bundle: Bundle) -> None:
    for index in range(1, 5):
        lines = [
            f"接龙册 {index:02d} 只收俱乐部活动。观星、面点、陶艺和合唱都写在活动室，不进入课题口径。",
            "接龙里的津贴数字是活动室随手写的。正式值班津贴以杂录夹注里标明正式口径的句子为准。",
            "不要把本册的菜谱、排练时段或醒发记录抄进课题通讯录、规程或电价卷。",
            "活动室不记载课题分机、纯度、合格比、班车时刻、挥发物限值或冷链电价。",
        ]
        lines.extend(club(f"NZ{index:02d}", 36))
        for offset in range(36):
            theme = [
                "观星小组只核对英仙座，记录留在活动室。",
                "面点卡只写醒发状态，不写课题津贴或电价。",
                "陶艺接龙留在活动室，不能改正式口径。",
                "合唱排练只排曲目顺序，不记载分机或纯度。",
            ][(index + offset) % 4]
            lines.append(f"活动室流水 NZ{index:02d}-B{offset:02d}：{theme} 本条没有课题代号。")
        for offset in range(12):
            number = 19000 + index * 100 + offset
            lines.append(
                f"俱乐部接龙 NZ-{index:02d}-{offset:02d} 把值班津贴写成 {number} 元每班，该接龙不能作为课题正式口径。"
            )
        bundle.add_doc(fname("nz", index), f"澄川俱乐部接龙（卷 {index:02d}）", "俱乐部杂录，不是课题口径", lines)


def add_noise(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "噪声干扰")
    count = PLAN["噪声干扰"][0]
    for chunk, members in groups.items():
        lines = ["本卷把俱乐部杂录和一条正式口径夹在一起。接龙句作废，正式口径才有效。"]
        lines.extend(club(f"卷{chunk:02d}前", 12))
        for entity, diff in members:
            lines.extend(dossier(entity, "杂录夹注"))
            number = bundle.nums.take()
            official, answer = sent(f"正式口径：{entity.code} 的值班津贴是 {number} 元每班")
            if diff == "困难":
                decoy = bundle.nums.take()
                lines.append(f"接龙误写 {entity.code} 的值班津贴为 {decoy} 元每班，该接龙作废。")
                query = f"接龙里的数作废。{entity.code} 的正式值班津贴是多少元每班？"
            elif diff == "中等":
                query = f"{entity.code} 的值班津贴以正式口径为准是多少元每班？俱乐部菜谱不算。"
            else:
                query = f"{entity.code} 正式口径里的值班津贴是多少元每班？"
            lines.append(official)
            lines.extend(club(entity.code, 3))
            relevant = [fname("ns", chunk)]
            bait_noise = fname("nz", ((chunk - 1) % 4) + 1)
            bundle.add_q(
                scenario="噪声干扰",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=[answer],
                evidence=[official],
                distractor=extras(bait_noise, neighbor("ns", chunk, count), relevant=relevant),
            )
        lines.extend(club(f"卷{chunk:02d}后", 12))
        bundle.add_doc(fname("ns", chunk), f"澄川实验室杂录夹注（卷 {chunk:02d}）", "当前有效", lines)


def add_conflict(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "知识冲突")
    count = PLAN["知识冲突"][0]
    for chunk, members in groups.items():
        poster_lines = ["本卷是宣传海报，不是现行规程。旧限值不能当作现行限值。"]
        spec_lines = ["本卷是现行规程。海报与规程不一致时，以本卷规程编号为准。"]
        for entity, diff in members:
            poster_lines.extend(dossier(entity, "挥发物海报"))
            spec_lines.extend(dossier(entity, "挥发物规程"))
            old_value = bundle.nums.take()
            new_value = bundle.nums.take()
            poster, poster_answer = sent(f"海报把 {entity.code} 的挥发物限值印成 {old_value} 毫克每升")
            current, current_answer = sent(f"{entity.code} 的现行挥发物限值是 {new_value} 毫克每升")
            rule, rule_answer = sent(f"海报若仍印着旧限值，{entity.code} 以规程 {entity.code}-SPEC-2026 为准")
            poster_lines.append(poster)
            spec_lines.extend([current, rule])
            file_old = fname("cfO", chunk)
            file_new = fname("cfN", chunk)
            if diff == "简单":
                query = f"{entity.code} 的宣传海报把挥发物限值印成多少毫克每升？"
                relevant = [file_old]
                answers = [poster_answer]
                evidence = [poster]
                distractor = extras(file_new, neighbor("cfO", chunk, count), relevant=relevant)
            elif diff == "中等":
                query = f"{entity.code} 的现行挥发物限值是多少毫克每升？"
                relevant = [file_new]
                answers = [current_answer]
                evidence = [current]
                distractor = extras(file_old, neighbor("cfN", chunk, count), relevant=relevant)
            else:
                query = f"海报和规程不一致时，{entity.code} 以哪一份规程编号为准？现行挥发物限值是多少毫克每升？"
                relevant = [file_new]
                answers = [rule_answer, current_answer]
                evidence = [rule, current]
                distractor = extras(file_old, neighbor("cfN", chunk, count), relevant=relevant)
            bundle.add_q(
                scenario="知识冲突",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=answers,
                evidence=evidence,
                distractor=distractor,
            )
        bundle.add_doc(fname("cfO", chunk), f"澄川实验室挥发物海报（卷 {chunk:02d}）", "已废止的宣传印刷品，不是现行规程", poster_lines)
        bundle.add_doc(fname("cfN", chunk), f"澄川实验室挥发物规程（卷 {chunk:02d}）", "当前有效", spec_lines)


def add_cross(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "跨语言")
    count = PLAN["跨语言"][0]
    for chunk, members in groups.items():
        lines = ["This volume mixes English questions with the Chinese record. 英文问题以本卷对应句子为准。"]
        for entity, diff in members:
            lines.extend(dossier(entity, "双语条款"))
            number = bundle.nums.take()
            if diff == "简单":
                line, answer = sent(f"The on-call stipend of {entity.code} is {number} CNY per shift")
                gloss, _ = sent(f"{entity.code} 的英文条目将值班津贴记为 {number} CNY per shift")
                lines.extend([line, gloss])
                query = f"What is the on-call stipend of {entity.code}?"
                answers = [answer]
                evidence = [line]
            elif diff == "中等":
                english, english_answer = sent(f"The rate limit of {entity.code} is {number} requests per hour")
                chinese, chinese_answer = sent(f"{entity.code} 的 rate limit 是每小时 {number} 个请求")
                lines.extend([english, chinese])
                if entity.i % 2 == 0:
                    query = f"{entity.code} 的 rate limit 是每小时多少个请求？"
                    answers = [chinese_answer]
                    evidence = [chinese]
                else:
                    query = f"What is the rate limit of {entity.code}?"
                    answers = [english_answer]
                    evidence = [english]
            else:
                chinese, answer = sent(f"{entity.code} 的值班津贴是 {number} 元每班")
                lines.append(f"The on-call stipend of {entity.code} is stated only in the Chinese sentence.")
                lines.append(chinese)
                query = f"What is the on-call stipend of {entity.code}?"
                answers = [answer]
                evidence = [chinese]
            relevant = [fname("xl", chunk)]
            bundle.add_q(
                scenario="跨语言",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=answers,
                evidence=evidence,
                distractor=extras(neighbor("xl", chunk, count), relevant=relevant),
            )
        bundle.add_doc(fname("xl", chunk), f"澄川实验室双语条款（卷 {chunk:02d}）", "当前有效", lines)


def add_multi(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "多文档综合")
    count = PLAN["多文档综合"][0]
    for chunk, members in groups.items():
        old_lines = ["本卷只记载 2024 年冷链电价，不记载 2026 年电价，也不写降幅。"]
        new_lines = ["本卷记载 2026 年冷链电价。降幅句只在需要核对差值的课题下出现。"]
        for entity, diff in members:
            old_lines.extend(dossier(entity, "二零二四年冷链电价"))
            new_lines.extend(dossier(entity, "现行冷链电价"))
            file_old = fname("mdO", chunk)
            file_new = fname("mdN", chunk)
            if diff == "中等":
                old_value = bundle.nums.take()
                new_value = bundle.nums.take()
                old_line, old_answer = sent(f"{entity.code} 在 2024 年的冷链电价是 {old_value} 分钱")
                new_line, new_answer = sent(f"{entity.code} 在 2026 年的冷链电价是 {new_value} 分钱")
                old_lines.append(old_line)
                new_lines.append(new_line)
                query = f"请分别给出 {entity.code} 在 2024 年和 2026 年的冷链电价。"
                answers = [old_answer, new_answer]
                evidence = [old_line, new_line]
            else:
                new_value = bundle.nums.take()
                old_value = bundle.nums.reserve_ahead(new_value + 200 + entity.i)
                gap = old_value - new_value
                old_line, _ = sent(f"{entity.code} 在 2024 年的冷链电价是 {old_value} 分钱")
                new_line, _ = sent(f"{entity.code} 在 2026 年的冷链电价是 {new_value} 分钱")
                delta_line, delta_answer = sent(
                    f"{entity.code} 的冷链电价从 2024 年到 2026 年下降了 {gap} 分钱"
                )
                old_lines.append(old_line)
                new_lines.extend([new_line, delta_line])
                query = f"{entity.code} 的冷链电价从 2024 年到 2026 年下降了多少分钱？"
                answers = [delta_answer]
                evidence = [old_line, delta_line]
            relevant = [file_old, file_new]
            bundle.add_q(
                scenario="多文档综合",
                entity=entity,
                diff=diff,
                query=query,
                relevant=relevant,
                answers=answers,
                evidence=evidence,
                distractor=extras(neighbor("mdN", chunk, count), relevant=relevant),
            )
        bundle.add_doc(fname("mdO", chunk), f"澄川实验室冷链电价 2024（卷 {chunk:02d}）", "2024 年档案", old_lines)
        bundle.add_doc(fname("mdN", chunk), f"澄川实验室冷链电价 2026（卷 {chunk:02d}）", "当前有效", new_lines)


def add_refusal(bundle: Bundle, entities: list) -> None:
    _, groups = plan_rows(entities, "拒答边界")
    count = PLAN["拒答边界"][0]
    hop_count = PLAN["多跳推理"][0]
    for chunk, members in groups.items():
        lines = ["本卷档案可以提到维护人或津贴读数，但下面点名的字段没有记载，不能用其他数字顶替。"]
        for entity, diff in members:
            lines.extend(dossier(entity, "档案摘录"))
            file_rj = fname("rj", chunk)
            if diff == "简单":
                line, _ = sent(f"{entity.code} 的档案不记载预算金额")
                lines.append(line)
                query = f"{entity.code} 的预算金额是多少？"
                bait = [file_rj]
                evidence = [line]
            elif diff == "中等":
                line, _ = sent(f"{entity.code} 的档案不记载{entity.owner}的身份证号和私人手机号")
                lines.append(f"档案提到 {entity.code} 的维护人是 {entity.owner}。")
                lines.append(line)
                query = f"{entity.owner}的身份证号是多少？" if entity.i % 2 else f"{entity.owner}的私人手机号是多少？"
                bait = [file_rj]
                if entity.i <= hop_count:
                    bait.append(fname("hpB", chunk_of(entity.i)))
                evidence = [line]
            else:
                tempt = bundle.nums.take()
                tease, _ = sent(f"档案摘录出现过 {entity.code} 的值班津贴读数 {tempt} 元每班，该读数不是预算")
                line, _ = sent(f"{entity.code} 的档案不记载预算金额，也不记载法人代表")
                lines.extend([tease, line])
                query = f"{entity.code} 明年的预算金额是多少？津贴读数不能拿来顶替。"
                bait = [file_rj, fname("dm", chunk_of(entity.i))]
                evidence = [line]
            bundle.add_q(
                scenario="拒答边界",
                entity=entity,
                diff=diff,
                query=query,
                relevant=[],
                answers=["资料不足，应拒答"],
                evidence=evidence,
                bait=bait,
            )
        bundle.add_doc(fname("rj", chunk), f"澄川实验室档案摘录（卷 {chunk:02d}）", "当前有效", lines)


def dump_query(item: dict) -> str:
    payload = {key: item[key] for key in KEY_ORDER}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate(bundle: Bundle, runner, old_queries: list[dict]) -> None:
    bundle.finalize()
    if sum(item[0] for item in PLAN.values()) != 900:
        raise SystemExit("新增题数不是 900")
    easy = sum(item[1] for item in PLAN.values())
    mid = sum(item[2] for item in PLAN.values())
    hard = sum(item[3] for item in PLAN.values())
    if (easy, mid, hard) != (267, 361, 272):
        raise SystemExit(f"新增难度不是 267/361/272：{(easy, mid, hard)}")
    old_scenario: dict[str, int] = defaultdict(int)
    old_diff: dict[str, int] = defaultdict(int)
    old_answers = []
    old_evidence = []
    old_ids = set()
    for item in old_queries:
        old_scenario[item["scenario"]] += 1
        old_diff[item["difficulty"]] += 1
        old_ids.add(item["id"])
        old_answers.extend(item.get("answers") or [])
        old_evidence.extend(item.get("evidence") or [])
    for name, count in PLAN.items():
        expected_old = FINAL_PER_SCENARIO[name] - count[0]
        if old_scenario[name] != expected_old:
            raise SystemExit(f"原题场景 {name} 为 {old_scenario[name]}，预期 {expected_old}")
    if dict(old_diff) != {"简单": 33, "中等": 39, "困难": 28}:
        raise SystemExit(f"原题难度变化：{dict(old_diff)}")
    if len(bundle.queries) != 900:
        raise SystemExit(f"新题 {len(bundle.queries)} 条，不是 900")
    ids = [item["id"] for item in bundle.queries]
    if len(set(ids)) != len(ids) or set(ids) & old_ids:
        raise SystemExit("新题 id 冲突")
    queries = [item["query"] for item in bundle.queries]
    old_query_text = {item["query"] for item in old_queries}
    if len(set(queries)) != len(queries) or set(queries) & old_query_text:
        raise SystemExit("问题文本重复")

    old_docs = {
        path.name: path.read_text(encoding="utf-8")
        for path in CORPUS.glob("*.md")
        if not path.name.startswith("60_扩_")
    }
    all_docs = {**old_docs, **bundle.docs}
    compacts = {name: compact(text) for name, text in all_docs.items()}
    stems = [Path(name).stem.lower() for name in all_docs]
    for left in stems:
        for right in stems:
            if left != right and (left in right or right in left):
                raise SystemExit(f"文件名互相包含，会污染文档级命中：{left} / {right}")

    for answer in old_answers:
        token = compact(answer)
        if len(token) < 2:
            continue
        for name, text in bundle.docs.items():
            if token in compact(text):
                raise SystemExit(f"旧答案「{answer}」出现在新文档 {name}")
    for evidence in old_evidence:
        token = compact(evidence)
        if len(evidence) < 16:
            continue
        for name, text in bundle.docs.items():
            if token in compact(text):
                raise SystemExit(f"旧证据出现在新文档 {name}：{evidence}")

    refusal = runner.REFUSAL_ANSWER
    seen_answers = {}
    three_hop = 0
    for item in bundle.queries:
        if item["task"] == "unanswerable" or item["scenario"] == "拒答边界":
            if item["task"] != "unanswerable" or item["scenario"] != "拒答边界":
                raise SystemExit(f"{item['id']} 拒答标记不一致")
            if item["relevant_sources"] or not item["bait_sources"] or item["answers"] != [refusal]:
                raise SystemExit(f"{item['id']} 拒答结构不正确")
            if any(refusal in text for text in bundle.docs.values()):
                raise SystemExit("语料正文写入了拒答标准答案")
            for span in item["evidence"]:
                homes = [name for name in item["bait_sources"] if compact(span) in compacts.get(name, "")]
                if not homes:
                    raise SystemExit(f"{item['id']} 的拒答证据不在诱饵文档中")
            continue
        rel = item["relevant_sources"]
        if not rel or not item["answers"] or not item["evidence"]:
            raise SystemExit(f"{item['id']} 缺少答案、证据或文档")
        gold = "\n".join(bundle.docs[name] for name in rel)
        q_tokens = {tok for tok in runner.tokenize(item["query"]) if len(tok) >= 2}
        if q_tokens and q_tokens.isdisjoint(set(runner.tokenize(gold))):
            raise SystemExit(f"{item['id']} 与 gold 无词法重叠：{item['query']}")
        if len(rel) == 3:
            three_hop += 1
        for answer in item["answers"]:
            if answer in seen_answers:
                raise SystemExit(f"答案重复：{answer} @ {seen_answers[answer]} / {item['id']}")
            seen_answers[answer] = item["id"]
            token = compact(answer)
            if token not in compact(gold):
                raise SystemExit(f"{item['id']} 答案不在 gold：{answer}")
            homes = [name for name, text in compacts.items() if token in text]
            if set(homes) - set(rel) or not homes:
                raise SystemExit(f"{item['id']} 答案落在 gold 之外：{homes} / {answer}")
            if len(rel) == 1 and set(homes) != set(rel):
                raise SystemExit(f"{item['id']} 答案未落在唯一 gold")
            if len(rel) > 1 and len(homes) != 1:
                raise SystemExit(f"{item['id']} 多文档答案应只出现在其中一篇：{homes}")
        for span in item["evidence"]:
            token = compact(span)
            if token not in compact(gold):
                raise SystemExit(f"{item['id']} 证据不在 gold：{span}")
            homes = [name for name, text in compacts.items() if token in text]
            if set(homes) - set(rel):
                raise SystemExit(f"{item['id']} 证据落在 gold 之外：{span}")
        if len(rel) > 1:
            local = {name: compacts[name] for name in rel}
            for name in rel:
                others = "".join(text for other, text in local.items() if other != name)
                if not any(compact(span) in local[name] and compact(span) not in others for span in item["evidence"]):
                    raise SystemExit(f"{item['id']} 的 {name} 没有专属证据")
    if three_hop != 15:
        raise SystemExit(f"三跳题应为 15，当前 {three_hop}")


def main() -> None:
    runner = load_runner()
    old_lines = []
    old_queries = []
    for line in QUERIES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if str(item["id"]).startswith("ex_"):
            continue
        old_lines.append(line)
        old_queries.append(item)
    if len(old_lines) != 100 or old_queries[0]["id"] != "nq_01" or old_queries[-1]["id"] != "sh_05":
        raise SystemExit("原 100 题边界不符，拒绝覆盖")

    entities = build_entities()
    bundle = Bundle()
    add_noise_books(bundle)
    add_single(bundle, entities)
    add_domain(bundle, entities)
    add_claim(bundle, entities)
    add_table(bundle, entities)
    add_fuzzy(bundle, entities)
    add_hop(bundle, entities)
    add_temporal(bundle, entities)
    add_long(bundle, entities)
    add_noise(bundle, entities)
    add_conflict(bundle, entities)
    add_cross(bundle, entities)
    add_multi(bundle, entities)
    add_refusal(bundle, entities)
    validate(bundle, runner, old_queries)

    for path in CORPUS.glob("60_扩_*.md"):
        path.unlink()
    for path in DEMO.glob("[0-9][0-9]_澄川实验室_*.md"):
        path.unlink()
    for path in DEMO.glob("60_扩_*.md"):
        path.unlink()
    for name in sorted(bundle.docs):
        (DEMO / name).write_text(bundle.docs[name], encoding="utf-8")
    new_lines = [dump_query(item) for item in bundle.queries]
    QUERIES.write_text("\n".join([*old_lines, *new_lines]) + "\n", encoding="utf-8")
    by_scenario: dict[str, int] = defaultdict(int)
    by_diff: dict[str, int] = defaultdict(int)
    for item in bundle.queries:
        by_scenario[item["scenario"]] += 1
        by_diff[item["difficulty"]] += 1
    print(f"docs {len(bundle.docs)}")
    print(f"new_queries {len(bundle.queries)}")
    for name in sorted(bundle.docs):
        text = bundle.docs[name]
        print(f"{name}\tlines={text.count(chr(10)) + 1}\tchars={len(text)}")
    print(json.dumps({"scenario": by_scenario, "difficulty": by_diff}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
