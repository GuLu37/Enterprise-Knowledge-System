"""DeepSeek 连接与纯文本输出测试"""

import argparse
import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_deepseek_llm


def _extract_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _build_llm():
    model = os.getenv("TEST_DEEPSEEK_MODEL") or None
    temperature = os.getenv("TEST_DEEPSEEK_TEMPERATURE")
    timeout = os.getenv("TEST_DEEPSEEK_TIMEOUT")

    return get_deepseek_llm(
        model=model,
        temperature=float(temperature) if temperature else None,
        timeout=int(timeout) if timeout else None,
    )


def _invoke(prompt: str, system_prompt: str = "") -> str:
    try:
        llm = _build_llm()
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = llm.invoke(messages)
        return _extract_text(response)
    except Exception as exc:
        pytest.fail(f"DeepSeek 连接或调用失败，请检查 API_KEY、BASE_URL、MODEL 和网络: {exc}")


def test_deepseek_one_sentence():
    """验证 DeepSeek 可连接并返回一句话"""
    prompt = os.getenv("TEST_DEEPSEEK_PROMPT", "").strip() or "请只用一句话回答：你好。"
    text = _invoke(
        prompt=prompt,
        system_prompt="你是一个只输出纯文本的中文助手，回答必须简洁，只输出一句话。",
    )

    print(f"DeepSeek response: {text}")
    assert text
    assert isinstance(text, str)
    assert len(text) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepSeek connection test.")
    parser.add_argument("--prompt", help="Custom prompt text.")
    parser.add_argument("--system-prompt", help="Optional system prompt.")
    parser.add_argument("--model", help="DeepSeek model override.")
    parser.add_argument("--temperature", type=float, help="Temperature override.")
    parser.add_argument("--timeout", type=int, help="Timeout override.")
    args = parser.parse_args()

    if args.model:
        os.environ["TEST_DEEPSEEK_MODEL"] = args.model
    if args.temperature is not None:
        os.environ["TEST_DEEPSEEK_TEMPERATURE"] = str(args.temperature)
    if args.timeout is not None:
        os.environ["TEST_DEEPSEEK_TIMEOUT"] = str(args.timeout)

    prompt = args.prompt or "请只用一句话回答：你好。"
    text = _invoke(prompt=prompt, system_prompt=args.system_prompt or "你是一个只输出纯文本的中文助手，回答必须简洁，只输出一句话。")
    print(text)
