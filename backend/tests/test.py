"""LLM 纯文本对话测试"""

import argparse
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.llm import get_llm


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
    provider = os.getenv("TEST_LLM_PROVIDER") or None
    model = os.getenv("TEST_LLM_MODEL") or None
    temperature = os.getenv("TEST_LLM_TEMPERATURE")
    timeout = os.getenv("TEST_LLM_TIMEOUT")

    return get_llm(
        provider=provider,
        model=model,
        temperature=float(temperature) if temperature else None,
        timeout=int(timeout) if timeout else None,
    )


def _invoke_llm(messages):
    try:
        return _build_llm().invoke(messages)
    except Exception as exc:
        pytest.fail(f"LLM 调用失败，请检查 .env、模型服务和网络配置: {exc}")


def _build_messages(prompt: str, system_prompt: str = ""):
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    return messages


def test_llm_plain_text_generation():
    """单轮纯文本生成测试"""
    response = _invoke_llm(
        [
            SystemMessage(content="你是一个只输出纯文本的中文助手，不要使用 markdown、列表符号或代码块。"),
            HumanMessage(content="用一句话介绍你自己。"),
        ]
    )
    text = _extract_text(response)

    print(f"Plain response: {text}")
    assert text
    assert isinstance(text, str)
    assert "```" not in text
    assert "*" not in text


def test_llm_custom_prompt():
    """自定义输入测试"""
    custom_prompt = os.getenv("CUSTOM_LLM_PROMPT", "").strip() or "用一句话解释什么是 RAG。"
    system_prompt = os.getenv("CUSTOM_LLM_SYSTEM_PROMPT", "").strip()
    response = _invoke_llm(_build_messages(custom_prompt, system_prompt=system_prompt))
    text = _extract_text(response)

    print(f"Custom response: {text}")
    assert text
    assert isinstance(text, str)


def test_llm_text_conversation_with_history():
    """两轮纯文本对话测试"""
    messages = [
        SystemMessage(content="你是一个只输出纯文本的中文助手，回答要简洁。"),
        HumanMessage(content="请用一句话说明什么是向量检索。"),
    ]
    first_reply = _extract_text(_invoke_llm(messages))
    print(f"First reply: {first_reply}")
    assert first_reply

    messages.append(AIMessage(content=first_reply))
    messages.append(HumanMessage(content="再用一句话说明它为什么适合语义搜索。"))
    second_reply = _extract_text(_invoke_llm(messages))
    print(f"Second reply: {second_reply}")

    assert second_reply
    assert isinstance(second_reply, str)
    assert "```" not in second_reply
    assert "*" not in second_reply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live LLM text chat tests.")
    parser.add_argument("--prompt", help="Custom prompt text for a one-off test.")
    parser.add_argument("--system-prompt", help="Optional system prompt for the custom test.")
    parser.add_argument("--provider", help="LLM provider override.")
    parser.add_argument("--model", help="LLM model override.")
    parser.add_argument("--temperature", type=float, help="LLM temperature override.")
    parser.add_argument("--timeout", type=int, help="LLM timeout override.")
    args = parser.parse_args()

    if args.provider:
        os.environ["TEST_LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["TEST_LLM_MODEL"] = args.model
    if args.temperature is not None:
        os.environ["TEST_LLM_TEMPERATURE"] = str(args.temperature)
    if args.timeout is not None:
        os.environ["TEST_LLM_TIMEOUT"] = str(args.timeout)

    if args.prompt:
        response = _invoke_llm(_build_messages(args.prompt, system_prompt=args.system_prompt or ""))
        print(_extract_text(response))
    else:
        test_llm_plain_text_generation()
        test_llm_custom_prompt()
        test_llm_text_conversation_with_history()
    print("LLM text chat tests passed.")
