"""Tests for the named chat-template registry.

Rendering tests use Jinja2 directly (the same engine transformers uses); they
skip automatically on a base install without it.
"""

from __future__ import annotations

import pytest

from hydratune.data.chat_templates import CHAT_TEMPLATES, resolve_chat_template
from hydratune.utils.errors import ChatTemplateError

MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]


def test_registry_covers_all_named_templates() -> None:
    assert set(CHAT_TEMPLATES) == {"chatml", "llama3", "mistral", "zephyr"}


def test_resolve_returns_jinja_source() -> None:
    assert "{% for message in messages %}" in resolve_chat_template("chatml")


def test_resolve_unknown_template_raises() -> None:
    with pytest.raises(ChatTemplateError, match="Unknown chat template"):
        resolve_chat_template("gpt-neo")


def render(name: str, add_generation_prompt: bool = False) -> str:
    jinja2 = pytest.importorskip("jinja2")
    template = jinja2.Environment().from_string(resolve_chat_template(name))
    return template.render(
        messages=MESSAGES,
        add_generation_prompt=add_generation_prompt,
        bos_token="<s>",
        eos_token="</s>",
    )


def test_chatml_rendering() -> None:
    text = render("chatml")
    assert "<|im_start|>user\nHi<|im_end|>" in text
    assert "<|im_start|>assistant\nHello!<|im_end|>" in text


def test_chatml_generation_prompt() -> None:
    assert render("chatml", add_generation_prompt=True).endswith("<|im_start|>assistant\n")


def test_llama3_rendering() -> None:
    text = render("llama3")
    assert text.startswith("<s>")
    assert "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>" in text


def test_mistral_rendering() -> None:
    text = render("mistral")
    assert "[INST] Hi [/INST]" in text
    assert "Hello!</s>" in text


def test_zephyr_rendering() -> None:
    text = render("zephyr")
    assert "<|user|>\nHi</s>" in text
    assert "<|assistant|>\nHello!</s>" in text
