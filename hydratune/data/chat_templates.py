"""Named chat templates that can be forced onto a tokenizer.

``dataset.chat_template: tokenizer_default`` keeps whatever template the
tokenizer ships with; any other name overwrites ``tokenizer.chat_template``
with the Jinja source below so the rendered training text is deterministic
regardless of which base model is loaded.
"""

from __future__ import annotations

from hydratune.utils.errors import ChatTemplateError

CHAT_TEMPLATES: dict[str, str] = {
    "chatml": (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    ),
    "llama3": (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
        "+ message['content'] | trim + '<|eot_id|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    ),
    "mistral": (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>\n' + message['content'] + eos_token + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
    ),
}


def resolve_chat_template(name: str) -> str:
    """Return the Jinja source for a named template.

    Raises:
        ChatTemplateError: if the name is not in the registry.
    """
    try:
        return CHAT_TEMPLATES[name]
    except KeyError:
        raise ChatTemplateError(
            f"Unknown chat template {name!r}; available: {sorted(CHAT_TEMPLATES)}"
        ) from None
