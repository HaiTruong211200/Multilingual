"""All prompt text used by both training stages."""
from __future__ import annotations

import random


TRANSLATION_INSTRUCTION_TEMPLATES = (
    "Translate this text from {src_lang} to {tgt_lang}.",
    "Convert this sentence from {src_lang} to {tgt_lang}.",
    "Change this paragraph from {src_lang} to {tgt_lang}.",
    "Render this message from {src_lang} to {tgt_lang}.",
    "Translate this phrase from {src_lang} to {tgt_lang}.",
    "Turn this text from {src_lang} to {tgt_lang}.",
    "Rewrite this statement from {src_lang} to {tgt_lang}.",
    "Provide a translation from {src_lang} to {tgt_lang} for this text.",
    "Offer a {tgt_lang} translation for this text from {src_lang}.",
    "Give a {tgt_lang} version of this text from {src_lang}.",
)


def translation_instruction(
    source_lang: str,
    target_lang: str,
    template_index: int | None = None,
) -> str:
    """Render a translation prompt; random by default, indexable for eval."""
    if template_index is None:
        template = random.choice(TRANSLATION_INSTRUCTION_TEMPLATES)
    else:
        template = TRANSLATION_INSTRUCTION_TEMPLATES[template_index]
    instruction = template.format(src_lang=source_lang, tgt_lang=target_lang)
    return f"{instruction}\nSource: "


TRANSLATION_TARGET_MARKER = "\nTranslation: "


def summarization_instruction(language: str) -> str:
    return f"Summarize the following article in {language}."


def instruction_user_prompt(instruction: str, input_text: str) -> str:
    return instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"


def instruction_prompt(user_prompt: str) -> str:
    return f"### Instruction:\n{user_prompt}\n\n### Response:\n"
