"""All prompt text used by both training stages."""


def translation_instruction(source_lang: str, target_lang: str) -> str:
    return f"Translate from {source_lang} to {target_lang}.\nSource: "


TRANSLATION_TARGET_MARKER = "\nTranslation: "


def summarization_instruction(language: str) -> str:
    return f"Summarize the following article in {language}."


def instruction_user_prompt(instruction: str, input_text: str) -> str:
    return instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"


def instruction_prompt(user_prompt: str) -> str:
    return f"### Instruction:\n{user_prompt}\n\n### Response:\n"
