"""Batch construction for language modeling and cross-lingual alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from .prompts import (
    TRANSLATION_TARGET_MARKER,
    instruction_prompt,
    instruction_user_prompt,
    translation_instruction,
)


@dataclass
class MultilingualDataCollator:
    """Tạo batch Stage 1 cho NTP và hai objective alignment.

    Mỗi sample có dạng:
    [instruction] [source] [translation marker] [target] [EOS]

    Ngoài input/label chuẩn của causal LM, collator trả vị trí đầu/cuối của
    source và target để model cắt đúng hai vùng từ cùng một hidden state.
    """

    tokenizer: PreTrainedTokenizerBase
    prompt_format: str = "plain"
    enable_thinking: bool = False

    def _chat_prompt_and_source_span(
        self, instruction: str, source: str
    ) -> tuple[list[int], int, int]:
        """Render the real chat template and locate source tokens via offsets."""
        if not self.tokenizer.chat_template:
            raise ValueError(
                "prompt_format='chat' requires a tokenizer with chat_template."
            )
        user_content = instruction + source
        try:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError as error:
            raise ValueError(
                "This tokenizer/chat template does not accept enable_thinking. "
                "Use --prompt_format plain or update the tokenizer template."
            ) from error
        user_start = rendered.rfind(user_content)
        if user_start < 0:
            raise ValueError("Chat template changed the user text; cannot locate source span.")
        char_start = user_start + len(instruction)
        char_end = char_start + len(source)
        encoded = self.tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise ValueError("Chat span detection requires a fast tokenizer with offsets.")
        indices = [
            index for index, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end
        ]
        if source and not indices:
            raise ValueError("Could not map source characters to chat-template tokens.")
        source_start = indices[0] if indices else len(encoded["input_ids"])
        source_end = indices[-1] + 1 if indices else source_start
        return encoded["input_ids"], source_start, source_end

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ------------------------------------------------------------------
        # Block 1: Lấy text nguồn/đích từ các sample đã được prepare_data.
        # ------------------------------------------------------------------
        sources = [str(item["source"]) for item in features]
        targets = [str(item["target"]) for item in features]

        # ------------------------------------------------------------------
        # Block 2: Tokenize từng thành phần và ghép thành một causal sequence.
        # Tokenize riêng từng phần giúp biết chính xác index của source/target.
        # ------------------------------------------------------------------
        lm_ids, lm_labels, spans = [], [], []
        eos_id = self.tokenizer.eos_token_id
        for item, source, target in zip(features, sources, targets):
            # Instruction chứa hướng dịch; marker phân cách source với target.
            instruction = translation_instruction(item["source_lang"], item["target_lang"])
            target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

            if self.prompt_format == "chat":
                prompt_ids, source_start, source_end = self._chat_prompt_and_source_span(
                    instruction, source
                )
                target_start = len(prompt_ids)
                ids = prompt_ids + target_ids
            else:
                instruction_ids = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
                marker_ids = self.tokenizer(TRANSLATION_TARGET_MARKER, add_special_tokens=False)["input_ids"]
                source_ids = self.tokenizer(source, add_special_tokens=False)["input_ids"]
                source_start = len(instruction_ids)
                source_end = source_start + len(source_ids)
                target_start = source_end + len(marker_ids)
                ids = instruction_ids + source_ids + marker_ids + target_ids

            # ------------------------------------------------------------------
            # Block 3: Xác định các interval [start, end). Stage 1 không truncate
            # source/target; toàn bộ cặp câu được giữ nguyên cho NTP/CL/OT.
            # Các index này trỏ trực tiếp vào sequence được đưa qua model.
            # ------------------------------------------------------------------
            target_end = target_start + len(target_ids)
            if eos_id is not None:
                ids.append(eos_id)

            # Không âm thầm cắt dữ liệu. Nếu model công bố context window hữu
            # hạn và sample vượt giới hạn, báo lỗi để người dùng lọc dữ liệu hoặc
            # chọn model có context dài hơn.
            model_limit = self.tokenizer.model_max_length
            if model_limit < 10**9 and len(ids) > model_limit:
                raise ValueError(
                    f"Translation sample has {len(ids)} tokens, exceeding "
                    f"model_max_length={model_limit}."
                )

            # ------------------------------------------------------------------
            # Block 4: Response-only NTP. Instruction, source và marker nhận
            # label=-100 nên CrossEntropyLoss bỏ qua; target và EOS được học.
            # ------------------------------------------------------------------
            labels = ([-100] * target_start) + ids[target_start:]
            lm_ids.append(ids)
            lm_labels.append(labels)
            spans.append((source_start, source_end, target_start, target_end))

        # ------------------------------------------------------------------
        # Block 5: Right padding theo sequence dài nhất trong batch. PAD có attention_mask=0 và
        # labels=-100, vì vậy không tham gia attention hợp lệ hay NTP loss.
        # ------------------------------------------------------------------
        batch_size, width = len(lm_ids), max(map(len, lm_ids))
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        labels = torch.full((batch_size, width), -100, dtype=torch.long)
        for i, (ids, row_labels) in enumerate(zip(lm_ids, lm_labels)):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids)
            attention_mask[i, :length] = 1
            labels[i, :length] = torch.tensor(row_labels)

        # ------------------------------------------------------------------
        # Block 6: Batch đầu ra. Bốn position tensors được model chuyển thành
        # source_mask/target_mask để mean pooling, contrastive và OT.
        # ------------------------------------------------------------------
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_start_positions": torch.tensor([x[0] for x in spans]),
            "source_end_positions": torch.tensor([x[1] for x in spans]),
            "target_start_positions": torch.tensor([x[2] for x in spans]),
            "target_end_positions": torch.tensor([x[3] for x in spans]),
        }


@dataclass
class InstructionDataCollator:
    """Tạo batch Stage 2 với NTP loss chỉ áp dụng lên response/output."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024
    prompt_format: str = "plain"
    enable_thinking: bool = False

    def _prompt_ids(self, instruction: str, input_text: str) -> list[int]:
        # ------------------------------------------------------------------
        # Block 1: Dựng plain-text prompt cho base model. Không dùng chat
        # template hay system/user/assistant special tokens.
        # ------------------------------------------------------------------
        user = instruction_user_prompt(instruction, input_text)
        if self.prompt_format == "chat":
            if not self.tokenizer.chat_template:
                raise ValueError(
                    "prompt_format='chat' requires a tokenizer with chat_template."
                )
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        return self.tokenizer(
            instruction_prompt(user), add_special_tokens=True
        )["input_ids"]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ------------------------------------------------------------------
        # Block 2: Tokenize prompt và response riêng để xác định ranh giới label.
        # ------------------------------------------------------------------
        rows = []
        eos_id = self.tokenizer.eos_token_id
        for row in features:
            prompt_ids = self._prompt_ids(
                str(row["instruction"]), str(row.get("input") or "")
            )
            response_ids = self.tokenizer(
                str(row["output"]), add_special_tokens=False
            )["input_ids"]
            if eos_id is not None:
                response_ids.append(eos_id)

            # ------------------------------------------------------------------
            # Block 3: Ưu tiên giữ response vì đây là vùng có supervision.
            # Response được cắt theo max_length trước; prompt chỉ dùng phần
            # ngân sách còn lại và giữ phần cuối gần response nhất.
            # ------------------------------------------------------------------
            response_ids = response_ids[: self.max_length]
            prompt_budget = max(0, self.max_length - len(response_ids))
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            ids = prompt_ids + response_ids

            # Prompt nhận -100; chỉ response và EOS được tính NTP loss.
            rows.append((ids, [-100] * len(prompt_ids) + response_ids))

        # ------------------------------------------------------------------
        # Block 4: Right padding giống Stage 1. Padding labels luôn là -100.
        # ------------------------------------------------------------------
        width = max(len(ids) for ids, _ in rows)
        input_ids = torch.full(
            (len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        for index, (ids, row_labels) in enumerate(rows):
            length = len(ids)
            input_ids[index, :length] = torch.tensor(ids)
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(row_labels)
        # Stage 2 không cần span positions vì không tính contrastive/OT.
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
