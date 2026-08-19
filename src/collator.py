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

    @staticmethod
    def _token_span(offsets, char_start: int, char_end: int, name: str):
        indices = [
            index for index, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end
        ]
        if not indices:
            raise ValueError(f"Could not map non-empty {name} text to tokens.")
        return indices[0], indices[-1] + 1

    def _encode_full_sample(
        self, instruction: str, source: str, target: str
    ) -> tuple[list[int], int, int, int, int]:
        """Render source+target, tokenize once, then derive spans from offsets."""
        if self.prompt_format == "chat":
            if not self.tokenizer.chat_template:
                raise ValueError("prompt_format='chat' requires tokenizer.chat_template")
            user_content = instruction + source
            rendered = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": target},
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
            )
            user_start = rendered.rfind(user_content)
            target_char_start = rendered.rfind(target)
            if user_start < 0 or target_char_start < 0:
                raise ValueError("Chat template changed source/target text; spans are unavailable")
            source_char_start = user_start + len(instruction)
            source_char_end = source_char_start + len(source)
            target_char_end = target_char_start + len(target)
            add_special_tokens = False  # chat template already rendered them
        else:
            rendered = instruction + source + TRANSLATION_TARGET_MARKER + target
            source_char_start = len(instruction)
            source_char_end = source_char_start + len(source)
            target_char_start = source_char_end + len(TRANSLATION_TARGET_MARKER)
            target_char_end = target_char_start + len(target)
            add_special_tokens = True

        encoded = self.tokenizer(
            rendered,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise ValueError("Alignment requires a fast tokenizer with offset_mapping")
        source_start, source_end = self._token_span(
            offsets, source_char_start, source_char_end, "source"
        )
        target_start, target_end = self._token_span(
            offsets, target_char_start, target_char_end, "target"
        )
        return encoded["input_ids"], source_start, source_end, target_start, target_end

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Tokenization, labels and alignment spans are prepared once in
        # prepare_alignment_dataset(). This collator only pads a ready batch.
        if features and "input_ids" in features[0] and "labels" in features[0]:
            width = max(len(item["input_ids"]) for item in features)
            input_ids = torch.full(
                (len(features), width), self.tokenizer.pad_token_id, dtype=torch.long
            )
            attention_mask = torch.zeros_like(input_ids)
            labels = torch.full_like(input_ids, -100)
            for index, item in enumerate(features):
                length = len(item["input_ids"])
                input_ids[index, :length] = torch.tensor(item["input_ids"])
                attention_mask[index, :length] = 1
                labels[index, :length] = torch.tensor(item["labels"])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "source_start_positions": torch.tensor(
                    [item["source_start_positions"] for item in features]
                ),
                "source_end_positions": torch.tensor(
                    [item["source_end_positions"] for item in features]
                ),
                "target_start_positions": torch.tensor(
                    [item["target_start_positions"] for item in features]
                ),
                "target_end_positions": torch.tensor(
                    [item["target_end_positions"] for item in features]
                ),
            }
        raise ValueError(
            "MultilingualDataCollator expects a dataset processed by "
            "prepare_alignment_dataset(); raw text is not accepted."
        )

        # Legacy implementation below is intentionally unreachable and will be
        # removed after old notebooks have migrated to prepare_alignment_dataset.
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
            ids, source_start, source_end, target_start, target_end = (
                self._encode_full_sample(instruction, source, target)
            )

            # ------------------------------------------------------------------
            # Block 3: Xác định các interval [start, end). Stage 1 không truncate
            # source/target; toàn bộ cặp câu được giữ nguyên cho NTP/CL/OT.
            # Các index này trỏ trực tiếp vào sequence được đưa qua model.
            # ------------------------------------------------------------------
            if eos_id is not None and (not ids or ids[-1] != eos_id):
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
    training_mode: str = "finetune"

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
            row_labels = (
                [-100] * len(prompt_ids) + response_ids
                if self.training_mode == "finetune"
                else ids.copy()
            )
            rows.append((ids, row_labels))

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


@dataclass
class TranslationInferenceCollator:
    """Only pad samples already tokenized by prepare_data; never tokenize here."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Left-pad to the longest pre-tokenized prompt in this batch."""
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full(
            (len(features), width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, feature in enumerate(features):
            ids = feature["input_ids"]
            length = len(ids)
            input_ids[index, -length:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[index, -length:] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "rows": [
                {key: value for key, value in feature.items() if key != "input_ids"}
                for feature in features
            ],
            "prompt_width": width,
        }
