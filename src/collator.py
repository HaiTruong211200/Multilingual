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
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    alignment_max_length: int = 128

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        sources = [str(item["source"]) for item in features]
        targets = [str(item["target"]) for item in features]

        # Build one causal sequence and retain the exact source/target spans in it.
        lm_ids, lm_labels, spans = [], [], []
        eos_id = self.tokenizer.eos_token_id
        for item, source, target in zip(features, sources, targets):
            instruction = translation_instruction(item["source_lang"], item["target_lang"])
            instruction_ids = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
            marker_ids = self.tokenizer(TRANSLATION_TARGET_MARKER, add_special_tokens=False)["input_ids"]
            source_ids = self.tokenizer(source, add_special_tokens=False)["input_ids"][: self.alignment_max_length]
            target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"][: self.alignment_max_length]

            fixed = len(instruction_ids) + len(marker_ids) + (1 if eos_id is not None else 0)
            token_budget = self.max_length - fixed
            if token_budget < 2:
                raise ValueError("max_length is too small for the instruction and two aligned spans")
            # Trim the longer span until both fit, retaining at least one token each.
            while len(source_ids) + len(target_ids) > token_budget:
                if len(source_ids) >= len(target_ids) and len(source_ids) > 1:
                    source_ids.pop()
                elif len(target_ids) > 1:
                    target_ids.pop()
                else:
                    break

            source_start = len(instruction_ids)
            source_end = source_start + len(source_ids)
            target_start = source_end + len(marker_ids)
            target_end = target_start + len(target_ids)
            ids = instruction_ids + source_ids + marker_ids + target_ids
            if eos_id is not None:
                ids.append(eos_id)
            labels = ([-100] * target_start) + ids[target_start:]
            lm_ids.append(ids)
            lm_labels.append(labels)
            spans.append((source_start, source_end, target_start, target_end))

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
    """Stage-2 collator whose NTP labels cover only the assistant response."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024

    def _prompt_ids(self, instruction: str, input_text: str) -> list[int]:
        user = instruction_user_prompt(instruction, input_text)
        return self.tokenizer(
            instruction_prompt(user), add_special_tokens=True
        )["input_ids"]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
            response_ids = response_ids[: self.max_length]
            prompt_budget = max(0, self.max_length - len(response_ids))
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            ids = prompt_ids + response_ids
            rows.append((ids, [-100] * len(prompt_ids) + response_ids))

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
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
