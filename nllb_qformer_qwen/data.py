from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


def read_json_records(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    return payload["data"] if isinstance(payload, dict) and "data" in payload else payload


class ParallelQADataset(Dataset):
    """Samples two different languages from each semantic group."""

    REQUIRED = {"question_id", "language", "question", "answer", "semantic_group_id"}

    def __init__(self, records: list[dict], seed: int = 42):
        groups = defaultdict(list)
        for row in records:
            missing = self.REQUIRED - row.keys()
            if missing:
                raise ValueError(f"Missing fields {sorted(missing)} in record {row}")
            groups[str(row["semantic_group_id"])].append(row)
        self.groups = [rows for rows in groups.values() if len({r["language"] for r in rows}) >= 2]
        if not self.groups:
            raise ValueError("No semantic group contains at least two languages")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict:
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        left, right = rng.sample(self.groups[index], 2)
        return {"left": left, "right": right}


class ParallelQACollator:
    def __init__(self, nllb_tokenizer, qwen_tokenizer, prompt: str, max_question_length: int = 256, max_answer_length: int = 128):
        self.nllb_tokenizer = nllb_tokenizer
        self.qwen_tokenizer = qwen_tokenizer
        self.prompt = prompt
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

    def __call__(self, examples: list[dict]) -> dict:
        rows = [example[side] for example in examples for side in ("left", "right")]
        questions = self.nllb_tokenizer(
            [str(row["question"]) for row in rows], padding=True, truncation=True,
            max_length=self.max_question_length, return_tensors="pt",
        )
        prompt_ids = self.qwen_tokenizer(
            self.prompt, add_special_tokens=True, truncation=True, max_length=self.max_question_length
        )["input_ids"]
        answer_ids, eos = [], self.qwen_tokenizer.eos_token_id
        for row in rows:
            ids = self.qwen_tokenizer(
                str(row["answer"]), add_special_tokens=False, truncation=True, max_length=self.max_answer_length
            )["input_ids"]
            if eos is not None:
                ids = ids + [eos]
            answer_ids.append(ids)
        max_answer = max(map(len, answer_ids))
        pad = self.qwen_tokenizer.pad_token_id
        answer_tensor = torch.full((len(rows), max_answer), pad, dtype=torch.long)
        answer_mask = torch.zeros_like(answer_tensor)
        for i, ids in enumerate(answer_ids):
            answer_tensor[i, :len(ids)] = torch.tensor(ids)
            answer_mask[i, :len(ids)] = 1
        return {
            "nllb_input_ids": questions["input_ids"],
            "nllb_attention_mask": questions["attention_mask"],
            "prompt_input_ids": torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).expand(len(rows), -1),
            "answer_input_ids": answer_tensor,
            "answer_attention_mask": answer_mask,
            "pair_batch_size": len(examples),
        }

