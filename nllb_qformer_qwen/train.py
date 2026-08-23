from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from .config import BridgeConfig
from .data import ParallelQACollator, ParallelQADataset, read_json_records
from .model import NLLBQFormerQwen


def parse_args():
    parser = argparse.ArgumentParser(description="Train the frozen NLLB -> Q-Former -> frozen Qwen bridge")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nllb_name", default=BridgeConfig.nllb_name)
    parser.add_argument("--qwen_name", default=BridgeConfig.qwen_name)
    parser.add_argument("--bridge_type", choices=("qformer", "transformer"), default="qformer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--ot_weight", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else None
    cfg = BridgeConfig(
        nllb_name=args.nllb_name, qwen_name=args.qwen_name, bridge_type=args.bridge_type,
        contrastive_weight=args.contrastive_weight, ot_weight=args.ot_weight,
    )
    nllb_tokenizer = AutoTokenizer.from_pretrained(cfg.nllb_name)
    qwen_tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_name)
    if qwen_tokenizer.pad_token_id is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    dataset = ParallelQADataset(read_json_records(args.train_file), seed=args.seed)
    collator = ParallelQACollator(nllb_tokenizer, qwen_tokenizer, cfg.prompt)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    model = NLLBQFormerQwen(cfg, torch_dtype=dtype).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    update_steps = max(1, (len(loader) * args.epochs + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(update_steps * args.warmup_ratio), update_steps)
    trainable, total = model.trainable_parameter_summary()
    print(f"trainable={trainable:,} total={total:,} ratio={trainable / total:.4%}")
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=dtype is not None):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation_steps
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
                print(f"epoch={epoch + 1} step={global_step} total={output.loss.item():.4f} qa={output.qa_loss.item():.4f} global={output.global_loss.item():.4f} ot={output.local_loss.item():.4f}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": model.adapter.state_dict(), "bridge": model.bridge.state_dict(), "projector": model.projector.state_dict()}, output_dir / "bridge.pt")
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    nllb_tokenizer.save_pretrained(output_dir / "nllb_tokenizer")
    qwen_tokenizer.save_pretrained(output_dir / "qwen_tokenizer")


if __name__ == "__main__":
    main()
