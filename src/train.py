"""Single training entry point for alignment (stage 1) and instruction FT (stage 2)."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from collections import defaultdict
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

from .collator import InstructionDataCollator, MultilingualDataCollator
from .model import MultilingualAlignmentModel
from .layer_scheduler import POTSAStyleLayerScheduler
from .prepare_data import (
    load_instruction_dataset,
    load_parallel_dataset,
    prepare_alignment_dataset,
)

LOGGER = logging.getLogger("multilingual_training")


class ComponentLoggingTrainer(Trainer):
    """Average every loss component over a Trainer logging interval."""

    COMPONENT_KEYS = (
        "model_total_loss", "ntp_loss", "contrastive_loss", "ot_loss",
        "weighted_contrastive_loss", "weighted_ot_loss",
    )

    def __init__(self, *args, stage: str, layer_scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage
        self.layer_scheduler = layer_scheduler
        self._scheduler_activation_step = 0
        self._scheduler_history = {
            "steps": [],
            "selected_layers": [],
            "next_layers": [],
            "task_losses": [],
            "contrastive_losses": [],
            "total_losses": [],
            "rewards": [],
            "q_values": [],
        }
        # The alignment wrapper accepts **kwargs for compatibility but does not
        # consume num_items_in_batch when reducing NTP/CL/OT. Tell Trainer not to
        # pass or assume handling of that argument. For the standard HF model in
        # stage 2, preserve Trainer's own signature-based detection.
        if stage == "alignment":
            self.model_accepts_loss_kwargs = False
        self._sums = {"train": defaultdict(float), "eval": defaultdict(float)}
        self._counts = {"train": 0, "eval": 0}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Delegate the actual loss computation to Hugging Face Trainer so label
        # smoothing, custom loss functions, item-count normalization, and future
        # Trainer behavior remain intact. This override only observes the output.
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        mode = "train" if model.training else "eval"
        # This is the authoritative loss returned by Trainer's own compute_loss
        # and therefore the value actually sent into backward.
        self._sums[mode]["trainer_loss"] += float(loss.detach().float().mean())
        if self.stage == "alignment":
            for key in self.COMPONENT_KEYS:
                value = outputs.get(key)
                if value is not None:
                    self._sums[mode][key] += float(value.float().mean())
            if mode == "train" and self.layer_scheduler is not None:
                selected_layer = self.layer_scheduler.current_layer
                task_loss = float(outputs.ntp_loss.float().mean())
                reward = self.layer_scheduler.observe(selected_layer, task_loss)
                state = self.layer_scheduler.states[selected_layer]
                self._sums[mode]["scheduler_layer"] += float(selected_layer)
                self._sums[mode]["scheduler_q_value"] += state.q_value
                if reward is not None:
                    self._sums[mode]["scheduler_reward"] += reward
                next_layer = self.layer_scheduler.sample_next(
                    self.state.global_step + 1
                )
                self._record_scheduler_history(
                    selected_layer=selected_layer,
                    next_layer=next_layer,
                    task_loss=task_loss,
                    contrastive_loss=float(outputs.contrastive_loss.float().mean()),
                    total_loss=float(loss.detach().float().mean()),
                    reward=reward,
                )
                self.accelerator.unwrap_model(model).align_layer = (
                    self.layer_scheduler.current_layer
                )
        else:
            self._sums[mode]["ntp_loss"] += float(loss.detach().float().mean())
        self._counts[mode] += 1
        if not return_outputs:
            return loss
        if self.stage == "alignment":
            # ModelOutput is immutable with respect to pop/delete. Return a clean
            # standard output instead of mutating the alignment output; this also
            # prevents scalar logging fields from being interpreted as logits.
            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return loss, outputs

    def _record_scheduler_history(
        self,
        selected_layer,
        next_layer,
        task_loss,
        contrastive_loss,
        total_loss,
        reward,
    ):
        """Keep compact per-activation scheduler history in RAM."""
        self._scheduler_activation_step += 1
        if not self.is_world_process_zero():
            return
        states = self.layer_scheduler.states
        history = self._scheduler_history
        history["steps"].append(self.state.global_step)
        history["selected_layers"].append(selected_layer)
        history["next_layers"].append(next_layer)
        history["task_losses"].append(task_loss)
        history["contrastive_losses"].append(contrastive_loss)
        history["total_losses"].append(total_loss)
        history["rewards"].append(float("nan") if reward is None else reward)
        history["q_values"].append(
            [states[layer].q_value for layer in self.layer_scheduler.candidate_layers]
        )

    def _save_scheduler_history(self, output_dir):
        """Write compact tensor columns once after training finishes."""
        if not self._scheduler_history["steps"] or not self.is_world_process_zero():
            return
        history = self._scheduler_history
        compact_history = {
            "candidate_layers": torch.tensor(
                self.layer_scheduler.candidate_layers, dtype=torch.int16
            ),
            "steps": torch.tensor(history["steps"], dtype=torch.int64),
            "selected_layers": torch.tensor(history["selected_layers"], dtype=torch.int16),
            "next_layers": torch.tensor(history["next_layers"], dtype=torch.int16),
            "task_losses": torch.tensor(history["task_losses"], dtype=torch.float32),
            "contrastive_losses": torch.tensor(
                history["contrastive_losses"], dtype=torch.float32
            ),
            "total_losses": torch.tensor(history["total_losses"], dtype=torch.float32),
            "rewards": torch.tensor(history["rewards"], dtype=torch.float32),
            "q_values": torch.tensor(history["q_values"], dtype=torch.float32),
        }
        os.makedirs(output_dir, exist_ok=True)
        torch.save(compact_history, os.path.join(output_dir, "scheduler_history.pt"))

    def log(self, logs, start_time=None):
        mode = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        count = self._counts[mode]
        if count:
            prefix = "eval" if mode == "eval" else "train"
            component_logs = {}
            for key, total in self._sums[mode].items():
                name = f"{prefix}/{key}"
                value = total / count
                logs[name] = value
                component_logs[name] = value
            # if component_logs:
            #     self.accelerator.print(
            #         "Loss components | "
            #         + " | ".join(
            #             f"{key}={value:.6f}"
            #             for key, value in component_logs.items()
            #         )
            #     )
            self._sums[mode].clear()
            self._counts[mode] = 0
        if start_time is None:
            return super().log(logs)
        return super().log(logs, start_time)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save only the nested Stage 1 LM/adapter, not the outer wrapper."""
        output_dir = output_dir or self.args.output_dir
        if not _internal_call:
            self._save_scheduler_history(output_dir)
        if self.stage != "alignment":
            return super().save_model(output_dir, _internal_call=_internal_call)

        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        language_model = unwrapped.lm
        language_model.save_pretrained(
            output_dir,
            safe_serialization=self.args.save_safetensors,
        )
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if self.layer_scheduler is not None:
            torch.save(
                self.layer_scheduler.state_dict(),
                os.path.join(output_dir, "layer_scheduler.pt"),
            )
        LOGGER.info(
            "Saved Stage 1 adapter via overridden save_model() to %s",
            output_dir,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Optional YAML config; CLI values override it.")
    parser.add_argument("--stage", required=True, choices=["alignment", "instruction"])
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="data/MT")
    parser.add_argument("--train_file")
    parser.add_argument("--validation_file")
    parser.add_argument("--language_pairs", default="all")
    parser.add_argument("--direction", choices=["forward", "reverse", "both"], default="forward")
    parser.add_argument("--xlsum_dir", default="data/XLSum/XLSum")
    parser.add_argument("--bactrian_dir", default="data/Bactrian-Multilingual_Instruction")
    parser.add_argument("--languages", default="en,km,my,th,vi")
    parser.add_argument("--bactrian_validation_ratio", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--prompt_format", choices=["plain", "chat"], default="plain")
    parser.add_argument(
        "--training_mode", choices=["finetune", "continue"], default="finetune"
    )
    parser.add_argument(
        "--enable_thinking", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--ot_weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--align_layer", type=int, default=-1)
    parser.add_argument(
        "--candidate_layers",
        default=None,
        help="Comma-separated hidden-state indices; enables reward-guided scheduling.",
    )
    parser.add_argument("--reward_ema_rho", type=float, default=0.1)
    parser.add_argument("--ucb_beta", type=float, default=0.5)
    parser.add_argument("--layer_temperature", type=float, default=1.0)
    parser.add_argument("--layer_warmup_steps", type=int, default=100)
    parser.add_argument(
        "--force_each_layer_once",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attention_mass_weight", type=float, default=0.5)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_seed", type=int, default=None,
        help="Seed for Trainer's train-dataloader shuffle; defaults to --seed.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated linear module names.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument(
        "--save_strategy", choices=["steps", "epoch"], default="epoch"
    )
    parser.add_argument(
        "--eval_strategy", choices=["steps", "epoch"], default="epoch"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        default="cosine",
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument(
        "--report_to", choices=["none", "tensorboard"], default="tensorboard"
    )
    config_args, _ = parser.parse_known_args()
    if config_args.config:
        config_path = Path(config_args.config)
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise ValueError("The YAML config root must be a mapping")

        flattened_config = {}
        for section, values in config.items():
            if isinstance(values, dict):
                flattened_config.update(values)
            else:
                flattened_config[section] = values
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(flattened_config) - valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown YAML config keys: {unknown_keys}")
        parser.set_defaults(**flattened_config)
    return parser.parse_args()


def apply_lora(model, args):
    """Attach trainable LoRA adapters while freezing the original LM weights."""
    if not args.use_lora:
        return model
    if isinstance(model, PeftModel):
        # A saved Stage 1 adapter is already attached and trainable.
        return model
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_causal_lm(model_name_or_path: str, trust_remote_code: bool = False):
    """Load either a regular HF checkpoint or a local PEFT adapter directory."""
    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    if adapter_config_path.exists():
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        return PeftModel.from_pretrained(
            base_model,
            model_name_or_path,
            is_trainable=True,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )


def build_stage(args, tokenizer):
    if args.stage == "alignment":
        dataset = load_parallel_dataset(
            args.data_dir, args.language_pairs, args.direction,
            args.train_file, args.validation_file,
        )
        dataset = prepare_alignment_dataset(
            dataset, tokenizer, args.prompt_format, args.enable_thinking,
            args.training_mode,
        )
        model = MultilingualAlignmentModel(
            args.model_name_or_path,
            contrastive_weight=args.contrastive_weight,
            ot_weight=args.ot_weight,
            temperature=args.temperature,
            align_layer=args.align_layer,
            attention_mass_weight=args.attention_mass_weight,
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_iterations=args.sinkhorn_iterations,
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )
        model.lm = apply_lora(model.lm, args)
        collator = MultilingualDataCollator(tokenizer)
        return dataset, model, collator

    dataset = load_instruction_dataset(
        args.xlsum_dir, args.bactrian_dir, args.languages,
        args.bactrian_validation_ratio, args.seed,
    )
    model = load_causal_lm(args.model_name_or_path, args.trust_remote_code)
    model = apply_lora(model, args)
    return dataset, model, InstructionDataCollator(
        tokenizer, args.max_length, args.prompt_format, args.enable_thinking,
        args.training_mode,
    )


def main() -> None:
    args = parse_args()
    if args.data_seed is None:
        args.data_seed = args.seed
    if args.logging_steps <= 0 or args.save_steps <= 0 or args.eval_steps <= 0:
        raise ValueError("logging_steps, save_steps, and eval_steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if args.save_strategy != args.eval_strategy:
        raise ValueError(
            "save_strategy and eval_strategy must match when load_best_model_at_end=True"
        )
    if args.save_strategy == "steps" and args.save_steps % args.eval_steps != 0:
        raise ValueError(
            "save_steps must be a multiple of eval_steps when using step strategies"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(args.seed)
    LOGGER.info("Building stage=%s", args.stage)
    LOGGER.info("Model/checkpoint=%s | output=%s", args.model_name_or_path, args.output_dir)
    LOGGER.info(
        "Prompt format=%s | thinking=%s | training_mode=%s",
        args.prompt_format, args.enable_thinking, args.training_mode,
    )
    if args.enable_thinking:
        LOGGER.warning(
            "Thinking is enabled, but the current datasets contain direct targets "
            "without supervised reasoning traces."
        )
    LOGGER.info(
        "Precision bf16=%s fp16=%s | batch=%d | accumulation=%d | epochs=%s | lr=%g",
        args.bf16, args.fp16, args.batch_size,
        args.gradient_accumulation_steps, args.epochs, args.learning_rate,
    )
    LOGGER.info(
        "Random seed=%d | data shuffle seed=%d | train shuffle=True | eval shuffle=False",
        args.seed, args.data_seed,
    )
    LOGGER.info(
        "Schedule=%s | warmup_steps=%d | logging_steps=%d | save=%s/%d | eval=%s/%d",
        args.lr_scheduler_type, args.warmup_steps, args.logging_steps,
        args.save_strategy, args.save_steps, args.eval_strategy, args.eval_steps,
    )
    if args.stage == "alignment":
        LOGGER.info(
            "Data MT=%s | pairs=%s | direction=%s | mode=%s | loss=NTP + %.4g*CL + %.4g*OT",
            args.data_dir, args.language_pairs, args.direction,
            args.training_mode, args.contrastive_weight, args.ot_weight,
        )
        LOGGER.info(
            "Alignment layer=%d | contrastive_temperature=%g | attention_mass_weight=%g | Sinkhorn eps=%g iterations=%d",
            args.align_layer, args.temperature, args.attention_mass_weight,
            args.sinkhorn_epsilon, args.sinkhorn_iterations,
        )
    else:
        LOGGER.info(
            "Data XLSum=%s | Bactrian=%s | languages=%s | mode=%s | loss=NTP",
            args.xlsum_dir, args.bactrian_dir, args.languages, args.training_mode,
        )
    LOGGER.info(
        "LoRA=%s | r=%d alpha=%d dropout=%g targets=%s",
        args.use_lora, args.lora_r, args.lora_alpha,
        args.lora_dropout, args.lora_target_modules,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.prompt_format == "chat" and not tokenizer.chat_template:
        raise ValueError(
            "--prompt_format chat was requested but tokenizer.chat_template is empty."
        )
    dataset, model, collator = build_stage(args, tokenizer)
    layer_scheduler = None
    if args.stage == "alignment" and args.candidate_layers:
        candidate_values = (
            args.candidate_layers
            if isinstance(args.candidate_layers, (list, tuple))
            else args.candidate_layers.split(",")
        )
        candidate_layers = tuple(
            int(value.strip()) if isinstance(value, str) else int(value)
            for value in candidate_values
            if not isinstance(value, str) or value.strip()
        )
        hidden_state_count = model.config.num_hidden_layers + 1
        invalid_layers = [
            layer for layer in candidate_layers
            if not -hidden_state_count <= layer < hidden_state_count
        ]
        if invalid_layers:
            raise ValueError(
                f"Invalid candidate hidden-state indices {invalid_layers}; "
                f"expected [{-hidden_state_count}, {hidden_state_count - 1}]"
            )
        layer_scheduler = POTSAStyleLayerScheduler(
            candidate_layers=candidate_layers,
            ema_rho=args.reward_ema_rho,
            ucb_beta=args.ucb_beta,
            temperature=args.layer_temperature,
            warmup_steps=args.layer_warmup_steps,
            force_each_layer_once=args.force_each_layer_once,
            seed=args.seed,
        )
        model.align_layer = layer_scheduler.current_layer
        LOGGER.info(
            "Reward-guided layer scheduling | candidates=%s rho=%g beta=%g "
            "temperature=%g warmup=%d force_once=%s",
            candidate_layers, args.reward_ema_rho, args.ucb_beta,
            args.layer_temperature, args.layer_warmup_steps,
            args.force_each_layer_once,
        )
    LOGGER.info(
        "Dataset built | train=%d | validation=%d | columns=%s",
        len(dataset["train"]), len(dataset["validation"]),
        dataset["train"].column_names,
    )
    sample = dataset["train"][0]
    sample_summary = {}
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 240:
            sample_summary[key] = value[:240] + "..."
        elif isinstance(value, list) and len(value) > 20:
            sample_summary[key] = {"length": len(value), "head": value[:20]}
        else:
            sample_summary[key] = value
    LOGGER.info("Processed sample=%s", sample_summary)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info(
        "Parameters trainable=%d total=%d ratio=%.6f%%",
        trainable, total, 100.0 * trainable / total,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if args.use_lora:
            trainable_lm = model.lm if args.stage == "alignment" else model
            trainable_lm.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_dir=f"{args.output_dir}/runs",
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to=args.report_to,
        load_best_model_at_end=True,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.data_seed,
    )
    trainer = ComponentLoggingTrainer(
        model=model, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        data_collator=collator,
        stage=args.stage,
        layer_scheduler=layer_scheduler,
    )
    trainer.train()
    trainer.save_state()
    export_model = model.lm if args.stage == "alignment" else model
    if hasattr(export_model, "gradient_checkpointing_disable"):
        export_model.gradient_checkpointing_disable()
    export_model.config.use_cache = True
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
