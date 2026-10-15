"""Zero-shot generation evaluation on MGSM and BeleBele test splits."""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


def parse_answer(text, task):
    """Parse an explicit final answer; ambiguous choice outputs remain invalid."""
    text = text.split('</think>')[-1].strip()
    if task == 'belebele':
        match = re.fullmatch(r'\s*([ABCD1-4])[.)]?\s*', text, re.I)
        if not match:
            match = re.search(r'(?:Answer|Final answer)\s*:\s*([ABCD1-4])\b', text, re.I)
        if not match:
            return None
        answer = match.group(1).upper()
        return str('ABCD'.index(answer) + 1) if answer in 'ABCD' else answer
    text = text.replace(',', '')
    marked = re.search(r'(?:####|Final answer\s*:)\s*([-+]?\d+(?:\.\d+)?)', text, re.I)
    numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', text)
    answer = marked.group(1) if marked else (numbers[-1] if numbers else None)
    if answer is None:
        return None
    try:
        return str(Decimal(answer).normalize())
    except InvalidOperation:
        return None


def load_benchmark_jsonl(data_dir, languages, task, max_samples=None):
    """Read <data_dir>/<language>/test.jsonl and validate before model loading."""
    if isinstance(languages, str):
        languages = [value.strip() for value in languages.split(',') if value.strip()]
    if not languages:
        return []
    if data_dir is None:
        raise ValueError(f'{task}_data_dir is required when selecting {task} languages')
    required = {'question', 'answer_number'} if task == 'mgsm' else {
        'question', 'flores_passage', 'correct_answer_num',
        'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4',
    }
    jobs = []
    for language in languages:
        path = Path(data_dir) / language / 'test.jsonl'
        records = []
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path}:{line_number}: invalid JSON') from error
                if not isinstance(row, dict) or not required.issubset(row):
                    raise ValueError(f'{path}:{line_number}: expected fields {sorted(required)}')
                if task == 'belebele' and str(row['correct_answer_num']) not in {'1', '2', '3', '4'}:
                    raise ValueError(f'{path}:{line_number}: correct_answer_num must be 1..4')
                if task == 'mgsm' and parse_answer(str(row['answer_number']), task) is None:
                    raise ValueError(f'{path}:{line_number}: invalid answer_number')
                records.append(row)
                if max_samples is not None and len(records) >= max_samples:
                    break
        if not records:
            raise ValueError(f'Empty test set: {path}')
        print(f'{task} [{language}]: {len(records)} samples from {path}')
        jobs.append((task, language, records))
    return jobs


def run_instruction_benchmarks(
    model_name_or_path,
    *,
    mgsm_data_dir=None,
    belebele_data_dir=None,
    mgsm_languages=(),
    belebele_languages=(),
    output_dir='outputs/instruction_benchmarks',
    batch_size=8,
    max_new_tokens=512,
    max_samples=None,
    use_chat_template=False,
    enable_thinking=False,
    trust_remote_code=False,
):
    """Evaluate local <benchmark_dir>/<language>/test.jsonl datasets.

    Pass the root folder and language subdirectory names for each benchmark.
    An empty language list skips that benchmark. No dataset is downloaded.
    """
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from tqdm.auto import tqdm

    if batch_size < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError('batch_size and max_samples must be positive')
    jobs = []
    for task, data_dir, languages in (
        ('mgsm', mgsm_data_dir, mgsm_languages),
        ('belebele', belebele_data_dir, belebele_languages),
    ):
        jobs.extend(load_benchmark_jsonl(data_dir, languages, task, max_samples))
    if not jobs:
        raise ValueError('Select at least one benchmark language and its data directory')

    model_path = str(model_name_or_path)
    if Path(model_path).is_dir():
        checkpoint = get_last_checkpoint(model_path)
        if checkpoint:
            model_path = checkpoint
    is_adapter = (Path(model_path) / 'adapter_config.json').is_file()
    base = PeftConfig.from_pretrained(model_path).base_model_name_or_path if is_adapter else model_path
    tokenizer_path = model_path if (Path(model_path) / 'tokenizer_config.json').is_file() else base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype='auto', device_map='auto', trust_remote_code=trust_remote_code,
    )
    if is_adapter:
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    device = model.get_input_embeddings().weight.device
    pending = []
    metrics = []
    for task, config, data in jobs:
        prompts = []
        for row in data:
            if task == 'mgsm':
                prompt = row['question'] + '\nSolve the problem. End with: Final answer: <number>.'
            else:
                options = '\n'.join(f'{letter}. {row["mc_answer" + str(i)]}' for i, letter in enumerate('ABCD', 1))
                prompt = f'{row["flores_passage"]}\n\n{row["question"]}\n{options}\nAnswer with only A, B, C, or D.'
            if use_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=enable_thinking,
                )
            prompts.append(prompt)
        # Tokenize all prompts first; then pad each batch before generation.
        encoded = tokenizer(prompts, padding=False, truncation=False,
                            add_special_tokens=not use_chat_template)
        batches = [tokenizer.pad(
            {key: values[start:start + batch_size] for key, values in encoded.items()},
            padding=True, return_tensors='pt',
        ) for start in range(0, len(prompts), batch_size)]
        predictions = []
        with torch.inference_mode():
            for batch in tqdm(batches, desc=f'{task} [{config}]', unit='batch'):
                batch = batch.to(device)
                output = model.generate(
                    **batch, max_new_tokens=max_new_tokens, do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                predictions.extend(tokenizer.batch_decode(
                    output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True,
                ))
        records = []
        for row, raw in zip(data, predictions):
            gold = parse_answer(str(row['answer_number']), task) if task == 'mgsm' else str(row['correct_answer_num'])
            pred = parse_answer(raw, task)
            records.append({**row, 'prediction_raw': raw, 'prediction': pred,
                            'gold': gold, 'is_correct': pred is not None and pred == gold})
        count = len(records)
        metrics.append(dict(task=task, language=config, samples=count,
                            accuracy=sum(r['is_correct'] for r in records) / count if count else None,
                            invalid=sum(r['prediction'] is None for r in records),
                            protocol='zero-shot generation', model=model_path))
        pending.append((task, config, records))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for task, config, records in pending:
        with (root / f'{task}.{config}.jsonl').open('w', encoding='utf-8') as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (root / 'metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return {'metrics': metrics, 'output_dir': root}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--mgsm_data_dir')
    parser.add_argument('--belebele_data_dir')
    parser.add_argument('--mgsm_languages', nargs='*', default=[])
    parser.add_argument('--belebele_languages', nargs='*', default=[])
    parser.add_argument('--output_dir', default='outputs/instruction_benchmarks')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--trust_remote_code', action='store_true')
    print(run_instruction_benchmarks(**vars(parser.parse_args())))
