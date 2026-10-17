"""Inspect the actual Stage 1 two-stream sampling without loading a model."""
from __future__ import annotations

import argparse
from collections import Counter


def _count_rows(dataset, columns):
    return Counter(zip(*(dataset[column] for column in columns)))


def _ratio_table(available, selected, usable, columns, extra_keys=()):
    import pandas as pd

    keys = sorted(set(available) | set(selected) | set(usable) | set(extra_keys))
    totals = [sum(counter.values()) for counter in (available, selected, usable)]
    records = []
    for key in keys:
        row = dict(zip(columns, key))
        for name, counter, total in zip(
            ('Available', 'Selected', 'Usable'), (available, selected, usable), totals,
        ):
            row[name] = counter[key]
            row[name + ' (%)'] = 100 * counter[key] / total if total else 0.0
        records.append(row)
    return pd.DataFrame(records)


def inspect_selected_streams(paired, alignment_available, instruction_available,
                             *, batch_size=2, world_size=1,
                             batch_by_language_pair=False, data_seed=42,
                             languages=(), epochs=1, gradient_accumulation_steps=1):
    """Summarize selected samples and the first epoch's dropped-tail effects."""
    from src.sampler import LanguagePairSampler

    sampler_data = paired.alignment
    if not batch_by_language_pair:
        sampler_data = sampler_data.remove_columns(['source_lang', 'target_lang'])
        sampler_data = sampler_data.add_column('source_lang', ['all'] * len(paired.alignment))
        sampler_data = sampler_data.add_column('target_lang', ['all'] * len(paired.alignment))
    sampler = LanguagePairSampler(sampler_data, batch_size, world_size, data_seed)
    if paired.sft_multiplier == 2:
        from src.two_stream import CurriculumObjectiveSampler
        planned = list(CurriculumObjectiveSampler(paired, sampler, epochs, gradient_accumulation_steps))
        boundary = len(paired.alignment)
        alignment_usable = paired.alignment.select([i for i in planned if i < boundary])
        instruction_usable = paired.instruction.select([i - boundary for i in planned if i >= boundary])
    else:
        indices = list(sampler)
        alignment_usable = paired.alignment.select(indices)
        instruction_usable = paired.instruction.select(indices)
    result = {}
    for name, before, after, usable, columns in (
        ('alignment_pairs', alignment_available, paired.alignment, alignment_usable,
         ['source_lang', 'target_lang']),
        ('sft_languages', instruction_available, paired.instruction, instruction_usable,
         ['language']),
        ('sft_datasets', instruction_available, paired.instruction, instruction_usable,
         ['dataset_name']),
        ('sft_language_dataset', instruction_available, paired.instruction, instruction_usable,
         ['language', 'dataset_name']),
    ):
        result[name] = _ratio_table(
            _count_rows(before, columns), _count_rows(after, columns),
            _count_rows(usable, columns), columns,
            extra_keys=[(lang,) for lang in languages] if name == 'sft_languages' else (),
        )
    print(f'Global batch size: {batch_size * world_size}')
    print(f'Selected alignment={len(paired.alignment)}, SFT={len(paired.instruction)}; '
          f'epoch 1 exposures: alignment={len(alignment_usable)}, SFT={len(instruction_usable)}')
    print('Available: training pools AFTER validation split. Selected: sampled pools. '
          'Usable: first epoch after dropping incomplete global batches.')
    print('SFT languages are NOT quota-balanced; shuffle preserves no exact language ratio.')
    for name, table in result.items():
        print(f'\n{name}')
        print(table.to_string(index=False, float_format=lambda value: f'{value:.2f}'))
    result['alignment_train'] = paired.alignment
    result['sft_train'] = paired.instruction
    return result


def inspect_stage1_data(
    data_dir,
    *,
    language_pairs='all',
    direction='both',
    train_file=None,
    validation_file=None,
    instruction_datasets='xlsum,bactrian',
    languages='en,km,my,th,vi',
    xlsum_dir='data/XLSum/XLSum',
    bactrian_dir='data/Bactrian-Multilingual_Instruction',
    globalmgsm_dir='data/GlobalMGSM',
    bactrian_validation_ratio=0.01,
    globalmgsm_validation_ratio=0.01,
    stage1_samples_per_stream=None,
    stage1_balance_pairs=True,
    sampling_order='sft_first',
    loss_schedule='alternate_microbatch_then_sft',
    gradient_accumulation_steps=1,
    epochs=1,
    batch_by_language_pair=False,
    batch_size=2,
    world_size=1,
    seed=42,
    data_seed=None,
):
    """Use training loaders to show ratios, with an optional SFT-first experiment.

    Does not tokenize, load a model, download benchmarks, or read test splits.
    May create the normal Hugging Face dataset preprocessing cache.
    """
    from src.prepare_data import load_parallel_dataset, load_instruction_dataset
    from src.two_stream import PairedObjectiveDataset

    if not isinstance(language_pairs, str):
        language_pairs = ','.join(language_pairs)
    if not isinstance(languages, str):
        languages = ','.join(languages)
    if not isinstance(instruction_datasets, str):
        instruction_datasets = ','.join(instruction_datasets)
    data_seed = seed if data_seed is None else data_seed
    alignment = load_parallel_dataset(
        data_dir, language_pairs, direction, train_file, validation_file,
    )
    instruction = load_instruction_dataset(
        xlsum_dir, bactrian_dir, languages, bactrian_validation_ratio, seed,
        instruction_datasets=instruction_datasets,
        globalmgsm_dir=globalmgsm_dir,
        globalmgsm_validation_ratio=globalmgsm_validation_ratio,
    )
    paired = PairedObjectiveDataset(
        alignment['train'], instruction['train'], stage1_samples_per_stream,
        data_seed, balance_pairs=stage1_balance_pairs,
        sampling_order=sampling_order,
        sft_multiplier=2 if loss_schedule == 'alternate_microbatch_then_sft' else 1,
        batch_quantum=batch_size * world_size * gradient_accumulation_steps,
    )
    return inspect_selected_streams(
        paired, alignment['train'], instruction['train'], batch_size=batch_size,
        world_size=world_size, batch_by_language_pair=batch_by_language_pair,
        data_seed=data_seed, languages=[lang.strip() for lang in languages.split(',') if lang.strip()],
        epochs=epochs, gradient_accumulation_steps=gradient_accumulation_steps,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--language_pairs', default='all')
    parser.add_argument('--direction', choices=['forward', 'reverse', 'both'], default='both')
    parser.add_argument('--train_file')
    parser.add_argument('--validation_file')
    parser.add_argument('--instruction_datasets', default='xlsum,bactrian')
    parser.add_argument('--languages', default='en,km,my,th,vi')
    parser.add_argument('--xlsum_dir', default='data/XLSum/XLSum')
    parser.add_argument('--bactrian_dir', default='data/Bactrian-Multilingual_Instruction')
    parser.add_argument('--globalmgsm_dir', default='data/GlobalMGSM')
    parser.add_argument('--bactrian_validation_ratio', type=float, default=0.01)
    parser.add_argument('--globalmgsm_validation_ratio', type=float, default=0.01)
    parser.add_argument('--stage1_samples_per_stream', type=int)
    parser.add_argument('--sampling_order', choices=['joint_limit', 'sft_first'], default='sft_first')
    parser.add_argument('--loss_schedule', choices=['alternate_microbatch', 'alternate_microbatch_then_sft'], default='alternate_microbatch_then_sft')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--stage1_balance_pairs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--batch_by_language_pair', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_seed', type=int)
    inspect_stage1_data(**vars(parser.parse_args()))
