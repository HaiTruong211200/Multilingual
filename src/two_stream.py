"""Equal-sized parallel alignment and instruction SFT streams for Stage 1."""
from itertools import islice
from collections import defaultdict
import logging
import random

from torch.utils.data import Dataset, Sampler


class PairedObjectiveDataset(Dataset):
    """Pair independent dataset indices without concatenating incompatible schemas."""

    column_names = ["alignment_batch", "sft_batch"]

    def __init__(self, alignment, instruction, samples=None, seed=42, balance_pairs=True,
                 sampling_order='joint_limit'):
        if sampling_order not in {'joint_limit', 'sft_first'}:
            raise ValueError('sampling_order must be joint_limit or sft_first')
        count = len(instruction) if sampling_order == 'sft_first' else min(len(alignment), len(instruction))
        if samples is not None:
            if samples <= 0 or samples > count:
                raise ValueError(f'stage1_samples_per_stream must be in [1, {count}]')
            count = samples
        if count == 0:
            raise ValueError('Both Stage 1 training streams must be nonempty')
        if count > len(alignment):
            raise ValueError(f'SFT requests {count} alignment samples, but only {len(alignment)} are available')
        if balance_pairs:
            groups = defaultdict(list)
            for index, pair in enumerate(zip(alignment['source_lang'], alignment['target_lang'])):
                groups[pair].append(index)
            # Sample without replacement. joint_limit rounds the total down;
            # sft_first preserves the SFT budget with pair quotas differing by
            # at most one sample when the budget is not divisible by pair count.
            per_pair = min(min(map(len, groups.values())), count // len(groups))
            if sampling_order == 'sft_first':
                per_pair, remainder = divmod(count, len(groups))
                quotas = {pair: per_pair + (i < remainder) for i, pair in enumerate(sorted(groups))}
                shortage = {f'{src}-{tgt}': {'needed': quota, 'available': len(groups[(src, tgt)])}
                            for (src, tgt), quota in quotas.items() if quota > len(groups[(src, tgt)])}
                if shortage:
                    raise ValueError(f'Insufficient alignment data for SFT-first balanced sampling: {shortage}')
            else:
                quotas = {pair: per_pair for pair in groups}
            if per_pair == 0:
                raise ValueError('Sample budget is too small to include every alignment pair')
            rng = random.Random(seed)
            selected = []
            for pair in sorted(groups):
                rng.shuffle(groups[pair])
                selected.extend(groups[pair][:quotas[pair]])
            rng.shuffle(selected)
            count = len(selected)
            self.alignment = alignment.select(selected)
            logging.getLogger('multilingual_training').info(
                'Balanced alignment | sampling_order=%s | counts=%s | total=%d',
                sampling_order, {f'{src}-{tgt}': quotas[(src, tgt)] for src, tgt in sorted(groups)}, count,
            )
        else:
            self.alignment = alignment.shuffle(seed=seed).select(range(count))
        self.instruction = instruction.shuffle(seed=seed + 1).select(range(count))

    def __len__(self):
        return len(self.alignment)

    def __getitem__(self, index):
        return {'alignment_batch': self.alignment[index], 'sft_batch': self.instruction[index]}


class RepeatObjectiveBatchSampler(Sampler):
    """Expose each global batch twice: once for alignment, once for instruction.

    Accelerate shards each global batch across ranks. Duplicating whole global
    batches (rather than individual indices) keeps both objectives synchronized.
    Incomplete tails are dropped. The wrapped sampler handles epoch shuffling.
    """

    def __init__(self, sampler, global_batch_size):
        self.sampler = sampler
        self.width = global_batch_size

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return (len(self.sampler) // self.width) * self.width * 2

    def __iter__(self):
        indices = iter(self.sampler)
        while True:
            batch = list(islice(indices, self.width))
            if len(batch) != self.width:
                return
            yield from batch
            yield from batch


class ObjectiveStreamCollator:
    """Collate each data source separately; eval uses instruction batches only."""

    def __init__(self, alignment_collator, instruction_collator):
        self.alignment_collator = alignment_collator
        self.instruction_collator = instruction_collator

    def __call__(self, features):
        if 'alignment_batch' not in features[0]:
            return self.instruction_collator(features)
        return {
            'alignment_batch': self.alignment_collator([row['alignment_batch'] for row in features]),
            'sft_batch': self.instruction_collator([row['sft_batch'] for row in features]),
        }
