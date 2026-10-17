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
                 sampling_order='joint_limit', sft_multiplier=1, batch_quantum=1):
        if sft_multiplier not in {1, 2} or batch_quantum < 1:
            raise ValueError('sft_multiplier must be 1 or 2; batch_quantum must be positive')
        self.sft_multiplier = sft_multiplier
        if sampling_order not in {'joint_limit', 'sft_first'}:
            raise ValueError('sampling_order must be joint_limit or sft_first')
        count = len(instruction) if sampling_order == 'sft_first' else min(sft_multiplier * len(alignment), len(instruction))
        if samples is not None:
            if samples <= 0 or samples > count:
                raise ValueError(f'stage1_samples_per_stream must be in [1, {count}]')
            count = samples
        if count == 0:
            raise ValueError('Both Stage 1 training streams must be nonempty')
        if sft_multiplier == 2:
            # The input budget is SFT samples. Reserve one alignment sample
            # for every two SFT samples, following CAROT's LM-batch clock.
            count //= 2
            pair_count = len(set(zip(alignment['source_lang'], alignment['target_lang']))) if balance_pairs else 1
            quantum = pair_count * batch_quantum
            count = (count // quantum) * quantum
            if count == 0:
                raise ValueError(f'Need at least {2 * quantum} SFT samples for complete 2:1 curriculum batches')
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
            if sft_multiplier == 2:
                # For joint_limit, the smallest pair can reduce the budget.
                per_pair = (min(quotas.values()) // batch_quantum) * batch_quantum
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
        self.instruction = instruction.shuffle(seed=seed + 1).select(range(count * sft_multiplier))
        if sft_multiplier == 2:
            logging.getLogger('multilingual_training').info(
                'Curriculum budget | unique alignment=%d | unique SFT=%d | SFT:alignment=2:1',
                len(self.alignment), len(self.instruction),
            )

    def __len__(self):
        return (len(self.alignment) + len(self.instruction)) if self.sft_multiplier == 2 else len(self.alignment)

    def __getitem__(self, index):
        if self.sft_multiplier == 2:
            if index < len(self.alignment):
                return {'stream_objective': 'alignment', 'features': self.alignment[index]}
            return {'stream_objective': 'sft', 'features': self.instruction[index - len(self.alignment)]}
        return {'alignment_batch': self.alignment[index], 'sft_batch': self.instruction[index]}


class CurriculumObjectiveSampler(Sampler):
    """First half of LM batches adds alignment; remaining LM batches are SFT.

    Plan the whole integer-epoch run. Each stream has independent shuffled
    traversal cycles. For one epoch, every selected sample is used once. For
    multiple epochs, each pool is traversed once per configured epoch in total,
    concentrated in the phase where its objective is active.
    """

    def __init__(self, dataset, alignment_sampler, epochs, accumulation_steps=1):
        if epochs < 1 or int(epochs) != epochs:
            raise ValueError('The 2:1 curriculum currently requires an integer number of epochs')
        self.dataset = dataset
        self.base = alignment_sampler
        self.width = alignment_sampler.global_batch_size
        self.epochs = int(epochs)
        self.epoch = 0
        self.batches = (len(dataset.alignment) + len(dataset.instruction)) // self.width
        if len(self.base) != len(dataset.alignment) or len(dataset.instruction) != 2 * len(dataset.alignment):
            raise ValueError('2:1 pools must consist of complete global batches for every alignment group')
        if self.batches % (3 * accumulation_steps):
            raise ValueError('Curriculum budget must be divisible by 3 * gradient_accumulation_steps batches')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.batches * self.width

    def __iter__(self):
        start = self.epoch * self.batches
        # Half of LM batches have an extra alignment batch. This alternating
        # phase occupies 2/3 of all batches, NOT half of all optimizer updates.
        cutoff = self.epochs * self.batches * 2 // 3
        alignment_used = min(start, cutoff) // 2
        used = {'alignment': alignment_used, 'sft': start - alignment_used}
        cache = {}
        sizes = {'alignment': len(self.dataset.alignment) // self.width,
                 'sft': len(self.dataset.instruction) // self.width}
        for position in range(start, start + self.batches):
            objective = 'alignment' if position < cutoff and position % 2 == 1 else 'sft'
            cycle, offset = divmod(used[objective], sizes[objective])
            key = (objective, cycle)
            if key not in cache:
                if objective == 'alignment':
                    self.base.set_epoch(cycle)
                    cache[key] = list(self.base)
                else:
                    indices = list(range(len(self.dataset.instruction)))
                    random.Random(self.base.seed + 1 + cycle).shuffle(indices)
                    cache[key] = [i + len(self.dataset.alignment) for i in indices]
            begin = offset * self.width
            yield from cache[key][begin:begin + self.width]
            used[objective] += 1


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
        if 'stream_objective' in features[0]:
            objective = features[0]['stream_objective']
            if any(row['stream_objective'] != objective for row in features):
                raise ValueError('A batch cannot mix alignment and SFT records')
            collator = self.alignment_collator if objective == 'alignment' else self.instruction_collator
            return {'stream_objective': objective,
                    'stream_batch': collator([row['features'] for row in features])}
        if 'alignment_batch' not in features[0]:
            return self.instruction_collator(features)
        return {
            'alignment_batch': self.alignment_collator([row['alignment_batch'] for row in features]),
            'sft_batch': self.instruction_collator([row['sft_batch'] for row in features]),
        }
