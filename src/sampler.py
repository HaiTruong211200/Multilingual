"""Optional language-pair grouping for Trainer's existing DataLoader."""
from collections import defaultdict
import random

from torch.utils.data import Sampler


class LanguagePairSampler(Sampler):
    """Shuffle homogeneous global batches, then expose their indices to Trainer.

    Each group drops its incomplete tail at a global-batch boundary.
    Consequently Accelerate can shard batches across ranks without mixing pairs
    or needing to fill a tail from another language. All ranks use the same seed.
    """

    def __init__(self, dataset, batch_size, world_size=1, seed=42):
        if batch_size < 1 or world_size < 1:
            raise ValueError("batch_size and world_size must be positive")
        self.global_batch_size = batch_size * world_size
        self.seed = seed
        self.epoch = 0
        self.groups = defaultdict(list)
        if not {"source_lang", "target_lang"}.issubset(dataset.column_names):
            raise ValueError("Pair batching requires source_lang and target_lang columns")
        for index, pair in enumerate(zip(dataset["source_lang"], dataset["target_lang"])):
            self.groups[pair].append(index)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        width = self.global_batch_size
        return sum((len(indices) // width) * width for indices in self.groups.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        width = self.global_batch_size
        batches = []
        for original in self.groups.values():
            indices = list(original)
            rng.shuffle(indices)
            usable_size = (len(indices) // width) * width
            batches.extend(indices[start:start + width] for start in range(0, usable_size, width))
        rng.shuffle(batches)
        return iter(index for batch in batches for index in batch)
