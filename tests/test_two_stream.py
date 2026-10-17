"""CPU checks for dataset routing, DDP batch boundaries and alternating backward."""
import tempfile
import unittest

import torch
from datasets import Dataset
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler
from transformers import PretrainedConfig, TrainingArguments

from src.model import AlignmentCausalLMOutputWithPast
from src.sampler import LanguagePairSampler
from src.train import ComponentLoggingTrainer
from src.two_stream import PairedObjectiveDataset, RepeatObjectiveBatchSampler, ObjectiveStreamCollator, CurriculumObjectiveSampler


def collate(features):
    ids = torch.tensor([[row['kind']] for row in features])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)


class TinyModel(torch.nn.Module):
    ot_weight = 1.0
    contrastive_weight = structure_weight = 0.0
    ot_solver = 'ipot'
    config = PretrainedConfig()

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def forward(self, input_ids, attention_mask=None, labels=None, loss_objective='joint'):
        kind = input_ids[:, 0]
        if self.training:
            self.seen.append((loss_objective, kind.tolist()))
            expected = 1 if loss_objective == 'alignment' else 2
            assert (kind == expected).all(), (loss_objective, kind)
        loss = self.weight.square()
        return AlignmentCausalLMOutputWithPast(
            loss=loss, logits=self.weight * torch.ones(len(kind), 1, 3),
            ntp_loss=loss.detach() if loss_objective != 'alignment' else loss.detach() * 0,
            ot_loss=loss.detach() if loss_objective == 'alignment' else loss.detach() * 0,
        )


class TwoStreamTests(unittest.TestCase):
    def test_curriculum_trainer_crosses_phase_boundary(self):
        alignment = Dataset.from_dict(dict(
            kind=[1]*16, source_lang=['en']*16, target_lang=['vi']*8+['zh']*8,
        ))
        instruction = Dataset.from_dict(dict(kind=[2]*24))
        paired = PairedObjectiveDataset(
            alignment, instruction, sampling_order='sft_first',
            sft_multiplier=2, batch_quantum=4,
        )
        for epochs in (1, 2):
            model = TinyModel()
            with tempfile.TemporaryDirectory() as output:
                args = TrainingArguments(
                    output_dir=output, use_cpu=True, num_train_epochs=epochs,
                    per_device_train_batch_size=2, gradient_accumulation_steps=2,
                    remove_unused_columns=False, report_to=[], save_strategy='no',
                    logging_strategy='no', disable_tqdm=True, dataloader_pin_memory=False,
                )
                trainer = ComponentLoggingTrainer(
                    model=model, args=args, stage='alignment',
                    loss_schedule='alternate_microbatch_then_sft',
                    stage1_data_mode='parallel_instruction', batch_by_language_pair=True,
                    train_dataset=paired, data_collator=ObjectiveStreamCollator(collate, collate),
                )
                trainer.train()
                objectives = [objective for objective, _ in model.seen]
                self.assertEqual(objectives, ['sft','alignment']*(4*epochs)+['sft']*(4*epochs))

    def test_curriculum_uses_unique_sft_and_two_to_one_budget(self):
        from collections import Counter
        alignment = Dataset.from_dict(dict(
            id=list(range(20)), kind=[1]*20,
            source_lang=['en']*20, target_lang=['vi']*10+['zh']*10,
        ))
        instruction = Dataset.from_dict(dict(id=list(range(32)), kind=[2]*32))
        paired = PairedObjectiveDataset(
            alignment, instruction, sampling_order='sft_first',
            sft_multiplier=2, batch_quantum=4,
        )
        self.assertEqual(len(paired.alignment), 16)
        self.assertEqual(len(paired.instruction), 32)
        base = LanguagePairSampler(paired.alignment, 2, 2)
        sampler = CurriculumObjectiveSampler(paired, base, 1, 1)
        indices = list(sampler)
        self.assertEqual(len(indices), 48)
        self.assertEqual(len(set(indices)), 48)
        kinds = [paired[i]['stream_objective'] for i in indices[::4]]
        self.assertEqual(kinds, ['sft','alignment']*4+['sft']*4)
        # Across two epochs, every record has exactly two total exposures.
        sampler = CurriculumObjectiveSampler(paired, base, 2, 1)
        all_indices = []
        for epoch in range(2):
            sampler.set_epoch(epoch)
            all_indices.extend(list(sampler))
        self.assertEqual(set(Counter(all_indices).values()), {2})

    def test_sft_first_keeps_sft_budget(self):
        from collections import Counter
        alignment = Dataset.from_dict(dict(
            id=list(range(24)), source_lang=['en'] * 24,
            target_lang=['vi'] * 12 + ['zh'] * 12,
        ))
        instruction = Dataset.from_dict(dict(language=['vi'] * 10 + ['zh'] * 5))
        data = PairedObjectiveDataset(alignment, instruction, sampling_order='sft_first')
        self.assertEqual(len(data), 15)
        self.assertEqual(Counter(data.instruction['language']), {'vi': 10, 'zh': 5})
        self.assertEqual(Counter(data.alignment['target_lang']), {'vi': 8, 'zh': 7})
        self.assertEqual(len(set(data.alignment['id'])), 15)
        with self.assertRaisesRegex(ValueError, 'SFT requests'):
            PairedObjectiveDataset(alignment.select(range(10)), instruction, sampling_order='sft_first')

    def test_language_report_exposes_unbalanced_sft(self):
        import io
        from contextlib import redirect_stdout
        from eval.inspect_stage1_data import inspect_selected_streams

        alignment = Dataset.from_dict(dict(
            id=list(range(12)), source_lang=['en'] * 12,
            target_lang=['vi'] * 8 + ['zh'] * 4,
        ))
        instruction = Dataset.from_dict(dict(
            language=['vi'] * 9 + ['zh'] * 3, dataset_name=['globalmgsm'] * 12,
        ))
        paired = PairedObjectiveDataset(alignment, instruction, samples=12)
        with redirect_stdout(io.StringIO()):
            result = inspect_selected_streams(
                paired, alignment, instruction, batch_size=2,
                batch_by_language_pair=True, languages=['vi', 'zh', 'en'],
            )
        table = result['sft_languages'].set_index('language')
        self.assertEqual(table.loc['vi', 'Available (%)'], 75)
        self.assertEqual(table.loc['zh', 'Available (%)'], 25)
        self.assertEqual(table.loc['en', 'Selected'], 0)
        self.assertEqual(table['Selected'].sum(), 8)
        self.assertEqual(result['alignment_pairs']['Selected'].tolist(), [4, 4])
        self.assertEqual(result['alignment_pairs']['Usable'].sum(), table['Usable'].sum())

    def test_unequal_pair_sizes_are_balanced_without_duplicates(self):
        from collections import Counter
        alignment = Dataset.from_dict(dict(
            id=list(range(18)), source_lang=['en'] * 18,
            target_lang=['vi'] * 12 + ['zh'] * 6,
        ))
        instruction = Dataset.from_dict(dict(kind=[2] * 30))
        data = PairedObjectiveDataset(alignment, instruction, samples=17)
        self.assertEqual(Counter(data.alignment['target_lang']), {'vi': 6, 'zh': 6})
        self.assertEqual(len(data.instruction), 12)
        self.assertEqual(len(set(data.alignment['id'])), 12)
        again = PairedObjectiveDataset(alignment, instruction, samples=17)
        self.assertEqual(data.alignment['id'], again.alignment['id'])
        old = PairedObjectiveDataset(alignment, instruction, samples=17, balance_pairs=False)
        self.assertEqual(len(old), 17)

    def make_data(self):
        alignment = Dataset.from_dict(dict(
            kind=[1] * 16, source_lang=['en'] * 8 + ['vi'] * 8,
            target_lang=['vi'] * 8 + ['en'] * 8,
        ))
        instruction = Dataset.from_dict(dict(kind=[2] * 20))
        return PairedObjectiveDataset(alignment, instruction, 16), instruction

    def test_distributed_batch_boundaries(self):
        data, _ = self.make_data()
        base = LanguagePairSampler(data.alignment, batch_size=2, world_size=2)
        repeated = RepeatObjectiveBatchSampler(base, 4)
        shards = []
        for rank in range(2):
            shards.append(list(BatchSamplerShard(
                BatchSampler(repeated, 2, False), num_processes=2, process_index=rank,
                split_batches=False,
            )))
        for index in range(0, len(shards[0]), 2):
            for rank in range(2):
                self.assertEqual(shards[rank][index], shards[rank][index + 1])
            indices = shards[0][index] + shards[1][index]
            pairs = {(data.alignment[i]['source_lang'], data.alignment[i]['target_lang']) for i in indices}
            self.assertEqual(len(pairs), 1)

    def test_trainer_routing_and_backward(self):
        for schedule in ['alternate_microbatch', 'alternate_microbatch_then_sft']:
            data, instruction = self.make_data()
            model = TinyModel()
            with tempfile.TemporaryDirectory() as output:
                args = TrainingArguments(
                    output_dir=output, use_cpu=True, max_steps=4,
                    per_device_train_batch_size=2, gradient_accumulation_steps=2,
                    remove_unused_columns=False, report_to=[], save_strategy='no',
                    logging_strategy='no', disable_tqdm=True, dataloader_pin_memory=False,
                )
                trainer = ComponentLoggingTrainer(
                    model=model, args=args, stage='alignment', loss_schedule=schedule,
                    stage1_data_mode='parallel_instruction', batch_by_language_pair=True,
                    train_dataset=data, eval_dataset=instruction,
                    data_collator=ObjectiveStreamCollator(collate, collate),
                )
                before = model.weight.item()
                trainer.train()
                self.assertNotEqual(before, model.weight.item())
                expected = ['alignment', 'sft'] * 4
                if schedule.endswith('then_sft'):
                    expected[4:] = ['sft'] * 4
                self.assertEqual([objective for objective, _ in model.seen], expected)
                metrics = trainer.evaluate()
                self.assertIn('eval_loss', metrics)


if __name__ == '__main__':
    unittest.main()
