"""Reparse saved BeleBele generations and recompute accuracy without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .instruction_benchmarks import parse_answer


def rescore_belebele(input_dir, *, languages=None, output_dir=None):
    """Save rescored copies; numeric first-line answers take precedence.

    Original predictions and input metrics are never overwritten. Letter answers
    are mapped A->1 through D->4. Gold is used only for scoring, never extraction.
    """
    root = Path(input_dir)
    destination = Path(output_dir) if output_dir else root / 'reparsed'
    if destination.resolve() == root.resolve():
        raise ValueError('output_dir must differ from input_dir to preserve original results')
    if isinstance(languages, str):
        languages = [lang.strip() for lang in languages.split(',') if lang.strip()]
    paths = (
        [root / f'belebele.{lang}.jsonl' for lang in languages]
        if languages else sorted(root.glob('belebele.*.jsonl'))
    )
    if not paths:
        raise FileNotFoundError(f'No belebele.*.jsonl files found in {root}')
    source_metrics = {}
    metrics_path = root / 'metrics.json'
    if metrics_path.exists():
        source_metrics = {
            (m['task'], m['language']): m
            for m in json.loads(metrics_path.read_text(encoding='utf-8'))
        }
    pending, metrics, comparison = [], [], []
    for path in paths:
        language = path.name[len('belebele.'):-len('.jsonl')]
        results = []
        old_correct = old_invalid = changed = recovered = 0
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get('prediction_raw'), str):
                    raise ValueError(f'{path}:{line_number}: missing string prediction_raw')
                gold = str(row.get('correct_answer_num', row.get('gold', ''))).strip()
                if gold not in {'1', '2', '3', '4'}:
                    raise ValueError(f'{path}:{line_number}: invalid numeric gold')
                old = row.get('prediction')
                old_correct += old is not None and str(old) == gold
                old_invalid += old is None
                prediction = parse_answer(row['prediction_raw'], 'belebele')
                changed += prediction != old
                recovered += old is None and prediction is not None
                results.append({**row, 'prediction': prediction, 'gold': gold,
                                'is_correct': prediction is not None and prediction == gold})
        if not results:
            raise ValueError(f'Empty result file: {path}')
        count = len(results)
        correct = sum(row['is_correct'] for row in results)
        invalid = sum(row['prediction'] is None for row in results)
        metric = {
            **source_metrics.get(('belebele', language), {}),
            'task': 'belebele', 'language': language, 'samples': count,
            'correct': correct, 'accuracy': correct / count, 'invalid': invalid,
            'parser': 'letter-or-numeric-first-response-v3', 'source_file': str(path),
        }
        metrics.append(metric)
        comparison.append({
            'Language': language, 'Samples': count,
            'Old correct': old_correct, 'New correct': correct,
            'Old accuracy (%)': old_correct / count * 100,
            'New accuracy (%)': correct / count * 100,
            'Old invalid': old_invalid, 'New invalid': invalid,
            'Recovered': recovered, 'Changed': changed,
        })
        pending.append((path.name, results))
    destination.mkdir(parents=True, exist_ok=True)
    # Exclusive writes protect both originals and previous rescoring runs.
    targets = [destination / name for name, _ in pending] + [
        destination / 'metrics.json', destination / 'rescore_comparison.json',
    ]
    for target in targets:
        if target.exists():
            raise FileExistsError(f'Output exists; choose another output_dir: {target}')
    for name, results in pending:
        with (destination / name).open('x', encoding='utf-8') as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    for name, value in [('metrics.json', metrics), ('rescore_comparison.json', comparison)]:
        with (destination / name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
    import pandas as pd
    table = pd.DataFrame(comparison)
    print(table.to_string(index=False, float_format=lambda value: f'{value:.2f}'))
    print(f'Saved rescored results: {destination.resolve()}')
    return {'metrics': metrics, 'comparison': table, 'output_dir': destination}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--languages', nargs='+')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    rescore_belebele(args.input_dir, languages=args.languages, output_dir=args.output_dir)
