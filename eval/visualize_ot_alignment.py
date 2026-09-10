"""Reusable OT-alignment visualization extracted from the notebook."""
from __future__ import annotations

import gc
import argparse
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.prepare_data import prepare_alignment_dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16 if torch.cuda.is_available() else torch.float32
)
sns.set_theme(style='white')

__all__ = ['visualize_ot_alignment']

def mixed_mass(scores, alpha):
    scores = scores.float().clamp_min(0)
    uniform = torch.full_like(scores, 1.0 / scores.numel())
    attention_mass = scores / scores.sum().clamp_min(1e-8)
    if scores.sum() <= 1e-8:
        attention_mass = uniform
    return alpha * attention_mass + (1.0 - alpha) * uniform


def sinkhorn_plan(cost, source_mass, target_mass, epsilon, iterations):
    log_a = source_mass.clamp_min(1e-38).log()
    log_b = target_mass.clamp_min(1e-38).log()
    log_k = -cost.float() / epsilon
    log_u = torch.zeros_like(source_mass, dtype=torch.float32)
    log_v = torch.zeros_like(target_mass, dtype=torch.float32)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_k.T + log_u.unsqueeze(0), dim=1)
    plan = torch.exp(log_u[:, None] + log_k + log_v[None, :])
    return plan / plan.sum().clamp_min(1e-8)


def ipot_plan(cost, source_mass, target_mass, beta, iterations, inner_iterations):
    log_a = source_mass.clamp_min(1e-38).log()
    log_b = target_mass.clamp_min(1e-38).log()
    log_kernel = -cost.float() / beta
    log_transport = log_a[:, None] + log_b[None, :]
    log_v = torch.zeros_like(target_mass, dtype=torch.float32)
    for _ in range(iterations):
        log_q = log_kernel + log_transport
        for _ in range(inner_iterations):
            log_u = log_a - torch.logsumexp(log_q + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_q.T + log_u.unsqueeze(0), dim=1)
        log_transport = log_u[:, None] + log_q + log_v[None, :]
    plan = torch.exp(log_transport)
    return plan / plan.sum().clamp_min(1e-8)


def solve_plan(cost, source_mass, target_mass, config):
    if config['ot_solver'] == 'sinkhorn':
        return sinkhorn_plan(cost, source_mass, target_mass,
                             config['sinkhorn_epsilon'], config['sinkhorn_iterations'])
    if config['ot_solver'] == 'ipot':
        return ipot_plan(cost, source_mass, target_mass, config['ipot_beta'],
                         config['ipot_iterations'], config['ipot_inner_iterations'])
    raise ValueError("ot_solver phải là 'sinkhorn' hoặc 'ipot'")

def _is_adapter(model_path):
    path = Path(str(model_path))
    return path.is_dir() and (path / 'adapter_config.json').exists()


def build_context(
    source_text,
    target_text,
    source_lang,
    target_lang,
    tokenizer_source,
    prompt_format='plain',
    enable_thinking=False,
    training_mode='finetune',
    trust_remote_code=False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw = DatasetDict({
        'test': Dataset.from_list([{
            'source': source_text,
            'target': target_text,
            'source_lang': source_lang,
            'target_lang': target_lang,
        }])
    })

    sample = prepare_alignment_dataset(
        raw,
        tokenizer,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
        training_mode=training_mode,
    )['test'][0]

    input_ids = torch.tensor(
        sample['input_ids'],
        dtype=torch.long,
    ).unsqueeze(0)

    ss = sample['source_start_positions']
    se = sample['source_end_positions']
    ts = sample['target_start_positions']
    te = sample['target_end_positions']

    assert 0 <= ss < se <= ts < te <= input_ids.shape[1]

    return {
        'tokenizer': tokenizer,
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'spans': (ss, se, ts, te),
        'source_text': source_text,
        'target_text': target_text,
    }


def readable_tokens(token_ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(list(token_ids))

    return [
        f'{index}: '
        + token.replace('Ġ', '␠')
        .replace('▁', '␠')
        .replace('\n', '↵')
        for index, token in enumerate(tokens)
    ]


def load_causal_lm(
    model_path,
    adapter,
    base_model_name,
    config,
):
    kwargs = {
        'trust_remote_code': config['trust_remote_code'],
        'attn_implementation': 'eager',
        'torch_dtype': DTYPE,
    }

    if DEVICE == 'cuda':
        kwargs['device_map'] = 'auto'

    if adapter:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **kwargs,
        )

        model = PeftModel.from_pretrained(
            base,
            model_path,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )

    if DEVICE == 'cpu':
        model = model.to(DEVICE)

    model.eval()
    return model


def _normalized_attention_entropy(attention):
    """
    Entropy trung bình của attention theo từng query.

    attention: [num_queries, num_keys]
    """
    if attention is None or attention.numel() == 0:
        return float('nan')

    probability = attention.float().clamp_min(0)

    row_sum = probability.sum(
        dim=-1,
        keepdim=True,
    )

    valid_rows = row_sum.squeeze(-1) > 1e-12

    if not valid_rows.any():
        return float('nan')

    probability = probability[valid_rows]

    probability = probability / probability.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)

    entropy = -(
        probability
        * probability.clamp_min(1e-38).log()
    ).sum(dim=-1)

    denominator = np.log(
        max(probability.shape[-1], 2)
    )

    return float(
        (entropy / denominator).mean()
    )


def _off_diagonal_statistics(similarity):
    """
    Thống kê cosine giữa các token khác nhau trong cùng chuỗi.

    similarity: [L, L]
    """
    length = similarity.shape[0]

    if length <= 1:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'minimum': float('nan'),
            'maximum': float('nan'),
        }

    diagonal_mask = torch.eye(
        length,
        dtype=torch.bool,
        device=similarity.device,
    )

    values = similarity[~diagonal_mask]

    return {
        'mean': float(values.mean()),
        'std': float(values.std(unbiased=False)),
        'minimum': float(values.min()),
        'maximum': float(values.max()),
    }


@torch.inference_mode()
def extract_alignment(model, context, config):
    tokenizer = context['tokenizer']
    input_ids = context['input_ids']
    attention_mask = context['attention_mask']

    ss, se, ts, te = context['spans']

    forward_mode = config['alignment_forward_mode']

    if forward_mode not in {'joint', 'independent'}:
        raise ValueError(
            "alignment_forward_mode phải là "
            "'joint' hoặc 'independent'"
        )

    input_device = (
        model.get_input_embeddings().weight.device
    )

    align_layer = config['align_layer']

    # hidden_states[16] là output sau block 16.
    # Attention tương ứng là attentions[15].
    if align_layer < 0:
        attention_layer = align_layer
    else:
        attention_layer = max(
            align_layer - 1,
            0,
        )

    def forward(ids, mask):
        return model(
            input_ids=ids.to(input_device),
            attention_mask=mask.to(input_device),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    if forward_mode == 'joint':
        outputs = forward(
            input_ids,
            attention_mask,
        )

        hidden = (
            outputs.hidden_states[align_layer][0]
            .float()
        )

        src_hidden = hidden[ss:se]
        tgt_hidden = hidden[ts:te]

        src_ids = input_ids[0, ss:se].tolist()
        tgt_ids = input_ids[0, ts:te].tolist()

        # [num_heads, sequence_length, sequence_length]
        layer_attention = (
            outputs.attentions[attention_layer][0]
            .float()
        )

        # Dùng trung bình head cho visualization.
        # Chiều 0 là query, chiều 1 là key.
        attention = layer_attention.mean(dim=0)

        source_self_attention = (
            attention[ss:se, ss:se]
        )

        target_to_source_attention = (
            attention[ts:te, ss:se]
        )

        target_self_attention = (
            attention[ts:te, ts:te]
        )

        # Tổng lượng attention target queries
        # truyền vào từng key.
        received = attention[ts:te, :].sum(dim=0)

        source_scores = received[ss:se]
        target_scores = received[ts:te]

    else:
        batches = [
            tokenizer(
                text,
                add_special_tokens=config[
                    'independent_add_special_tokens'
                ],
                return_tensors='pt',
                truncation=False,
            )
            for text in (
                context['source_text'],
                context['target_text'],
            )
        ]

        outputs = [
            forward(
                batch['input_ids'],
                batch['attention_mask'],
            )
            for batch in batches
        ]

        special_ids = set(
            tokenizer.all_special_ids
        )

        all_ids = [
            batch['input_ids'][0].tolist()
            for batch in batches
        ]

        indices = [
            [
                index
                for index, token_id in enumerate(ids)
                if token_id not in special_ids
            ]
            for ids in all_ids
        ]

        if not indices[0] or not indices[1]:
            raise ValueError(
                'Source hoặc target không có '
                'non-special token'
            )

        selected_hidden = []
        selected_scores = []
        selected_attentions = []

        for output, chosen in zip(
            outputs,
            indices,
        ):
            hidden = (
                output.hidden_states[align_layer][0]
                .float()
            )

            hidden_index = torch.tensor(
                chosen,
                dtype=torch.long,
                device=hidden.device,
            )

            selected_hidden.append(
                hidden.index_select(
                    0,
                    hidden_index,
                )
            )

            attention = (
                output.attentions[attention_layer][0]
                .float()
                .mean(dim=0)
            )

            attention_index = torch.tensor(
                chosen,
                dtype=torch.long,
                device=attention.device,
            )

            selected_attention = (
                attention
                .index_select(0, attention_index)
                .index_select(1, attention_index)
            )

            selected_attentions.append(
                selected_attention
            )

            selected_scores.append(
                selected_attention.sum(dim=0)
            )

        src_hidden, tgt_hidden = selected_hidden
        source_scores, target_scores = selected_scores

        source_self_attention = (
            selected_attentions[0]
        )

        target_self_attention = (
            selected_attentions[1]
        )

        # Không có cross-attention khi hai câu
        # được forward độc lập.
        target_to_source_attention = None

        src_ids = [
            all_ids[0][index]
            for index in indices[0]
        ]

        tgt_ids = [
            all_ids[1][index]
            for index in indices[1]
        ]

    normalized_src = F.normalize(
        src_hidden,
        dim=-1,
        eps=1e-8,
    )

    normalized_tgt = F.normalize(
        tgt_hidden,
        dim=-1,
        eps=1e-8,
    )

    # Cross-language cosine: [Ls, Lt]
    similarity = (
        normalized_src @ normalized_tgt.T
    )

    # Intra-source cosine: [Ls, Ls]
    source_intra_similarity = (
        normalized_src @ normalized_src.T
    )

    # Intra-target cosine: [Lt, Lt]
    target_intra_similarity = (
        normalized_tgt @ normalized_tgt.T
    )

    cost = 1.0 - similarity

    source_mass = mixed_mass(
        source_scores,
        config['attention_mass_weight'],
    ).to(cost.device)

    target_mass = mixed_mass(
        target_scores,
        config['attention_mass_weight'],
    ).to(cost.device)

    plan = solve_plan(
        cost,
        source_mass,
        target_mass,
        config,
    )

    probability = (
        plan / plan.sum().clamp_min(1e-8)
    )

    plan_entropy = -(
        probability
        * probability.clamp_min(1e-38).log()
    ).sum()

    plan_entropy = (
        plan_entropy
        / np.log(max(probability.numel(), 2))
    )

    src_pos = torch.linspace(
        0,
        1,
        len(src_ids),
        device=plan.device,
    )[:, None]

    tgt_pos = torch.linspace(
        0,
        1,
        len(tgt_ids),
        device=plan.device,
    )[None, :]

    band_mask = (
        (src_pos - tgt_pos).abs() <= 0.15
    )

    band_mass = (
        plan * band_mask
    ).sum()

    source_intra_stats = (
        _off_diagonal_statistics(
            source_intra_similarity
        )
    )

    target_intra_stats = (
        _off_diagonal_statistics(
            target_intra_similarity
        )
    )

    metrics = {
        'OT expected cost ↓':
            float((plan * cost).sum()),

        'Mean best cross cosine ↑':
            float(
                similarity
                .max(dim=1)
                .values
                .mean()
            ),

        'Normalized plan entropy ↓':
            float(plan_entropy),

        'Monotonic-band mass ↑ (heuristic)':
            float(band_mass),

        'Source self-attention entropy ↓':
            _normalized_attention_entropy(
                source_self_attention
            ),

        'Target self-attention entropy ↓':
            _normalized_attention_entropy(
                target_self_attention
            ),

        'Target-to-source attention entropy ↓':
            _normalized_attention_entropy(
                target_to_source_attention
            ),

        'Source intra cosine mean':
            source_intra_stats['mean'],

        'Source intra cosine std':
            source_intra_stats['std'],

        'Source intra cosine min':
            source_intra_stats['minimum'],

        'Source intra cosine max':
            source_intra_stats['maximum'],

        'Target intra cosine mean':
            target_intra_stats['mean'],

        'Target intra cosine std':
            target_intra_stats['std'],

        'Target intra cosine min':
            target_intra_stats['minimum'],

        'Target intra cosine max':
            target_intra_stats['maximum'],
    }

    return {
        'similarity':
            similarity.cpu().numpy(),

        'plan':
            plan.cpu().numpy(),

        'source_intra_similarity':
            source_intra_similarity.cpu().numpy(),

        'target_intra_similarity':
            target_intra_similarity.cpu().numpy(),

        'source_self_attention': (
            None
            if source_self_attention is None
            else source_self_attention.cpu().numpy()
        ),

        'target_self_attention': (
            None
            if target_self_attention is None
            else target_self_attention.cpu().numpy()
        ),

        'target_to_source_attention': (
            None
            if target_to_source_attention is None
            else target_to_source_attention.cpu().numpy()
        ),

        'source_mass':
            source_mass.cpu().numpy(),

        'target_mass':
            target_mass.cpu().numpy(),

        'source_tokens':
            readable_tokens(src_ids, tokenizer),

        'target_tokens':
            readable_tokens(tgt_ids, tokenizer),

        'source_token_ids':
            src_ids,

        'target_token_ids':
            tgt_ids,

        'metrics':
            metrics,
    }


def analyze_checkpoint(
    model_path,
    context,
    config,
    name,
    adapter='auto',
):
    adapter = (
        _is_adapter(model_path)
        if adapter == 'auto'
        else bool(adapter)
    )

    base_model_name = config['base_model_name']

    print(
        f'Đang đo {name}: {model_path} '
        f'| adapter={adapter}'
    )

    model = load_causal_lm(
        model_path=str(model_path),
        adapter=adapter,
        base_model_name=base_model_name,
        config=config,
    )

    try:
        result = extract_alignment(
            model,
            context,
            config,
        )
    finally:
        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result.update(
        name=name,
        model_path=str(model_path),
    )

    return result

def _word_groups(token_ids, tokenizer):
    token_strings = tokenizer.convert_ids_to_tokens(
        token_ids
    )

    groups = []
    special_ids = set(tokenizer.all_special_ids)

    for index, (token_id, token_string) in enumerate(
        zip(token_ids, token_strings)
    ):
        piece = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )

        stripped = piece.strip()

        starts_word = (
            token_string.startswith(('Ġ', '▁'))
            or piece[:1].isspace()
        )

        is_continuation = (
            token_string.startswith('##')
        )

        has_cjk = any(
            ('\u3400' <= char <= '\u9fff')
            or ('\uac00' <= char <= '\ud7af')
            for char in stripped
        )

        is_punctuation = (
            bool(stripped)
            and all(
                unicodedata.category(char)[0]
                in {'P', 'S'}
                for char in stripped
            )
        )

        start_new = (
            not groups
            or token_id in special_ids
            or starts_word
            or is_punctuation
            or (
                has_cjk
                and not is_continuation
            )
        )

        if start_new:
            groups.append([index])
        else:
            groups[-1].append(index)

    labels = []

    for group in groups:
        ids = [
            token_ids[index]
            for index in group
        ]

        label = tokenizer.decode(
            ids,
            clean_up_tokenization_spaces=False,
        ).strip()

        if not label:
            label = ''.join(
                token_strings[index]
                for index in group
            )

        labels.append(label)

    return groups, labels


def merge_subword_matrix(
    matrix,
    source_ids,
    target_ids,
    tokenizer,
    reducer='sum',
):
    source_groups, source_labels = (
        _word_groups(
            source_ids,
            tokenizer,
        )
    )

    target_groups, target_labels = (
        _word_groups(
            target_ids,
            tokenizer,
        )
    )

    merged = np.zeros(
        (
            len(source_groups),
            len(target_groups),
        ),
        dtype=matrix.dtype,
    )

    for row, source_group in enumerate(
        source_groups
    ):
        for column, target_group in enumerate(
            target_groups
        ):
            values = matrix[
                np.ix_(
                    source_group,
                    target_group,
                )
            ]

            if reducer == 'sum':
                merged[row, column] = values.sum()

            elif reducer == 'mean':
                merged[row, column] = values.mean()

            else:
                raise ValueError(
                    "reducer phải là "
                    "'sum' hoặc 'mean'"
                )

    return (
        merged,
        source_labels,
        target_labels,
    )


def merge_subword_attention(
    attention,
    query_ids,
    key_ids,
    tokenizer,
):
    """
    Query subwords: mean.
    Key subwords: sum.
    """
    query_groups, query_labels = _word_groups(
        query_ids,
        tokenizer,
    )

    key_groups, key_labels = _word_groups(
        key_ids,
        tokenizer,
    )

    merged = np.zeros(
        (
            len(query_groups),
            len(key_groups),
        ),
        dtype=np.float32,
    )

    for row, query_group in enumerate(
        query_groups
    ):
        for column, key_group in enumerate(
            key_groups
        ):
            values = attention[
                np.ix_(
                    query_group,
                    key_group,
                )
            ]

            merged[row, column] = (
                values
                .sum(axis=1)
                .mean()
            )

    return merged, query_labels, key_labels


def _safe_row_normalize(matrix):
    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    denominator = matrix.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix),
        where=denominator > 1e-12,
    )


def _plot_data(
    result,
    tokenizer,
    merge_subwords,
):
    if not merge_subwords:
        return (
            result['similarity'],
            result['plan'],
            result['source_tokens'],
            result['target_tokens'],
        )

    similarity, source_labels, target_labels = (
        merge_subword_matrix(
            matrix=result['similarity'],
            source_ids=result['source_token_ids'],
            target_ids=result['target_token_ids'],
            tokenizer=tokenizer,
            reducer='mean',
        )
    )

    plan, _, _ = merge_subword_matrix(
        matrix=result['plan'],
        source_ids=result['source_token_ids'],
        target_ids=result['target_token_ids'],
        tokenizer=tokenizer,
        reducer='sum',
    )

    return (
        similarity,
        plan,
        source_labels,
        target_labels,
    )


def _prepare_attention(
    result,
    attention_name,
    tokenizer,
    merge_subwords=True,
    normalize_rows=True,
):
    matrix = result.get(attention_name)

    if matrix is None:
        return None, None, None

    if attention_name == 'source_self_attention':
        query_ids = result['source_token_ids']
        key_ids = result['source_token_ids']
        query_labels = result['source_tokens']
        key_labels = result['source_tokens']

    elif attention_name == 'target_self_attention':
        query_ids = result['target_token_ids']
        key_ids = result['target_token_ids']
        query_labels = result['target_tokens']
        key_labels = result['target_tokens']

    elif attention_name == 'target_to_source_attention':
        query_ids = result['target_token_ids']
        key_ids = result['source_token_ids']
        query_labels = result['target_tokens']
        key_labels = result['source_tokens']

    else:
        raise ValueError(
            f'Unknown attention matrix: '
            f'{attention_name}'
        )

    if merge_subwords:
        matrix, query_labels, key_labels = (
            merge_subword_attention(
                attention=matrix,
                query_ids=query_ids,
                key_ids=key_ids,
                tokenizer=tokenizer,
            )
        )

    if normalize_rows:
        matrix = _safe_row_normalize(matrix)

    return matrix, query_labels, key_labels


def _prepare_intra_similarity(
    result,
    side,
    tokenizer,
    merge_subwords=True,
    mask_diagonal=True,
):
    if side == 'source':
        matrix = result[
            'source_intra_similarity'
        ]

        token_ids = result[
            'source_token_ids'
        ]

        labels = result[
            'source_tokens'
        ]

    elif side == 'target':
        matrix = result[
            'target_intra_similarity'
        ]

        token_ids = result[
            'target_token_ids'
        ]

        labels = result[
            'target_tokens'
        ]

    else:
        raise ValueError(
            "side phải là 'source' hoặc 'target'"
        )

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    ).copy()

    if merge_subwords:
        matrix, labels, _ = merge_subword_matrix(
            matrix=matrix,
            source_ids=token_ids,
            target_ids=token_ids,
            tokenizer=tokenizer,
            reducer='mean',
        )

    if mask_diagonal:
        np.fill_diagonal(
            matrix,
            np.nan,
        )

    return matrix, labels


def plot_alignment_comparison(
    before,
    after,
    tokenizer,
    output_path=None,
    merge_subwords=True,
    show=True,
    dpi=180,
    similarity_cmap='coolwarm',
    transport_cmap='YlOrRd',
):
    (
        before_sim,
        before_plan,
        before_src,
        before_tgt,
    ) = _plot_data(
        before,
        tokenizer,
        merge_subwords,
    )

    (
        after_sim,
        after_plan,
        after_src,
        after_tgt,
    ) = _plot_data(
        after,
        tokenizer,
        merge_subwords,
    )

    sim_min = min(
        before_sim.min(),
        after_sim.min(),
    )

    sim_max = max(
        before_sim.max(),
        after_sim.max(),
    )

    plan_max = max(
        before_plan.max(),
        after_plan.max(),
    )

    width = max(
        15,
        0.45 * max(
            len(before_tgt),
            len(after_tgt),
        ) * 2,
    )

    height = max(
        10,
        0.35 * (
            len(before_src)
            + len(after_src)
        ),
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width, height),
        constrained_layout=True,
    )

    plots = [
        (
            axes[0, 0],
            before_sim,
            'BEFORE · cross cosine similarity',
            similarity_cmap,
            sim_min,
            sim_max,
            before_src,
            before_tgt,
        ),
        (
            axes[0, 1],
            after_sim,
            'AFTER · cross cosine similarity',
            similarity_cmap,
            sim_min,
            sim_max,
            after_src,
            after_tgt,
        ),
        (
            axes[1, 0],
            before_plan,
            'BEFORE · transport plan',
            transport_cmap,
            0,
            plan_max,
            before_src,
            before_tgt,
        ),
        (
            axes[1, 1],
            after_plan,
            'AFTER · transport plan',
            transport_cmap,
            0,
            plan_max,
            after_src,
            after_tgt,
        ),
    ]

    for (
        axis,
        matrix,
        title,
        cmap,
        value_min,
        value_max,
        source_labels,
        target_labels,
    ) in plots:
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=cmap,
            vmin=value_min,
            vmax=value_max,
            xticklabels=target_labels,
            yticklabels=source_labels,
        )

        axis.set(
            title=title,
            xlabel='Target',
            ylabel='Source',
        )

        axis.tick_params(
            axis='x',
            rotation=70,
            labelsize=8,
        )

        axis.tick_params(
            axis='y',
            rotation=0,
            labelsize=8,
        )

    saved_to = None

    if output_path is not None:
        saved_to = Path(output_path)

        if not saved_to.is_absolute():
            saved_to = REPO_ROOT / saved_to

        saved_to.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            saved_to,
            dpi=dpi,
            bbox_inches='tight',
        )

        print(
            f'Đã lưu ảnh OT: '
            f'{saved_to.resolve()}'
        )

    if show:
        plt.show()

    return fig, saved_to


def plot_attention_comparison(
    before,
    after,
    tokenizer,
    output_path=None,
    merge_subwords=True,
    normalize_rows=True,
    cmap='magma',
    show=True,
    dpi=180,
):
    specifications = [
        (
            'source_self_attention',
            r'$A_{\mathrm{src}\rightarrow'
            r'\mathrm{src}}$',
        ),
        (
            'target_to_source_attention',
            r'$A_{\mathrm{tgt}\rightarrow'
            r'\mathrm{src}}$',
        ),
        (
            'target_self_attention',
            r'$A_{\mathrm{tgt}\rightarrow'
            r'\mathrm{tgt}}$',
        ),
    ]

    prepared = []

    for attention_name, display_name in specifications:
        before_data = _prepare_attention(
            before,
            attention_name,
            tokenizer,
            merge_subwords=merge_subwords,
            normalize_rows=normalize_rows,
        )

        after_data = _prepare_attention(
            after,
            attention_name,
            tokenizer,
            merge_subwords=merge_subwords,
            normalize_rows=normalize_rows,
        )

        if (
            before_data[0] is not None
            and after_data[0] is not None
        ):
            prepared.append((
                display_name,
                before_data,
                after_data,
            ))

    if not prepared:
        raise ValueError(
            'Không tìm thấy attention matrix '
            'để visualize.'
        )

    fig, axes = plt.subplots(
        2,
        len(prepared),
        figsize=(
            max(14, 6 * len(prepared)),
            12,
        ),
        constrained_layout=True,
        squeeze=False,
    )

    for column, (
        display_name,
        before_data,
        after_data,
    ) in enumerate(prepared):
        (
            before_matrix,
            before_queries,
            before_keys,
        ) = before_data

        (
            after_matrix,
            after_queries,
            after_keys,
        ) = after_data

        value_min = min(
            before_matrix.min(),
            after_matrix.min(),
        )

        value_max = max(
            before_matrix.max(),
            after_matrix.max(),
        )

        row_data = [
            (
                before_matrix,
                before_queries,
                before_keys,
                'BEFORE',
            ),
            (
                after_matrix,
                after_queries,
                after_keys,
                'AFTER',
            ),
        ]

        for row, (
            matrix,
            query_labels,
            key_labels,
            phase,
        ) in enumerate(row_data):
            axis = axes[row, column]

            sns.heatmap(
                matrix,
                ax=axis,
                cmap=cmap,
                vmin=value_min,
                vmax=value_max,
                xticklabels=key_labels,
                yticklabels=query_labels,
            )

            normalization_label = (
                'row-normalized'
                if normalize_rows
                else 'raw'
            )

            axis.set_title(
                f'{phase} · {display_name}\n'
                f'{normalization_label}'
            )

            axis.set_xlabel('Key tokens')
            axis.set_ylabel('Query tokens')

            axis.tick_params(
                axis='x',
                rotation=70,
                labelsize=8,
            )

            axis.tick_params(
                axis='y',
                rotation=0,
                labelsize=8,
            )

    saved_to = None

    if output_path is not None:
        saved_to = Path(output_path)

        if not saved_to.is_absolute():
            saved_to = REPO_ROOT / saved_to

        saved_to.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            saved_to,
            dpi=dpi,
            bbox_inches='tight',
        )

        print(
            'Đã lưu attention figure: '
            f'{saved_to.resolve()}'
        )

    if show:
        plt.show()

    return fig, saved_to


def plot_intra_cosine_comparison(
    before,
    after,
    tokenizer,
    output_path=None,
    merge_subwords=True,
    mask_diagonal=True,
    cmap='coolwarm',
    show=True,
    dpi=180,
):
    before_source, before_source_labels = (
        _prepare_intra_similarity(
            before,
            'source',
            tokenizer,
            merge_subwords=merge_subwords,
            mask_diagonal=mask_diagonal,
        )
    )

    after_source, after_source_labels = (
        _prepare_intra_similarity(
            after,
            'source',
            tokenizer,
            merge_subwords=merge_subwords,
            mask_diagonal=mask_diagonal,
        )
    )

    before_target, before_target_labels = (
        _prepare_intra_similarity(
            before,
            'target',
            tokenizer,
            merge_subwords=merge_subwords,
            mask_diagonal=mask_diagonal,
        )
    )

    after_target, after_target_labels = (
        _prepare_intra_similarity(
            after,
            'target',
            tokenizer,
            merge_subwords=merge_subwords,
            mask_diagonal=mask_diagonal,
        )
    )

    source_min = min(
        np.nanmin(before_source),
        np.nanmin(after_source),
    )

    source_max = max(
        np.nanmax(before_source),
        np.nanmax(after_source),
    )

    target_min = min(
        np.nanmin(before_target),
        np.nanmin(after_target),
    )

    target_max = max(
        np.nanmax(before_target),
        np.nanmax(after_target),
    )

    source_size = max(
        len(before_source_labels),
        len(after_source_labels),
    )

    target_size = max(
        len(before_target_labels),
        len(after_target_labels),
    )

    width = max(
        16,
        0.35 * (
            source_size + target_size
        ),
    )

    height = max(
        11,
        0.35
        * max(source_size, target_size)
        * 2,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width, height),
        constrained_layout=True,
    )

    plots = [
        (
            axes[0, 0],
            before_source,
            'BEFORE · source intra-cosine',
            before_source_labels,
            source_min,
            source_max,
        ),
        (
            axes[0, 1],
            before_target,
            'BEFORE · target intra-cosine',
            before_target_labels,
            target_min,
            target_max,
        ),
        (
            axes[1, 0],
            after_source,
            'AFTER · source intra-cosine',
            after_source_labels,
            source_min,
            source_max,
        ),
        (
            axes[1, 1],
            after_target,
            'AFTER · target intra-cosine',
            after_target_labels,
            target_min,
            target_max,
        ),
    ]

    for (
        axis,
        matrix,
        title,
        labels,
        value_min,
        value_max,
    ) in plots:
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=cmap,
            vmin=value_min,
            vmax=value_max,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
        )

        axis.set_title(title)
        axis.set_xlabel('Token')
        axis.set_ylabel('Token')

        axis.tick_params(
            axis='x',
            rotation=70,
            labelsize=8,
        )

        axis.tick_params(
            axis='y',
            rotation=0,
            labelsize=8,
        )

    saved_to = None

    if output_path is not None:
        saved_to = Path(output_path)

        if not saved_to.is_absolute():
            saved_to = REPO_ROOT / saved_to

        saved_to.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            saved_to,
            dpi=dpi,
            bbox_inches='tight',
        )

        print(
            'Đã lưu intra-cosine figure: '
            f'{saved_to.resolve()}'
        )

    if show:
        plt.show()

    return fig, saved_to


def visualize_ot_alignment(
    source_text,
    target_text,
    after_model_or_adapter,
    before_model_name_or_path=None,
    source_lang='en',
    target_lang='vi',
    output_path=(
        'outputs/ot-visualization/alignment.png'
    ),
    *,
    prompt_format='plain',
    enable_thinking=False,
    training_mode='finetune',
    alignment_forward_mode='joint',
    independent_add_special_tokens=True,
    align_layer=-1,
    ot_solver='sinkhorn',
    attention_mass_weight=0,
    sinkhorn_epsilon=0.03,
    sinkhorn_iterations=20,
    ipot_beta=0.5,
    ipot_iterations=50,
    ipot_inner_iterations=1,
    trust_remote_code=False,
    merge_subwords=True,
    mask_intra_diagonal=True,
    normalize_attention_rows=True,
    similarity_cmap='coolwarm',
    transport_cmap='YlOrRd',
    attention_cmap='magma',
    show=True,
    dpi=180,
):
    after_is_adapter = _is_adapter(
        after_model_or_adapter
    )

    adapter_base = (
        PeftConfig.from_pretrained(
            str(after_model_or_adapter)
        ).base_model_name_or_path
        if after_is_adapter
        else None
    )

    base_model_name = (
        before_model_name_or_path
        or adapter_base
    )

    if base_model_name is None:
        raise ValueError(
            'Hãy truyền before_model_name_or_path '
            'khi AFTER không phải PEFT adapter.'
        )

    if (
        adapter_base
        and before_model_name_or_path
        and before_model_name_or_path
        != adapter_base
    ):
        print(
            f'Cảnh báo: adapter dùng base '
            f'{adapter_base}, nhưng BEFORE là '
            f'{before_model_name_or_path}.'
        )

    after_path = Path(
        str(after_model_or_adapter)
    )

    has_tokenizer = (
        after_path.is_dir()
        and any(
            (after_path / name).exists()
            for name in (
                'tokenizer.json',
                'tokenizer_config.json',
            )
        )
    )

    tokenizer_source = (
        str(after_model_or_adapter)
        if has_tokenizer
        else base_model_name
    )

    context = build_context(
        source_text=source_text,
        target_text=target_text,
        source_lang=source_lang,
        target_lang=target_lang,
        tokenizer_source=tokenizer_source,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
        training_mode=training_mode,
        trust_remote_code=trust_remote_code,
    )

    config = {
        'base_model_name':
            base_model_name,

        'alignment_forward_mode':
            alignment_forward_mode,

        'independent_add_special_tokens':
            independent_add_special_tokens,

        'align_layer':
            align_layer,

        'ot_solver':
            ot_solver,

        'attention_mass_weight':
            attention_mass_weight,

        'sinkhorn_epsilon':
            sinkhorn_epsilon,

        'sinkhorn_iterations':
            sinkhorn_iterations,

        'ipot_beta':
            ipot_beta,

        'ipot_iterations':
            ipot_iterations,

        'ipot_inner_iterations':
            ipot_inner_iterations,

        'trust_remote_code':
            trust_remote_code,
    }

    ss, se, ts, te = context['spans']
    ids = context['input_ids'][0]

    print(
        'Source span:',
        context['tokenizer'].decode(
            ids[ss:se]
        ),
    )

    print(
        'Target span:',
        context['tokenizer'].decode(
            ids[ts:te]
        ),
    )

    before = analyze_checkpoint(
        model_path=base_model_name,
        context=context,
        config=config,
        name='BEFORE',
        adapter=False,
    )

    after = analyze_checkpoint(
        model_path=after_model_or_adapter,
        context=context,
        config=config,
        name='AFTER',
        adapter='auto',
    )

    metrics = pd.DataFrame({
        'BEFORE': before['metrics'],
        'AFTER': after['metrics'],
    })

    metrics['DELTA (AFTER - BEFORE)'] = (
        metrics['AFTER']
        - metrics['BEFORE']
    )

    output_path_object = (
        None if output_path is None else Path(output_path)
    )

    figure, saved_to = plot_alignment_comparison(
        before=before,
        after=after,
        tokenizer=context['tokenizer'],
        output_path=output_path_object,
        merge_subwords=merge_subwords,
        show=show,
        dpi=dpi,
        similarity_cmap=similarity_cmap,
        transport_cmap=transport_cmap,
    )

    attention_output_path = (
        None
        if output_path_object is None
        else output_path_object.parent
        / (
            output_path_object.stem
            + '_attention'
            + output_path_object.suffix
        )
    )

    attention_figure, attention_saved_to = (
        plot_attention_comparison(
            before=before,
            after=after,
            tokenizer=context['tokenizer'],
            output_path=attention_output_path,
            merge_subwords=merge_subwords,
            normalize_rows=(
                normalize_attention_rows
            ),
            cmap=attention_cmap,
            show=show,
            dpi=dpi,
        )
    )

    intra_cosine_output_path = (
        None
        if output_path_object is None
        else output_path_object.parent
        / (
            output_path_object.stem
            + '_intra_cosine'
            + output_path_object.suffix
        )
    )

    (
        intra_cosine_figure,
        intra_cosine_saved_to,
    ) = plot_intra_cosine_comparison(
        before=before,
        after=after,
        tokenizer=context['tokenizer'],
        output_path=intra_cosine_output_path,
        merge_subwords=merge_subwords,
        mask_diagonal=mask_intra_diagonal,
        cmap=similarity_cmap,
        show=show,
        dpi=dpi,
    )

    print(metrics.to_string())

    return {
        'before': before,
        'after': after,
        'metrics': metrics,

        'figure': figure,
        'saved_to': saved_to,

        'attention_figure':
            attention_figure,

        'attention_saved_to':
            attention_saved_to,

        'intra_cosine_figure':
            intra_cosine_figure,

        'intra_cosine_saved_to':
            intra_cosine_saved_to,

        'context': context,
        'config': config,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Visualize cosine similarity, OT transport, attention and '
            'intra-sequence cosine structure before/after alignment.'
        )
    )
    parser.add_argument('--source-text', required=True)
    parser.add_argument('--target-text', required=True)
    parser.add_argument('--after-model', required=True)
    parser.add_argument('--before-model', default=None)
    parser.add_argument('--source-lang', default='en')
    parser.add_argument('--target-lang', default='vi')
    parser.add_argument(
        '--output-path',
        default='outputs/ot-visualization/alignment.png',
    )
    parser.add_argument('--prompt-format', choices=['plain', 'chat'], default='plain')
    parser.add_argument('--enable-thinking', action='store_true')
    parser.add_argument(
        '--alignment-forward-mode',
        choices=['joint', 'independent'],
        default='joint',
    )
    parser.add_argument('--align-layer', type=int, default=-1)
    parser.add_argument('--ot-solver', choices=['sinkhorn', 'ipot'], default='sinkhorn')
    parser.add_argument('--attention-mass-weight', type=float, default=0.0)
    parser.add_argument('--sinkhorn-epsilon', type=float, default=0.03)
    parser.add_argument('--sinkhorn-iterations', type=int, default=20)
    parser.add_argument('--ipot-beta', type=float, default=0.5)
    parser.add_argument('--ipot-iterations', type=int, default=50)
    parser.add_argument('--ipot-inner-iterations', type=int, default=1)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--no-merge-subwords', action='store_true')
    parser.add_argument('--no-normalize-attention-rows', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_ot_alignment(
        source_text=args.source_text,
        target_text=args.target_text,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        after_model_or_adapter=args.after_model,
        before_model_name_or_path=args.before_model,
        output_path=args.output_path,
        prompt_format=args.prompt_format,
        enable_thinking=args.enable_thinking,
        alignment_forward_mode=args.alignment_forward_mode,
        align_layer=args.align_layer,
        ot_solver=args.ot_solver,
        attention_mass_weight=args.attention_mass_weight,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iterations=args.sinkhorn_iterations,
        ipot_beta=args.ipot_beta,
        ipot_iterations=args.ipot_iterations,
        ipot_inner_iterations=args.ipot_inner_iterations,
        trust_remote_code=args.trust_remote_code,
        merge_subwords=not args.no_merge_subwords,
        normalize_attention_rows=not args.no_normalize_attention_rows,
        show=not args.no_show,
    )


if __name__ == '__main__':
    main()
