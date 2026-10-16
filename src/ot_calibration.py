"""Fixed sentence covariance for OT, using CAROT pooling without LEACE."""
import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm

LOGGER = logging.getLogger('multilingual_training')


def position_weighted_pool(hidden, content_mask):
    """Weight valid tokens 1..n, excluding special tokens and padding."""
    mask = content_mask.to(device=hidden.device, dtype=torch.float32)
    if (mask.sum(-1) == 0).any():
        raise ValueError('Calibration sentence has no content tokens')
    weights = mask.cumsum(-1) * mask
    weights = weights / weights.sum(-1, keepdim=True)
    return (hidden.float() * weights.unsqueeze(-1)).sum(1)


@torch.no_grad()
def calibrate_covariance(
    model, tokenizer, train_dataset, path, *, layer=-1,
    samples_per_language=1024, batch_size=8, max_length=256, seed=42,
    model_name='',
):
    """Estimate centered covariance on train-only standalone sentences.

    Cache metadata is checked before reuse. Each DDP rank computes/loads the
    same calibration independently; only rank zero writes the cache. Sentence
    counts are capped per language after deterministic shuffling and deduping.
    """
    if min(samples_per_language, batch_size, max_length) < 1:
        raise ValueError('Calibration sample count, batch size and max length must be positive')
    metadata = dict(
        model=model_name, layer=layer, dataset_fingerprint=train_dataset._fingerprint,
        samples_per_language=samples_per_language, max_length=max_length, seed=seed,
        pooling='position_weighted', leace=False,
        tokenizer_name=tokenizer.name_or_path, vocabulary_size=len(tokenizer),
        excluded_tokens='special,padding,first_if_no_bos',
    )
    path = Path(path)
    if path.exists():
        cached = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(cached, dict) or cached.get('metadata') != metadata:
            raise ValueError(f'Calibration cache does not match this run: {path}. Use another cache path.')
        LOGGER.info('Reusing OT covariance cache: %s', path)
        return cached['covariance']

    grouped = {}
    seen = {}
    for row in train_dataset.shuffle(seed=seed):
        for text_key, lang_key in [('source', 'source_lang'), ('target', 'target_lang')]:
            lang, text = row[lang_key], row[text_key]
            texts = grouped.setdefault(lang, [])
            unique = seen.setdefault(lang, set())
            if len(texts) < samples_per_language and text not in unique:
                texts.append(text)
                unique.add(text)
    texts = [text for lang in sorted(grouped) for text in grouped[lang]]
    LOGGER.info('Calibrating raw-hidden covariance | layer=%s | sentences=%s | LEACE=False',
                layer, {lang: len(values) for lang, values in grouped.items()})
    count, mean, scatter = 0, None, None
    training = model.training
    model.eval()
    device = model.get_input_embeddings().weight.device
    try:
        for start in tqdm(range(0, len(texts), batch_size), desc='OT covariance calibration'):
            encoded = tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors='pt', return_special_tokens_mask=True,
            )
            special = encoded.pop('special_tokens_mask').to(device).bool()
            encoded = {key: value.to(device) for key, value in encoded.items()}
            mask = encoded['attention_mask'].bool() & ~special
            for token in tokenizer.all_special_ids:
                mask &= encoded['input_ids'].ne(token)
            if tokenizer.bos_token_id is None:
                first = encoded['attention_mask'].long().argmax(-1)
                mask[torch.arange(len(mask), device=device), first] = False
            output = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
            hidden = output.hidden_states[layer]
            valid = mask.any(-1)
            if not valid.any():
                continue
            values = position_weighted_pool(hidden[valid], mask[valid]).cpu().double()
            n = len(values)
            batch_mean = values.mean(0)
            centered = values - batch_mean
            batch_scatter = centered.T @ centered
            if count == 0:
                mean, scatter = batch_mean, batch_scatter
            else:
                delta = batch_mean - mean
                scatter += batch_scatter + torch.outer(delta, delta) * (count * n / (count + n))
                mean += delta * (n / (count + n))
            count += n
            del output, hidden, encoded
    finally:
        model.train(training)
    if count < 2:
        raise ValueError('Need at least two valid sentences to fit OT covariance')
    covariance = scatter / (count - 1)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed or torch.distributed.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(dict(covariance=covariance, metadata=metadata, sentence_count=count), temporary)
        temporary.replace(path)
        LOGGER.info('Saved fixed OT covariance: %s | sentences=%d', path, count)
    return covariance
