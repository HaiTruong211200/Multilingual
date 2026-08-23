import torch

from nllb_qformer_qwen.losses import sinkhorn_hidden_state_ot, symmetric_info_nce
from nllb_qformer_qwen.modules import GlobalAdapter, QFormer, TransformerBridge


def test_shapes_and_gradients():
    adapter = GlobalAdapter(16, 4)
    qformer = QFormer(16, 12, 5, 2, 3, dropout=0.0)
    hidden = torch.randn(4, 7, 16)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]] * 4)
    adapted = adapter(hidden)
    latent, states = qformer(adapted, mask, return_hidden_states=True)
    loss = latent.square().mean()
    loss.backward()
    assert latent.shape == (4, 5, 12)
    assert len(states) == 2
    assert adapter.up.weight.grad is not None


def test_alignment_losses_are_finite():
    left, right = torch.randn(3, 5, 12), torch.randn(3, 5, 12)
    mask = torch.ones(3, 5, dtype=torch.bool)
    assert torch.isfinite(sinkhorn_hidden_state_ot(left, right, mask, mask))
    assert torch.isfinite(symmetric_info_nce(left.mean(1), right.mean(1)))


def test_identical_queries_have_lower_ot():
    left = torch.randn(3, 5, 12)
    mask = torch.ones(3, 5, dtype=torch.bool)
    identical = sinkhorn_hidden_state_ot(left, left, mask, mask)
    unrelated = sinkhorn_hidden_state_ot(left, torch.randn_like(left), mask, mask)
    assert identical < unrelated


def test_ot_ignores_masked_hidden_states_per_batch_item():
    left, right = torch.randn(2, 5, 12), torch.randn(2, 6, 12)
    left_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    right_mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    baseline = sinkhorn_hidden_state_ot(left, right, left_mask, right_mask)
    changed = left.clone()
    changed[~left_mask] = 1_000_000
    assert torch.allclose(baseline, sinkhorn_hidden_state_ot(changed, right, left_mask, right_mask))


def test_transformer_bridge_preserves_token_mask_and_returns_layers():
    bridge = TransformerBridge(16, 12, layers=2, heads=3, dropout=0.0)
    hidden = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    output, states = bridge(hidden, mask, return_hidden_states=True)
    assert output.shape == (2, 6, 12)
    assert len(states) == 2
    assert torch.count_nonzero(output[~mask.bool()]) == 0
