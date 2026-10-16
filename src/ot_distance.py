"""Transport costs shared by Sinkhorn and IPOT."""
import torch
import torch.nn.functional as F


def stop_gradient_rows(values, selected_rows):
    """Detach selected batch rows without changing their values or other rows."""
    selected = selected_rows.to(device=values.device, dtype=torch.bool)
    if selected.shape != (values.shape[0],):
        raise ValueError('English language flags must have shape [batch_size]')
    selected = selected.reshape((values.shape[0],) + (1,) * (values.ndim - 1))
    return torch.where(selected, values.detach(), values)


def load_whitening(path, hidden_size, regularization=1e-4,
                   method='ridge', relative_threshold=1e-5):
    """Build whitening from covariance (tensor, .npy, .pt, or calibration cache).

    Covariance must be precomputed at the alignment layer in the same
    representation space as the tokens being aligned. It is fixed during SFT.
    'truncated' discards small eigenvalues as CAROT does; 'ridge' adds lambda I.
    """
    if method not in {'ridge', 'truncated'}:
        raise ValueError('Whitening method must be ridge or truncated')
    if method == 'ridge' and regularization <= 0:
        raise ValueError("ot_covariance_regularization must be positive")
    if not 0 <= relative_threshold < 1:
        raise ValueError('Eigenvalue relative threshold must be in [0, 1)')
    if isinstance(path, torch.Tensor):
        covariance = path
    elif str(path).lower().endswith('.npy'):
        import numpy as np
        covariance = torch.from_numpy(np.load(path, allow_pickle=False))
    else:
        covariance = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(covariance, dict) and 'covariance' in covariance:
        covariance = covariance['covariance']
    if not isinstance(covariance, torch.Tensor):
        raise ValueError("Expected a covariance tensor or a calibration dictionary with 'covariance'")
    covariance = covariance.detach().to(device='cpu', dtype=torch.float64)
    if covariance.shape != (hidden_size, hidden_size) or not torch.isfinite(covariance).all():
        raise ValueError(f"Covariance must be a finite [{hidden_size}, {hidden_size}] matrix")
    if not torch.allclose(covariance, covariance.T, atol=1e-6, rtol=1e-5):
        raise ValueError("Covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    tolerance = 1e-8 * max(1.0, eigenvalues.abs().max().item())
    if eigenvalues.min().item() < -tolerance:
        raise ValueError("Covariance must be positive semidefinite")
    if method == 'truncated':
        # CAROT's spectral truncation, applied here WITHOUT language erasure.
        keep = eigenvalues > eigenvalues.max() * relative_threshold
        if not keep.any():
            raise ValueError('Covariance has no positive eigenvalues above the threshold')
        return (eigenvectors[:, keep] / eigenvalues[keep].sqrt()).float()
    scales = (eigenvalues.clamp_min(0) + regularization).rsqrt()
    return ((eigenvectors * scales) @ eigenvectors.T).float()


def transport_cost(source, target, distance='cosine', whitening=None):
    """Return [batch, source_tokens, target_tokens] costs in float32."""
    source, target = source.float(), target.float()
    if distance == 'cosine':
        source = F.normalize(source, dim=-1, eps=1e-6)
        target = F.normalize(target, dim=-1, eps=1e-6)
        return (1 - torch.bmm(source, target.transpose(1, 2))).clamp(0, 2)
    if distance == 'mahalanobis':
        if whitening is None:
            raise ValueError("Mahalanobis distance requires a covariance file")
        whitening = whitening.to(device=source.device, dtype=torch.float32)
        source, target = source @ whitening, target @ whitening
    elif distance != 'squared_euclidean':
        raise ValueError(f"Unsupported OT distance: {distance}")
    return (
        source.square().sum(-1, keepdim=True)
        + target.square().sum(-1).unsqueeze(1)
        - 2 * torch.bmm(source, target.transpose(1, 2))
    ).clamp_min(0)
