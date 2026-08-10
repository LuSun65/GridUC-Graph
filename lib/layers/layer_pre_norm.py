import torch
import torch.nn as nn


class PreNormLayer(nn.Module):
    """
    Normalizes features using a learned shift and scale fitted on the first batch.

    x [B, ..., F]  ->  (x + shift) * scale  ->  [B, ..., F]

    NOTE: normalization is applied only over the last dimension (F). All leading
    dimensions (batch, sequence, graph nodes, etc.) are treated as independent samples.

    Usage:
        layer.update_on()
        with torch.no_grad():
            model(calibration_batch)   # shift/scale fitted on this batch
        layer.update_off()             # frozen for normal inference/training
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer('shift', torch.zeros(n_features))
        self.register_buffer('scale', torch.ones(n_features))
        self.n_features = n_features
        self.is_updating = False

    def _update_stats(self, x: torch.Tensor):
        # flatten all dims except last, compute mean/var over F
        x_flat = x.reshape(-1, self.n_features)
        mean = x_flat.mean(dim=0)
        var  = x_flat.var(dim=0, unbiased=False)
        self.shift = -mean
        self.scale = 1.0 / torch.sqrt(var + 1e-8)

    def update_on(self):
        self.is_updating = True

    def update_off(self):
        self.is_updating = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [..., F]
        if self.is_updating:
            self._update_stats(x)
        return (x + self.shift) * self.scale
