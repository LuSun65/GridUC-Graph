import torch
import torch.nn as nn


class PreNormLayer(nn.Module):
    """
    Normalize features using fixed statistics collected during calibration.

    x [B, ..., F]  ->  (x + shift) * scale  ->  [B, ..., F]

    NOTE: normalization is applied only over the last dimension (F). All leading
    dimensions (batch, sequence, graph nodes, etc.) are treated as independent samples.

    Usage:
        layer.update_on()
        with torch.no_grad():
            for calibration_batch in calibration_loader:
                model(calibration_batch)
        layer.update_off()             # frozen for normal inference/training
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer('shift', torch.zeros(n_features))
        self.register_buffer('scale', torch.ones(n_features))
        # Calibration accumulators follow the module device but are not part of
        # checkpoints; only the finalized shift and scale are persistent.
        self.register_buffer(
            '_count', torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            '_running_mean', torch.zeros(n_features), persistent=False
        )
        self.register_buffer(
            '_running_m2', torch.zeros(n_features), persistent=False
        )
        self.n_features = n_features
        self.is_updating = False

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor):
        """Merge one batch into the full-dataset feature statistics."""
        x_flat = x.reshape(-1, self.n_features).to(self._running_mean.dtype)
        batch_count = x_flat.shape[0]
        if batch_count == 0:
            return

        batch_mean = x_flat.mean(dim=0)
        batch_m2 = ((x_flat - batch_mean) ** 2).sum(dim=0)

        previous_count = self._count.to(batch_mean.dtype)
        batch_count_tensor = batch_mean.new_tensor(batch_count)
        total_count = previous_count + batch_count_tensor
        delta = batch_mean - self._running_mean

        self._running_mean.add_(delta * batch_count_tensor / total_count)
        self._running_m2.add_(
            batch_m2
            + delta.square()
            * previous_count
            * batch_count_tensor
            / total_count
        )
        self._count.add_(batch_count)
        self._set_normalization()

    @torch.no_grad()
    def _set_normalization(self):
        count = self._count.to(self._running_m2.dtype)
        variance = (self._running_m2 / count).clamp_min(0.0)
        self.shift.copy_(-self._running_mean.to(self.shift.dtype))
        self.scale.copy_(
            torch.rsqrt(variance + 1e-8).to(self.scale.dtype)
        )

    def update_on(self):
        self._count.zero_()
        self._running_mean.zero_()
        self._running_m2.zero_()
        self.is_updating = True

    def update_off(self):
        if self._count.item() == 0:
            raise RuntimeError("cannot finalize PreNorm without calibration data")
        self._set_normalization()
        self.is_updating = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [..., F]
        if self.is_updating:
            self._update_stats(x)
        return (x + self.shift) * self.scale
