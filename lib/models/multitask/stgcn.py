import torch

from lib.models.stgcn.stgcn import STGCN
from .module_fusion import MultitaskFusion


class STGCN_V1_Multitask(STGCN):
    """Reuse V1's configuration, encoders and PreNorm with UC/LMP output."""

    fusion_type = MultitaskFusion

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("lmp_mean", torch.tensor(0.0))
        self.register_buffer("lmp_std", torch.tensor(1.0))
