from lib.models.stgcn.module_fusion import FusionModule
from .task_heads import LMPHead, STGCNOutput, UCHead


class MultitaskFusion(FusionModule):
    """Share V1's bus fusion and branch before generator-bus selection."""

    def __init__(self, config):
        super().__init__(config)
        self.uc_head = UCHead(self.map_layer, self.out_layer, config.n_period)
        # The existing layers now belong solely to the UC head.
        del self.map_layer, self.out_layer
        self.lmp_head = LMPHead(config.f_hidden, config.n_period, config.n_layers)

    def forward(self, data) -> STGCNOutput:
        x = self.encode(data)
        return STGCNOutput(
            uc_logits=self.uc_head(x, data.gen_bus),
            lmp_pred=self.lmp_head(x),
        )
