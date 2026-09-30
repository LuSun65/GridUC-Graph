"""Standard SmallMLP adapted from upstream edge-set-attention."""
from torch import nn
from torch.nn import functional as F


class SmallMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        num_layers=2,
        use_ln=False,
    ):
        super().__init__()

        self.mlp = []

        if num_layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.mlp.append(nn.Linear(in_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                elif i != num_layers - 1:
                    self.mlp.append(nn.Linear(inter_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                else:
                    self.mlp.append(nn.Linear(inter_dim, out_dim))

                if dropout_p > 0:
                    self.mlp.append(nn.Dropout(p=dropout_p))

            self.mlp = nn.Sequential(*self.mlp)


    def forward(self, x):
        return self.mlp(x)
