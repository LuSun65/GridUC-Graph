# ESA encoder provenance

Source: https://github.com/davidbuterez/edge-set-attention
Pinned commit: `861a0b12d445ae80cc69b3948af623b4331321d4`
License: MIT, Copyright (c) 2024 David Buterez; included in `LICENSE`.

`attention.py` adapts upstream `esa/mha.py` (MAB/SAB).
`mlp.py` extracts `SmallMLP` from `esa/mlp_utils.py`.
`encoder.py` adapts the constant-width, standard-MLP, Pre-LN path of
`esa/masked_layers.py` (`SABComplete` and encoder-only `ESA`) and the
LayerNorm definition in `utils/norm_layers.py`.

Preserved: independent Q/K/V projections, Xavier-normal weight initialization,
SDPA attention, `out + mish(fc_o(out))`, upstream's normalized-input Pre-LN
residual, feed-forward residual, outer encoder input skip, and final
linear + Mish projection. No PMA or graph pooling is used.

Intentional adaptations:

- Only PyTorch SDPA; CPU uses the math backend (PyTorch 2.3.1 fused CPU
  attention produces NaNs on large sparse masks); CUDA selects available kernels
  automatically. No xformers, flash-attn, PyG, Lightning, or admin-torch dependency.
- A shared `[1,1,L,L]` boolean mask has **True = allowed**, for edges sharing
  either endpoint, including the diagonal. Upstream's torch mask builder
  complements adjacency before SDPA; this adapter corrects that polarity and
  retains self-attention. It is not bitwise equivalent to that upstream mask.
- Residual dropout follows `self.training`; upstream calls functional dropout
  without passing the mode. Evaluation here is deterministic.
- Fixed graph batches need neither padding nor token mixing between samples.
  The topology mask is cached as nonpersistent buffers, not repeated over B/T.
- Optional per-block gradient checkpointing uses PyTorch's non-reentrant API.
- UC input adaptation and endpoint-message/mean/node readout are project code.

Default experiment: H=64, four heads, `MSMS` (depth = sequence length),
Pre-LN (epsilon 1e-8), standard two-layer Mish MLPs of hidden width 128,
all dropout zero, full-horizon inputs, first-half physical-line tokens,
endpoint mean readout. Configuration and upstream revision are serialized by
existing checkpoint code. Registry architecture version `3.1` rejects old
`3.0` three-branch ESA checkpoints.

Run the portable contract checks with:

```sh
OMP_NUM_THREADS=1 python -m unittest discover -s tests/models -p 'test_esa_contract.py' -v
```
