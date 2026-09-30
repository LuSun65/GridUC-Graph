from dataclasses import dataclass
from typing import Any, Dict, Mapping, Type

import torch.nn as nn

from lib.models.mlp import MLP, MLPConfig
from lib.models.stgcn.stgcn import STGCN, STGCNConfig
from lib.models.model_input import STGCNInput
from lib.models.stgcn_attn.stgcn_attn_0 import STGCNAttn0, STGCNAttn0Config
from lib.models.stgcn_attn.stgcn_attn_1 import STGCNAttn1, STGCNAttn1Config
from lib.models.stgcn_attn.stgcn_attn_2 import STGCNAttn2, STGCNAttn2Config
from lib.models.multitask import STGCN_V1_Multitask
from lib.models.esa.esa import ESA, ESAConfig


@dataclass(frozen=True)
class ModelSpec:
    model_class: Type[nn.Module]
    config_class: Type
    architecture_version: str
    prenorm_warmup: bool = False


# Names match model families; architecture_version describes checkpoint schemas.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "mlp": ModelSpec(MLP, MLPConfig, "1.0"),
    "stgcn": ModelSpec(STGCN, STGCNConfig, "1.0", prenorm_warmup=True),
    "stgcn_attn_0": ModelSpec(STGCNAttn0, STGCNAttn0Config, "2.0", prenorm_warmup=True),
    "stgcn_attn_1": ModelSpec(STGCNAttn1, STGCNAttn1Config, "2.1", prenorm_warmup=True),
    "stgcn_attn_2": ModelSpec(STGCNAttn2, STGCNAttn2Config, "2.2", prenorm_warmup=True),
    "esa": ModelSpec(ESA, ESAConfig, "3.1", prenorm_warmup=True),
    "multitask": ModelSpec(STGCN_V1_Multitask, STGCNConfig, "1.0", prenorm_warmup=True),
}

# Read old checkpoint metadata without rewriting serialized data. These are
# not accepted as model names in new calls or used in output filenames.
_LEGACY_CHECKPOINT_NAMES = {
    "stgcn_v1": "stgcn",
    "stgcn_v2": "stgcn_attn_0",
    "stgcn_v3": "esa",
    "stgcn_v4": "multitask",
    "stgcn-v2": "stgcn_attn_0",
    "stgcn_attn": "stgcn_attn_0",
    "stgcn_v2.1": "stgcn_attn_1",
    "stgcn-v2.1": "stgcn_attn_1",
    "stgcn_v2.2": "stgcn_attn_2",
    "stgcn-v2.2": "stgcn_attn_2",
    "stgcn_v1_mtl": "multitask",
}


def canonical_model_type(model_type: str) -> str:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type: {model_type!r}; expected one of {sorted(MODEL_REGISTRY)}"
        )
    return model_type


def checkpoint_model_type(model_type: str) -> str:
    return canonical_model_type(_LEGACY_CHECKPOINT_NAMES.get(model_type, model_type))


def get_model_spec(model_type: str) -> ModelSpec:
    return MODEL_REGISTRY[canonical_model_type(model_type)]


def build_model(
    model_type: str,
    data: STGCNInput,
    config_overrides: Mapping[str, Any] = None,
) -> nn.Module:
    spec = get_model_spec(model_type)
    config = spec.model_class.default_config(data)
    for name, value in (config_overrides or {}).items():
        if not hasattr(config, name):
            raise ValueError(
                f"{canonical_model_type(model_type)} config has no field {name!r}"
            )
        setattr(config, name, value)
    return spec.model_class(config)


def model_from_config(model_type: str, config) -> nn.Module:
    spec = get_model_spec(model_type)
    if isinstance(config, dict):
        config = spec.config_class(**config)
    return spec.model_class(config)
