from dataclasses import dataclass
from typing import Any, Dict, Mapping, Type

import torch.nn as nn

from lib.models.mlp import MLP, MLPConfig
from lib.models.stgcn.stgcn import STGCN, STGCNConfig
from lib.models.model_input import STGCNInput
from lib.models.stgcn_attn.stgcn_v2 import STGCN_V2, STGCNV2Config
from lib.models.stgcn_attn.stgcn_v2_1 import STGCN_V2_1, STGCNV21Config
from lib.models.stgcn_attn.stgcn_v2_2 import STGCN_V2_2, STGCNV22Config
from lib.models.esa.stgcn_v3 import STGCN_V3, STGCNV3Config
from lib.models.multitask import STGCN_V1_Multitask


@dataclass(frozen=True)
class ModelSpec:
    model_class: Type[nn.Module]
    config_class: Type
    architecture_version: str
    prenorm_warmup: bool = False


# Family/version -> source directory -> canonical model_type:
# V1       -> stgcn/      -> stgcn_v1
# V2/2.1/2.2 -> stgcn_attn/ -> stgcn_v2 / stgcn_v2.1 / stgcn_v2.2
# V3       -> esa/        -> stgcn_v3 (interface only)
# V4       -> multitask/  -> stgcn_v1_mtl (alias: stgcn_v4)
# V4 retains its existing name and checkpoint architecture_version="1.0";
# the family label changes neither its V1 encoder nor its serialized weights.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "mlp": ModelSpec(MLP, MLPConfig, "1.0"),
    "stgcn_v1": ModelSpec(STGCN, STGCNConfig, "1.0", prenorm_warmup=True),
    "stgcn_v1_mtl": ModelSpec(
        STGCN_V1_Multitask, STGCNConfig, "1.0", prenorm_warmup=True
    ),
    "stgcn_v2": ModelSpec(STGCN_V2, STGCNV2Config, "2.0", prenorm_warmup=True),
    "stgcn_v2.1": ModelSpec(
        STGCN_V2_1, STGCNV21Config, "2.1", prenorm_warmup=True
    ),
    "stgcn_v2.2": ModelSpec(
        STGCN_V2_2, STGCNV22Config, "2.2", prenorm_warmup=True
    ),
    "stgcn_v3": ModelSpec(STGCN_V3, STGCNV3Config, "3.0"),
}

# Directory aliases select a fixed version; stgcn_attn means V2.0, not latest.
# Historical experiment names remain supported.
MODEL_ALIASES = {
    "stgcn": "stgcn_v1",
    "stgcn_attn": "stgcn_v2",
    "esa": "stgcn_v3",
    "multitask": "stgcn_v1_mtl",
    "stgcn_v4": "stgcn_v1_mtl",
    "stgcn-v2": "stgcn_v2",
    "stgcn-v2.1": "stgcn_v2.1",
    "stgcn-v2.2": "stgcn_v2.2",
}


def canonical_model_type(model_type: str) -> str:
    canonical = MODEL_ALIASES.get(model_type, model_type)
    if canonical not in MODEL_REGISTRY:
        choices = sorted((*MODEL_REGISTRY.keys(), *MODEL_ALIASES.keys()))
        raise ValueError(
            f"Unknown model_type: {model_type!r}; expected one of {choices}"
        )
    return canonical


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
