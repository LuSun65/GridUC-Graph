"""Add fixed-commitment pricing labels to existing samples in place."""

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

from lib.lmp_sover import (
    PRICING_RULE,
    _validate_uc_solution, commitment_hash, solve_pricing,
)
from lib.toolkit import load_pkl


def valid_pricing_labels(data, uc_type: str) -> bool:
    """Only skip complete labels generated with the current pricing contract."""
    try:
        commitment = _validate_uc_solution(data, data.uc_sol, uc_type)
        for name, shape in (
            ("lmp_target", (data.num_bus, data.num_period)),
            ("p_target", (data.num_gen, data.num_period)),
        ):
            value = getattr(data, name, None)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                return False
            if not np.isfinite(value).all():
                return False
        for name in ("pricing_obj", "pricing_solve_time"):
            value = getattr(data, name, None)
            if value is None or not np.isscalar(value) or not np.isfinite(value):
                return False
        if data.pricing_solve_time < 0:
            return False
        metadata = getattr(data, "pricing_metadata", None)
        expected = {
            "pricing_rule": PRICING_RULE,
            "uc_sol_sha256": commitment_hash(commitment),
            "uc_type": uc_type,
        }
        return (isinstance(metadata, dict)
                and all(metadata.get(key) == value for key, value in expected.items()))
    except (TypeError, ValueError, AttributeError):
        return False


def _atomic_save(data, path: Path) -> None:
    # The temporary file must share the destination filesystem for atomic replace.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def label_sample(path, uc_type: str, overwrite: bool = False) -> str:
    """Return 'saved' or 'skipped'; failures leave the original file untouched."""
    path = Path(path)
    data = load_pkl(path)
    _validate_uc_solution(data, data.uc_sol, uc_type)
    if not overwrite and valid_pricing_labels(data, uc_type):
        return "skipped"
    result = solve_pricing(data, data.uc_sol, uc_type)
    data.lmp_target = result.lmp_target
    data.p_target = result.p_target
    data.pricing_obj = result.obj
    data.pricing_solve_time = result.solve_time
    data.pricing_metadata = result.metadata
    if not valid_pricing_labels(data, uc_type):
        raise ValueError("Pricing result does not satisfy the label contract")
    _atomic_save(data, path)
    return "saved"
