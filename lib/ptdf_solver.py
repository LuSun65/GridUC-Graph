from collections import OrderedDict
import hashlib

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from lib.case_loader import UCData


# Per-process, byte-bounded cache. Limits are deliberately not cached: they may
# change between samples even when the electrical network is identical.
_PTDF_CACHE = OrderedDict()
_PTDF_CACHE_BYTES = 128 * 1024**2


def _network_ptdf(data, reactance, ref_bus):
    line_from = np.asarray(data.line_from, dtype=np.int64)
    line_to = np.asarray(data.line_to, dtype=np.int64)
    reactance = np.asarray(reactance, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray([data.num_bus, data.num_line, ref_bus], dtype=np.int64).tobytes())
    for value in (line_from, line_to, reactance):
        digest.update(value.tobytes())
    key = digest.digest()
    if key in _PTDF_CACHE:
        _PTDF_CACHE.move_to_end(key)
        return _PTDF_CACHE[key]
    lines = np.arange(data.num_line)
    incidence = sparse.csc_matrix(
        (np.r_[np.ones(data.num_line), -np.ones(data.num_line)],
         (np.r_[line_from, line_to], np.r_[lines, lines])),
        shape=(data.num_bus, data.num_line),
    )
    keep = np.arange(data.num_bus) != ref_bus
    reduced = incidence[keep, :]
    weighted = sparse.diags(1.0 / reactance) @ reduced.T
    try:
        factor = splu((reduced @ weighted).tocsc())
    except RuntimeError as error:
        raise np.linalg.LinAlgError("PTDF network factorization failed") from error
    result = np.zeros((data.num_line, data.num_bus))
    # Solve B.T X = (diag(b) A.T).T in blocks; never form B inverse.
    buses = np.flatnonzero(keep)
    for start in range(0, data.num_line, 128):
        end = min(start + 128, data.num_line)
        result[start:end, buses] = factor.solve(
            weighted[start:end].toarray().T, trans="T"
        ).T
    result[np.abs(result) < 1e-6] = 0
    result.setflags(write=False)
    if result.nbytes <= _PTDF_CACHE_BYTES:
        while _PTDF_CACHE and sum(v.nbytes for v in _PTDF_CACHE.values()) + result.nbytes > _PTDF_CACHE_BYTES:
            _PTDF_CACHE.popitem(last=False)
        _PTDF_CACHE[key] = result
    return result


def compute_ptdf(data: UCData, ref_bus: int = 0) -> np.ndarray:
    """Return a cached read-only PTDF, using sparse network factorization."""
    return _network_ptdf(data, data.line_reactance, ref_bus)


def compute_post_ptdf(
    data: UCData,
    outage_line: int,
    factor: float = 1.05,
    ref_bus: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute post-contingency PTDF using a soft-outage approach.
    Assumes double-circuit lines: doubles reactance and halves
    capacity for the outaged line, then recomputes PTDF.
    Does not modify data in-place.

    Inputs:
        data:          UCData with network topology.
        outage_line:   int, line index to apply soft outage.
        factor:        multiplier for post_fmax (default 1.05).
        ref_bus:       reference bus index (default 0).
    Outputs:
        (post_ptdf, post_fmax) - ([num_line, num_bus], [num_line]).
    """
    # copy to avoid mutating data
    line_reactance = np.array(data.line_reactance, dtype=float)
    line_fmax = np.array(data.line_fmax, dtype=float)

    line_reactance[outage_line] *= 2.0
    line_fmax[outage_line] *= 0.5

    post_ptdf = _network_ptdf(data, line_reactance, ref_bus)

    post_fmax = factor * line_fmax
    return post_ptdf, post_fmax
