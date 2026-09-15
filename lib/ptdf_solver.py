import numpy as np
from lib.case_loader import UCData


def compute_ptdf(data: UCData, ref_bus: int = 0) -> np.ndarray:
    """
    Compute the PTDF matrix from network topology in UCData.

    Inputs:
        data:     UCData with line_from, line_to, line_reactance, num_bus, num_line.
        ref_bus:  reference bus index (default 0).
    Outputs:
        np.ndarray - [num_line, num_bus] PTDF matrix.
    """
    line_from = np.asarray(data.line_from, int)
    line_to = np.asarray(data.line_to, int)
    line_reactance = np.asarray(data.line_reactance, float)
    num_bus = data.num_bus
    num_line = data.num_line

    bl = np.diag(1.0 / line_reactance)
    a = np.zeros((num_bus, num_line))
    a[line_from, range(num_line)] = 1
    a[line_to, range(num_line)] = -1

    idx = np.arange(num_bus) != ref_bus
    bn_red = (a @ bl @ a.T)[np.ix_(idx, idx)]
    try:
        inv_bn_red = np.linalg.inv(bn_red)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError("Base PTDF matrix inversion failed") from error

    inv_bn_full = np.zeros((num_bus, num_bus))
    inv_bn_full[np.ix_(idx, idx)] = inv_bn_red
    ptdf = bl @ a.T @ inv_bn_full
    ptdf[np.abs(ptdf) < 1e-6] = 0
    return ptdf


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

    line_from = np.asarray(data.line_from, int)
    line_to = np.asarray(data.line_to, int)
    num_bus = data.num_bus
    num_line = data.num_line

    bl = np.diag(1.0 / line_reactance)
    a = np.zeros((num_bus, num_line))
    a[line_from, range(num_line)] = 1
    a[line_to, range(num_line)] = -1

    idx = np.arange(num_bus) != ref_bus
    bn_red = (a @ bl @ a.T)[np.ix_(idx, idx)]
    try:
        inv_bn_red = np.linalg.inv(bn_red)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            f"Post-contingency PTDF matrix inversion failed (outage_line={outage_line})"
        ) from error

    inv_bn_full = np.zeros((num_bus, num_bus))
    inv_bn_full[np.ix_(idx, idx)] = inv_bn_red
    post_ptdf = bl @ a.T @ inv_bn_full
    post_ptdf[np.abs(post_ptdf) < 1e-6] = 0

    post_fmax = factor * line_fmax
    return post_ptdf, post_fmax
