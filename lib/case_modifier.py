import copy
import numpy as np
from lib.case_loader import UCData

# Capacity band the maintenance line is drawn from: sort lines by fmax
# descending, drop the largest DROP_TOP (backbone, too disruptive to take out)
# and the smallest DROP_BOTTOM (tail lines, taking them out means nothing).
DROP_TOP = 0.20
DROP_BOTTOM = 0.30


def _clone(input_uc: UCData) -> UCData:
    return copy.deepcopy(input_uc)


def capacity_pool(uc: UCData) -> np.ndarray:
    """Line indices in the middle capacity band, cached per loaded case."""
    pool = uc.__dict__.get("_pool_cache")
    if pool is None:
        fmax = np.asarray(uc.line_fmax, float)
        order = np.argsort(-fmax)
        lo = int(round(len(fmax) * DROP_TOP))
        hi = int(round(len(fmax) * (1.0 - DROP_BOTTOM)))
        pool = np.sort(order[lo:hi])
        uc._pool_cache = pool
    return pool


def _randomize_demand_and_bid(uc: UCData) -> None:
    """Randomly assign loadshapes and compute time-varying demand & bid prices."""
    num_lds = uc.lds_pool.shape[0]
    uc.bid_lds_idx = np.random.randint(0, num_lds, size=uc.num_gen)
    uc.load_lds_idx = np.random.randint(0, num_lds, size=uc.num_bus)

    load_shapes = uc.lds_pool[uc.load_lds_idx]          # [num_bus, num_period]
    uc.demand = uc.nodal_peak[:, None] * load_shapes     # [num_bus, num_period]

    bid_shapes = uc.lds_pool[uc.bid_lds_idx]             # [num_gen, num_period]
    uc.bid_price = uc.bid_price[:, :, None] * bid_shapes[:, None, :]  # [num_gen, num_segments, num_period]


def case_tcuc_rnd(input_uc: UCData) -> UCData:
    """N-0 on the base topology."""
    uc = _clone(input_uc)
    _randomize_demand_and_bid(uc)
    return uc


def case_topo_rnd(input_uc: UCData) -> UCData:
    """N-0 on a topology with one line out for maintenance."""
    uc = _clone(input_uc)
    uc.maintenance_line = int(np.random.choice(capacity_pool(input_uc)))
    _randomize_demand_and_bid(uc)
    return uc


def case_scuc_rnd(input_uc: UCData) -> UCData:
    """N-0 plus N-1 over one contingency list from the case's dync_monitor."""
    uc = _clone(input_uc)
    _randomize_demand_and_bid(uc)

    pick = np.random.randint(len(uc.dync_monitor))
    uc.cc_monitor = list(uc.dync_monitor[pick]["lines"])
    return uc
