import os
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from lib.toolkit import load_pkl
from lib.case_loader import UCData
from lib.models.model_input import STGCNInput
from lib.experiment import ExperimentPaths


def to_stgcn_input(uc: UCData, uc_type: str) -> STGCNInput:
    num_line = uc.num_line

    edge_index = torch.tensor(
        [[*uc.line_from, *uc.line_to], [*uc.line_to, *uc.line_from]], dtype=torch.long
    )

    edge_attr = torch.tensor(
        [[r, f] for r, f in zip(uc.line_reactance, uc.line_fmax)] * 2,
        dtype=torch.float32,
    ).unsqueeze(0)

    if uc_type == "scuc":
        contingency = torch.zeros(1, 2 * num_line, 1, dtype=torch.float32)
        if getattr(uc, "cc_monitor", None):
            for line_idx in uc.cc_monitor:
                contingency[0, line_idx, 0] = 1.0
                contingency[0, line_idx + num_line, 0] = 1.0
        edge_attr = torch.cat([edge_attr, contingency], dim=-1)

    edge_mask = torch.ones(1, 2 * num_line, dtype=torch.float32)
    if uc.maintenance_line is not None:
        ml = uc.maintenance_line
        ml_list = [int(ml)] if not hasattr(ml, '__iter__') else [int(i) for i in ml]
        for line_idx in ml_list:
            edge_mask[0, line_idx] = 0.0
            edge_mask[0, line_idx + num_line] = 0.0

    gen_feat = np.stack([
        uc.gen_pmax, uc.gen_pmin, uc.gen_ramp_up, uc.gen_ramp_down,
        uc.gen_min_up.astype(float), uc.gen_min_down.astype(float),
        uc.gen_cost, uc.gen_startup_cost,
    ], axis=1)
    node_feat_s = torch.tensor(gen_feat, dtype=torch.float32).unsqueeze(0)

    demand_t = torch.tensor(uc.demand, dtype=torch.float32).T

    N = uc.demand.shape[0]
    bid_price = torch.tensor(uc.bid_price, dtype=torch.float32)
    G, Nseg, T = bid_price.shape
    gen_bus_idx = torch.tensor(uc.gen_bus, dtype=torch.long)
    bp = bid_price.permute(2, 0, 1)
    bus_bid = torch.zeros(T, N, Nseg, dtype=torch.float32)
    idx = gen_bus_idx[None, :, None].expand(T, G, Nseg)
    bus_bid.scatter_add_(1, idx, bp)

    node_feat_d = torch.cat([demand_t.unsqueeze(-1), bus_bid], dim=-1).unsqueeze(0)

    gen_bus = torch.tensor(uc.gen_bus, dtype=torch.long)
    uc_target = torch.tensor(uc.uc_sol, dtype=torch.float32).unsqueeze(0)

    return STGCNInput(
        node_feat_s=node_feat_s,
        node_feat_d=node_feat_d,
        edge_index=edge_index,
        edge_attr=edge_attr,
        gen_bus=gen_bus,
        edge_mask=edge_mask,
        uc_target=uc_target,
        lmp_target=torch.tensor(uc.lmp_target, dtype=torch.float32).unsqueeze(0)
                   if uc.lmp_target is not None else None,
    )


def concat_stgcn_inputs(inputs: List[STGCNInput]) -> STGCNInput:
    # None means this label is absent for the entire collection. Mixed
    # availability reaches torch.cat and raises instead of dropping labels.
    return STGCNInput(
        node_feat_s=torch.cat([d.node_feat_s for d in inputs], dim=0),
        node_feat_d=torch.cat([d.node_feat_d for d in inputs], dim=0),
        edge_attr=torch.cat([d.edge_attr for d in inputs], dim=0),
        edge_mask=torch.cat([d.edge_mask for d in inputs], dim=0),
        edge_index=inputs[0].edge_index,
        gen_bus=inputs[0].gen_bus,
        uc_target=torch.cat([d.uc_target for d in inputs], dim=0)
                  if inputs[0].uc_target is not None else None,
        lmp_target=torch.cat([d.lmp_target for d in inputs], dim=0)
                   if any(d.lmp_target is not None for d in inputs) else None,
    )


def list_sample_files(casename: str, uc_type: str,
                      paths: ExperimentPaths) -> Tuple[str, List[str]]:
    from lib.sampler import sample_dir
    data_dir = sample_dir(casename, uc_type, paths)
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pkl")],
        key=lambda f: int(f.replace(".pkl", ""))
    )
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    return data_dir, files


def _process_split(data_dir: str, files: List[str], uc_type: str,
                   chunk_size: int, label: str, require_lmp: bool = False) -> STGCNInput:
    n_total = len(files)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    print(f"[process] {label}: {n_total} files -> {n_chunks} chunks of up to {chunk_size}")

    chunks = []
    for ci in range(n_chunks):
        batch_files = files[ci * chunk_size: (ci + 1) * chunk_size]
        print(f"[process] {label} chunk {ci + 1}/{n_chunks}: {len(batch_files)} files")

        stgcn_list = []
        for fname in batch_files:
            uc = load_pkl(os.path.join(data_dir, fname))
            if require_lmp and uc.lmp_target is None:
                raise ValueError(f"{fname}: missing lmp_target; run label_pricing.py first")
            stgcn_list.append(to_stgcn_input(uc, uc_type))
            del uc

        chunks.append(concat_stgcn_inputs(stgcn_list))
        del stgcn_list
        gc.collect()

    data = concat_stgcn_inputs(chunks)
    del chunks
    gc.collect()
    return data


def process_data(
    casename: str,
    uc_type: str,
    paths: ExperimentPaths,
    chunk_size: int = 200,
    train_ratio: float = 0.8,
    save: bool = True,
    require_lmp: bool = False,
) -> Tuple[STGCNInput, STGCNInput]:
    data_dir, files = list_sample_files(casename, uc_type, paths)

    # Split the file list first and build each side separately: concatenating the
    # full set and slicing afterwards holds the entire dataset twice in memory.
    n_train = min(max(1, int(len(files) * train_ratio)), len(files) - 1)
    train_data = _process_split(data_dir, files[:n_train], uc_type, chunk_size,
                                f"{casename}/{uc_type} train", require_lmp)
    test_data = _process_split(data_dir, files[n_train:], uc_type, chunk_size,
                               f"{casename}/{uc_type} test", require_lmp)

    lmp_stats = None
    if train_data.lmp_target is not None:
        values = train_data.lmp_target
        values = values.double()
        lmp_stats = {"mean": values.mean().item(),
                     "std": values.std(unbiased=False).item()}

    if save:
        save_path = paths.processed_path(casename, uc_type)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        temporary = save_path.with_name(f".{save_path.name}.{os.getpid()}.tmp")
        try:
            torch.save({'train': train_data, 'test': test_data,
                        'lmp_normalization': lmp_stats}, temporary)
            os.replace(temporary, save_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"[process] saved processed data to {save_path}")

    return train_data, test_data


def load_processed_data(
    path,
    expected_sample_count: int = None,
    require_lmp: bool = False,
) -> Tuple[STGCNInput, STGCNInput, dict]:
    path = os.fspath(path)
    data = torch.load(path, weights_only=False)
    validate_processed_data(data, expected_sample_count, require_lmp)
    print(f"[load] processed data <- {path}")
    return data['train'], data['test'], data['lmp_normalization'] if require_lmp else None


def validate_processed_data(data: dict, expected_sample_count: int = None,
                            require_lmp: bool = False) -> None:
    """Perform lightweight compatibility checks for new or legacy tensors."""
    if not isinstance(data, dict) or "train" not in data or "test" not in data:
        raise ValueError("Processed file must contain train and test entries")

    required = ("node_feat_s", "node_feat_d", "edge_index", "edge_attr",
                "gen_bus", "edge_mask", "uc_target")
    counts = []
    for split_name in ("train", "test"):
        split = data[split_name]
        if require_lmp and split.lmp_target is None:
            raise ValueError(f"{split_name}: missing lmp_target; label samples and reprocess")
        missing = [field for field in required if not hasattr(split, field)]
        if missing:
            raise ValueError(
                f"Processed {split_name} split is missing fields: {missing}"
            )
        count = split.node_feat_s.shape[0]
        counts.append(count)
        for field in ("node_feat_d", "edge_attr", "edge_mask", "uc_target"):
            value = getattr(split, field)
            if value is not None and value.shape[0] != count:
                raise ValueError(
                    f"Processed {split_name}.{field} has inconsistent batch size"
                )
    if expected_sample_count is not None and sum(counts) != expected_sample_count:
        raise ValueError(
            f"Processed sample count mismatch: expected {expected_sample_count}, "
            f"got {sum(counts)}"
        )
    for field in ("edge_index", "gen_bus"):
        if not torch.equal(getattr(data["train"], field),
                           getattr(data["test"], field)):
            raise ValueError(
                f"Processed train/test {field} values do not match"
            )


class STGCNDataset(Dataset):
    def __init__(self, data: STGCNInput):
        self.node_feat_s = data.node_feat_s
        self.node_feat_d = data.node_feat_d
        self.edge_attr = data.edge_attr
        self.edge_mask = data.edge_mask
        self.uc_target = data.uc_target
        self.lmp_target = data.lmp_target
        self.edge_index = data.edge_index
        self.gen_bus = data.gen_bus

    def __len__(self) -> int:
        return self.node_feat_s.shape[0]

    def __getitem__(self, idx: int):
        return {
            'node_feat_s': self.node_feat_s[idx],
            'node_feat_d': self.node_feat_d[idx],
            'edge_attr': self.edge_attr[idx],
            'edge_mask': self.edge_mask[idx],
            'uc_target': self.uc_target[idx] if self.uc_target is not None else None,
            'lmp_target': self.lmp_target[idx] if self.lmp_target is not None else None,
        }

    def _collate_fn(self, samples: list) -> STGCNInput:
        return STGCNInput(
            node_feat_s=torch.stack([s['node_feat_s'] for s in samples]),
            node_feat_d=torch.stack([s['node_feat_d'] for s in samples]),
            edge_attr=torch.stack([s['edge_attr'] for s in samples]),
            edge_mask=torch.stack([s['edge_mask'] for s in samples]),
            edge_index=self.edge_index,
            gen_bus=self.gen_bus,
            uc_target=torch.stack([s['uc_target'] for s in samples])
                      if samples[0]['uc_target'] is not None else None,
            lmp_target=torch.stack([s['lmp_target'] for s in samples])
                       if self.lmp_target is not None else None,
        )


def make_dataloader(data: STGCNInput, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = STGCNDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=dataset._collate_fn,
    )
