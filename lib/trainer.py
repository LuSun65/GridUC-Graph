import os
import random
import time
from dataclasses import asdict, is_dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from lib.data_loader import load_processed_data, make_dataloader
from lib.experiment import ExperimentPaths
from lib.models.model_registry import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    build_model,
    canonical_model_type,
    get_model_spec,
    model_from_config,
)


# Compatibility view for code that imported the old class-only registry.
MODEL_CLASSES = {
    name: get_model_spec(name).model_class
    for name in (*MODEL_REGISTRY.keys(), *MODEL_ALIASES.keys())
}


def model_path(model_type: str, casename: str, uc_type: str,
               paths: ExperimentPaths):
    return paths.checkpoint_path(casename, uc_type, model_type)


def save_model(model, model_type: str, casename: str, uc_type: str,
               paths: ExperimentPaths):
    path = model_path(model_type, casename, uc_type, paths)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canonical = canonical_model_type(model_type)
    spec = get_model_spec(canonical)
    config = asdict(model.config) if is_dataclass(model.config) else model.config
    torch.save({
        'checkpoint_schema': 2,
        'model_type': canonical,
        'architecture_version': spec.architecture_version,
        'config': config,
        'state_dict': model.state_dict(),
    }, path)
    print(f"[SAVED] {model_type} model -> {path}")


def load_model(model_type: str, casename: str, uc_type: str,
               paths: ExperimentPaths, device: str = "cpu"):
    path = paths.test_checkpoint_path(casename, uc_type, model_type)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    requested_type = canonical_model_type(model_type)
    checkpoint_type = checkpoint.get('model_type')
    if checkpoint_type is not None:
        checkpoint_type = canonical_model_type(checkpoint_type)
        if checkpoint_type != requested_type:
            raise ValueError(
                f"Checkpoint model_type is {checkpoint_type!r}, "
                f"not requested type {requested_type!r}"
            )
    schema = checkpoint.get('checkpoint_schema', 1)
    if schema not in (1, 2):
        raise ValueError(f"Unsupported checkpoint schema: {schema}")
    checkpoint_version = checkpoint.get('architecture_version')
    expected_version = get_model_spec(requested_type).architecture_version
    if checkpoint_version is not None and checkpoint_version != expected_version:
        raise ValueError(
            f"Checkpoint architecture_version is {checkpoint_version!r}, "
            f"not expected version {expected_version!r}"
        )
    model = model_from_config(requested_type, checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    print(f"[LOADED] {model_type} model <- {path}  device={device}")
    return model


def _run_epoch(model, loader, criterion, optimizer=None, device="cpu") -> float:
    total_loss = 0.0
    for batch in loader:
        # Datasets stay on CPU and move one batch at a time: the big cases hold
        # gigabytes of processed tensors that would otherwise pin the whole set
        # in GPU memory alongside the model.
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.uc_target)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.node_feat_s.shape[0]
    return total_loss / len(loader.dataset)


def _run_prenorm_warmup(model, loader, device="cpu") -> None:
    """Collect full-loader PreNorm statistics without training the model."""
    model.eval()
    model.prenorm_update_on()
    with torch.no_grad():
        for batch in loader:
            model(batch.to(device))
    model.prenorm_update_off()


def _set_model_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model_type: str,
    casename: str,
    uc_type: str,
    paths: ExperimentPaths,
    processed_path,
    expected_sample_count: int = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    print_interval: int = 1,
    save_best: bool = True,
    n_hidden: int = 3,
    hidden_dim: int = 512,
    random_seed: int = 42,
    config_overrides: dict = None,
):
    dev = torch.device(device)

    # Reset before model construction and before the shuffled loader is iterated,
    # so model order in a comparison run does not affect initialization/shuffling.
    _set_model_seed(random_seed)

    train_data, test_data = load_processed_data(
        processed_path, expected_sample_count=expected_sample_count
    )
    train_loader = make_dataloader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = make_dataloader(test_data, batch_size=batch_size, shuffle=False)

    overrides = dict(config_overrides or {})
    if canonical_model_type(model_type) == "mlp":
        overrides.setdefault("n_hidden", n_hidden)
        overrides.setdefault("hidden_dim", hidden_dim)
    model = build_model(model_type, train_data, overrides).to(dev)
    spec = get_model_spec(model_type)

    if spec.prenorm_warmup:
        print("[PRENORM] collecting normalisation statistics ...")
        _run_prenorm_warmup(model, train_loader, dev)
        print("[PRENORM] statistics frozen")

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n[TRAIN] {model_type} | {casename} / {uc_type}  epochs={epochs}  batch={batch_size}  lr={lr}")

    best_test_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, dev)

        model.eval()
        with torch.no_grad():
            test_loss = _run_epoch(model, test_loader, criterion, device=dev)

        elapsed = time.time() - t0

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % print_interval == 0:
            print(f"  epoch {epoch:4d}/{epochs}"
                  f"  train={train_loss:.4f}"
                  f"  test={test_loss:.4f}"
                  f"  time={elapsed:.1f}s"
                  f"  {'* saved best' if test_loss == best_test_loss else ''}")

    model.load_state_dict({k: v.to(dev) for k, v in best_state_dict.items()})
    model.eval()
    # Written once, at the end: on the big cases a checkpoint is over a gigabyte
    # and saving on every improvement dominates the epoch time.
    if save_best:
        save_model(model, model_type, casename, uc_type, paths)
    print(f"\n[DONE] {model_type} | {casename} / {uc_type}  best_test_loss={best_test_loss:.4f}")
    return model
