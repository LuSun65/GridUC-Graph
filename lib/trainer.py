import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam

from lib.stgcn import STGCN
from lib.mlp import MLP
from lib.data_loader import load_processed_data, make_dataloader
from lib.experiment import ExperimentPaths

MODEL_CLASSES = {
    "stgcn": STGCN,
    "mlp": MLP,
}


def model_path(model_type: str, casename: str, uc_type: str,
               paths: ExperimentPaths):
    return paths.checkpoint_path(casename, uc_type, model_type)


def save_model(model, model_type: str, casename: str, uc_type: str,
               paths: ExperimentPaths):
    path = model_path(model_type, casename, uc_type, paths)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'config': model.config, 'state_dict': model.state_dict()}, path)
    print(f"[SAVED] {model_type} model -> {path}")


def load_model(model_type: str, casename: str, uc_type: str,
               paths: ExperimentPaths, device: str = "cpu"):
    path = model_path(model_type, casename, uc_type, paths)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model = MODEL_CLASSES[model_type](checkpoint['config'])
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
):
    dev = torch.device(device)

    train_data, test_data = load_processed_data(
        processed_path, expected_sample_count=expected_sample_count
    )
    train_loader = make_dataloader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = make_dataloader(test_data, batch_size=batch_size, shuffle=False)

    if model_type == "stgcn":
        model = STGCN(STGCN.default_config(train_data)).to(dev)
    elif model_type == "mlp":
        config = MLP.default_config(train_data)
        config.n_hidden = n_hidden
        config.hidden_dim = hidden_dim
        model = MLP(config).to(dev)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}, expected 'stgcn' or 'mlp'")

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    if model_type == "stgcn":
        print("[PRENORM] warming up normalisation statistics ...")
        model.train()
        model.prenorm_update_on()
        warmup_loss = _run_epoch(model, train_loader, criterion, optimizer, dev)
        model.prenorm_update_off()
        print(f"[PRENORM] done  warmup_loss={warmup_loss:.4f}")

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
