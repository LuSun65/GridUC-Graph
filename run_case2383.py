from pathlib import Path

from lib.runner import run_all

PROJECT_ROOT = Path(__file__).resolve().parent
V1_INPUT_ROOT = PROJECT_ROOT.parent / "GridUC-Graph" / "data"

# sample_solver / test_solver: "none" | "lazy" | "dense"

if __name__ == "__main__":
    run_all(
        case="case2383",
        stages=["summary"],
        n_samples=1000,
        n_test=20,
        epochs=50,
        batch_size=1,
        device="cuda",
        train_devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        sample_solver="dense",
        test_solver="dense",
        workers=None,
        uc_types=("tcuc", "topo", "scuc"),
        models=("stgcn-v2.2", "stgcn_v2", "stgcn_v1", "mlp"),
        input_root=V1_INPUT_ROOT,
        output_root=PROJECT_ROOT,
        run_id="test",
        checkpoint_run_id='checkpoints',
    )
