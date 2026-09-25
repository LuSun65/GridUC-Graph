from pathlib import Path

from lib.runner import run_all

# Project directory containing this script.
PROJECT_ROOT = Path(__file__).resolve().parent
# Local data: GridUC-Graph-v2/data.
LOCAL_INPUT_ROOT = PROJECT_ROOT / "data"
# Legacy data: GridUC-Graph/data.
V1_INPUT_ROOT = PROJECT_ROOT.parent / "GridUC-Graph" / "data"

# sample_solver / test_solver: "none" | "lazy" | "dense"

if __name__ == "__main__":
    run_all(
        # Shared.
        case="case6515",
        stages=["train", "test", "summary"],
        uc_types=("tcuc", "topo", "scuc"),
        input_root=V1_INPUT_ROOT,
        output_root=PROJECT_ROOT,

        # Train / test / summary.
        models=("stgcn_attn_0", "stgcn", "mlp"),

        # Train / test.
        device="cuda",

        # Sample.
        n_samples=800,
        sample_solver="dense",
        workers=None,

        # Train.
        epochs=30,
        batch_size=4,
        train_devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),

        # Test.
        n_test=20,
        test_solver="dense",
    )
