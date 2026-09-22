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
        case="case2383",
        stages=["sample",],
        uc_types=("tcuc", "scuc"),
        input_root=LOCAL_INPUT_ROOT,
        output_root=PROJECT_ROOT,
        run_id="samples",

        # Train / test / summary.
        models=("stgcn-v2.2", "stgcn_v2", "stgcn_v1", "mlp"),

        # Train / test.
        device="cuda",

        # Sample.
        n_samples=1000,
        sample_solver="dense",
        workers=4,

        # Train.
        epochs=50,
        batch_size=1,
        train_devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),

        # Test.
        n_test=20,
        test_solver="dense",
        checkpoint_run_id='checkpoints',
    )
