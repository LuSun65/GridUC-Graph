from pathlib import Path

from lib.runner import run_all

PROJECT_ROOT = Path(__file__).resolve().parent
V1_INPUT_ROOT = PROJECT_ROOT.parent / "GridUC-Graph" / "data"

# sample_solver / test_solver: "none" | "lazy" | "dense"

if __name__ == "__main__":
    run_all(
        case="case6515",
        stages=["train", "test", "summary"],
        n_samples=800,
        n_test=1,
        epochs=30,
        batch_size=4,
        device="cuda",
        sample_solver="dense",
        test_solver="dense",
        workers=None,
        input_root=V1_INPUT_ROOT,
        output_root=PROJECT_ROOT,
        run_id="r004",
    )
