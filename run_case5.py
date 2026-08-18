from pathlib import Path

from lib.runner import run_all

PROJECT_ROOT = Path(__file__).resolve().parent
V1_INPUT_ROOT = PROJECT_ROOT.parent / "GridUC-Graph" / "data"

# sample_solver / test_solver: "none" | "lazy" | "dense"
if __name__ == "__main__":
    run_all(
        case="case5",
        stages=["train", "test", "summary"],
        n_samples=200,
        n_test=5,
        epochs=50,
        batch_size=32,
        device="cpu",
        sample_solver="dense",
        test_solver="dense",
        workers=4,
        input_root=V1_INPUT_ROOT,
        output_root=PROJECT_ROOT,
        run_id="r001",
    )
