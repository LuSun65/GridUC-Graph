from pathlib import Path

from lib.runner import run_all

PROJECT_ROOT = Path(__file__).resolve().parent
V1_INPUT_ROOT = PROJECT_ROOT.parent / "GridUC-Graph" / "data"

# sample_solver / test_solver: "none" | "lazy" | "dense"

if __name__ == "__main__":
    run_all(
        case="case118",
        stages=["train", "test", "summary"],
        n_samples=1000,
        n_test=10,
        epochs=50,
        batch_size=64,
        device="cuda",
        sample_solver="dense",
        test_solver="dense",
        workers=4,
        input_root=V1_INPUT_ROOT,
        output_root=PROJECT_ROOT,
        run_id="v2-test",
    )
