from lib.runner import run_all

# sample_solver / test_solver: "none" | "lazy" | "dense"

if __name__ == "__main__":
    run_all(
        case="case2383",
        stages=["train", "test", "summary"],
        n_samples=1000,
        n_test=1,
        epochs=50,
        batch_size=16,
        device="mps",
        sample_solver="dense",
        test_solver="dense",
        workers=None,
    )
