from lib.runner import run_all

# sample_solver / test_solver: "none" | "lazy" | "dense"

if __name__ == "__main__":
    run_all(
        case="case6515",
        stages=["train", "test", "summary"],
        n_samples=800,
        n_test=1,
        epochs=30,
        batch_size=4,
        device="mps",
        sample_solver="dense",
        test_solver="dense",
        workers=None,
    )
