from lib.runner import run_all

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
    )
