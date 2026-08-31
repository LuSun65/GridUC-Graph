from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


def _validate_id(name: str, value: str) -> None:
    """Require IDs to be one safe, non-special path component."""
    if not value or value in {".", ".."}:
        raise ValueError(f"{name} must be a non-empty path component")
    if os.path.basename(value) != value or "/" in value or "\\" in value:
        raise ValueError(f"{name} must not contain path separators: {value!r}")


@dataclass(frozen=True)
class ExperimentPaths:
    """Input and per-run output locations for one experiment run.

    ``input_root`` contains case definitions and reusable raw samples. It is
    read-only for derived stages; the optional sampling stage populates it.
    Existing processed tensors may be read from ``input_root`` or an explicit
    path. Every newly generated file is placed below the isolated ``run_root``.
    """

    input_root: Path
    output_root: Path
    run_id: str
    checkpoint_run_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_root", Path(self.input_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        _validate_id("run_id", self.run_id)
        if self.checkpoint_run_id is not None:
            _validate_id("checkpoint_run_id", self.checkpoint_run_id)

    @property
    def run_root(self) -> Path:
        return self.output_root / "runs" / self.run_id

    @property
    def checkpoint_run_root(self) -> Path:
        run_id = self.checkpoint_run_id or self.run_id
        return self.output_root / "runs" / run_id

    def run_case_root(self, casename: str) -> Path:
        _validate_id("casename", casename)
        return self.run_root / casename

    def case_dir(self) -> Path:
        return self.input_root / "case"

    def sample_dir(self, casename: str, uc_type: str) -> Path:
        _validate_id("casename", casename)
        _validate_id("uc_type", uc_type)
        return self.input_root / casename / "samples" / uc_type

    def processed_path(self, casename: str, uc_type: str) -> Path:
        _validate_id("casename", casename)
        _validate_id("uc_type", uc_type)
        return self.run_case_root(casename) / "processed" / f"{uc_type}.pt"

    def default_input_processed_path(self, casename: str, uc_type: str) -> Path:
        _validate_id("casename", casename)
        _validate_id("uc_type", uc_type)
        return self.input_root / casename / "processed" / f"{uc_type}.pt"

    def checkpoint_path(self, casename: str, uc_type: str, model_type: str) -> Path:
        _validate_id("casename", casename)
        _validate_id("uc_type", uc_type)
        _validate_id("model_type", model_type)
        return self.run_case_root(casename) / "checkpoints" / uc_type / f"{model_type}.pt"

    def test_checkpoint_path(self, casename: str, uc_type: str,
                             model_type: str) -> Path:
        """Checkpoint selected for testing; defaults to the current run."""
        _validate_id("casename", casename)
        _validate_id("uc_type", uc_type)
        _validate_id("model_type", model_type)
        return (self.checkpoint_run_root / casename /
                "checkpoints" / uc_type / f"{model_type}.pt")

    def result_path(self, casename: str, uc_type: str, model_type: str) -> Path:
        _validate_id("casename", casename)
        _validate_id("uc_type", uc_type)
        _validate_id("model_type", model_type)
        return self.run_case_root(casename) / "results" / uc_type / f"{model_type}.pkl"

    def benchmark_path(self, casename: str, uc_type: str, solver_mode: str,
                       seed_base: int, n_test: int) -> Path:
        _validate_id("casename", casename)
        _validate_id("uc_type", uc_type)
        _validate_id("solver_mode", solver_mode)
        if seed_base < 0:
            raise ValueError("seed_base must be non-negative")
        if n_test < 1:
            raise ValueError("n_test must be positive")
        filename = f"{uc_type}_{solver_mode}_seed{seed_base}_n{n_test}.pkl"
        return self.run_case_root(casename) / "benchmarks" / filename

    def log_path(self, casename: str, *parts: str) -> Path:
        _validate_id("casename", casename)
        for part in parts:
            _validate_id("log path component", part)
        return self.run_case_root(casename) / "logs" / Path(*parts)
