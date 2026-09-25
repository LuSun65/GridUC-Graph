from dataclasses import dataclass
from pathlib import Path
import os


def _validate_id(name: str, value: str) -> None:
    """Require IDs to be one safe, non-special path component."""
    if not value or value in {".", ".."}:
        raise ValueError(f"{name} must be a non-empty path component")
    if os.path.basename(value) != value or "/" in value or "\\" in value:
        raise ValueError(f"{name} must not contain path separators: {value!r}")


@dataclass(frozen=True)
class ExperimentPaths:
    """Input and output locations using the original GridUC-Graph layout.

    ``input_root`` contains case definitions, reusable raw samples, and
    processed tensors. Sampling and processing write their outputs there.
    Models, results, and benchmarks use ``output_root/data/<case>``;
    ordinary logs use ``output_root/log/<case>``.
    """

    input_root: Path
    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_root", Path(self.input_root))
        object.__setattr__(self, "output_root", Path(self.output_root))

    def output_case_root(self, casename: str) -> Path:
        _validate_id("casename", casename)
        return self.output_root / "data" / casename

    def case_dir(self) -> Path:
        return self.input_root / "case"

    def sample_dir(self, casename: str, uc_type: str) -> Path:
        _validate_id("casename", casename)
        _validate_id("uc_type", uc_type)
        return self.input_root / casename / "samples" / uc_type

    def processed_path(self, casename: str, uc_type: str) -> Path:
        _validate_id("uc_type", uc_type)
        return self.default_input_processed_path(casename, uc_type)

    def default_input_processed_path(self, casename: str, uc_type: str) -> Path:
        _validate_id("casename", casename)
        _validate_id("uc_type", uc_type)
        return self.input_root / casename / "processed" / f"{uc_type}.pt"

    def checkpoint_path(self, casename: str, uc_type: str, model_type: str) -> Path:
        _validate_id("uc_type", uc_type)
        _validate_id("model_type", model_type)
        return self.output_case_root(casename) / "model" / f"{uc_type}_{model_type}.pt"

    def test_checkpoint_path(self, casename: str, uc_type: str,
                             model_type: str) -> Path:
        """Read the same case/model path used by training."""
        return self.checkpoint_path(casename, uc_type, model_type)

    def result_path(self, casename: str, uc_type: str, model_type: str) -> Path:
        _validate_id("uc_type", uc_type)
        _validate_id("model_type", model_type)
        return self.output_case_root(casename) / "result" / f"{uc_type}_{model_type}.pkl"

    def benchmark_path(self, casename: str, uc_type: str, solver_mode: str,
                       seed_base: int, n_test: int) -> Path:
        _validate_id("uc_type", uc_type)
        _validate_id("solver_mode", solver_mode)
        if seed_base < 0:
            raise ValueError("seed_base must be non-negative")
        if n_test < 1:
            raise ValueError("n_test must be positive")
        filename = f"{uc_type}_{solver_mode}_seed{seed_base}_n{n_test}.pkl"
        return self.output_case_root(casename) / "benchmarks" / filename

    def log_path(self, casename: str, *parts: str) -> Path:
        _validate_id("casename", casename)
        for part in parts:
            _validate_id("log path component", part)
        return self.output_root / "log" / casename / Path(*parts)
