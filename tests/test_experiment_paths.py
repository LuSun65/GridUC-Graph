import unittest
from pathlib import Path

from lib.experiment import ExperimentPaths


class ExperimentPathsTest(unittest.TestCase):
    def setUp(self):
        self.paths = ExperimentPaths(
            input_root="/shared/v1/data",
            output_root="/work/v2",
            run_id="r001",
        )

    def test_input_paths_stay_under_input_root(self):
        self.assertEqual(self.paths.case_dir(), Path("/shared/v1/data/case"))
        self.assertEqual(
            self.paths.sample_dir("case118", "tcuc"),
            Path("/shared/v1/data/case118/samples/tcuc"),
        )
        self.assertEqual(
            self.paths.default_input_processed_path("case118", "tcuc"),
            Path("/shared/v1/data/case118/processed/tcuc.pt"),
        )

    def test_derived_paths_stay_under_run_root(self):
        root = Path("/work/v2/runs/r001")
        self.assertEqual(self.paths.run_root, root)
        self.assertEqual(
            self.paths.processed_path("case118", "tcuc"),
            root / "case118/processed/tcuc.pt",
        )
        self.assertEqual(
            self.paths.checkpoint_path("case118", "tcuc", "stgcn"),
            root / "case118/checkpoints/tcuc/stgcn.pt",
        )
        self.assertEqual(
            self.paths.result_path("case118", "tcuc", "stgcn"),
            root / "case118/results/tcuc/stgcn.pkl",
        )
        self.assertEqual(
            self.paths.log_path("case118", "train", "train_tcuc_stgcn.log"),
            root / "case118/logs/train/train_tcuc_stgcn.log",
        )

    def test_different_runs_have_different_output_paths(self):
        other_run = ExperimentPaths(
            input_root=self.paths.input_root,
            output_root=self.paths.output_root,
            run_id="r002",
        )
        self.assertNotEqual(
            self.paths.processed_path("case118", "tcuc"),
            other_run.processed_path("case118", "tcuc"),
        )
        self.assertNotEqual(self.paths.run_root, other_run.run_root)

    def test_ids_cannot_escape_output_namespace(self):
        for bad_id in ("", ".", "..", "nested/run", "nested\\run"):
            with self.subTest(bad_id=bad_id), self.assertRaises(ValueError):
                ExperimentPaths("data", ".", bad_id)

    def test_output_components_cannot_escape_run_root(self):
        with self.assertRaises(ValueError):
            self.paths.processed_path("../case118", "tcuc")
        with self.assertRaises(ValueError):
            self.paths.checkpoint_path("case118", "tcuc", "../stgcn")
        with self.assertRaises(ValueError):
            self.paths.log_path("case118", "../legacy", "train.log")


if __name__ == "__main__":
    unittest.main()
