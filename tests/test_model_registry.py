import tempfile
import unittest
from pathlib import Path

import torch

from lib.experiment import ExperimentPaths
from lib.model_registry import (
    MODEL_REGISTRY,
    build_model,
    canonical_model_type,
    get_model_spec,
)
from lib.stgcn import STGCN, STGCNInput
from lib.stgcn_v2 import STGCN_V2
from lib.trainer import _set_model_seed, load_model, save_model


def _input() -> STGCNInput:
    return STGCNInput(
        node_feat_s=torch.randn(2, 2, 3),
        node_feat_d=torch.randn(2, 2, 3, 2),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.randn(2, 4, 2),
        gen_bus=torch.tensor([0, 2]),
        edge_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        uc_target=torch.randint(0, 2, (2, 2, 2)).float(),
    )


class ModelRegistryTest(unittest.TestCase):
    def test_canonical_models_and_legacy_aliases_are_registered(self):
        self.assertEqual(
            set(MODEL_REGISTRY), {"mlp", "stgcn_v1", "stgcn_v2"}
        )
        self.assertEqual(canonical_model_type("stgcn"), "stgcn_v1")
        self.assertEqual(canonical_model_type("stgcn-v2"), "stgcn_v2")
        self.assertIs(get_model_spec("stgcn").model_class, STGCN)
        self.assertIs(get_model_spec("stgcn-v2").model_class, STGCN_V2)

    def test_every_model_obeys_the_common_output_contract(self):
        data = _input()
        overrides = {
            "mlp": {"hidden_dim": 8, "n_hidden": 1},
            "stgcn_v1": {"f_hidden": 8},
            "stgcn_v2": {"f_hidden": 8, "attention_heads": 2},
        }
        for model_type in MODEL_REGISTRY:
            with self.subTest(model_type=model_type):
                model = build_model(model_type, data, overrides[model_type])
                model.eval()
                output = model(data)
                self.assertEqual(output.shape, (2, 2, 2))
                self.assertTrue(torch.isfinite(output).all())
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output, data.uc_target
                )
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))

    def test_unknown_config_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "has no field"):
            build_model("stgcn_v2", _input(), {"unknown_option": 1})

    def test_seed_reset_makes_initialization_reproducible(self):
        data = _input()
        _set_model_seed(17)
        first = build_model("stgcn_v2", data, {
            "f_hidden": 8,
            "attention_heads": 2,
        })
        _set_model_seed(17)
        second = build_model("stgcn_v2", data, {
            "f_hidden": 8,
            "attention_heads": 2,
        })
        for first_value, second_value in zip(
            first.state_dict().values(), second.state_dict().values()
        ):
            self.assertTrue(torch.equal(first_value, second_value))


class ModelCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.paths = ExperimentPaths(root / "input", root, "comparison")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_v2_checkpoint_round_trip_uses_versioned_schema(self):
        model = build_model("stgcn_v2", _input(), {
            "f_hidden": 8,
            "attention_heads": 2,
        })
        save_model(model, "stgcn_v2", "case5", "tcuc", self.paths)

        checkpoint_path = self.paths.checkpoint_path(
            "case5", "tcuc", "stgcn_v2"
        )
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location="cpu"
        )
        self.assertEqual(checkpoint["checkpoint_schema"], 2)
        self.assertEqual(checkpoint["model_type"], "stgcn_v2")
        self.assertEqual(checkpoint["architecture_version"], "2.0")
        self.assertIsInstance(checkpoint["config"], dict)

        loaded = load_model(
            "stgcn_v2", "case5", "tcuc", self.paths, device="cpu"
        )
        for expected, actual in zip(
            model.state_dict().values(), loaded.state_dict().values()
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_legacy_checkpoint_and_alias_still_load(self):
        model = build_model("stgcn", _input(), {"f_hidden": 8})
        path = self.paths.checkpoint_path("case5", "tcuc", "stgcn")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": model.config,
            "state_dict": model.state_dict(),
        }, path)

        loaded = load_model("stgcn", "case5", "tcuc", self.paths)
        self.assertIsInstance(loaded, STGCN)

    def test_checkpoint_run_id_selects_checkpoint(self):
        model = build_model("stgcn_v2", _input(), {
            "f_hidden": 8,
            "attention_heads": 2,
        })
        save_model(model, "stgcn_v2", "case5", "tcuc", self.paths)
        selected_paths = ExperimentPaths(
            self.paths.input_root,
            self.paths.output_root,
            "test-results",
            checkpoint_run_id="comparison",
        )

        loaded = load_model(
            "stgcn_v2", "case5", "tcuc", selected_paths,
        )
        for expected, actual in zip(
            model.state_dict().values(), loaded.state_dict().values()
        ):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
