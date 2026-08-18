import unittest

import torch

from lib.data_loader import concat_stgcn_inputs, validate_processed_data
from lib.stgcn import STGCNInput


def _sample(value: float) -> STGCNInput:
    return STGCNInput(
        node_feat_s=torch.tensor([[[value]]]),
        node_feat_d=torch.tensor([[[[value]]]]),
        edge_index=torch.tensor([[0], [0]]),
        edge_attr=torch.tensor([[[value]]]),
        gen_bus=torch.tensor([0]),
        edge_mask=torch.tensor([[1.0]]),
        uc_target=torch.tensor([[[value]]]),
    )


class ProcessedDataValidationTest(unittest.TestCase):
    def setUp(self):
        self.data = {
            "train": concat_stgcn_inputs([_sample(0.0)]),
            "test": concat_stgcn_inputs([_sample(1.0)]),
        }

    def test_compatible_data_is_accepted(self):
        validate_processed_data(self.data, expected_sample_count=2)

    def test_sample_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            validate_processed_data(self.data, expected_sample_count=3)

    def test_inconsistent_tensor_batch_is_rejected(self):
        self.data["test"].edge_attr = torch.zeros(2, 1, 1)
        with self.assertRaisesRegex(ValueError, "inconsistent batch size"):
            validate_processed_data(self.data)


if __name__ == "__main__":
    unittest.main()
