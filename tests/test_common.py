"""
Unit tests for ti64 package core functionality.
Tests deterministic preprocessing, disjoint fold enforcement, metric calculations, and model shape.
Self-contained with standard asserts and unittest.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ti64.common import (
    preprocess,
    assert_disjoint,
    metrics,
    set_seed,
)
from ti64.model import build_model, get_gradcam_layer


class TestCommon(unittest.TestCase):
    def test_preprocess_deterministic(self):
        set_seed(42)
        # Create synthetic BGR tile
        tile = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        out1 = preprocess(tile, size=256)
        out2 = preprocess(tile, size=256)

        self.assertIsInstance(out1, torch.Tensor)
        self.assertEqual(out1.shape, (3, 256, 256))
        self.assertTrue(torch.equal(out1, out2), "preprocess must be strictly deterministic")
        self.assertEqual(out1.dtype, torch.float32)

    def test_assert_disjoint(self):
        # Disjoint sets should succeed
        train_ids = [101, 102, 103, 104]
        val_ids = [105, 106]
        assert_disjoint(train_ids, val_ids)  # Must not raise

        # Overlapping sets must raise AssertionError (Rule A1)
        leaky_val = [104, 107]
        with self.assertRaises(AssertionError):
            assert_disjoint(train_ids, leaky_val)

    def test_metrics_match_sklearn(self):
        set_seed(42)
        y_true = np.array([0.05, 0.12, 0.85, 1.45, 0.33, 0.67, 2.10, 0.95])
        y_pred = np.array([0.04, 0.15, 0.78, 1.30, 0.40, 0.70, 2.00, 1.05])

        res = metrics(y_true, y_pred)

        expected_r2 = float(r2_score(y_true, y_pred))
        expected_mae = float(mean_absolute_error(y_true, y_pred))
        expected_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        expected_bias = float(np.mean(y_pred - y_true))

        self.assertTrue(np.isclose(res["R2"], expected_r2, atol=1e-6), "R2 does not match sklearn")
        self.assertTrue(np.isclose(res["MAE"], expected_mae, atol=1e-6), "MAE does not match sklearn")
        self.assertTrue(np.isclose(res["RMSE"], expected_rmse, atol=1e-6), "RMSE does not match sklearn")
        self.assertTrue(np.isclose(res["Mean_bias"], expected_bias, atol=1e-6), "Mean bias does not match numpy")
        self.assertTrue(-1.0 <= res["Pearson_r"] <= 1.0)

    def test_model_output_shape(self):
        model_single = build_model(n_outputs=1, backbone="tf_efficientnetv2_s", pretrained=False)
        dummy_input = torch.randn(2, 3, 256, 256)
        out_single = model_single(dummy_input)
        self.assertEqual(out_single.shape, (2, 1), f"Expected shape (2, 1), got {out_single.shape}")

        model_multi = build_model(n_outputs=3, backbone="tf_efficientnetv2_s", pretrained=False)
        out_multi = model_multi(dummy_input)
        self.assertEqual(out_multi.shape, (2, 3), f"Expected shape (2, 3), got {out_multi.shape}")

        # Verify Grad-CAM layer extraction
        layer, layer_name = get_gradcam_layer(model_single)
        self.assertIsInstance(layer, torch.nn.Module)
        self.assertIsInstance(layer_name, str)
        self.assertGreater(len(layer_name), 0)


if __name__ == "__main__":
    unittest.main()
