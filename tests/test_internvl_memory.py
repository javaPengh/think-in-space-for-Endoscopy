"""CPU regression tests; run with the evaluation environment's torch/accelerate."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

# Import only the helper; the model registry has unrelated optional dependencies.
HELPER = Path(__file__).resolve().parents[1] / "lmms_eval/models/model_utils/internvl_memory.py"
spec = importlib.util.spec_from_file_location("internvl_memory", HELPER)
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


class MemoryBudgetTests(unittest.TestCase):
    def test_reserves_on_all_visible_gpus_and_keeps_cpu_budget(self):
        gib = 1024**3
        with patch.object(memory, "get_max_memory", return_value={0: 23 * gib, 1: 21 * gib, 2: 23 * gib, 3: 22 * gib, "cpu": 100 * gib}):
            self.assertEqual(memory.inference_max_memory(4), {0: 19 * gib, 1: 17 * gib, 2: 19 * gib, 3: 18 * gib, "cpu": 100 * gib})

    def test_rejects_invalid_reserve(self):
        for reserve in [0, -1, float("nan"), float("inf")]:
            with self.subTest(reserve=reserve), self.assertRaises(ValueError):
                memory.inference_max_memory(reserve)

    def test_fails_before_loading_on_busy_gpu(self):
        with patch.object(memory, "get_max_memory", return_value={0: 3 * 1024**3}), self.assertRaisesRegex(ValueError, "Visible GPU 0"):
            memory.inference_max_memory(4)


class VisionChunkTests(unittest.TestCase):
    def test_keeps_features_order_and_values_with_bounded_batches(self):
        # Include a partial last chunk and multiple patches per conceptual frame.
        for count, batch_size in [(32, 1), (35, 4), (1, 4)]:
            with self.subTest(count=count, batch_size=batch_size):
                encoder = torch.nn.Linear(3, 5).eval()
                pixels = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
                expected = encoder(pixels).detach()
                calls = []

                class Model:
                    def extract_feature(self, chunk):
                        calls.append((len(chunk), torch.is_grad_enabled()))
                        return encoder(chunk)

                model = Model()
                memory.enable_chunked_vision(model, batch_size)
                actual = model.extract_feature(pixel_values=pixels)
                torch.testing.assert_close(actual, expected)
                self.assertFalse(actual.requires_grad)
                self.assertEqual(sum(size for size, _ in calls), count)
                self.assertTrue(all(size <= batch_size and not grad for size, grad in calls))

    def test_rejects_nonpositive_batch_size(self):
        for batch_size in [0, -1]:
            with self.assertRaises(ValueError):
                memory.enable_chunked_vision(object(), batch_size)


if __name__ == "__main__":
    unittest.main()
