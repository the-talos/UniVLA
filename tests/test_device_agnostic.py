"""Device selection for the inference path (XPU > CUDA > CPU).

Everything except the last two tests runs without an accelerator, by faking which backends exist.
The parity tests need an XPU and use tiny random-weight modules, so no checkpoint is needed.
"""

import importlib.util
import unittest
from unittest import mock

import torch

try:  # keep the torchvision import ahead of tensorflow (imported by the eval utils)
    import torchvision  # noqa: F401
except ImportError:
    pass

from prismatic.util import device as dev


def _fake_availability(*, xpu: bool, cuda: bool):
    def fake(device_type: str) -> bool:
        return {"xpu": xpu, "cuda": cuda}.get(device_type, device_type == "cpu")

    return mock.patch.object(dev, "is_available", side_effect=fake)


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _fake_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.default_device(), torch.device("xpu", 0))

    def test_cuda_default_is_unchanged(self):
        with _fake_availability(xpu=False, cuda=True):
            self.assertEqual(dev.default_device(), torch.device("cuda:0"))

    def test_cpu_without_accelerator(self):
        with _fake_availability(xpu=False, cuda=False):
            self.assertEqual(dev.default_device(), torch.device("cpu"))
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_explicit_device_wins_over_detection(self):
        with _fake_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device(torch.device("xpu:1")), torch.device("xpu:1"))


class AttentionBackendTest(unittest.TestCase):
    def test_flash_attention_kept_on_cuda(self):
        with _fake_availability(xpu=False, cuda=True):
            self.assertEqual(dev.attn_implementation(), "flash_attention_2")

    def test_sdpa_off_cuda(self):
        for xpu in (True, False):
            with _fake_availability(xpu=xpu, cuda=False):
                self.assertEqual(dev.attn_implementation(), "sdpa")


class SeedingTest(unittest.TestCase):
    def test_seed_goes_to_detected_backend_only(self):
        with _fake_availability(xpu=True, cuda=False):
            with mock.patch.object(torch.xpu, "manual_seed_all") as xpu_seed, mock.patch.object(
                torch.cuda, "manual_seed_all"
            ) as cuda_seed:
                dev.manual_seed_all(3)
                xpu_seed.assert_called_once_with(3)
                cuda_seed.assert_not_called()
        with _fake_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "manual_seed_all") as cuda_seed:
                dev.manual_seed_all(3)
                cuda_seed.assert_called_once_with(3)

    def test_seed_is_a_noop_on_cpu(self):
        with _fake_availability(xpu=False, cuda=False):
            with mock.patch.object(torch.cuda, "manual_seed_all") as cuda_seed:
                dev.manual_seed_all(3)
                cuda_seed.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "tensorflow is needed to import the eval utils")
class EvalUtilsTest(unittest.TestCase):
    def test_module_devices_follow_detection(self):
        from experiments.robot import openvla_utils, robot_utils

        self.assertEqual(openvla_utils.DEVICE, dev.default_device())
        self.assertEqual(robot_utils.DEVICE, dev.default_device())

    def test_set_seed_everywhere_seeds_through_device_module(self):
        from experiments.robot import robot_utils

        with mock.patch.object(robot_utils, "manual_seed_all") as seed_all:
            robot_utils.set_seed_everywhere(5)
            seed_all.assert_called_once_with(5)

    def test_get_vla_loads_with_resolved_attention_and_device(self):
        from experiments.robot import openvla_utils

        cfg = mock.Mock(pretrained_checkpoint="ckpt", load_in_8bit=False, load_in_4bit=False, task_suite_name="libero_spatial")
        model = mock.MagicMock()
        model.to.return_value = model
        with mock.patch.object(openvla_utils.AutoModelForVision2Seq, "from_pretrained", return_value=model) as load:
            with mock.patch.object(openvla_utils, "DEVICE", torch.device("xpu", 0)):
                with mock.patch.object(openvla_utils, "attn_implementation", return_value="sdpa"):
                    openvla_utils.get_vla(cfg)
        self.assertEqual(load.call_args.kwargs["attn_implementation"], "sdpa")
        model.to.assert_called_once_with(torch.device("xpu", 0))


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an Intel XPU")
class XpuParityTest(unittest.TestCase):
    @staticmethod
    def _randomise(module):
        for param in module.parameters():  # no zero-initialised layer may hide a mismatch
            torch.nn.init.normal_(param, std=0.1)

    def test_action_decoder_pooling_cpu_matches_xpu(self):
        from prismatic.models.policy.transformer_utils import MAPBlock

        torch.manual_seed(0)
        block = MAPBlock(n_latents=1, vis_dim=64, embed_dim=32, n_heads=4).eval()
        self._randomise(block)
        tokens = torch.randn(1, 6, 64)

        def run(device):
            with torch.no_grad():
                return block.to(device, torch.bfloat16)(tokens.to(device, torch.bfloat16)).float().cpu()

        cpu_out, xpu_out = run("cpu"), run("xpu")
        self.assertGreater(cpu_out.std().item(), 0)
        torch.testing.assert_close(xpu_out, cpu_out, atol=5e-2, rtol=5e-2)

    def test_tiny_llama_sdpa_cpu_matches_xpu(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4
        )
        torch.manual_seed(0)
        model = LlamaForCausalLM._from_config(config, attn_implementation="sdpa", torch_dtype=torch.bfloat16).eval()
        self._randomise(model)
        ids = torch.randint(0, 128, (1, 10))

        def run(device):
            with torch.no_grad():
                return model.to(device)(input_ids=ids.to(device)).logits.float().cpu()

        cpu_out, xpu_out = run("cpu"), run("xpu")
        self.assertGreater(cpu_out.std().item(), 0)
        torch.testing.assert_close(xpu_out, cpu_out, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    unittest.main()
