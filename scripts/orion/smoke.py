"""Validate the installed model stack on an allocated GPU; no model downloads."""

import importlib
from importlib.metadata import version
import json
import os


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Submit scripts/orion/smoke.sbatch to run this check.")
    import torch

    expected = {
        "torch": "2.8.0", "torchvision": "0.23.0", "torchaudio": "2.8.0",
        "vllm": "0.11.0", "transformers": "4.57.3", "ray": "2.48.0",
        "peft": "0.17.1", "tensordict": "0.10.0", "numpy": "2.2.6",
    }
    versions = {name: version(name) for name in expected}
    for name, wanted in expected.items():
        assert versions[name].split("+")[0] == wanted, (name, versions[name], wanted)
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert torch.cuda.is_available(), "CUDA is unavailable in this allocation"
    # Device 0 means the first device exposed by Slurm's CUDA_VISIBLE_DEVICES.
    assert torch.cuda.device_count() == 1, "Expected exactly one allocated GPU"
    assert torch.cuda.is_bf16_supported(), "BF16 is unavailable"
    x = torch.randn(512, 512, device="cuda:0", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    assert torch.isfinite(y).all().item(), "GPU matrix multiplication failed"

    for name in (
        "torchvision", "torchaudio", "vllm", "ray", "peft", "tensordict",
        "qwen_vl_utils", "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "verl", "verl.models.transformers.qwen3_vl",
    ):
        importlib.import_module(name)
    # Check a compiled vLLM extension as well as Python imports.
    importlib.import_module("vllm._C")
    from vllm.model_executor.models.registry import ModelRegistry

    assert "Qwen3VLForConditionalGeneration" in ModelRegistry.get_supported_archs()
    report = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "versions": versions,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_bf16_matmul": "passed",
        "qwen3_vl_imports_and_vllm_registration": "passed",
        "scope": "Model-stack check only; no model generation or Android emulator run",
    }
    print(json.dumps(report, indent=2))
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
