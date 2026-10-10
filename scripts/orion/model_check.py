"""Load the selected Qwen3-VL snapshot and run text and synthetic-image inference."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Submit scripts/orion/model-check.sbatch to run this check.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.model_dir / "config.json").read_text())
    if "Qwen3VLForConditionalGeneration" not in config.get("architectures", []):
        raise RuntimeError("Expected a Qwen3-VL dense checkpoint")
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        if not (args.model_dir / shard).is_file():
            raise FileNotFoundError(f"Model download incomplete: {shard}")

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This check expects one GPU exposed by Slurm")
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    text_question = "What is 2 plus 3? Answer with only the digit."
    image_question = "What is the dominant color in this image? Answer with one English color word."
    image = Image.new("RGB", (256, 256), color="blue")
    image.save(args.output_dir / "input-blue.png")
    requests = [
        {"prompt": processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": text_question}]}],
            tokenize=False, add_generation_prompt=True,
        )},
        {
            "prompt": processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": image_question},
                ]}], tokenize=False, add_generation_prompt=True,
            ),
            "multi_modal_data": {"image": image},
        },
    ]
    start = time.monotonic()
    llm = LLM(
        model=str(args.model_dir), tokenizer=str(args.model_dir),
        dtype="bfloat16", tensor_parallel_size=1, seed=0,
        max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=0.5,
        enforce_eager=True, trust_remote_code=False,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={"min_pixels": 64 * 64, "max_pixels": 512 * 512},
    )
    outputs = llm.generate(requests, SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
    answers = [output.outputs[0].text.strip() for output in outputs]
    checks = {
        "text_generation": len(answers) == 2 and bool(re.fullmatch(r"5[.!]?", answers[0])),
        "image_understanding": len(answers) == 2 and bool(re.search(r"\bblue\b", answers[1], re.I)),
    }
    report = {
        "job_id": os.environ["SLURM_JOB_ID"], "hostname": platform.node(),
        "model_id": args.model_id, "revision": args.revision,
        "local_snapshot": str(args.model_dir),
        "versions": {name: version(name) for name in ("torch", "transformers", "vllm")},
        "gpu": torch.cuda.get_device_name(0),
        "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "questions": [text_question, image_question], "answers": answers,
        "checks": checks, "load_and_inference_seconds": round(time.monotonic() - start, 2),
        "scope": "Text and synthetic-image inference only; no Android tasks or training",
    }
    rendered = json.dumps(report, indent=2)
    (args.output_dir / "result.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)
    if not all(checks.values()):
        raise SystemExit("Model generated unexpected answers; review result.json")
    print("MODEL_CHECK_OK", flush=True)


if __name__ == "__main__":
    main()
