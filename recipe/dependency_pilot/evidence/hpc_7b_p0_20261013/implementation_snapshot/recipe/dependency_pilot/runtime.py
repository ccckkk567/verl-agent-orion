import os
import time
from pathlib import Path

from .common import write_json, digest, file_hash


class TransformersWorker:
    """Single-process frozen actor implementing TrajectoryCollector's worker API.

    Uses the same collector preprocessing; generation is sequential to fit a small GPU.
    This is frozen inference, never advertised as a distributed training rollout.
    """
    world_size = 1

    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config, self.observer = config, None
        revision = config.get("model_revision")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"], revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        options = dict(revision=revision, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                       attn_implementation="sdpa", low_cpu_mem_usage=True)
        if config.get("device") == "cpu" or not torch.cuda.is_available():
            options["device_map"] = "cpu"
        else:
            options["device_map"] = "auto"
            if config.get("gpu_memory_gib"):
                options["max_memory"] = {0: f"{config['gpu_memory_gib']}GiB", "cpu": "16GiB"}
        self.model = AutoModelForCausalLM.from_pretrained(config["model_path"], **options).eval()
        config["model_revision"] = getattr(self.model.config, "_commit_hash", None) or revision
        config["checkpoint_id"] = config["model_path"] + "@" + str(config["model_revision"])
        config["chat_template"] = self.tokenizer.chat_template
        config["device_map_resolved"] = {str(k): str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()}
        if config.get('phase') == 'p0_7b_repeat':
            import json
            pinned = json.loads(Path(__file__).with_name('configs').joinpath('model_7b.json').read_text())
            count = sum(p.numel() for p in self.model.parameters())
            if (count != pinned['expected_num_parameters'] or self.model.config.hidden_size != pinned['expected_hidden_size']
                or config['model_revision'] != pinned['model_revision']):
                raise RuntimeError('Loaded model does not match pinned Qwen2.5-7B-Instruct identity')
            for parameter in self.model.parameters(): parameter.requires_grad_(False)
            config.update(actual_model_revision=config['model_revision'], tokenizer_id=config['model_id'],
                tokenizer_revision=config['model_revision'], chat_template_hash=digest(self.tokenizer.chat_template),
                actual_num_parameters=count, actual_dtype=str(next(self.model.parameters()).dtype),
                policy_updated=False, training_updates=0)
            base = Path(config['model_path'])
            identity = {k: config[k] for k in ['model_id','model_path','actual_model_revision','checkpoint_id','tokenizer_id',
                'tokenizer_revision','chat_template_hash','actual_num_parameters','actual_dtype','device_map_resolved','policy_updated','training_updates']}
            identity['file_hashes'] = {p.name: file_hash(p) for p in base.iterdir() if p.name in
                ['config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','download_manifest.json']}
            write_json(Path(config['run_dir'])/'model_identity.json',identity)
            write_json(Path(config['run_dir'])/'actual_model_config.json',self.model.config.to_dict())
            print(f"ACTUAL_MODEL {config['model_id']} revision={config['model_revision']} parameters={count} dtype={config['actual_dtype']}",flush=True)

    def generate_sequences(self, batch):
        import numpy as np
        import torch
        from verl import DataProto
        from verl.utils.model import compute_position_id_with_mask
        started = time.monotonic()
        responses, masks = [], []
        max_new = self.config["response_tokens"]
        device = self.model.get_input_embeddings().weight.device
        for i in range(len(batch)):
            active = self.observer is None or self.observer.active[i]
            prompt = batch.batch["input_ids"][i][batch.batch["attention_mask"][i].bool()].unsqueeze(0).to(device)
            if active:
                # A recorded independent stream for each trajectory and turn.
                turn = self.observer.episodes[i]["agent_turns"] if self.observer else 0
                seed = self.config["seed"] + i + 100003 * turn
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    output = self.model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                        max_new_tokens=max_new, do_sample=True, temperature=self.config["temperature"],
                        top_p=self.config["top_p"], pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
                ids = output[0, prompt.shape[1]:].cpu()
            else:
                ids = torch.tensor([self.tokenizer.eos_token_id])
            mask = torch.zeros(max_new, dtype=torch.long)
            mask[:len(ids)] = 1
            padded = torch.full((max_new,), self.tokenizer.pad_token_id, dtype=torch.long)
            padded[:len(ids)] = ids
            responses.append(padded); masks.append(mask)
        responses = torch.stack(responses)
        prompts = batch.batch["input_ids"].cpu()
        attention = torch.cat([batch.batch["attention_mask"].cpu(), torch.stack(masks)], dim=-1)
        out = DataProto.from_single_dict({"prompts": prompts, "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=-1), "attention_mask": attention,
            "position_ids": compute_position_id_with_mask(attention)})
        print(f"generated batch in {time.monotonic()-started:.1f}s; active={sum(self.observer.active) if self.observer else len(batch)}", flush=True)
        return out


class PilotTracking:
    """One driver / one existing Tracking instance. Local outputs always come first."""
    def __init__(self, run_dir, config):
        import sys
        if sys.platform == "win32":
            for stream in (sys.stdout, sys.stderr):
                if hasattr(stream, "reconfigure"):
                    stream.reconfigure(encoding="utf-8", errors="replace")
        self.run_dir, self.tracking, self.sdk = Path(run_dir), None, None
        mode = config.get("swanlab_mode", "auto")
        if mode == "auto":
            mode = "cloud" if os.environ.get("SWANLAB_API_KEY") else "offline"
        self.state = {"backend": "swanlab", "mode_requested": mode, "mode": "disabled", "url": None, "summary_status": "pending"}
        if mode != "disabled":
            try:
                os.environ["SWANLAB_MODE"] = mode
                os.environ["SWANLAB_LOG_DIR"] = str(self.run_dir / "swanlog")
                from verl.utils.tracking import Tracking
                saved_key = os.environ.pop("SWANLAB_API_KEY", None) if mode == "offline" else None
                try:
                    self.tracking = Tracking(config["swanlab_project"], config["run_id"], default_backend=["swanlab"], config=config)
                finally:
                    if saved_key is not None:
                        os.environ["SWANLAB_API_KEY"] = saved_key
                self.sdk = self.tracking.logger["swanlab"]
                self.state["mode"] = mode
                run = self.sdk.get_run()
                # Explicit analysis axis. No training-update axis is claimed for frozen data.
                self.sdk.define_metric("analysis/batch_idx")
                for prefix in ["eval", "data", "phase", "coverage", "anchor", "adv", "cost"]:
                    self.sdk.define_metric(prefix + "/*", step_metric="analysis/batch_idx")
                url = getattr(run, "url", None) if mode == "cloud" else None
                if callable(url):
                    url = url()
                if mode == "cloud" and isinstance(url, str) and url.startswith("https://"):
                    self.state["url"] = url
            except Exception as exc:
                self.state["error"] = f"{type(exc).__name__}: {exc}"
        write_json(self.run_dir / "tracking.json", self.state)

    def log_summary(self, metrics, figures):
        if self.tracking is None or self.state["mode"] == "disabled":
            return
        data = {"analysis/batch_idx": 0}
        for key, record in metrics.items():
            if isinstance(record, dict) and isinstance(record.get("value"), (float, int)):
                data[key] = record["value"]
                if record.get("denominator") is not None:
                    data[key + "_denominator"] = record["denominator"]
        try:
            for figure in figures:
                data["figures/" + figure.stem] = self.sdk.Image(str(figure))
            self.tracking.log(data, step=0)
            self.state["summary_status"] = "submitted"
            write_json(self.run_dir / "tracking.json", self.state)
        except Exception as exc:
            self.state["log_error"] = f"{type(exc).__name__}: {exc}"
            write_json(self.run_dir / "tracking.json", self.state)

    def close(self):
        if self.tracking:
            try:
                self.sdk.finish()
                self.state["finish_status"] = "completed"
                write_json(self.run_dir / "tracking.json", self.state)
            except Exception as exc:
                self.state["finish_error"] = f"{type(exc).__name__}: {exc}"
                write_json(self.run_dir / "tracking.json", self.state)
            finally:
                self.tracking.logger.clear()  # avoid a second finish from upstream destructor
