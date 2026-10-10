"""Launch one official GiGPO arm on one idle GPU. No hyperparameter fallback."""
import os,sys,json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];arm=sys.argv[1];assert arm in ('obs','mem')
# GPU selection uses memory and active compute UUIDs, rechecked immediately before launch.
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
available=[int(i) for i,u,m in (line.split(', ') for line in lines) if int(m)<100 and u not in jobs]
assert available,'NO_IDLE_GPU'
gpu=3 if 3 in available else available[0]
run=R/'runs'/arm;run.mkdir(parents=True,exist_ok=True)
assert not list(run.glob('update_*')),'Existing rollouts require reviewed recovery'
env=dict(os.environ,PILOT_ROOT=str(R),PILOT_ARM=arm,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=f'{R}/code:{R}/deps:{R}/repo',RAY_ADDRESS='',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',VLLM_USE_V1='1')
python='/mnt/data/cjk/sol_qwen3vl8b_ms_swift_sft_v3/envs/sft_swift/bin/python'
args=['algorithm.adv_estimator=gigpo','algorithm.gamma=1.0','algorithm.gigpo.step_advantage_w=1.0','algorithm.gigpo.mode=mean_norm','algorithm.use_kl_in_reward=False','algorithm.filter_groups.enable=False',f'data.train_files={R}/configs/tasks.parquet',f'data.val_files={R}/configs/tasks.parquet','data.train_batch_size=2','data.val_batch_size=2','data.shuffle=False','+data.seed=2026092700','data.max_prompt_length=16384','data.max_response_length=512','data.return_raw_chat=True','data.truncation=error',
'actor_rollout_ref.model.path=/mnt/data/cjk/interpreter_qwen3vl8b_v1/models/Qwen3-VL-8B-Instruct','actor_rollout_ref.model.lora_rank=64','actor_rollout_ref.model.lora_alpha=64','actor_rollout_ref.model.target_modules=all-linear','actor_rollout_ref.model.enable_gradient_checkpointing=True','actor_rollout_ref.model.use_remove_padding=False','+actor_rollout_ref.model.override_config._attn_implementation=sdpa','+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16','actor_rollout_ref.actor.optim.lr=3e-6','actor_rollout_ref.actor.ppo_mini_batch_size=16','actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1','actor_rollout_ref.actor.use_kl_loss=False','actor_rollout_ref.actor.use_invalid_action_penalty=False','actor_rollout_ref.actor.fsdp_config.param_offload=True','actor_rollout_ref.actor.fsdp_config.optimizer_offload=True',
'actor_rollout_ref.rollout.name=vllm','actor_rollout_ref.rollout.tensor_model_parallel_size=1','actor_rollout_ref.rollout.gpu_memory_utilization=0.4','actor_rollout_ref.rollout.enforce_eager=True','actor_rollout_ref.rollout.free_cache_engine=True','actor_rollout_ref.rollout.load_format=safetensors','actor_rollout_ref.rollout.max_model_len=16896','actor_rollout_ref.rollout.max_num_batched_tokens=16896','actor_rollout_ref.rollout.max_num_seqs=1','actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1','actor_rollout_ref.rollout.temperature=1.0','actor_rollout_ref.rollout.top_p=0.95','actor_rollout_ref.rollout.n=1','+actor_rollout_ref.rollout.seed=2026092700',
'env.env_name=AndroidWorld','env.seed=2026092700','env.max_steps=34','env.rollout.n=8','trainer.n_gpus_per_node=1','trainer.nnodes=1','trainer.total_training_steps=3','trainer.total_epochs=3','trainer.val_before_train=False','trainer.test_freq=-1','trainer.save_freq=1','trainer.resume_mode=disable','trainer.logger=[console]',f'trainer.default_local_dir={run}/checkpoints','ray_init.num_cpus=8','+ray_init.num_gpus=1',f'+ray_init._temp_dir=/tmp/cjk_gmp_{arm}']
manifest=dict(arm=arm,gpu=gpu,gpu_snapshot=lines,argv=args,initial_checkpoint='Qwen3-VL-8B-Instruct',seed=2026092700,notes=['SDPA via existing model override; no remove-padding; both arms identical.','Official LoRA rank64 alpha64 all-linear; VL compatibility subject to smoke.','Invalid action penalty disabled to preserve binary task reward.','No fallback on OOM/nonfinite; checkpoint reload and 3-update checks required before extension.'],created_at=time.time())
(R/f'run_manifest_{arm}.json').write_text(json.dumps(manifest,indent=2))
log=R/'logs'/f'smoke_{arm}.log'
if log.exists():log.rename(R/'logs'/f'smoke_{arm}_{time.time_ns()}.log')
with log.open('w') as f:
 p=subprocess.Popen([python,'-m','verl.trainer.main_ppo',*args],env=env,cwd=R/'repo',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'status.json').write_text(json.dumps(dict(stage='SMOKE_LAUNCH',arm=arm,pid=p.pid,gpu=gpu,launched_at=time.time(),training_updates_verified=0),indent=2));print('Launched',p.pid,'GPU',gpu)
