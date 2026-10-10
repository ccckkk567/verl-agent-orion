"""Exercise repository Ray game wrappers, not just third-party imports."""
import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import sys


def check_alfworld(repo):
    from agent_system.environments.env_package.alfworld import build_alfworld_envs

    config = repo / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
    results = {}
    for split in ['train', 'eval_in_distribution', 'eval_out_of_distribution']:
        env = build_alfworld_envs(str(config), seed=42, env_num=1, group_n=1,
                                 resources_per_worker={'num_cpus': 1}, is_train=split == 'train',
                                 env_kwargs={'eval_dataset': split})
        try:
            obs, image, infos = env.reset()
            assert len(obs) == 1 and obs[0] and image is None
            commands = env.get_admissible_commands[0]
            assert commands
            action = next((a for a in commands if a.startswith('go to ')), commands[0])
            obs, image, rewards, dones, infos = env.step([action])
            assert obs[0] and math.isfinite(float(rewards[0]))
            results[split] = {'reset_and_step': 'passed', 'action': action,
                              'gamefile': str(infos[0].get('extra.gamefile', ''))}
        finally:
            env.close()
    return results


def check_webshop(repo):
    from agent_system.environments.env_package.webshop import build_webshop_envs

    data = Path(os.environ['WEBSHOP_DATA'])
    manifest = json.loads((data / 'index-manifest.json').read_text())
    assert manifest['indexed_products'] >= 1_000_000
    env = build_webshop_envs(seed=42, env_num=1, group_n=1,
                            resources_per_worker={'num_cpus': 2}, is_train=True,
                            env_kwargs={'observation_mode': 'text', 'num_products': None,
                                        'human_goals': False,
                                        'file_path': str(data / 'items_shuffle.json'),
                                        'attr_path': str(data / 'items_ins_v2.json')})
    try:
        assert len(env.goal_idxs) > 0
        training_goals = len(env.goal_idxs)
        results = {}
        # Reuse one full-catalog actor to avoid loading a second catalog into RAM.
        for split, indices in [('train', env.goal_idxs), ('eval', range(500))]:
            env.goal_idxs = indices
            obs, infos = env.reset()
            assert obs[0] and infos[0]['available_actions']['has_search_bar']
            query = str(manifest['probe_query'])
            obs, rewards, dones, infos = env.step([f'search[{query}]'])
            clickables = infos[0]['available_actions']['clickables']
            asin = next((x for x in clickables if re.fullmatch(r'[a-z0-9]{10}', x)), None)
            assert asin, f'No product links after search: {clickables}'
            obs, rewards, dones, infos = env.step([f'click[{asin}]'])
            assert 'buy now' in infos[0]['available_actions']['clickables']
            obs, rewards, dones, infos = env.step(['click[buy now]'])
            assert dones[0] and math.isfinite(float(infos[0]['task_score']))
            results[split] = {'search_open_buy': 'passed', 'product': asin,
                              'task_score': float(infos[0]['task_score'])}
        return {'indexed_products': manifest['indexed_products'],
                'training_goals': training_goals, 'episodes': results}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('game', choices=['alfworld', 'webshop'])
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise SystemExit('Run inside a Slurm allocation')
    repo = Path(os.environ['ORION_REPO'])
    sys.path.insert(0, str(repo))
    import torch
    import ray
    import vllm  # noqa: F401 -- validate the installed inference stack imports

    assert torch.cuda.is_available()
    a = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
    assert (a @ a).float().mean().item() == 32.0
    # Ray's Unix socket paths require a short node-local directory.
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False,
             _node_ip_address='127.0.0.1', _temp_dir=f'/tmp/cjk-ray-{job}-{args.game}',
             object_store_memory=512 * 1024 * 1024)
    try:
        result = check_alfworld(repo) if args.game == 'alfworld' else check_webshop(repo)
    finally:
        ray.shutdown()
    packages = ['torch', 'vllm', 'ray', 'numpy', 'transformers']
    packages += ['alfworld', 'textworld'] if args.game == 'alfworld' else ['pyserini', 'spacy', 'gym']
    report = {'job_id': job, 'game': args.game, 'python': sys.version,
              'versions': {name: version(name) for name in packages},
              'gpu_bf16_matmul': 'passed', 'environment': result,
              'scope': 'Text game interaction and GPU stack; no model rollout or RL training'}
    path = Path(os.environ['ORION_ROOT']) / 'logs' / f'{args.game}-check-{job}.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'GAME_SMOKE_OK: {args.game}')


if __name__ == '__main__':
    main()
