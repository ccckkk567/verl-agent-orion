"""Exercise the repaired packages in the existing prepare job, without a model."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path


def main():
    import decord
    import numpy as np
    import textworld
    from agent_system.environments.env_package.alfworld.envs import AlfworldWorker, load_config_file
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

    # Executes a native libdecord allocation/copy, beyond merely importing Python.
    sample = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(decord.ndarray.array(sample).asnumpy(), sample)
    config_path = Path('agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
    config = load_config_file(config_path)
    base = get_environment(config['env']['type'])(config, train_eval='train')
    if base.num_games <= 0:
        raise RuntimeError('No ALFWorld training games found')
    worker = AlfworldWorker(config, seed=123, base_env=base)
    try:
        obs, info = worker.reset()
        if len(obs) != 1 or not obs[0] or not info['admissible_commands'][0]:
            raise RuntimeError('ALFWorld reset did not return an observation and action menu')
        after, scores, dones, after_info = worker.step('look')
        if len(after) != 1 or not after[0]:
            raise RuntimeError('ALFWorld look did not return an observation')
        result = dict(status='passed', textworld=metadata.version('textworld'), decord=metadata.version('decord'),
                      textworld_path=textworld.__file__, decord_path=decord.__file__, games=base.num_games,
                      gamefile=info.get('extra.gamefile'), reset_observation=obs, action='look', observation=after,
                      admissible_commands=after_info['admissible_commands'], model_loaded=False, training_updates=0)
        (Path(os.environ['TMPDIR']) / 'alfworld-runtime-check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print('ALFWORLD_RUNTIME_CHECK_PASSED ' + json.dumps({k: result[k] for k in ('textworld', 'decord', 'games', 'model_loaded', 'training_updates')}), flush=True)
    finally:
        worker.env.close()


if __name__ == '__main__':
    main()
