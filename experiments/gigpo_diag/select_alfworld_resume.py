"""Select a committed, structurally complete native two-rank checkpoint."""
import argparse
import gzip
import json
import sys
from pathlib import Path

def select(run):
    run = Path(run).resolve()
    manifest = json.loads((run/'RUN_MANIFEST.json').read_text())
    if manifest.get('env') != 'alfworld' or str(manifest.get('job_id')) != '160764':
        raise ValueError('Expected ALFWorld parent job 160764')
    root = run/'checkpoints'
    committed = set()
    marker = root/'latest_checkpointed_iteration.txt'
    if marker.exists():
        value = marker.read_text().strip()
        if value.isdigit():
            committed.add(int(value))
    for f in (run/'diagnostics').glob('checkpoints.pid-*.jsonl.gz'):
        try:
            with gzip.open(f,'rt') as stream:
                for line in stream:
                    if not line.endswith('\n'):
                        break
                    r = json.loads(line)
                    if r.get('run_id') != manifest['run_id'] or r.get('code_commit') != manifest['code_commit']:
                        raise ValueError('Mixed checkpoint provenance')
                    if r.get('native_save_returned'):
                        committed.add(int(r['global_step']))
        except EOFError:
            print('Ignoring incomplete checkpoint-log tail: '+str(f),file=sys.stderr)
    for step in sorted(committed,reverse=True):
        folder = root/f'global_step_{step}'
        required = [folder/'data.pt'] + [folder/'actor'/f'{kind}_world_size_2_rank_{rank}.pt' for kind in ('model','optim','extra_state') for rank in (0,1)]
        if all(f.is_file() and f.stat().st_size > 0 for f in required):
            return step,folder,manifest
        print(f'Skipping incomplete checkpoint {folder}',file=sys.stderr)
    raise ValueError('No committed complete checkpoint; refusing to train from scratch')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run')
    a=p.parse_args()
    step,folder,manifest=select(a.run)
    print(folder)
