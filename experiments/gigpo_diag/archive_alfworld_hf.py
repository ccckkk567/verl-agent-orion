"""Upload completed ALFWorld runs; remove verified older checkpoints only."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

BASE = Path('/scratch/prjsnnn0000269/Students/shuming/cjk')
ROOT = BASE/'logs/gigpo-reproduction/alfworld'
JOBS = {'160764', '161471'}
REPO = 'Fyvoid/alfworld-qwen2.5-1.5b-gigpo-checkpoints'

def idle():
    output = subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i %j'],text=True)
    if any(line.split()[0] in JOBS or 'ggp-alf' in line for line in output.splitlines() if line.strip()):
        raise RuntimeError('ALFWorld job still queued/running; refusing archive cleanup')

def files(folder):
    if folder.is_symlink() or ROOT.resolve() not in folder.resolve().parents:
        raise ValueError('Unsafe checkpoint path: '+str(folder))
    result = []
    for f in folder.rglob('*'):
        if f.is_symlink():
            raise ValueError('Symlink in checkpoint: '+str(f))
        if f.is_file():
            result.append(f)
    return sorted(result)

def hashes(f):
    size = f.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(f'blob {size}\0'.encode())
    with f.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            sha.update(block); git.update(block)
    return dict(size=size,sha256=sha.hexdigest(),git_blob=git.hexdigest())

def complete(folder):
    expected = [folder/'data.pt'] + [folder/'actor'/f'{kind}_world_size_2_rank_{rank}.pt' for kind in ('model','optim','extra_state') for rank in (0,1)]
    return all(f.is_file() and not f.is_symlink() and f.stat().st_size>0 for f in expected)

def verify(local, remote):
    if remote.size != local['size']:
        raise ValueError('Remote size mismatch')
    lfs = remote.lfs
    digest = (lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
    if (digest and digest != local['sha256']) or (not digest and remote.blob_id != local['git_blob']):
        raise ValueError('Remote content hash mismatch or unavailable')

def main():
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    idle()
    api = HfApi()
    if api.whoami()['name'] != 'Fyvoid':
        raise ValueError('HF credential must belong to Fyvoid')
    runs=[]; checkpoints=[]
    for f in sorted(ROOT.glob('*/RUN_MANIFEST.json')):
        m=json.loads(f.read_text())
        if str(m.get('job_id')) not in JOBS:
            continue
        if m.get('env') != 'alfworld':
            raise ValueError('Wrong environment')
        runs.append((f.parent,m))
        for p in (f.parent/'checkpoints').glob('global_step_*'):
            if p.is_dir() and re.fullmatch(r'global_step_\d+',p.name):
                files(p)
                checkpoints.append((int(p.name.rsplit('_',1)[1]),p))
    if {str(m['job_id']) for _,m in runs} != JOBS:
        raise ValueError('Expected exactly original and continuation run identities')
    committed=[]
    for step,p in checkpoints:
        marker=p.parent/'latest_checkpointed_iteration.txt'
        if complete(p) and marker.exists() and marker.read_text().strip().isdigit() and step<=int(marker.read_text().strip()):
            committed.append((step,p))
    if not committed:
        raise ValueError('No complete committed checkpoint to retain')
    latest=max(step for step,p in committed)
    keep=[p for step,p in checkpoints if step>=latest]
    print('KEEP',*[str(p) for p in keep],sep='\n',flush=True)
    api.create_repo(REPO,repo_type='model',private=False,exist_ok=True)
    if api.repo_info(REPO,repo_type='model').private:
        raise ValueError('Existing repository is private; no visibility change made')
    audit=BASE/'analysis/gigpo-reproduction/alfworld'/('hf-archive-'+os.environ['SLURM_JOB_ID'])
    audit.mkdir(parents=True,exist_ok=False)
    records=[]
    for step,p in sorted(checkpoints):
        prefix=p.parent.parent.name+'/checkpoints/'+p.name
        print('HASH_AND_UPLOAD',p,flush=True)
        inventory={f.relative_to(p).as_posix():hashes(f) for f in files(p)}
        if not inventory:
            raise ValueError('Empty checkpoint directory')
        commit=api.upload_folder(repo_id=REPO,repo_type='model',folder_path=str(p),path_in_repo=prefix,commit_message='Archive '+prefix)
        remote={x.path:x for x in api.list_repo_tree(REPO,path_in_repo=prefix,revision=commit.oid,recursive=True) if isinstance(x,RepoFile)}
        for name,value in inventory.items():
            verify(value,remote[prefix+'/'+name])
        record=dict(local=str(p),step=step,remote_prefix=prefix,revision=commit.oid,files=inventory,keep=p in keep)
        records.append(record)
        (audit/'verified.json').write_text(json.dumps(records,indent=2))
        print('VERIFIED',p,commit.oid,flush=True)
    # Upload only explicit provenance files, never the full run/log directory.
    for run,m in runs:
        for name in ('RUN_MANIFEST.json','model_identity.json','CONFIG_DIFF.md'):
            f=run/name
            if f.is_file():
                api.upload_file(repo_id=REPO,path_or_fileobj=str(f),path_in_repo=run.name+'/'+name)
    api.upload_file(repo_id=REPO,path_or_fileobj=str(audit/'verified.json'),path_in_repo='archive_receipts/'+audit.name+'.json')
    idle()
    # Revalidate every source before any removal. New/mutated files abort cleanup.
    for r in records:
        p=Path(r['local'])
        current={f.relative_to(p).as_posix():hashes(f) for f in files(p)}
        if current!=r['files']:
            raise ValueError('Checkpoint changed; no cleanup: '+str(p))
    removed=[]
    for r in records:
        if r['keep']:
            continue
        idle()
        p=Path(r['local'])
        for name in r['files']:
            (p/name).unlink()
        for d in sorted((x for x in p.rglob('*') if x.is_dir()),key=lambda x:len(x.parts),reverse=True):
            d.rmdir()
        p.rmdir()
        removed.append(str(p))
        (audit/'deleted.json').write_text(json.dumps(removed,indent=2))
        print('DELETED_VERIFIED_OLD',p,flush=True)
    print('DONE',REPO,'RETAINED',[str(p) for p in keep],'AUDIT',audit,flush=True)

if __name__=='__main__': main()
