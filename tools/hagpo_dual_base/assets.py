"""Pin and verify public assets. Downloads use hf CLI, never execute remote code."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(os.environ.get('HAGPO_ROOT','/mnt/data/cjk/hagpo_20260928'))
BASE = ROOT/'dual_base'
CANDIDATES = {
    'openmobile': ('cckevinn/OpenMobile-8B','0cd9fc56c7f359ebe62ae273b731f3213fd62b40'),
    'uitars': ('ByteDance-Seed/UI-TARS-7B-SFT','3434901a9dd04dd3625617d839a5724fe5e2db20'),
}


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False));tmp.replace(path)


def fetch(url):
    op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:10809','http':'http://127.0.0.1:10809'}))
    with op.open(url,timeout=45) as f:return f.read()


def lock_metadata(candidate):
    path=BASE/'manifests'/candidate/'model_lock.json'
    if path.exists():return json.loads(path.read_text())
    repo,revision=CANDIDATES[candidate]
    metadata=json.loads(fetch(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'))
    if metadata['sha']!=revision:raise RuntimeError('revision mismatch')
    lock={'repo':repo,'revision':revision,'source':'official_huggingface','created_at':time.time(),
          'modelscope_check':'checked candidate namespaces; no validated mirror available; official HF fallback',
          'files':[x for x in metadata['siblings'] if x['rfilename']!='.gitattributes']}
    save(path,lock);return lock


def digest_file(path, item):
    if not path.is_file() or path.stat().st_size!=item['size']:raise RuntimeError('missing/size mismatch '+str(path))
    sha=hashlib.sha256();git=hashlib.sha1()
    git.update(('blob '+str(item['size'])+'\0').encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            sha.update(block);git.update(block)
    if item.get('lfs'):
        if sha.hexdigest()!=item['lfs']['sha256']:raise RuntimeError('LFS hash mismatch '+str(path))
    elif git.hexdigest()!=item['blobId']:raise RuntimeError('git blob mismatch '+str(path))
    return sha.hexdigest()


def download(candidate, weights):
    lock=lock_metadata(candidate);out=ROOT/'models'/('OpenMobile-8B' if candidate=='openmobile' else 'UI-TARS-7B-SFT')
    items=[x for x in lock['files'] if weights or not x['rfilename'].endswith('.safetensors')]
    env=os.environ.copy();env.update(HTTPS_PROXY='http://127.0.0.1:10809',HTTP_PROXY='http://127.0.0.1:10809',
        HF_HUB_DISABLE_XET='1',HF_HUB_OFFLINE='0',HF_HUB_DOWNLOAD_TIMEOUT='90',HF_HUB_ETAG_TIMEOUT='30')
    cmd=[str(ROOT/'envs/learner/bin/hf'),'download',lock['repo'],* [x['rfilename'] for x in items],
         '--revision',lock['revision'],'--local-dir',str(out),'--max-workers','2']
    # At most two attempts; a downloaded-file resume is not experiment trajectory resume.
    for attempt in range(2):
        print('DOWNLOAD',candidate,'weights' if weights else 'configs','attempt',attempt+1,flush=True)
        result=subprocess.run(cmd,env=env,timeout=7200)
        if result.returncode==0:break
    else:raise RuntimeError('bounded asset download attempts exhausted')
    hashes={x['rfilename']:digest_file(out/x['rfilename'],x) for x in items}
    report={'status':'verified','repo':lock['repo'],'revision':lock['revision'],'path':str(out),
            'files_sha256':hashes,'weights_verified':weights,'time':time.time()}
    save(BASE/'manifests'/candidate/('weights_ready.json' if weights else 'config_ready.json'),report)
    print('VERIFIED',candidate,len(hashes),'files',flush=True)


def sources():
    path=BASE/'sources/source_lock.json'
    if path.exists():return
    specs={
        'njucckevin/OpenMobile-Code':['README.md','LlamaFactory/qwen3vl_full_sft.yaml',
            'AndroidWorld/android_world/agents/PROMPT.py','AndroidWorld/android_world/agents/seeact_v.py',
            'AndroidWorld/android_world/agents/utils.py'],
        'bytedance/UI-TARS':['README_v1.md','codes/ui_tars/action_parser.py'],
    }
    lock={}
    for repo,files in specs.items():
        revision=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/main'))['sha']
        entries={}
        for name in files:
            url=f'https://raw.githubusercontent.com/{repo}/{revision}/{name}'
            data=fetch(url);target=BASE/'sources'/repo.split('/')[-1]/name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            entries[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
        lock[repo]={'revision':revision,'files':entries}
    save(path,lock)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','weights']);a=p.parse_args()
    BASE.mkdir(exist_ok=True)
    with (BASE/'assets.lock').open('a') as file:
        fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.mode=='prepare':sources()
            for c in CANDIDATES:download(c,a.mode=='weights')
        except Exception as exc:
            save(BASE/'reports'/('assets_'+a.mode+'_failure.json'),{'error':repr(exc),'time':time.time()})
            raise


if __name__=='__main__':main()
