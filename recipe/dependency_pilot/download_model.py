"""Pin and download the public model, with a verified GET fallback for broken HEAD proxies."""
import json
import hashlib
import os
from pathlib import Path

from .common import file_hash, write_json


def download(destination, repo="Qwen/Qwen2.5-1.5B-Instruct", revision=None):
    import requests
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    existing = destination/'download_manifest.json'
    if existing.exists():
        saved=json.loads(existing.read_text())
        if saved['repo_id']!=repo or (revision and saved['revision']!=revision):
            raise ValueError('Refusing to overwrite a different model or revision')
    headers = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    endpoint = f"https://huggingface.co/api/models/{repo}"
    if revision:
        endpoint += f"/revision/{revision}"
    response = requests.get(endpoint, params={"blobs": "true"}, headers=headers, timeout=45)
    response.raise_for_status()
    meta = response.json()
    revision = meta["sha"]
    records = []
    for item in meta["siblings"]:
        name = item["rfilename"]
        if name not in {"config.json", "generation_config.json", "merges.txt", "model.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "LICENSE"} and not (name.startswith('model-') and name.endswith('.safetensors') and '/' not in name):
            continue
        target = destination / name
        expected = item.get("lfs", {}).get("sha256")
        size = item.get("size") or item.get("lfs", {}).get("size")
        def verified(path):
            if not path.exists() or (size and path.stat().st_size != size): return False
            if expected: return file_hash(path) == expected
            blob = item.get('blobId')
            if blob:
                content = path.read_bytes()
                return hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest() == blob
            return False
        if verified(target):
            records.append({"file": name, "sha256": file_hash(target), "bytes": target.stat().st_size})
            continue
        partial = target.with_suffix(target.suffix + ".part")
        if verified(partial):
            partial.replace(target)
            records.append({'file':name,'sha256':file_hash(target),'bytes':target.stat().st_size})
            continue
        offset = partial.stat().st_size if partial.exists() else 0
        if size and offset>=size: offset=0
        request_headers = dict(headers)
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        # requests strips authorization on redirects to another host.
        with requests.get(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", params={"download": "true"},
                          headers=request_headers, stream=True, timeout=(30, 120)) as stream:
            stream.raise_for_status()
            if stream.status_code != 206:
                offset = 0
            with partial.open("ab" if offset else "wb") as out:
                for chunk in stream.iter_content(4 * 1024 * 1024):
                    out.write(chunk)
        if size and partial.stat().st_size != size:
            raise RuntimeError(f"Size verification failed for {name}")
        sha = file_hash(partial)
        if expected and sha != expected:
            raise RuntimeError(f"SHA256 verification failed for {name}")
        if not verified(partial):
            raise RuntimeError(f"Hub content verification failed for {name}")
        partial.replace(target)
        records.append({"file": name, "sha256": sha, "bytes": target.stat().st_size})
        print(f"Verified {name}: {target.stat().st_size} bytes", flush=True)
    index = destination / 'model.safetensors.index.json'
    required = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not required.issubset({r['file'] for r in records}):
        raise RuntimeError('Incomplete model weight shards')
    write_json(destination / "download_manifest.json", {"repo_id": repo, "revision": revision, "files": records})
    return str(destination)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    parser.add_argument("--revision")
    parser.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()
    print(download(args.destination, repo=args.repo, revision=args.revision))
