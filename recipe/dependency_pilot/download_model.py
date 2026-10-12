"""Pin and download the public model, with a verified GET fallback for broken HEAD proxies."""
import json
import os
from pathlib import Path

from .common import file_hash, write_json


def download(destination, repo="Qwen/Qwen2.5-1.5B-Instruct", revision=None):
    import requests
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
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
        if name not in {"config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "LICENSE"}:
            continue
        target = destination / name
        expected = item.get("lfs", {}).get("sha256")
        size = item.get("size") or item.get("lfs", {}).get("size")
        if target.exists() and (not size or target.stat().st_size == size) and (not expected or file_hash(target) == expected):
            records.append({"file": name, "sha256": file_hash(target), "bytes": target.stat().st_size})
            continue
        partial = target.with_suffix(target.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
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
        partial.replace(target)
        records.append({"file": name, "sha256": sha, "bytes": target.stat().st_size})
        print(f"Verified {name}: {target.stat().st_size} bytes", flush=True)
    write_json(destination / "download_manifest.json", {"repo_id": repo, "revision": revision, "files": records})
    return str(destination)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    parser.add_argument("--revision")
    args = parser.parse_args()
    print(download(args.destination, revision=args.revision))
