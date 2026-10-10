"""Prepare public text-game assets inside Scratch; run only in a Slurm job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                    url, '-o', str(temporary)], check=True)
    temporary.replace(path)


def link_directory(link, target):
    """Wire legacy repository paths to Scratch without replacing existing data."""
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError(f'{link} already exists; inspect it before linking to {target}')
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def prepare_alfworld(root):
    from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH

    data = Path(os.environ['ALFWORLD_DATA'])
    data.mkdir(parents=True, exist_ok=True)
    sources = [
        ('0.2.2', 'json_2.1.1_json.zip'),
        ('0.2.2', 'json_2.1.1_pddl.zip'),
        ('0.4.2', 'json_2.1.3_tw-pddl.zip'),
    ]
    records = []
    for release, name in sources:
        url = f'https://github.com/alfworld/alfworld/releases/download/{release}/{name}'
        archive = root / 'downloads' / 'alfworld' / name
        download(url, archive)
        marker = data / (name + '.extracted.json')
        digest = sha256(archive)
        if not marker.exists() or json.loads(marker.read_text())['sha256'] != digest:
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    destination = (data / member.filename).resolve()
                    if not destination.is_relative_to(data.resolve()):
                        raise ValueError(f'Unsafe archive member: {member.filename}')
                source.extractall(data)
            save_json(marker, {'sha256': digest, 'url': url})
        records.append({'url': url, 'sha256': digest})
    logic = data / 'logic'
    logic.mkdir(exist_ok=True)
    shutil.copyfile(ALFRED_PDDL_PATH, logic / 'alfred.pddl')
    shutil.copyfile(ALFRED_TWL2_PATH, logic / 'alfred.twl2')
    counts = {}
    for split in ['train', 'valid_seen', 'valid_unseen']:
        counts[split] = len(list((data / 'json_2.1.1' / split).rglob('game.tw-pddl')))
        if counts[split] == 0:
            raise RuntimeError(f'No TextWorld games in {split}')
    save_json(data / 'manifest.json', {'sources': records, 'game_files': counts})
    print('ALFWORLD_DATA_OK', counts, flush=True)


def prepare_webshop(root, repo):
    import gdown

    data = Path(os.environ['WEBSHOP_DATA'])
    data.mkdir(parents=True, exist_ok=True)
    base = repo / 'agent_system/environments/env_package/webshop/webshop'
    link_directory(base / 'data', data)
    sources = {
        'items_shuffle.json': '1A2whVgOO0euk5O13n2iYDM0bQRkkRduB',
        'items_ins_v2.json': '1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi',
        'items_human_ins.json': '14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O',
    }
    records = {}
    for name, file_id in sources.items():
        path = data / name
        if not path.is_file():
            temporary = data / (name + '.part')
            # Public downloads only; never use a borrowed account's cookies.
            result = gdown.download(id=file_id, output=str(temporary), quiet=False,
                                    use_cookies=False, resume=True)
            if not result:
                raise RuntimeError(f'Download failed for {name}; check Google Drive quota in the log')
            with temporary.open(encoding='utf-8') as stream:
                value = json.load(stream)
            if not isinstance(value, (list, dict)) or not value:
                raise ValueError(f'Invalid dataset: {name}')
            del value
            temporary.replace(path)
        records[name] = {'google_drive_id': file_id, 'bytes': path.stat().st_size,
                         'sha256': sha256(path)}

    sys.path.insert(0, str(base))
    from web_agent_site.engine.engine import load_products
    from pyserini.search.lucene import LuceneSearcher

    index = data / 'indexes'
    marker = data / 'index-manifest.json'
    if marker.is_file() and index.is_dir():
        previous = json.loads(marker.read_text())
        if previous['sources'] == records:
            searcher = LuceneSearcher(str(index))
            assert searcher.num_docs == previous['indexed_products']
            searcher.close()
            link_directory(base / 'search_engine/indexes', index)
            print('WEBSHOP_INDEX_REUSED', previous['indexed_products'], flush=True)
            return

    products, *_ = load_products(str(data / 'items_shuffle.json'),
                                 str(data / 'items_ins_v2.json'), human_goals=False)
    count = len(products)
    if count < 1_000_000:
        raise RuntimeError(f'Expected full WebShop catalog, got only {count} products')
    # Build only the full index. Preserve upstream normalization/search fields.
    resources = data / f'resources-{os.environ["SLURM_JOB_ID"]}'
    resources.mkdir()
    query = next((p['query'] for p in products if p.get('query')), 'shirt')
    with (resources / 'documents.jsonl').open('w', encoding='utf-8') as stream:
        for product in products:
            options = ', and '.join(f'{key}: {", ".join(values)}'
                                    for key, values in product.get('options', {}).items())
            bullets = product['BulletPoints']
            content = ' '.join([product['Title'], product['Description'],
                                bullets[0] if bullets else '', options]).lower()
            stream.write(json.dumps({'id': product['asin'], 'contents': content,
                                     'product': product}) + '\n')
    del products, _
    # A fresh staging index prevents duplicate documents after interrupted runs.
    stage = data / f'indexes-build-{os.environ["SLURM_JOB_ID"]}'
    if stage.exists():
        raise RuntimeError(f'Index staging directory already exists: {stage}')
    subprocess.run([sys.executable, '-m', 'pyserini.index.lucene',
                    '--collection', 'JsonCollection', '--input', str(resources),
                    '--index', str(stage), '--generator', 'DefaultLuceneDocumentGenerator',
                    '--threads', '4', '--storePositions', '--storeDocvectors', '--storeRaw'], check=True)
    searcher = LuceneSearcher(str(stage))
    assert searcher.num_docs == count, (searcher.num_docs, count)
    assert searcher.search(str(query), k=5), f'No search results for {query}'
    searcher.close()
    if index.exists():
        index.rename(data / f'indexes-previous-{os.environ["SLURM_JOB_ID"]}')
    stage.rename(index)
    save_json(marker, {'sources': records, 'indexed_products': count, 'probe_query': query})
    link_directory(base / 'search_engine/indexes', index)
    # Only delete the generated JSONL for this job after index validation.
    (resources / 'documents.jsonl').unlink()
    resources.rmdir()
    print('WEBSHOP_DATA_AND_INDEX_OK', count, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('game', choices=['alfworld', 'webshop'])
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run inside a Slurm allocation')
    root = Path(os.environ['ORION_ROOT'])
    if args.game == 'alfworld':
        prepare_alfworld(root)
    else:
        prepare_webshop(root, Path(os.environ['ORION_REPO']))


if __name__ == '__main__':
    main()
