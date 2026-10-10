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


# Public community mirror, pinned to a revision. Large-file hashes agree across
# sparklabutah/timewarp-env-data, YWZBrandon/webshop-data and HongbangYuan/webshop.
# The instruction file also matches WebShop's upstream baseline_models/data file.
WEBSHOP_MIRROR = (
    'https://huggingface.co/datasets/sparklabutah/timewarp-env-data/resolve/'
    'bfed490a8b4b5dae94b126be6105b84445dcbc4b/webshop'
)
WEBSHOP_FILES = {
    'items_shuffle.json': (5479720229, '2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8'),
    'items_ins_v2.json': (186295270, '1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e'),
    'items_human_ins.json': (5137548, 'cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3'),
}

# DefaultLuceneDocumentGenerator (Anserini 0.14.3, bundled by Pyserini 0.17)
# rejects contents.trim().length() == 0. Java trim removes only U+0000..U+0020,
# unlike Python str.strip(), which also removes many Unicode whitespace chars.
JAVA_TRIM_CHARS = ''.join(chr(code) for code in range(33))


def empty_index_document(contents):
    return not contents.strip(JAVA_TRIM_CHARS)


def validate_index_count(catalog_count, empty_count, indexed_count):
    expected = catalog_count - empty_count
    if not 0 <= empty_count <= catalog_count or indexed_count != expected:
        raise RuntimeError(
            f'Index count mismatch: catalog={catalog_count}, empty={empty_count}, '
            f'expected_indexed={expected}, actual_indexed={indexed_count}'
        )


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


def download_verified(url, path, expected_size, expected_sha256):
    """Resume a public mirror download; publish only size/hash-verified bytes."""
    def verify(candidate):
        if candidate.stat().st_size != expected_size or sha256(candidate) != expected_sha256:
            raise RuntimeError(f'Dataset size/SHA-256 mismatch: {candidate}; inspect before retrying')

    if path.is_file():
        verify(path)
        print(f'DATA_VERIFIED: {path.name} (existing)', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never append mirror bytes to a partial Google Drive/HTML response.
    temporary = path.with_name(path.name + '.hf.part')
    if not temporary.exists() or temporary.stat().st_size < expected_size:
        subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                        '--connect-timeout', '30', '--continue-at', '-',
                        url, '-o', str(temporary)], check=True)
    verify(temporary)
    temporary.replace(path)
    print(f'DATA_VERIFIED: {path.name}', flush=True)


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
        size, digest = WEBSHOP_FILES[name]
        download_verified(f'{WEBSHOP_MIRROR}/{name}?download=true', path, size, digest)
        # Keep the content identity stable for indexes made from Drive downloads.
        records[name] = {'google_drive_id': file_id, 'bytes': path.stat().st_size,
                         'sha256': digest}
    save_json(data / 'download-source.json', {'mirror': WEBSHOP_MIRROR, 'files': records})

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
    empty_count = 0
    with (resources / 'documents.jsonl').open('w', encoding='utf-8') as stream:
        for product in products:
            options = ', and '.join(f'{key}: {", ".join(values)}'
                                    for key, values in product.get('options', {}).items())
            bullets = product['BulletPoints']
            content = ' '.join([product['Title'], product['Description'],
                                bullets[0] if bullets else '', options]).lower()
            empty_count += int(empty_index_document(content))
            stream.write(json.dumps({'id': product['asin'], 'contents': content,
                                     'product': product}) + '\n')
    del products, _
    print(f'WEBSHOP_INDEX_INPUT: catalog={count}, empty={empty_count}, '
          f'expected_indexed={count - empty_count}', flush=True)
    # A fresh staging index prevents duplicate documents after interrupted runs.
    stage = data / f'indexes-build-{os.environ["SLURM_JOB_ID"]}'
    if stage.exists():
        raise RuntimeError(f'Index staging directory already exists: {stage}')
    subprocess.run([sys.executable, '-m', 'pyserini.index.lucene',
                    '--collection', 'JsonCollection', '--input', str(resources),
                    '--index', str(stage), '--generator', 'DefaultLuceneDocumentGenerator',
                    '--threads', '4', '--storePositions', '--storeDocvectors', '--storeRaw'], check=True)
    searcher = LuceneSearcher(str(stage))
    indexed_count = searcher.num_docs
    try:
        validate_index_count(count, empty_count, indexed_count)
        assert searcher.search(str(query), k=5), f'No search results for {query}'
    finally:
        searcher.close()
    if index.exists():
        index.rename(data / f'indexes-previous-{os.environ["SLURM_JOB_ID"]}')
    stage.rename(index)
    save_json(marker, {'sources': records, 'catalog_products': count,
                       'empty_documents': empty_count, 'indexed_products': indexed_count,
                       'probe_query': query})
    link_directory(base / 'search_engine/indexes', index)
    # Only delete the generated JSONL for this job after index validation.
    (resources / 'documents.jsonl').unlink()
    resources.rmdir()
    print('WEBSHOP_DATA_AND_INDEX_OK', indexed_count, flush=True)


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
