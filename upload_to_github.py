"""
upload_to_github.py
===================
Colab-only script. Pushes training results and checkpoints from
Google Drive to GitHub under results/colab_t4/ and checkpoints/colab_t4/.

Usage — paste into a Colab cell (Cell 6):
    exec(open(f'{DRIVE_BASE}/upload_to_github.py').read())

Local PC: don't run this. Just use  git add . / git commit / git push.
"""

import os, base64, json
from datetime import datetime
from google.colab import userdata
import urllib.request, urllib.error

# ═══════════════════════════════════════════════════════════════
# CONFIG — nothing here should ever need changing
# ═══════════════════════════════════════════════════════════════
GITHUB_USERNAME  = 'henibela'
GITHUB_REPO      = 'Neural-codes'
GITHUB_BRANCH    = 'main'
GITHUB_PAT       = userdata.get('GITHUB_PAT')

# Google Drive source — must match DRIVE_BASE in Cell 1
DRIVE_BASE       = '/content/drive/MyDrive/[04]Projects/BSc-Thesis-project/ae_lite'

# GitHub destination subfolders — hardcoded to colab_t4, never changes
RESULTS_DEST     = 'results/colab_t4'
CHECKPOINTS_DEST = 'checkpoints/colab_t4'
# ═══════════════════════════════════════════════════════════════

API_BASE = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents'
HEADERS  = {
    'Authorization': f'token {GITHUB_PAT}',
    'Accept':        'application/vnd.github.v3+json',
    'Content-Type':  'application/json',
}


def github_request(method, url, data=None):
    body = json.dumps(data).encode() if data else None
    req  = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code


def get_existing_sha(repo_path):
    resp, status = github_request('GET', f'{API_BASE}/{repo_path}')
    return resp.get('sha') if status == 200 else None


def upload_file(local_path, repo_path, commit_message):
    with open(local_path, 'rb') as f:
        content_b64 = base64.b64encode(f.read()).decode()

    sha  = get_existing_sha(repo_path)
    data = {'message': commit_message, 'content': content_b64, 'branch': GITHUB_BRANCH}
    if sha:
        data['sha'] = sha

    resp, status = github_request('PUT', f'{API_BASE}/{repo_path}', data)

    if status in (200, 201):
        print(f"  ✓ {'Updated' if sha else 'Created'}: {repo_path}")
        return True
    else:
        print(f"  ✗ Failed ({status}): {repo_path} — {resp.get('message', '')}")
        return False


def collect_files():
    files = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ── Results: .png plots and .npy arrays from DRIVE_BASE root ─
    for fname in os.listdir(DRIVE_BASE):
        fpath = os.path.join(DRIVE_BASE, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith('.png') or fname.endswith('.npy'):
            files.append((
                fpath,
                f'{RESULTS_DEST}/{fname}',
                f'colab_t4 results: {fname} [{ts}]'
            ))

    # ── Checkpoints: final .pt models in DRIVE_BASE root ─────────
    for fname in os.listdir(DRIVE_BASE):
        fpath = os.path.join(DRIVE_BASE, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith('.pt'):
            files.append((
                fpath,
                f'{CHECKPOINTS_DEST}/{fname}',
                f'colab_t4 model: {fname} [{ts}]'
            ))

    # ── Checkpoints: intermediate .pt files in checkpoints/ ──────
    ckpt_dir = os.path.join(DRIVE_BASE, 'checkpoints')
    if os.path.isdir(ckpt_dir):
        for fname in sorted(os.listdir(ckpt_dir)):
            fpath = os.path.join(ckpt_dir, fname)
            if os.path.isfile(fpath) and fname.endswith('.pt'):
                files.append((
                    fpath,
                    f'{CHECKPOINTS_DEST}/{fname}',
                    f'colab_t4 checkpoint: {fname} [{ts}]'
                ))

    return files


# ── Main ─────────────────────────────────────────────────────────

print('=' * 54)
print('AE-Lite → GitHub uploader  (Colab T4)')
print(f'Repo        : {GITHUB_USERNAME}/{GITHUB_REPO}  [{GITHUB_BRANCH}]')
print(f'Source      : {DRIVE_BASE}')
print(f'Results  →  : {RESULTS_DEST}/')
print(f'Checkpts →  : {CHECKPOINTS_DEST}/')
print('=' * 54)

if not GITHUB_PAT:
    print('\n✗  GITHUB_PAT not found in Colab Secrets.')
    print('   Left sidebar → 🔑 Secrets → Add new secret')
    print('   Name: GITHUB_PAT    Value: your ghp_xxx token')
else:
    files = collect_files()
    if not files:
        print('\nNothing to upload.')
        print('Run Cell 4 (BER evaluation) and Cell 5 (plots) first.')
    else:
        # Summary before uploading
        results_files  = [f for f in files if f[1].startswith('results/')]
        ckpt_files     = [f for f in files if f[1].startswith('checkpoints/')]
        print(f'\nFound:')
        print(f'  {len(results_files)} result file(s)  → {RESULTS_DEST}/')
        print(f'  {len(ckpt_files)} checkpoint file(s) → {CHECKPOINTS_DEST}/')
        print(f'\nUploading {len(files)} file(s) total...\n')

        ok = 0
        for local_path, repo_path, message in files:
            if os.path.exists(local_path):
                if upload_file(local_path, repo_path, message):
                    ok += 1
            else:
                print(f'  - Not found, skipped: {local_path}')

        print(f'\nDone.  {ok}/{len(files)} files uploaded.')
        print(f'\nView results     : https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/{RESULTS_DEST}')
        print(f'View checkpoints : https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/{CHECKPOINTS_DEST}')
