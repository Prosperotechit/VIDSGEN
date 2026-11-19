#!/usr/bin/env python3
"""Download the Wan 2.2 TI2V-5B model files for ComfyUI, from workflows/comfyui/models.yaml.

SAFE BY DEFAULT: without --execute this is a dry run. It only prints which files would be
downloaded, their sizes and where they would go. Nothing is downloaded and no network
connection is made.

    python scripts/download_models.py --comfyui-dir /path/to/ComfyUI            # dry run
    python scripts/download_models.py --comfyui-dir /path/to/ComfyUI --execute  # download (~18 GB)

--execute needs `pip install huggingface_hub`. An HF token is optional for these public
repositories; set HF_TOKEN in the environment (e.g. a Kaggle Secret) to avoid rate limits.
Never pass tokens on the command line. Files that already exist with the expected size are
skipped. Weights must never be committed to Git.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "workflows" / "comfyui" / "models.yaml"


def load_manifest(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")
    data = yaml.safe_load(path.read_text())
    files = data.get("files") or []
    for f in files:
        for key in ("role", "repo", "path", "comfyui_dir", "size_bytes"):
            if key not in f:
                sys.exit(f"Manifest entry is missing '{key}': {f}")
    return data


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def plan(manifest: dict, comfyui_dir: Path) -> list[dict]:
    rows = []
    for f in manifest["files"]:
        dest = comfyui_dir / f["comfyui_dir"] / Path(f["path"]).name
        size = dest.stat().st_size if dest.is_file() else None
        status = "present" if size == f["size_bytes"] else ("wrong size" if size is not None else "missing")
        rows.append({**f, "dest": dest, "status": status})
    return rows


def download(rows: list[dict]) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("--execute needs huggingface_hub: pip install huggingface_hub")
    for r in rows:
        if r["status"] == "present":
            print(f"skip  {r['dest']} (already present, size matches)")
            continue
        r["dest"].parent.mkdir(parents=True, exist_ok=True)
        staging = r["dest"].parent / ".vidsgen-download"
        print(f"get   {r['repo']}/{r['path']} -> {r['dest']} ({gb(r['size_bytes'])})")
        local = Path(hf_hub_download(repo_id=r["repo"], filename=r["path"], local_dir=staging))
        if local.stat().st_size != r["size_bytes"]:
            sys.exit(f"Size mismatch for {local}: {local.stat().st_size} != {r['size_bytes']}. Not installing it.")
        shutil.move(str(local), r["dest"])
        shutil.rmtree(staging, ignore_errors=True)
        print(f"ok    {r['dest']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfyui-dir", type=Path, default=Path("ComfyUI"),
                    help="ComfyUI installation (models go into <dir>/models/...). Default: ./ComfyUI")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--execute", action="store_true", help="Actually download. Without it: dry run only.")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    comfyui_dir = args.comfyui_dir.resolve()
    rows = plan(manifest, comfyui_dir)
    todo = [r for r in rows if r["status"] != "present"]

    print(f"Model: {manifest.get('model')}   manifest: {args.manifest}")
    print(f"ComfyUI directory: {comfyui_dir}\n")
    for r in rows:
        print(f"  [{r['status']:>10}] {r['role']:<16} {gb(r['size_bytes']):>8}  {r['repo']}/{r['path']}")
        print(f"               -> {r['dest']}")
    total = sum(r["size_bytes"] for r in todo)
    print(f"\nTo download: {len(todo)} file(s), {gb(total)}")

    if not args.execute:
        print("\nDRY RUN: nothing was downloaded. Re-run with --execute to download.")
        return 0
    if not todo:
        print("All model files are already present.")
        return 0
    download(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
