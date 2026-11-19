#!/usr/bin/env bash
# Fails if Git tracks (or is about to track) anything that must never be committed:
# model weights, secrets, generated media, or unexpectedly large files.
# Usage: bash scripts/check_repo_hygiene.sh      (checks tracked + staged + untracked-not-ignored)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MAX_BYTES=$((2 * 1024 * 1024))   # 2 MB per file
fail=0

# Files Git would include: tracked, staged, and untracked files that are not ignored.
mapfile -t files < <(git ls-files --cached --others --exclude-standard | sort -u)

forbidden_ext='\.(safetensors|ckpt|pt|pth|bin|gguf|onnx|mp4|webm|mov|mkv|avi|gif|db|sqlite3?|pem|key)$'
forbidden_name='(^|/)(\.env(\..*)?|kaggle\.json|hf_token.*)$'

for f in "${files[@]}"; do
  [[ -f "$f" ]] || continue
  if [[ "$f" =~ $forbidden_ext ]]; then
    echo "FORBIDDEN FILE TYPE: $f"; fail=1
  fi
  if [[ "$f" =~ $forbidden_name && "$f" != ".env.example" && "$f" != */.env.example ]]; then
    echo "SECRET FILE: $f"; fail=1
  fi
  size=$(stat -c %s "$f")
  if (( size > MAX_BYTES )); then
    echo "TOO LARGE ($size bytes): $f"; fail=1
  fi
  # Token-like strings (Hugging Face, GitHub, Cloudflare/generic bearer, AWS keys).
  if grep -IEq '(hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)' "$f"; then
    echo "POSSIBLE SECRET IN: $f"; fail=1
  fi
done

# Notebooks must be committed with outputs cleared (outputs can leak tokens/URLs).
for f in "${files[@]}"; do
  if [[ "$f" == *.ipynb ]] && ! python3 -c '
import json, sys
nb = json.load(open(sys.argv[1]))
sys.exit(any(c.get("outputs") for c in nb.get("cells", [])))' "$f"; then
    echo "NOTEBOOK HAS OUTPUTS (clear them): $f"; fail=1
  fi
done

if (( fail )); then
  echo "Repo hygiene check FAILED"; exit 1
fi
echo "Repo hygiene check passed (${#files[@]} files checked)"
