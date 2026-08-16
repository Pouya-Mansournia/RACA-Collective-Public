"""Lightweight structural smoke test over every committed experiment summary JSON file.

This is NOT a full audit that every number in every markdown phase report matches its source
JSON (that would be a much bigger undertaking, out of scope here). It is a cheap regression
check: confirm every `experiments/phase_*/summary*.json` and
`experiments/phase_*/results/*summary*.json` file (a) is present, (b) parses as valid JSON, and
(c) is non-empty (a dict with at least one key, or a non-empty list). This catches the concrete
regression this check is meant to catch -- someone later corrupts a summary file, forgets to
commit one, or a run script writes an empty/malformed file -- without trying to verify every
individual number against its report.

Usage: python3 -m analysis.check_summary_json
"""
from __future__ import annotations

import glob
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def find_summary_files():
    patterns = [
        "experiments/phase_*/summary*.json",
        "experiments/phase_*/results/*summary*.json",
    ]
    files = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(REPO_ROOT, pattern)):
            files.add(os.path.relpath(path, REPO_ROOT))
    return sorted(files)


def check_file(rel_path):
    abs_path = os.path.join(REPO_ROOT, rel_path)
    try:
        with open(abs_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return f"FAIL  {rel_path}: {exc}"

    if isinstance(data, dict):
        if not data:
            return f"FAIL  {rel_path}: parses but is an empty dict (no keys)"
    elif isinstance(data, list):
        if not data:
            return f"FAIL  {rel_path}: parses but is an empty list"
    else:
        return f"FAIL  {rel_path}: top-level JSON is neither an object nor an array"

    return f"OK    {rel_path}"


def run():
    files = find_summary_files()
    if not files:
        print("FAIL  no experiments/phase_*/*summary*.json files found at all")
        return 1

    results = [check_file(f) for f in files]
    for r in results:
        print(r)

    n_fail = sum(1 for r in results if r.startswith("FAIL"))
    print(f"\n{len(files)} summary JSON files checked, {n_fail} failed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(run())
