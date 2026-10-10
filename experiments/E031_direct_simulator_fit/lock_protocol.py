#!/usr/bin/env python3
"""Hash the prospective protocol and executable source before data generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("PROTOCOL.md", "prototype_fit.py", "generate_data.py", "fit_and_freeze.py",
         "plan_and_freeze.py", "evaluate.py", "lock_protocol.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    (HERE / "audit").mkdir(exist_ok=True)
    record = {"status": "prospective_source_lock", "fresh_data_generated": False,
        "test_targets_accessed": False,
        "source_sha256": {name: sha256(HERE / name) for name in FILES}}
    path = HERE / "audit/prerun_lock.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

