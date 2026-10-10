#!/usr/bin/env python3
"""Build a metadata-only catalog from pinned TartanDrive 2.0 public indexes."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
COMMIT = "fe893b9b4132bfbd30ba84a839872b3a855e8176"
INDEX_URL = f"https://raw.githubusercontent.com/castacks/tartan_drive_2.0/{COMMIT}/assets/files.yaml"
INDEX_SHA256 = "7da5d791b163c40501a4e047e9da57b097dc1b2e1b41f4b6a353e93b1b61f1e1"
OBJECT_BASE = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartandrive2"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main() -> None:
    raw_index = get(INDEX_URL)
    if hashlib.sha256(raw_index).hexdigest() != INDEX_SHA256:
        raise RuntimeError("upstream index does not match pinned hash")
    index = yaml.safe_load(raw_index)
    kitti = index["kitti"]
    kitti_by_run = {
        key.rstrip("/").split("/")[-1]: value
        for key, value in kitti.items()
    }
    runs = []
    failures = []
    for bag_key in sorted(index["bags"]):
        run = bag_key.rstrip("/").split("/")[-1]
        try:
            metadata = yaml.safe_load(get(f"{OBJECT_BASE}/{bag_key}info.yaml"))
        except Exception as error:  # catalog records inaccessible entries
            failures.append({"run": run, "error": type(error).__name__})
            continue
        kitti_entry = kitti_by_run.get(run, {})
        modalities = sorted(
            key.rstrip("/").split("/")[-1]
            for key in kitti_entry
            if key != "files"
        )
        pre = metadata.get("pre") or {}
        runs.append({
            "run": run,
            "duration_seconds": metadata.get("duration"),
            "average_speed_mps": metadata.get("average_speed"),
            "top_speed_mps": metadata.get("top_speed"),
            "course": pre.get("course", []),
            "conditions": pre.get("conditions", []),
            "context": pre.get("context", []),
            "driver": pre.get("driver"),
            "robot": pre.get("robot"),
            "sensors": metadata.get("sensors", []),
            "kitti_available": bool(kitti_entry),
            "modalities": modalities,
        })
    catalog = {
        "status": "metadata_only_pre_modeling_catalog",
        "upstream_repository": "https://github.com/castacks/tartan_drive_2.0",
        "upstream_commit": COMMIT,
        "index_url": INDEX_URL,
        "index_sha256": INDEX_SHA256,
        "run_count": len(runs),
        "failure_count": len(failures),
        "total_duration_seconds": sum(float(item["duration_seconds"] or 0) for item in runs),
        "runs": runs,
        "failures": failures,
    }
    (HERE / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(json.dumps({key: catalog[key] for key in ("run_count", "failure_count", "total_duration_seconds")}))


if __name__ == "__main__":
    main()
