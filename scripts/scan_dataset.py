import argparse
import json
from pathlib import Path

import numpy as np


def scan_npz(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    record = {
        "path": str(npz_path),
        "keys": list(data.files),
    }
    for key in ("poses", "trans", "betas", "incontact", "ground_contact_mask"):
        if key in data.files:
            value = data[key]
            record[f"{key}_shape"] = list(value.shape)
            record[f"{key}_dtype"] = str(value.dtype)
    parent_name = npz_path.parent.name
    if parent_name.endswith("_non_contact_sequences"):
        record["contact_bucket"] = "non_contact"
    elif parent_name.endswith("_contact_sequences"):
        record["contact_bucket"] = "contact"
    elif parent_name.endswith("_neutral_sequences"):
        record["contact_bucket"] = "neutral"
    else:
        record["contact_bucket"] = "unknown"
    record["sequence_group"] = parent_name
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    npz_files = sorted(dataset_root.glob("*/*.npz"))
    if args.limit > 0:
        npz_files = npz_files[:args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for npz_path in npz_files:
            try:
                record = scan_npz(npz_path)
            except Exception as exc:
                record = {"path": str(npz_path), "error": str(exc)}
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
