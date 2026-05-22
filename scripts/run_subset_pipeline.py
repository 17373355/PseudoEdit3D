import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np

from pseudoedit3d.artifacts import JsonlArtifactManager, count_jsonl_records
from pseudoedit3d.config import load_simple_yaml
from pseudoedit3d.edit.mining import build_attribute_record, dump_jsonl, mine_triplets
from pseudoedit3d.training.train_stage1 import train_from_config


def scan_subset(dataset_root: Path, parent_glob: str) -> list[dict]:
    npz_files = sorted(dataset_root.glob(f"{parent_glob}/*.npz"))
    records = []
    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
            record = {
                "path": str(npz_path),
                "keys": list(data.files),
                "sequence_group": npz_path.parent.name,
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
        except Exception as exc:
            record = {"path": str(npz_path), "error": str(exc), "sequence_group": npz_path.parent.name}
        records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--parent-glob", required=True)
    parser.add_argument("--dataset-root", default="/mnt/data/home/guoruoxi/code/CharRet_multi/dataset")
    parser.add_argument("--project-root", default=str(ROOT_DIR))
    parser.add_argument("--base-config", default=str(ROOT_DIR / "configs" / "stage1_mined_cmu_hybrid.yaml"))
    parser.add_argument("--run-train", action="store_true")
    parser.add_argument("--mine-candidate-limit", type=int, default=64)
    parser.add_argument("--mine-max-pairs-per-clip", type=int, default=4)
    parser.add_argument("--mine-distance-threshold", type=float, default=2.5)
    parser.add_argument("--mine-min-delta-deg", type=float, default=8.0)
    parser.add_argument("--mine-max-delta-deg", type=float, default=45.0)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    dataset_root = Path(args.dataset_root)
    manager = JsonlArtifactManager.from_project_root(project_root)

    manifest_path = manager.artifact_path(args.subset, "scan", "manifest", split="full")
    attribute_cache_path = manager.artifact_path(args.subset, "attributes", "attribute_cache", split="full")
    mined_pairs_path = manager.artifact_path(args.subset, "mining", "mined_pairs", split="full")

    print(f"[1/4] scanning subset {args.subset} with parent_glob={args.parent_glob}")
    manifest_records = scan_subset(dataset_root=dataset_root, parent_glob=args.parent_glob)
    write_jsonl(manifest_path, manifest_records)
    manager.write_registry_entry(
        subset=args.subset,
        stage="scan",
        purpose="manifest",
        path=manifest_path,
        split="full",
        num_records=len(manifest_records),
        extra={"parent_glob": args.parent_glob},
    )
    print(f"manifest_records={len(manifest_records)} path={manifest_path}")

    print(f"[2/4] building attribute cache")
    attribute_records = []
    for record in manifest_records:
        if "error" in record:
            continue
        if record.get("poses_shape") != [60, 156] or record.get("trans_shape") != [60, 3]:
            continue
        attribute_records.append(
            build_attribute_record(
                npz_path=record["path"],
                contact_bucket=record.get("contact_bucket", "unknown"),
                sequence_group=record.get("sequence_group", "unknown"),
            )
        )
    dump_jsonl(attribute_cache_path, attribute_records)
    manager.write_registry_entry(
        subset=args.subset,
        stage="attributes",
        purpose="attribute_cache",
        path=attribute_cache_path,
        split="full",
        num_records=len(attribute_records),
    )
    print(f"attribute_records={len(attribute_records)} path={attribute_cache_path}")

    print(f"[3/4] mining pairs")
    mined_pairs = mine_triplets(
        records=attribute_records,
        min_delta_deg=args.mine_min_delta_deg,
        max_delta_deg=args.mine_max_delta_deg,
        candidate_limit=args.mine_candidate_limit,
        max_pairs_per_clip=args.mine_max_pairs_per_clip,
        distance_threshold=args.mine_distance_threshold,
    )
    dump_jsonl(mined_pairs_path, mined_pairs)
    manager.write_registry_entry(
        subset=args.subset,
        stage="mining",
        purpose="mined_pairs",
        path=mined_pairs_path,
        split="full",
        num_records=len(mined_pairs),
    )
    print(f"mined_pairs={len(mined_pairs)} path={mined_pairs_path}")

    if not args.run_train:
        print("[4/4] skip training")
        return

    print("[4/4] training")
    cfg = load_simple_yaml(args.base_config)
    cfg = replace(
        cfg,
        manifest_path=str(manifest_path),
        pair_manifest_path=str(mined_pairs_path),
        save_dir=str(project_root / "outputs" / args.subset / "stage1_mined_cmu_hybrid"),
    )
    train_result = train_from_config(cfg, checkpoint_name="cmu_first_pass_last.pt")
    manager.write_registry_entry(
        subset=args.subset,
        stage="training",
        purpose="checkpoint",
        path=train_result["checkpoint_path"],
        split="full",
        num_records=train_result["num_samples"],
        extra={
            "save_dir": train_result["save_dir"],
            "last_loss": train_result["last_loss"],
            "tensorboard_dir": train_result.get("tensorboard_dir"),
        },
    )
    print(
        f"checkpoint={train_result['checkpoint_path']} "
        f"last_loss={train_result['last_loss']} "
        f"tensorboard_dir={train_result.get('tensorboard_dir')}"
    )


if __name__ == "__main__":
    main()
