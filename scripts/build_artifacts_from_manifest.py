import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.artifacts import JsonlArtifactManager, load_jsonl_records
from pseudoedit3d.edit.mining import build_attribute_record, dump_jsonl, mine_triplets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=str(ROOT_DIR))
    parser.add_argument("--mine-candidate-limit", type=int, default=64)
    parser.add_argument("--mine-max-pairs-per-clip", type=int, default=4)
    parser.add_argument("--mine-distance-threshold", type=float, default=2.5)
    parser.add_argument("--mine-min-delta-deg", type=float, default=8.0)
    parser.add_argument("--mine-max-delta-deg", type=float, default=45.0)
    args = parser.parse_args()

    manager = JsonlArtifactManager.from_project_root(args.project_root)
    records = load_jsonl_records(args.manifest)
    attribute_cache_path = manager.artifact_path(args.subset, "attributes", "attribute_cache", split="full")
    mined_pairs_path = manager.artifact_path(args.subset, "mining", "mined_pairs", split="full")

    attribute_records = []
    for record in records:
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
        extra={"source_manifest": args.manifest},
    )

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
        extra={"source_manifest": args.manifest},
    )

    print(f"attribute_cache={attribute_cache_path} num_records={len(attribute_records)}")
    print(f"mined_pairs={mined_pairs_path} num_records={len(mined_pairs)}")


if __name__ == "__main__":
    main()
