import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.artifacts import (
    JsonlArtifactManager,
    build_group_split,
    dump_jsonl_records,
    load_jsonl_records,
    split_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=str(ROOT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    records = load_jsonl_records(args.manifest)
    result = build_group_split(records, seed=args.seed, test_ratio=args.test_ratio)
    report = split_report(result)

    manager = JsonlArtifactManager.from_project_root(args.project_root)
    train_manifest = manager.artifact_path(f"{args.subset}-train", "scan", "manifest", split="full")
    test_manifest = manager.artifact_path(f"{args.subset}-test", "scan", "manifest", split="full")
    split_report_path = manager.artifact_path(args.subset, "splits", "group_split_report", split="full")

    dump_jsonl_records(train_manifest, result["train_records"])
    dump_jsonl_records(test_manifest, result["test_records"])
    with split_report_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=True, indent=2) + "\n")

    manager.write_registry_entry(
        subset=f"{args.subset}-train",
        stage="scan",
        purpose="manifest",
        path=train_manifest,
        split="full",
        num_records=len(result["train_records"]),
        extra={"seed": args.seed, "test_ratio": args.test_ratio, "source_manifest": args.manifest},
    )
    manager.write_registry_entry(
        subset=f"{args.subset}-test",
        stage="scan",
        purpose="manifest",
        path=test_manifest,
        split="full",
        num_records=len(result["test_records"]),
        extra={"seed": args.seed, "test_ratio": args.test_ratio, "source_manifest": args.manifest},
    )

    print(f"train_manifest={train_manifest}")
    print(f"test_manifest={test_manifest}")
    print(f"split_report={split_report_path}")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
