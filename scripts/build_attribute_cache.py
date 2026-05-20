import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit.mining import build_attribute_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest_records = [json.loads(line) for line in f if line.strip() and "error" not in line]
    if args.limit > 0:
        manifest_records = manifest_records[:args.limit]

    with open(args.output, "w", encoding="utf-8") as out_f:
        for record in manifest_records:
            if record.get("poses_shape") != [60, 156] or record.get("trans_shape") != [60, 3]:
                continue
            attribute_record = build_attribute_record(
                npz_path=record["path"],
                contact_bucket=record.get("contact_bucket", "unknown"),
                sequence_group=record.get("sequence_group", "unknown"),
            )
            out_f.write(json.dumps(attribute_record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
