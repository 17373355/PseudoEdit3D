import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit.mining import dump_jsonl, load_jsonl, mine_triplets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-delta-deg", type=float, default=8.0)
    parser.add_argument("--max-delta-deg", type=float, default=45.0)
    parser.add_argument("--candidate-limit", type=int, default=64)
    parser.add_argument("--max-pairs-per-clip", type=int, default=4)
    parser.add_argument("--distance-threshold", type=float, default=2.5)
    args = parser.parse_args()

    records = load_jsonl(args.attribute_cache)
    pairs = mine_triplets(
        records=records,
        min_delta_deg=args.min_delta_deg,
        max_delta_deg=args.max_delta_deg,
        candidate_limit=args.candidate_limit,
        max_pairs_per_clip=args.max_pairs_per_clip,
        distance_threshold=args.distance_threshold,
    )
    dump_jsonl(args.output, pairs)
    print(f"mined_pairs={len(pairs)}")


if __name__ == "__main__":
    main()
