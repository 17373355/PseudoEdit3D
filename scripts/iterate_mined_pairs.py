import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.config import load_simple_yaml
from pseudoedit3d.edit.iterative import run_iterative_refinement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-config", default=str(ROOT_DIR / "configs" / "stage1_mined_upper_body.yaml"))
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--initial-keep-ratio", type=float, default=0.6)
    parser.add_argument("--refine-keep-ratio", type=float, default=0.6)
    parser.add_argument("--max-pairs-per-source", type=int, default=2)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--train-num-workers", type=int, default=None)
    parser.add_argument("--mine-min-delta-deg", type=float, default=8.0)
    parser.add_argument("--mine-max-delta-deg", type=float, default=45.0)
    parser.add_argument("--mine-candidate-limit", type=int, default=64)
    parser.add_argument("--mine-max-pairs-per-clip", type=int, default=4)
    parser.add_argument("--mine-distance-threshold", type=float, default=2.5)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.base_config)
    if args.train_epochs is not None:
        cfg.epochs = args.train_epochs
    if args.train_batch_size is not None:
        cfg.batch_size = args.train_batch_size
    if args.train_num_workers is not None:
        cfg.num_workers = args.train_num_workers

    summary = run_iterative_refinement(
        attribute_cache_path=args.attribute_cache,
        output_dir=args.output_dir,
        base_cfg=cfg,
        num_rounds=args.num_rounds,
        initial_keep_ratio=args.initial_keep_ratio,
        refine_keep_ratio=args.refine_keep_ratio,
        max_pairs_per_source=args.max_pairs_per_source,
        max_train_pairs=args.max_train_pairs,
        mine_min_delta_deg=args.mine_min_delta_deg,
        mine_max_delta_deg=args.mine_max_delta_deg,
        mine_candidate_limit=args.mine_candidate_limit,
        mine_max_pairs_per_clip=args.mine_max_pairs_per_clip,
        mine_distance_threshold=args.mine_distance_threshold,
    )
    print(f"raw_pairs={summary['num_raw_pairs']}")
    for round_info in summary["rounds"]:
        print(
            f"round={round_info['round_idx']} "
            f"input_pairs={round_info['input_pairs']} "
            f"refined_pairs={round_info['refined_pairs']} "
            f"last_loss={round_info['last_loss']}"
        )
    print(f"summary_path={summary['summary_path']}")


if __name__ == "__main__":
    main()
