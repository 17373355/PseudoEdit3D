import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.data import MinedMotionEditDataset, MotionEditDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "mined"], default="synthetic")
    parser.add_argument("--manifest", default=str(ROOT_DIR / "data_manifest.jsonl"))
    parser.add_argument("--pair-manifest", default=str(ROOT_DIR / "mined_pairs_smoke.jsonl"))
    parser.add_argument("--dataset-root", default="/mnt/data/home/guoruoxi/code/CharRet_multi/dataset")
    parser.add_argument("--label-schema-path", default=str(ROOT_DIR / "configs" / "label_schema.yaml"))
    parser.add_argument("--prompt-style", default="template")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "mined":
        dataset = MinedMotionEditDataset(
            pair_manifest_path=args.pair_manifest,
            max_pairs=args.num_samples,
            label_schema_path=args.label_schema_path,
            prompt_style=args.prompt_style,
        )
    else:
        dataset = MotionEditDataset(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            max_clips=args.num_samples,
            label_schema_path=args.label_schema_path,
            prompt_style=args.prompt_style,
        )

    for idx in range(min(args.num_samples, len(dataset))):
        sample = dataset[idx]
        print(f"[{idx}] {sample['prompt_text']}")


if __name__ == "__main__":
    main()
