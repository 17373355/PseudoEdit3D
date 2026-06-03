import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.config import load_simple_yaml
from pseudoedit3d.inference import run_mined_case_inference
from pseudoedit3d.inference.predict import run_prefix_case_inference, select_case_indices, select_prefix_case_indices
from pseudoedit3d.visualization import export_case_gif, export_case_summary


def parse_indices(indices_text: str, limit: int) -> list[int]:
    if not indices_text:
        return []
    return [int(x.strip()) for x in indices_text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair-manifest", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--selection", choices=["first", "salient"], default="salient")
    parser.add_argument("--min-delta-deg", type=float, default=12.0)
    parser.add_argument("--min-duration-frames", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-limit", type=int, default=60)
    parser.add_argument("--smplh-model-path", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    explicit_indices = parse_indices(args.indices, args.num_cases)
    cfg = load_simple_yaml(args.config)
    if cfg.data_mode == "prefix":
        selection = "first" if cfg.prefix_task_mode == "continue" and args.selection == "salient" else args.selection
        manifest_path = args.manifest or cfg.manifest_path
        case_indices = select_prefix_case_indices(
            manifest_path=manifest_path,
            num_cases=args.num_cases,
            selection=selection,
            explicit_indices=explicit_indices,
        )
        results = run_prefix_case_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            manifest_path=manifest_path,
            case_indices=case_indices[: args.num_cases],
            device=args.device,
        )
    else:
        case_indices = select_case_indices(
            pair_manifest_path=args.pair_manifest,
            num_cases=args.num_cases,
            selection=args.selection,
            explicit_indices=explicit_indices,
            min_delta_deg=args.min_delta_deg,
            min_duration_frames=args.min_duration_frames,
        )
        results = run_mined_case_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            pair_manifest_path=args.pair_manifest,
            case_indices=case_indices[: args.num_cases],
            device=args.device,
        )

    for result in results:
        gif_path = output_dir / f"case_{result['case_idx']:04d}.gif"
        result["gif_path"] = export_case_gif(
            result,
            str(gif_path),
            smplh_model_path=args.smplh_model_path,
            fps=args.fps,
            frame_limit=args.frame_limit,
        )
        print(f"saved_gif={result['gif_path']}")

    summary_path = output_dir / "summary.jsonl"
    export_case_summary(results, str(summary_path))
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
