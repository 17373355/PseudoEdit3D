from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CASE_LIST_GLOB = "outputs/aml_regression_testset_v2/group_[0-9][0-9]_case_ids.txt"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _case_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _group_name(case_list: Path) -> str:
    stem = case_list.stem
    if stem.endswith("_case_ids"):
        stem = stem[: -len("_case_ids")]
    return stem


def _run(cmd: list[str], cwd: Path = ROOT_DIR) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _canonical_ids(case: dict[str, Any]) -> list[str]:
    out = []
    for action in case.get("canonical_actions") or []:
        family = action.get("semantic_family") or {}
        slots = action.get("slots") or {}
        if isinstance(family, dict) and family.get("probe_visible") is False:
            continue
        if isinstance(slots, dict) and slots.get("hidden_by_semantic_family"):
            continue
        out.append(str(action.get("canonical_id") or action.get("family_id") or "UNKNOWN"))
    return out


def _sanity_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    return {str(row["case_id"]): row for row in payload.get("rows") or []}


def _gif_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _load_json(path)
    return {str(row["case_id"]): row for row in rows}


def _rel_link(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def write_group_index(
    group_dir: Path,
    group_name: str,
    case_list: Path,
    probe_summary: Path,
    gif_dir: Path,
    sanity_json: Path,
) -> dict[str, Any]:
    cases = _load_json(probe_summary) if probe_summary.exists() else []
    gifs = _gif_rows(gif_dir / "summary.json")
    sanity = _sanity_rows(sanity_json)
    lines = [
        f"# AML MoMask Review Pack - {group_name}",
        "",
        "GT motion vs motion-only AutoPrompt-conditioned MoMask. HML3D captions are reference only.",
        "",
        f"- case list: `{_rel_link(case_list, group_dir)}`",
        f"- probe summary: `{_rel_link(probe_summary, group_dir)}`",
        f"- GIF dir: `{_rel_link(gif_dir, group_dir)}`",
        f"- kinematic sanity: `{_rel_link(sanity_json, group_dir)}`",
        "",
        "| # | case | HML3D reference | AutoPrompt | words | canonical ids | flags | GIF |",
        "| ---: | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    flag_counts: dict[str, int] = {}
    gif_count = 0
    for idx, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        prompt = str(case.get("auto_prompt") or "")
        words = len(prompt.split())
        flags = sanity.get(case_id, {}).get("flags") or []
        if not flags:
            flag_text = "ok"
        else:
            for flag in flags:
                flag_counts[str(flag)] = flag_counts.get(str(flag), 0) + 1
            flag_text = ", ".join(str(flag) for flag in flags)
        gif_path = gif_dir / f"case_{case_id}.gif"
        gif_link = f"[gif]({_rel_link(gif_path, group_dir)})" if gif_path.exists() else "missing"
        if gif_path.exists():
            gif_count += 1
        canonical = ", ".join(_canonical_ids(case)[:12])
        if len(_canonical_ids(case)) > 12:
            canonical += ", ..."
        lines.append(
            "| {idx} | `{case_id}` | {hml} | {prompt} | {words} | `{canonical}` | {flags} | {gif} |".format(
                idx=idx,
                case_id=case_id,
                hml=str(case.get("gt_prompt") or "").replace("|", "/"),
                prompt=prompt.replace("|", "/"),
                words=words,
                canonical=canonical,
                flags=flag_text,
                gif=gif_link,
            )
        )
    group_dir.mkdir(parents=True, exist_ok=True)
    index_path = group_dir / "index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "group": group_name,
        "case_list": str(case_list),
        "num_cases": len(cases),
        "gif_count": gif_count,
        "index": str(index_path),
        "probe_summary": str(probe_summary),
        "gif_dir": str(gif_dir),
        "sanity_json": str(sanity_json) if sanity_json.exists() else None,
        "flag_counts": sorted(flag_counts.items()),
    }


def write_master_index(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    total_cases = sum(int(row["num_cases"]) for row in rows)
    total_gifs = sum(int(row["gif_count"]) for row in rows)
    flag_counts: dict[str, int] = {}
    for row in rows:
        for flag, count in row.get("flag_counts") or []:
            flag_counts[str(flag)] = flag_counts.get(str(flag), 0) + int(count)
    lines = [
        "# AML MoMask Review Pack",
        "",
        "GT motion vs motion-only AutoPrompt-conditioned MoMask review bundle.",
        "",
        f"- review name: `{args.review_name}`",
        f"- prompt mode: `{args.prompt_mode}`",
        f"- max events: `{args.max_events}`",
        f"- time steps: `{args.time_steps}`",
        f"- cond scale: `{args.cond_scale}`",
        f"- caption semantic aliases: `{bool(args.caption_semantic_aliases)}`",
        f"- caption alias source: `{args.caption_alias_source if args.caption_semantic_aliases else 'none'}`",
        f"- cases: `{total_cases}`",
        f"- GIFs: `{total_gifs}`",
        f"- generation skipped: `{bool(args.skip_generation)}`",
        "",
        "## Groups",
        "",
        "| group | cases | gifs | flags | index |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        flags = ", ".join(f"{flag}:{count}" for flag, count in row.get("flag_counts") or []) or "none"
        lines.append(
            f"| `{row['group']}` | {row['num_cases']} | {row['gif_count']} | {flags} | "
            f"[index]({_rel_link(Path(row['index']), output_root)}) |"
        )
    if flag_counts:
        lines.extend(["", "## Aggregate Flags", "", "| flag | count |", "| --- | ---: |"])
        for flag, count in sorted(flag_counts.items()):
            lines.append(f"| {flag} | {count} |")
    lines.extend(
        [
            "",
            "## Manual Review Notes",
            "",
            "- Check whether the AutoPrompt preserves the main action semantics visible in GT.",
            "- Treat HML3D captions as reference text only; they are not used to generate AML labels.",
            "- Mark cases where the prompt is semantically wrong separately from cases where MoMask fails to realize a reasonable prompt.",
        ]
    )
    (output_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output_root / "review_manifest.json", {"args": vars(args), "groups": rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-list", action="append", default=[])
    parser.add_argument("--case-list-glob", default=DEFAULT_CASE_LIST_GLOB)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--review-name", default="aml_review250_semantic_v1")
    parser.add_argument("--ext-prefix", default="aml_review250_semantic_v1")
    parser.add_argument("--only-groups", default="")
    parser.add_argument("--prompt-mode", choices=["coarse", "event_stream"], default="coarse")
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--cond-scale", type=int, default=4)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--caption-semantic-aliases",
        action="store_true",
        help="Use HML3D captions only to name compatible AML geometry patterns during AutoPrompt generation.",
    )
    parser.add_argument(
        "--caption-alias-source",
        choices=["first", "all"],
        default="first",
        help="Caption source for --caption-semantic-aliases.",
    )
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--skip-kinematic", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-render-frames", type=int, default=None)
    args = parser.parse_args()

    if args.case_list:
        case_lists = [Path(p) for p in args.case_list]
    else:
        case_lists = sorted(ROOT_DIR.glob(args.case_list_glob))
    if args.only_groups:
        selected = {item.strip() for item in args.only_groups.split(",") if item.strip()}
        case_lists = [path for path in case_lists if _group_name(path) in selected]
    if not case_lists:
        raise SystemExit("No case lists found")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_list in case_lists:
        group_name = _group_name(case_list)
        group_dir = output_root / group_name
        probe_dir = group_dir / "probe"
        gif_dir = group_dir / "gifs"
        sanity_json = group_dir / "kinematic_sanity.json"
        sanity_md = group_dir / "kinematic_sanity.md"
        probe_summary = probe_dir / "summary.json"

        print(f"=== {group_name}: {len(_case_ids(case_list))} cases ===", flush=True)
        probe_cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "run_momask_aml_autoprompt_probe.py"),
            "--case-list",
            str(case_list),
            "--output-dir",
            str(probe_dir),
            "--max-events",
            str(args.max_events),
            "--prompt-mode",
            args.prompt_mode,
            "--ext-prefix",
            f"{args.ext_prefix}_{group_name}",
            "--gpu-id",
            str(args.gpu_id),
            "--time-steps",
            str(args.time_steps),
            "--cond-scale",
            str(args.cond_scale),
        ]
        if args.skip_generation:
            probe_cmd.append("--skip-generation")
        if args.reuse_existing:
            probe_cmd.append("--reuse-existing")
        if args.caption_semantic_aliases:
            probe_cmd.append("--caption-semantic-aliases")
            probe_cmd.extend(["--caption-alias-source", args.caption_alias_source])
        _run(probe_cmd)

        if not args.skip_visualization:
            vis_cmd = [
                sys.executable,
                str(ROOT_DIR / "scripts" / "visualize_momask_auto_gt.py"),
                "--summary",
                str(probe_summary),
                "--output-dir",
                str(gif_dir),
                "--fps",
                str(args.fps),
                "--frame-stride",
                str(args.frame_stride),
                "--show-hml3d-reference",
            ]
            if args.max_render_frames:
                vis_cmd.extend(["--max-render-frames", str(args.max_render_frames)])
            _run(vis_cmd)

        if not args.skip_kinematic:
            sanity_cmd = [
                sys.executable,
                str(ROOT_DIR / "scripts" / "analyze_momask_probe_kinematics.py"),
                "--summary",
                str(probe_summary),
                "--output",
                str(sanity_json),
                "--report",
                str(sanity_md),
            ]
            _run(sanity_cmd)

        rows.append(write_group_index(group_dir, group_name, case_list, probe_summary, gif_dir, sanity_json))
        write_master_index(output_root, rows, args)

    write_master_index(output_root, rows, args)
    print(f"review_index={output_root / 'index.md'}", flush=True)
    print(f"review_manifest={output_root / 'review_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
