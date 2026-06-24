"""Render static review sheets for the v4 closure draft pattern forest.

This renderer is for human review only. It shows GT HumanML3D keyframes for
the source examples attached to each closure-family node. It does not change
the pattern forest, Motion-BPE, or AML runtime logic.

Typical use:
    python scripts/render_v4_closure_pattern_forest_review_pack.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.visualization.skeleton_gif import (  # noqa: E402
    _draw_skeleton,
    _load_font,
    _normalize_points,
    _project_points,
    _wrap_text,
)


DEFAULT_FOREST = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_forest_v1_from_v4_closure_draft/"
    "aml_pattern_forest_v1_draft.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_forest_v1_from_v4_closure_review_pack"
)
DEFAULT_HML3D_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_gt_pack(hml3d_root: Path) -> dict[str, Any]:
    return torch.load(hml3d_root / "joints3d.pth", map_location="cpu")


def _case_key(case_id: str) -> str:
    return f"{case_id}.npy"


def _gt_joints(pack: dict[str, Any], case_id: str) -> np.ndarray | None:
    item = pack.get(_case_key(case_id))
    if item is None:
        return None
    joints = item.get("joints3d") if isinstance(item, dict) else item
    if hasattr(joints, "cpu"):
        joints = joints.cpu().numpy()
    return np.asarray(joints, dtype=np.float32)


def _caption_lines(hml3d_root: Path, case_id: str, fallback: str = "", limit: int = 3) -> list[str]:
    path = hml3d_root / "texts" / f"{case_id}.txt"
    lines: list[str] = []
    if path.exists():
        seen: set[str] = set()
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = raw.split("#", 1)[0].strip()
            if text and text not in seen:
                lines.append(text)
                seen.add(text)
            if len(lines) >= limit:
                break
    if not lines and fallback:
        lines.append(fallback)
    return lines[:limit]


def _child_index(forest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    nodes = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = nodes.get(str(edge.get("child_node_id") or ""))
        if parent and child:
            children[parent].append(child)
    for rows in children.values():
        rows.sort(key=_node_sort_key)
    return children


def _node_sort_key(node: dict[str, Any]) -> tuple[int, int, float, str]:
    evidence = node.get("evidence") or {}
    status = str(node.get("status") or "")
    status_rank = {
        "review_candidate": 0,
        "promote_review": 0,
        "split_required": 1,
        "composition_needs_closure": 2,
        "composition_review": 3,
    }.get(status, 9)
    blocker_count = len(evidence.get("promotion_blockers") or [])
    return (
        status_rank,
        blocker_count,
        -float(evidence.get("support_cases") or evidence.get("support_cases_max") or 0),
        str(node.get("node_id") or ""),
    )


def _family_nodes(forest: dict[str, Any], statuses: set[str]) -> list[dict[str, Any]]:
    rows = [node for node in forest.get("nodes") or [] if node.get("node_kind") == "pattern_family_candidate"]
    if statuses:
        rows = [node for node in rows if str(node.get("status") or "") in statuses]
    return sorted(rows, key=_node_sort_key)


def _frame_indices(num_frames: int, span: list[int] | None, count: int) -> list[int]:
    if num_frames <= 0:
        return []
    if span and len(span) >= 2:
        start = max(0, min(num_frames - 1, int(span[0])))
        end = max(start, min(num_frames - 1, int(span[1])))
    else:
        start, end = 0, num_frames - 1
    if count <= 1 or start == end:
        return [start]
    indices = np.linspace(start, end, count, dtype=np.int32).tolist()
    out: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _normalise_frames(joints: np.ndarray, indices: list[int], width: int, height: int) -> list[np.ndarray]:
    frames = joints[indices]
    points2d = _normalize_points(_project_points(frames), width=width, height=height, margin=18)
    return [points2d[i] for i in range(points2d.shape[0])]


def _draw_frame_strip(
    draw: ImageDraw.ImageDraw,
    joints: np.ndarray,
    indices: list[int],
    *,
    x: int,
    y: int,
    frame_w: int,
    frame_h: int,
) -> None:
    panels = _normalise_frames(joints, indices, frame_w, frame_h)
    for panel_idx, points in enumerate(panels):
        ox = x + panel_idx * frame_w
        draw.rectangle((ox, y, ox + frame_w - 1, y + frame_h - 1), outline=(218, 221, 228), fill=(255, 255, 255))
        shifted = points.copy()
        shifted[:, 0] += ox
        shifted[:, 1] += y
        _draw_skeleton(
            draw,
            shifted,
            base_color=(150, 156, 170),
            highlight_color=(32, 115, 90),
            highlight_joints=set(),
            highlight_edges=set(),
            radius=2,
            width=2,
        )
        draw.text((ox + 4, y + frame_h - 15), f"f{indices[panel_idx] + 1}", fill=(92, 96, 108), font=_load_font(9))


def _unique_examples(source_nodes: list[dict[str, Any]], *, max_sources: int, examples_per_source: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in source_nodes[:max_sources]:
        evidence = source.get("evidence") or {}
        for example in (source.get("source_examples") or [])[: max(1, examples_per_source * 3)]:
            case_id = str(example.get("case_id") or "")
            caption = str(example.get("caption") or "")
            key = (case_id, caption)
            if not case_id or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source": source,
                    "source_candidate_id": evidence.get("candidate_id"),
                    "example": example,
                }
            )
            source_seen = sum(1 for row in rows if row["source"] is source)
            if source_seen >= examples_per_source:
                break
    return rows


def _aliases_text(node: dict[str, Any], limit: int = 6) -> str:
    return ", ".join(f"{item.get('id')}:{item.get('count')}" for item in (node.get("language_aliases") or [])[:limit])


def _roles_text(node: dict[str, Any], limit: int = 8) -> str:
    motion = node.get("motion_summary") or {}
    roles = motion.get("canonical_role_items") or []
    if roles and isinstance(roles[0], dict):
        role_ids = [str(item.get("id") or "") for item in roles[:limit]]
    else:
        role_ids = [str(item) for item in roles[:limit]]
    return " + ".join(role_ids)


def _blockers_text(source: dict[str, Any]) -> str:
    evidence = source.get("evidence") or {}
    blockers = [str(item) for item in evidence.get("promotion_blockers") or []]
    return ", ".join(blockers) if blockers else "none"


def render_family_sheet(
    family: dict[str, Any],
    source_nodes: list[dict[str, Any]],
    *,
    gt_pack: dict[str, Any],
    hml3d_root: Path,
    output_path: Path,
    max_sources: int,
    examples_per_source: int,
    frames_per_example: int,
) -> dict[str, Any]:
    examples = _unique_examples(source_nodes, max_sources=max_sources, examples_per_source=examples_per_source)
    row_h = 186
    canvas_w = 1700
    header_h = 154
    canvas_h = max(header_h + row_h, header_h + row_h * max(1, len(examples)))
    img = Image.new("RGB", (canvas_w, canvas_h), color=(247, 248, 251))
    draw = ImageDraw.Draw(img)
    font_title = _load_font(22)
    font_body = _load_font(13)
    font_small = _load_font(10)
    font_caption = _load_font(11)

    evidence = family.get("evidence") or {}
    motion = family.get("motion_summary") or {}
    title = f"{family.get('node_id')}  [{family.get('status')}]"
    draw.text((24, 16), title, fill=(28, 32, 40), font=font_title)
    draw.text(
        (24, 48),
        f"name={family.get('accepted_name')}  scope={family.get('scope')}  sources={evidence.get('source_candidate_count')}  support_max={evidence.get('support_cases_max')}",
        fill=(58, 64, 78),
        font=font_body,
    )
    draw.text((24, 72), "aliases: " + (_aliases_text(family) or "none"), fill=(92, 72, 45), font=font_small)
    draw.text((24, 94), "roles: " + (_roles_text(family, limit=12) or "none"), fill=(42, 82, 110), font=font_small)
    blocker_counts = evidence.get("promotion_blocker_counts") or {}
    draw.text((24, 116), "family blockers: " + json.dumps(blocker_counts, ensure_ascii=True), fill=(120, 64, 58), font=font_small)

    frame_w = 96
    frame_h = 132
    strip_x = 24
    text_x = strip_x + frame_w * frames_per_example + 22
    max_text_w = canvas_w - text_x - 28
    saved_examples: list[dict[str, Any]] = []

    for row_idx, row in enumerate(examples):
        y = header_h + row_idx * row_h
        draw.rectangle((16, y + 8, canvas_w - 16, y + row_h - 8), outline=(224, 227, 235), fill=(255, 255, 255))
        source = row["source"]
        example = row["example"]
        source_evidence = source.get("evidence") or {}
        case_id = str(example.get("case_id") or "")
        joints = _gt_joints(gt_pack, case_id)
        span = example.get("span") or None
        if joints is not None:
            indices = _frame_indices(len(joints), span, frames_per_example)
            _draw_frame_strip(draw, joints, indices, x=strip_x, y=y + 26, frame_w=frame_w, frame_h=frame_h)
        else:
            indices = []
            draw.rectangle((strip_x, y + 26, strip_x + frame_w * frames_per_example - 1, y + 26 + frame_h), outline=(224, 227, 235), fill=(250, 250, 252))
            draw.text((strip_x + 12, y + 74), "missing joints", fill=(120, 65, 65), font=font_body)

        captions = _caption_lines(hml3d_root, case_id, str(example.get("caption") or ""))
        meta = (
            f"{case_id}  {source_evidence.get('candidate_id')}  {source.get('status')}  "
            f"support={source_evidence.get('support_cases')}  span={span}  blockers={_blockers_text(source)}"
        )
        draw.text((text_x, y + 20), meta, fill=(35, 40, 52), font=font_body)
        draw.text((text_x, y + 42), "roles: " + (_roles_text(source) or "none"), fill=(42, 82, 110), font=font_small)
        draw.text((text_x, y + 62), "aliases: " + (_aliases_text(source) or "none"), fill=(92, 72, 45), font=font_small)
        text_y = y + 82
        for idx, caption in enumerate(captions, start=1):
            for line in _wrap_text(draw, f"{idx}. {caption}", font_caption, max_text_w):
                if text_y > y + row_h - 22:
                    break
                draw.text((text_x, text_y), line, fill=(45, 83, 185), font=font_caption)
                text_y += 14
            text_y += 2
        saved_examples.append(
            {
                "case_id": case_id,
                "source_candidate_id": row.get("source_candidate_id"),
                "span": span,
                "frame_indices": indices,
                "captions": captions,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return {
        "family_id": family.get("node_id"),
        "status": family.get("status"),
        "scope": family.get("scope"),
        "name_candidate": family.get("accepted_name"),
        "language_aliases": family.get("language_aliases") or [],
        "evidence": family.get("evidence") or {},
        "motion_summary": family.get("motion_summary") or {},
        "image_path": str(output_path),
        "example_count": len(saved_examples),
        "examples": saved_examples,
    }


def write_index(path: Path, records: list[dict[str, Any]], source_forest: Path) -> None:
    lines: list[str] = [
        "# v4 Closure Pattern Forest Review Pack",
        "",
        f"- source forest: `{source_forest}`",
        "- each image shows GT HumanML3D keyframes, not MoMask generations.",
        "- captions are reference text for human review; they are not structural evidence.",
        "",
    ]
    for row in records:
        image_name = Path(str(row["image_path"])).name
        lines.extend(
            [
                f"## {row['family_id']}",
                "",
                f"- status: `{row['status']}`",
                f"- scope: `{row['scope']}`",
                f"- examples: `{row['example_count']}`",
                "",
                f"![{row['family_id']}](families/{image_name})",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _review_options(status: str) -> list[str]:
    if status == "review_candidate":
        return ["promote", "split", "merge", "downgrade_to_component"]
    if status == "split_required":
        return ["split_axis_confirmed", "merge_with_existing", "downgrade_to_component", "needs_new_observable"]
    if status == "composition_needs_closure":
        return ["needs_closure", "merge_with_review_candidate", "downgrade_to_component", "discard"]
    return ["keep_for_review", "downgrade_to_component", "discard"]


def _review_focus(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "")
    aliases = row.get("language_aliases") or []
    top_alias = str((aliases[0] or {}).get("id") or "") if aliases else ""
    if status == "review_candidate":
        return "Check whether the examples share one complete motion pattern, not just a reusable component."
    if status == "split_required" and top_alias in {"swim_like_motion", "fly_like_motion"}:
        return "Check whether this is floor/prone swimming/flying rather than inverted acrobatics."
    if status == "split_required":
        return "Check which structural axis should split this family before promotion."
    if status == "composition_needs_closure":
        return "Check whether one missing role should be added to close the pattern or whether this is only a component."
    return "Check whether this should remain a diagnostic candidate."


def write_review_queue(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# v4 Closure Pattern Forest Review Queue",
        "",
        "Fill decisions after looking at the PNG sheets. The default decision is `pending`.",
        "",
        "| family | status | name | options | focus | image |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in records:
        image = Path(str(row.get("image_path") or "")).name
        options = ", ".join(_review_options(str(row.get("status") or "")))
        focus = _review_focus(row)
        lines.append(
            f"| `{row.get('family_id')}` | `{row.get('status')}` | `{row.get('name_candidate')}` | `{options}` | {focus} | `families/{image}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def review_decision_template(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "v4_closure_pattern_forest_review_decisions_v1",
        "instructions": "Set decision after visual review. Captions are naming hints only; judge whether the motion examples share a stable pattern.",
        "decisions": [
            {
                "family_id": row.get("family_id"),
                "status": row.get("status"),
                "scope": row.get("scope"),
                "name_candidate": row.get("name_candidate"),
                "decision": "pending",
                "allowed_decisions": _review_options(str(row.get("status") or "")),
                "review_focus": _review_focus(row),
                "image_path": row.get("image_path"),
                "example_case_ids": [example.get("case_id") for example in row.get("examples") or []],
                "notes": "",
            }
            for row in records
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forest", type=Path, default=DEFAULT_FOREST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hml3d-root", type=Path, default=DEFAULT_HML3D_ROOT)
    parser.add_argument("--statuses", default="review_candidate,split_required,composition_needs_closure")
    parser.add_argument("--max-families", type=int, default=16)
    parser.add_argument("--max-source-candidates-per-family", type=int, default=8)
    parser.add_argument("--examples-per-source", type=int, default=2)
    parser.add_argument("--frames-per-example", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    families_dir = output_dir / "families"
    output_dir.mkdir(parents=True, exist_ok=True)
    families_dir.mkdir(parents=True, exist_ok=True)

    forest = _read_json(args.forest)
    statuses = {item.strip() for item in str(args.statuses).split(",") if item.strip()}
    families = _family_nodes(forest, statuses)[: max(1, int(args.max_families))]
    children = _child_index(forest)
    gt_pack = _load_gt_pack(args.hml3d_root)

    records: list[dict[str, Any]] = []
    for family in families:
        family_id = str(family.get("node_id") or "family")
        record = render_family_sheet(
            family,
            children.get(family_id, []),
            gt_pack=gt_pack,
            hml3d_root=args.hml3d_root,
            output_path=families_dir / f"{family_id}.png",
            max_sources=max(1, int(args.max_source_candidates_per_family)),
            examples_per_source=max(1, int(args.examples_per_source)),
            frames_per_example=max(2, int(args.frames_per_example)),
        )
        records.append(record)
        print(f"saved={record['image_path']}", flush=True)

    _write_json(output_dir / "review_pack_manifest.json", records)
    _write_json(output_dir / "review_decision_template.json", review_decision_template(records))
    _write_json(
        output_dir / "summary.json",
        {
            "source_forest": str(args.forest),
            "family_count": len(records),
            "statuses": sorted(statuses),
            "output_dir": str(output_dir),
        },
    )
    write_index(output_dir / "index.md", records, args.forest)
    write_review_queue(output_dir / "review_queue.md", records)
    print(output_dir)


if __name__ == "__main__":
    main()
