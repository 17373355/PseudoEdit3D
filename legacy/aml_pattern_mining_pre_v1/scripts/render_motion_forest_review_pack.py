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


DEFAULT_FOREST = Path("outputs/aml_regression_testset_v2/full_candidate_motion_forest_v1/full_candidate_motion_forest.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/full_candidate_motion_forest_review_pack_v1")
DEFAULT_HML3D_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = raw.split("#", 1)[0].strip()
            if text:
                lines.append(text)
            if len(lines) >= limit:
                break
    if not lines and fallback:
        lines.append(fallback)
    return lines[:limit]


def _family_children(forest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    nodes = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = nodes.get(str(edge.get("child_node_id") or ""))
        if parent and child:
            children[parent].append(child)
    for rows in children.values():
        rows.sort(key=lambda node: (-int((node.get("support") or {}).get("support_cases_reported") or 0), str(node.get("motif_id") or "")))
    return children


def _family_nodes(forest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [node for node in forest.get("nodes") or [] if node.get("node_kind") == "geometry_family"]
    rows.sort(key=lambda node: (-int((node.get("support") or {}).get("support_cases_sum") or 0), str(node.get("node_id") or "")))
    return rows


def _frame_indices(num_frames: int, span: list[int] | None, count: int) -> list[int]:
    if num_frames <= 0:
        return []
    if span and len(span) == 2:
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


def _sample_examples(children: list[dict[str, Any]], *, max_motifs: int, examples_per_motif: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child in children[:max_motifs]:
        for ex in (child.get("example_occurrences") or [])[:examples_per_motif]:
            rows.append(
                {
                    "motif": child,
                    "example": ex,
                }
            )
    return rows


def render_family_sheet(
    family: dict[str, Any],
    children: list[dict[str, Any]],
    *,
    gt_pack: dict[str, Any],
    hml3d_root: Path,
    output_path: Path,
    max_motifs: int,
    examples_per_motif: int,
    frames_per_example: int,
) -> dict[str, Any]:
    examples = _sample_examples(children, max_motifs=max_motifs, examples_per_motif=examples_per_motif)
    row_h = 176
    canvas_w = 1640
    header_h = 132
    canvas_h = max(header_h + row_h, header_h + row_h * max(1, len(examples)))
    img = Image.new("RGB", (canvas_w, canvas_h), color=(247, 248, 251))
    draw = ImageDraw.Draw(img)
    font_title = _load_font(24)
    font_body = _load_font(13)
    font_small = _load_font(10)
    font_caption = _load_font(11)

    support = family.get("support") or {}
    geometry = family.get("motion_evidence", {}).get("required_geometry_clusters") or []
    aliases = family.get("naming_diagnostics", {}).get("top_caption_aliases") or []
    legacy = family.get("legacy_diagnostics", {}).get("top_tree_families") or []
    draw.text((24, 18), f"{family.get('node_id')}  {family.get('status')}", fill=(28, 32, 40), font=font_title)
    draw.text(
        (24, 52),
        f"motifs={support.get('motif_count')}  coverage={support.get('unique_case_coverage')}  support_sum={support.get('support_cases_sum')}",
        fill=(58, 64, 78),
        font=font_body,
    )
    draw.text((24, 76), "geometry: " + " + ".join(geometry), fill=(42, 82, 110), font=font_body)
    draw.text(
        (24, 100),
        "aliases: " + ", ".join(f"{item.get('id')}:{item.get('count')}" for item in aliases[:5]) + "    old: " + ", ".join(f"{item.get('id')}:{item.get('count')}" for item in legacy[:3]),
        fill=(92, 72, 45),
        font=font_small,
    )

    saved_examples: list[dict[str, Any]] = []
    frame_w = 96
    frame_h = 130
    strip_x = 24
    text_x = strip_x + frame_w * frames_per_example + 22
    max_text_w = canvas_w - text_x - 28
    for row_idx, row in enumerate(examples):
        y = header_h + row_idx * row_h
        draw.rectangle((16, y + 8, canvas_w - 16, y + row_h - 8), outline=(224, 227, 235), fill=(255, 255, 255))
        motif = row["motif"]
        ex = row["example"]
        case_id = str(ex.get("case_id") or "")
        joints = _gt_joints(gt_pack, case_id)
        span = ex.get("span") or None
        if joints is not None:
            indices = _frame_indices(len(joints), span, frames_per_example)
            _draw_frame_strip(draw, joints, indices, x=strip_x, y=y + 24, frame_w=frame_w, frame_h=frame_h)
        else:
            indices = []
            draw.rectangle((strip_x, y + 24, strip_x + frame_w * frames_per_example - 1, y + 24 + frame_h), outline=(224, 227, 235), fill=(250, 250, 252))
            draw.text((strip_x + 12, y + 72), "missing joints", fill=(120, 65, 65), font=font_body)

        captions = _caption_lines(hml3d_root, case_id, str(ex.get("caption") or ""))
        meta = (
            f"{case_id}  {motif.get('motif_id')}  {motif.get('tier')}  "
            f"support={motif.get('support', {}).get('support_cases_reported')}  span={span}"
        )
        draw.text((text_x, y + 22), meta, fill=(35, 40, 52), font=font_body)
        draw.text(
            (text_x, y + 42),
            "clusters: " + " + ".join(motif.get("motion_evidence", {}).get("required_geometry_clusters") or []),
            fill=(42, 82, 110),
            font=font_small,
        )
        text_y = y + 62
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
                "motif_id": motif.get("motif_id"),
                "span": span,
                "frame_indices": indices,
                "captions": captions,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return {
        "family_id": family.get("node_id"),
        "image_path": str(output_path),
        "example_count": len(saved_examples),
        "examples": saved_examples,
    }


def write_index(path: Path, records: list[dict[str, Any]], source_forest: Path) -> None:
    lines: list[str] = []
    lines.append("# Motion Forest Review Pack")
    lines.append("")
    lines.append(f"- source forest: `{source_forest}`")
    lines.append("- each image shows GT HumanML3D keyframes, not MoMask generations.")
    lines.append("- captions are reference text for human review; they are not structural evidence.")
    lines.append("")
    for row in records:
        image_name = Path(str(row["image_path"])).name
        lines.append(f"## {row['family_id']}")
        lines.append("")
        lines.append(f"examples: `{row['example_count']}`")
        lines.append("")
        lines.append(f"![{row['family_id']}](families/{image_name})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render static PNG review sheets for a motion-BPE candidate forest.")
    parser.add_argument("--forest", default=str(DEFAULT_FOREST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hml3d-root", default=str(DEFAULT_HML3D_ROOT))
    parser.add_argument("--max-families", type=int, default=14)
    parser.add_argument("--max-motifs-per-family", type=int, default=4)
    parser.add_argument("--examples-per-motif", type=int, default=2)
    parser.add_argument("--frames-per-example", type=int, default=5)
    args = parser.parse_args()

    forest_path = Path(args.forest)
    output_dir = Path(args.output_dir)
    families_dir = output_dir / "families"
    output_dir.mkdir(parents=True, exist_ok=True)
    families_dir.mkdir(parents=True, exist_ok=True)

    forest = _read_json(forest_path)
    children = _family_children(forest)
    family_rows = _family_nodes(forest)[: max(1, int(args.max_families))]
    gt_pack = _load_gt_pack(Path(args.hml3d_root))

    records = []
    for family in family_rows:
        family_id = str(family.get("node_id") or "family")
        record = render_family_sheet(
            family,
            children.get(family_id, []),
            gt_pack=gt_pack,
            hml3d_root=Path(args.hml3d_root),
            output_path=families_dir / f"{family_id}.png",
            max_motifs=max(1, int(args.max_motifs_per_family)),
            examples_per_motif=max(1, int(args.examples_per_motif)),
            frames_per_example=max(2, int(args.frames_per_example)),
        )
        records.append(record)
        print(f"saved={record['image_path']}", flush=True)

    (output_dir / "review_pack_manifest.json").write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")
    write_index(output_dir / "index.md", records, forest_path)
    print(output_dir)


if __name__ == "__main__":
    main()
