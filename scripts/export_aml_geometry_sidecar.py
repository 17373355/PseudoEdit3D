from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import build_coarse_action_program
from pseudoedit3d.edit.geometry_sidecar import build_geometry_signature, summarize_geometry_sidecars

SOURCE = ROOT_DIR / "scripts" / "run_momask_aml_prompt_probe.py"
HML_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")


def _load_source_module():
    spec = importlib.util.spec_from_file_location("run_momask_aml_prompt_probe", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


src = _load_source_module()


def _read_prompts(case_id: str) -> list[str]:
    path = HML_ROOT / "texts" / f"{case_id}.txt"
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(line.split("#")[0].strip())
    return out


def _case_ids_from_args(args: argparse.Namespace) -> list[str]:
    case_ids: list[str] = []
    if args.case_ids:
        case_ids.extend(x.strip() for x in args.case_ids.split(",") if x.strip())
    if args.case_list:
        for line in Path(args.case_list).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                case_ids.append(line)
    if args.manifest:
        with Path(args.manifest).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                case_ids.append(str(row.get("case_id") or row.get("id")))
                if args.max_cases and len(case_ids) >= args.max_cases:
                    break
    seen: set[str] = set()
    out: list[str] = []
    for case_id in case_ids:
        if case_id in seen:
            continue
        seen.add(case_id)
        out.append(case_id)
        if args.max_cases and len(out) >= args.max_cases:
            break
    return out


def _extract_case(case_id: str, packed: dict[str, Any], max_residual_events: int) -> dict[str, Any] | None:
    key = f"{case_id}.npy"
    if key not in packed:
        return None
    joints = packed[key]["joints3d"]
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    joints = np.asarray(joints, dtype=np.float32)
    if len(joints) <= 20:
        return None
    aml = src.extract_aml_program(joints)
    coarse = build_coarse_action_program(aml["layer3"], max_residual_events=max_residual_events)
    geometry_signature = build_geometry_signature(aml["layer3"], coarse)
    return {
        "schema_version": "aml_geometry_sidecar_case_v1",
        "case_id": case_id,
        "num_frames": int(len(joints)),
        "selected_hml3d_prompt_for_reference_only": (_read_prompts(case_id) or [""])[0],
        "family_ids": [
            str((action.get("semantic_family") or {}).get("family_id") or action.get("canonical_id") or "UNKNOWN")
            for action in coarse.get("canonical_actions") or []
        ],
        "geometry_signature": geometry_signature,
    }


def _extract_cases(case_ids: list[str], max_residual_events: int, progress_every: int) -> list[dict[str, Any]]:
    packed = src.load_joints3d_pack()
    out: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, case_id in enumerate(case_ids, start=1):
        item = _extract_case(case_id, packed, max_residual_events)
        if item is not None:
            out.append(item)
        if progress_every and idx % progress_every == 0:
            print(f"processed {idx}/{len(case_ids)}, valid={len(out)}, elapsed={time.time() - t0:.1f}s", flush=True)
    return out


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return "\n".join(out)


def _top_family_counts(item: dict[str, Any], limit: int = 4) -> str:
    return ", ".join(f"{family}:{count}" for family, count in (item.get("family_counts") or [])[:limit])


def _top_context_counts(item: dict[str, Any], limit: int = 4) -> str:
    return ", ".join(f"{family}:{count}" for family, count in (item.get("context_family_counts") or [])[:limit])


def _write_markdown(summary: dict[str, Any], output: Path, top_n: int) -> None:
    stable_rows = [
        [item["geometry_cluster_id"], item["stable_family_id"], item["event_count"], item["case_support"], item.get("covered_context_link_count", 0)]
        for item in (summary.get("stable_geometry_cluster_mappings") or [])[:top_n]
    ]
    one_to_many_rows = [
        [item["geometry_cluster_id"], item["event_count"], item["case_support"], item.get("covered_context_link_count", 0), _top_family_counts(item)]
        for item in (summary.get("one_to_many_geometry_clusters") or [])[:top_n]
    ]
    multi_cluster_rows = [
        [
            item["family_id"],
            item["cluster_count"],
            item["total_links"],
            ", ".join(f"{cluster}:{count}" for cluster, count in (item.get("cluster_counts") or [])[:4]),
        ]
        for item in (summary.get("multi_cluster_semantic_families") or [])[:top_n]
    ]
    unnamed_rows = [
        [
            item["geometry_cluster_id"],
            item["event_count"],
            item["unable_to_name_count"],
            item["unable_to_name_share"],
            _top_family_counts(item),
            ", ".join(item.get("example_case_ids") or []),
        ]
        for item in (summary.get("aml_unable_to_name_clusters") or [])[:top_n]
    ]
    context_only_rows = [
        [
            item["geometry_cluster_id"],
            item["event_count"],
            item["context_only_event_count"],
            item["context_only_share"],
            _top_context_counts(item),
            ", ".join(item.get("example_case_ids") or []),
        ]
        for item in (summary.get("context_only_geometry_clusters") or [])[:top_n]
    ]
    lines = [
        "# AML Geometry Sidecar Summary",
        "",
        f"- cases: {summary.get('num_cases')}",
        f"- geometry events: {summary.get('total_geometry_events')}",
        "",
        "## Stable Geometry To Semantic",
        "",
        "Direct source links only; covered-context links are counted separately.",
        "",
        _table(["geometry_cluster", "stable_family", "events", "cases", "context_links"], stable_rows),
        "",
        "## Geometry One To Many",
        "",
        "Direct source links only. One-to-many here means the same geometry cluster directly supports multiple AML families.",
        "",
        _table(["geometry_cluster", "events", "cases", "context_links", "family_counts"], one_to_many_rows),
        "",
        "## Semantic Covered By Multiple Geometry Clusters",
        "",
        "Direct source links only.",
        "",
        _table(["family", "clusters", "links", "top_clusters"], multi_cluster_rows),
        "",
        "## Context-Only Geometry",
        "",
        "Events here are covered by AML actions as context but are not direct source evidence for a condition.",
        "",
        _table(["geometry_cluster", "events", "context_only", "share", "context_families", "examples"], context_only_rows),
        "",
        "## AML Unable To Name Geometry",
        "",
        "Events here have no AML link, or only unknown direct links. Context-only events are listed separately above.",
        "",
        _table(["geometry_cluster", "events", "unable", "share", "family_counts", "examples"], unnamed_rows),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--case-list", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-residual-events", type=int, default=8)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    case_ids = _case_ids_from_args(args)
    if not case_ids:
        raise SystemExit("No cases found. Provide --case-ids, --case-list, or --manifest.")

    records = _extract_cases(case_ids, args.max_residual_events, args.progress_every)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = summarize_geometry_sidecars(records, top_n=args.top_n)
    summary_path = Path(args.output_summary_json) if args.output_summary_json else output_jsonl.with_name("summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(summary, md_path, args.top_n)
    print(f"saved_jsonl={output_jsonl}")
    print(f"saved_summary_json={summary_path}")
    if args.output_md:
        print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
