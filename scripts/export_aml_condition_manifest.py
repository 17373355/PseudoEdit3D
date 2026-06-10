from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import build_coarse_action_program
from pseudoedit3d.edit.aml_condition_schema import (
    action_condition_weight,
    missing_required_slots,
    slot_confidences,
    slot_qualities,
    slot_values,
)

SOURCE = ROOT_DIR / "scripts" / "run_momask_aml_prompt_probe.py"
HML_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")


def _load_source_module():
    spec = importlib.util.spec_from_file_location("run_momask_aml_prompt_probe", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


src = _load_source_module()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _first_prompt(case_id: str) -> str:
    prompts = _read_prompts(case_id)
    return prompts[0] if prompts else ""


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


def _load_summary_cases(paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        items = _load_json(path)
        if isinstance(items, dict) and "cases" in items:
            items = items["cases"]
        if not isinstance(items, list):
            raise ValueError(f"Unsupported summary format: {path}")
        for item in items:
            copied = dict(item)
            copied["_source_summary"] = str(path)
            out.append(copied)
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
    return {
        "case_id": case_id,
        "num_frames": int(len(joints)),
        "selected_hml3d_prompt_for_reference_only": _first_prompt(case_id),
        "canonical_actions": coarse.get("canonical_actions") or [],
        "coarse_action_program": coarse,
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


def _case_num_frames(case: dict[str, Any]) -> int | None:
    for key in ("num_frames", "source_num_frames", "generated_num_frames"):
        value = case.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _condition_from_action(action: dict[str, Any], action_index: int) -> dict[str, Any]:
    family = dict(action.get("semantic_family") or {})
    family_id = str(family.get("family_id") or action.get("canonical_id") or "UNKNOWN")
    approx_slots = dict(action.get("approx_slots") or (action.get("slots") or {}).get("approx_slots") or {})
    missing_slots = missing_required_slots(family_id, approx_slots)
    return {
        "action_index": int(action_index),
        "canonical_id": str(action.get("canonical_id") or family_id),
        "family_id": family_id,
        "source_family": str(family.get("source_family") or action.get("family") or action.get("canonical_id") or family_id),
        "status": str(family.get("status") or (action.get("slots") or {}).get("semantic_family_status") or "unknown"),
        "condition_weight": action_condition_weight(family, missing_slots),
        "probe_visible": family.get("probe_visible", True) is not False,
        "motion_only": family.get("motion_only", True) is not False,
        "label_confidence": family.get("label_confidence"),
        "slot_values": slot_values(approx_slots),
        "slot_confidences": slot_confidences(approx_slots),
        "slot_qualities": slot_qualities(approx_slots),
        "approx_slots": approx_slots,
        "missing_required_slots": missing_slots,
        "surface_name_hint": action.get("surface_name_hint"),
        "probe_alias": action.get("probe_alias"),
    }


def _record_from_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("case_id"))
    actions = list(case.get("canonical_actions") or [])
    conditions = [_condition_from_action(action, idx) for idx, action in enumerate(actions)]
    return {
        "schema_version": "aml_condition_manifest_v1",
        "case_id": case_id,
        "num_frames": _case_num_frames(case),
        "selected_hml3d_prompt_for_reference_only": (
            case.get("selected_hml3d_prompt_for_reference_only")
            or case.get("selected_hml3d_prompt")
            or case.get("gt_prompt")
            or ""
        ),
        "auto_prompt_for_probe_only": case.get("auto_prompt") or "",
        "num_conditions": len(conditions),
        "status_counts": dict(Counter(cond["status"] for cond in conditions)),
        "family_ids": [cond["family_id"] for cond in conditions],
        "conditions": conditions,
    }


def _summarize(records: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    missing_required = 0
    zero_weight = 0
    total_conditions = 0
    for record in records:
        for cond in record.get("conditions") or []:
            total_conditions += 1
            status_counts[str(cond.get("status"))] += 1
            family_counts[str(cond.get("family_id"))] += 1
            if cond.get("missing_required_slots"):
                missing_required += 1
            if float(cond.get("condition_weight") or 0.0) <= 0.0:
                zero_weight += 1
    return {
        "schema_version": "aml_condition_manifest_v1",
        "source": source,
        "num_cases": len(records),
        "total_conditions": total_conditions,
        "status_counts": status_counts.most_common(),
        "family_counts": family_counts.most_common(),
        "missing_required_condition_count": missing_required,
        "zero_weight_condition_count": zero_weight,
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return "\n".join(out)


def _write_markdown(summary: dict[str, Any], output: Path, top_n: int) -> None:
    lines = [
        "# AML Condition Manifest Summary",
        "",
        f"- cases: {summary.get('num_cases')}",
        f"- conditions: {summary.get('total_conditions')}",
        f"- missing required conditions: {summary.get('missing_required_condition_count')}",
        f"- zero-weight conditions: {summary.get('zero_weight_condition_count')}",
        "",
        "## Status Counts",
        "",
        _table(["status", "conditions"], [[k, v] for k, v in summary.get("status_counts", [])]),
        "",
        "## Top Families",
        "",
        _table(["family", "conditions"], [[k, v] for k, v in (summary.get("family_counts") or [])[:top_n]]),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", action="append", default=[])
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

    cases = _load_summary_cases(args.summary_json)
    case_ids = _case_ids_from_args(args)
    if case_ids:
        cases.extend(_extract_cases(case_ids, args.max_residual_events, args.progress_every))
    if not cases:
        raise SystemExit("No cases found. Provide --summary-json, --case-ids, --case-list, or --manifest.")

    records = [_record_from_case(case) for case in cases]
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    source = {
        "summary_json_inputs": list(args.summary_json),
        "case_id_inputs": case_ids,
        "max_residual_events": int(args.max_residual_events),
    }
    summary = _summarize(records, source)
    summary_path = Path(args.output_summary_json) if args.output_summary_json else output_jsonl.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.output_md:
        _write_markdown(summary, Path(args.output_md), args.top_n)
    print(f"saved_jsonl={output_jsonl}")
    print(f"saved_summary_json={summary_path}")
    if args.output_md:
        print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
