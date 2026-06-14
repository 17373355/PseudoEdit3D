from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys


HML_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit.aml_family_taxonomy import active_family_id, family_taxonomy_metadata, taxonomy_parent_for_family


SPEC_PATH = ROOT_DIR / "pseudoedit3d" / "edit" / "aml_language_coverage_specs.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _load_specs(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    specs = payload.get("specs") if isinstance(payload, dict) else payload
    if not isinstance(specs, list):
        raise ValueError(f"Unsupported coverage spec format: {path}")
    return [dict(spec) for spec in specs if isinstance(spec, dict)]


def _load_summary_cases(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        payload = _load_json(path)
        if isinstance(payload, dict) and "cases" in payload:
            payload = payload["cases"]
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported summary-json format: {path}")
        for row in payload:
            copied = dict(row)
            copied["_source"] = str(path)
            copied["_source_kind"] = "summary_json"
            rows.append(copied)
    return rows


def _load_condition_cases(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                actions = []
                for cond in record.get("conditions") or []:
                    source_family = str(cond.get("family_id") or cond.get("canonical_id") or "UNKNOWN")
                    family_id = active_family_id(source_family)
                    taxonomy = family_taxonomy_metadata(family_id)
                    semantic_family = {
                        "family_id": family_id,
                        "source_family": cond.get("source_family") or source_family,
                        "status": cond.get("status") or "unknown",
                        "probe_visible": cond.get("probe_visible", True) is not False,
                        "motion_only": cond.get("motion_only", True) is not False,
                        "label_confidence": cond.get("label_confidence"),
                        "taxonomy_parent_id": cond.get("taxonomy_parent_id") or taxonomy.get("taxonomy_parent_id"),
                        "taxonomy_parent_label": cond.get("taxonomy_parent_label") or taxonomy.get("taxonomy_parent_label"),
                        "taxonomy_recoverability": cond.get("taxonomy_recoverability") or taxonomy.get("taxonomy_recoverability"),
                        "taxonomy_evidence_axes": cond.get("taxonomy_evidence_axes") or taxonomy.get("taxonomy_evidence_axes") or [],
                        "taxonomy_secondary_parent_ids": cond.get("taxonomy_secondary_parent_ids") or taxonomy.get("taxonomy_secondary_parent_ids") or [],
                        "ambiguity_boundary": cond.get("ambiguity_boundary") or taxonomy.get("ambiguity_boundary"),
                    }
                    actions.append(
                        {
                            "canonical_id": family_id,
                            "semantic_family": semantic_family,
                            "slots": {
                                "span": (cond.get("slot_values") or {}).get("span"),
                                "semantic_family_id": family_id,
                                "semantic_family_status": cond.get("status") or "unknown",
                            },
                            "approx_slots": cond.get("approx_slots") or {},
                        }
                    )
                rows.append(
                    {
                        "case_id": record.get("case_id"),
                        "gt_prompt": record.get("selected_hml3d_prompt_for_reference_only") or "",
                        "auto_prompt": record.get("auto_prompt_for_probe_only") or "",
                        "canonical_actions": actions,
                        "_source": str(path),
                        "_source_kind": "condition_jsonl",
                    }
                )
    return rows


def _load_kinematic_flags(paths: list[str]) -> dict[str, list[str]]:
    flags: dict[str, list[str]] = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        payload = _load_json(path)
        for row in payload.get("rows") or []:
            case_id = str(row.get("case_id") or "")
            if not case_id:
                continue
            flags.setdefault(case_id, [])
            for flag in row.get("flags") or []:
                if flag not in flags[case_id]:
                    flags[case_id].append(str(flag))
    return flags


def _read_hml_prompts(case_id: str, hml_root: Path) -> list[str]:
    path = hml_root / "texts" / f"{case_id}.txt"
    if not path.exists():
        return []
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        prompts.append(line.split("#")[0].strip())
    return prompts


def _caption_text(case: dict[str, Any], hml_root: Path) -> tuple[str, list[str]]:
    prompts: list[str] = []
    for key in ("gt_prompt", "selected_hml3d_prompt", "selected_hml3d_prompt_for_reference_only", "reference_prompt"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            prompts.append(value.strip())
    for item in case.get("raw_prompt_segments") or []:
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
            prompts.append(item[0].strip())
        elif isinstance(item, str):
            prompts.append(item.strip())
    case_id = str(case.get("case_id") or "")
    if case_id:
        prompts.extend(_read_hml_prompts(case_id, hml_root))
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        normalized = " ".join(prompt.lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(prompt)
    return " ".join(prompt.lower() for prompt in deduped), deduped


def _action_family_ids(case: dict[str, Any]) -> tuple[set[str], set[str], set[str], set[str]]:
    all_ids: set[str] = set()
    visible_ids: set[str] = set()
    all_taxonomy_parents: set[str] = set()
    visible_taxonomy_parents: set[str] = set()
    for action in case.get("canonical_actions") or []:
        family = action.get("semantic_family") or {}
        slots = action.get("slots") or {}
        family_id = (
            family.get("family_id")
            if isinstance(family, dict)
            else None
        ) or action.get("canonical_id") or action.get("family") or slots.get("semantic_family_id")
        if not family_id:
            continue
        family_id = active_family_id(str(family_id))
        all_ids.add(family_id)
        taxonomy = family_taxonomy_metadata(family_id)
        parent_id = str((family if isinstance(family, dict) else {}).get("taxonomy_parent_id") or taxonomy.get("taxonomy_parent_id") or "UNKNOWN_OR_FALLBACK")
        all_taxonomy_parents.add(parent_id)
        probe_visible = True
        if isinstance(family, dict) and family.get("probe_visible") is False:
            probe_visible = False
        if isinstance(slots, dict) and slots.get("hidden_by_semantic_family"):
            probe_visible = False
        if probe_visible:
            visible_ids.add(family_id)
            visible_taxonomy_parents.add(parent_id)
    return all_ids, visible_ids, all_taxonomy_parents, visible_taxonomy_parents


def _matches(patterns: list[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]


def _spec_issue(
    spec: dict[str, Any],
    *,
    caption_text: str,
    auto_prompt: str,
    all_families: set[str],
    visible_families: set[str],
) -> dict[str, Any] | None:
    matched = _matches(spec["caption_patterns"], caption_text)
    if not matched:
        return None
    if _matches(list(spec.get("negative_caption_patterns") or []), caption_text):
        return None
    expected = set(spec.get("expected_families") or [])
    substitutes = set(spec.get("substitute_families") or [])
    present = sorted(expected & all_families)
    visible = sorted(expected & visible_families)
    substitute_present = sorted(substitutes & all_families)
    prompt_hits = _matches(list(spec.get("prompt_patterns") or spec.get("caption_patterns") or []), auto_prompt)
    substitute_prompt_hits = _matches(list(spec.get("substitute_prompt_patterns") or []), auto_prompt)
    issue_type = str(spec["issue_type"])

    if present and (visible or prompt_hits) and not substitute_prompt_hits:
        return {
            "label": spec["label"],
            "matched_caption_patterns": matched,
            "status": "covered",
            "issue_type": "covered",
            "recoverability": spec.get("recoverability"),
            "expected_families_present": present,
            "visible_expected_families": visible,
            "substitute_families_present": substitute_present,
            "severity": 0,
        }

    if present and (not visible or substitute_prompt_hits):
        issue_type = "prompt_priority_error"
        severity = 2 if substitute_prompt_hits else 1
        reason = "expected family exists but prompt-visible rendering is weak or dominated by lower-level wording"
    elif not present:
        severity = 3 if substitute_present or substitute_prompt_hits else 2
        reason = "caption suggests a semantic family that is missing from AML output"
    else:
        severity = 1
        reason = "caption pattern matched but coverage is uncertain"

    if spec.get("recoverability") == "geometry_recoverable" and not present:
        severity += 1
    if spec.get("issue_type") == "object_or_intent_ambiguous" and not present:
        severity = max(2, severity - 1)

    return {
        "label": spec["label"],
        "matched_caption_patterns": matched,
        "status": "issue",
        "issue_type": issue_type,
        "recoverability": spec.get("recoverability"),
        "reason": reason,
        "expected_families": sorted(expected),
        "expected_families_present": present,
        "visible_expected_families": visible,
        "substitute_families_present": substitute_present,
        "substitute_prompt_hits": substitute_prompt_hits,
        "severity": int(severity),
    }


def _kinematic_issues(case_id: str, flags: dict[str, list[str]]) -> list[dict[str, Any]]:
    out = []
    for flag in flags.get(case_id, []):
        if flag in {"root_path_mismatch", "vertical_amp_mismatch", "unexpected_jumpiness", "unexpected_translation"}:
            severity = 2
        else:
            severity = 1
        out.append(
            {
                "label": flag,
                "status": "issue",
                "issue_type": "momask_realization_or_scale_review",
                "recoverability": "generation_review",
                "reason": "kinematic sanity flag from generated MoMask motion",
                "severity": severity,
            }
        )
    return out


def audit_cases(cases: list[dict[str, Any]], flags: dict[str, list[str]], hml_root: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("case_id") or "")
        caption_text, captions = _caption_text(case, hml_root)
        auto_prompt = str(case.get("auto_prompt") or "").lower()
        all_families, visible_families, all_taxonomy_parents, visible_taxonomy_parents = _action_family_ids(case)
        coverage_rows = []
        issues = []
        for spec in specs:
            result = _spec_issue(
                spec,
                caption_text=caption_text,
                auto_prompt=auto_prompt,
                all_families=all_families,
                visible_families=visible_families,
            )
            if not result:
                continue
            coverage_rows.append(result)
            if result["status"] == "issue":
                issues.append(result)
        issues.extend(_kinematic_issues(case_id, flags))
        issue_types = sorted({str(issue["issue_type"]) for issue in issues})
        rows.append(
            {
                "schema_version": "aml_language_coverage_case_v1",
                "case_id": case_id,
                "source": case.get("_source"),
                "source_kind": case.get("_source_kind"),
                "captions": captions,
                "auto_prompt": case.get("auto_prompt") or "",
                "all_family_ids": sorted(all_families),
                "visible_family_ids": sorted(visible_families),
                "all_taxonomy_parent_ids": sorted(all_taxonomy_parents),
                "visible_taxonomy_parent_ids": sorted(visible_taxonomy_parents),
                "matched_specs": coverage_rows,
                "issues": issues,
                "issue_types": issue_types,
                "issue_score": int(sum(int(issue.get("severity") or 0) for issue in issues)),
                "kinematic_flags": flags.get(case_id, []),
            }
        )
    return rows


def _active_samples(rows: list[dict[str, Any]], per_type: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for issue_type in row.get("issue_types") or []:
            buckets[issue_type].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for issue_type, bucket in sorted(buckets.items()):
        ranked = sorted(
            bucket,
            key=lambda row: (
                -int(row.get("issue_score") or 0),
                str(row.get("case_id") or ""),
            ),
        )
        for row in ranked[:per_type]:
            cid = str(row["case_id"])
            selected.setdefault(
                cid,
                {
                    "case_id": cid,
                    "issue_score": row.get("issue_score"),
                    "issue_types": row.get("issue_types"),
                    "labels": sorted({str(issue["label"]) for issue in row.get("issues") or []}),
                    "auto_prompt": row.get("auto_prompt"),
                    "captions": row.get("captions"),
                },
            )
    return sorted(selected.values(), key=lambda row: (-int(row.get("issue_score") or 0), str(row["case_id"])))


def summarize(rows: list[dict[str, Any]], active: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    issue_type_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    recoverability_counts: Counter[str] = Counter()
    covered_label_counts: Counter[str] = Counter()
    issue_taxonomy_parent_counts: Counter[str] = Counter()
    visible_taxonomy_parent_counts: Counter[str] = Counter()
    no_issue = 0
    for row in rows:
        if not row.get("issues"):
            no_issue += 1
        for parent_id in row.get("visible_taxonomy_parent_ids") or []:
            visible_taxonomy_parent_counts[str(parent_id)] += 1
        for matched in row.get("matched_specs") or []:
            if matched.get("status") == "covered":
                covered_label_counts[str(matched.get("label"))] += 1
        for issue in row.get("issues") or []:
            issue_type_counts[str(issue.get("issue_type"))] += 1
            label_counts[str(issue.get("label"))] += 1
            recoverability_counts[str(issue.get("recoverability"))] += 1
            for family_id in issue.get("expected_families") or []:
                issue_taxonomy_parent_counts[taxonomy_parent_for_family(str(family_id))] += 1
    return {
        "schema_version": "aml_language_coverage_summary_v1",
        "source": {
            "summary_json": list(args.summary_json or []),
            "condition_jsonl": list(args.condition_jsonl or []),
            "kinematic_json": list(args.kinematic_json or []),
            "spec_json": str(args.spec_json),
        },
        "num_cases": len(rows),
        "num_cases_with_issues": sum(1 for row in rows if row.get("issues")),
        "num_cases_without_issues": no_issue,
        "issue_type_counts": issue_type_counts.most_common(),
        "issue_label_counts": label_counts.most_common(),
        "recoverability_counts": recoverability_counts.most_common(),
        "covered_label_counts": covered_label_counts.most_common(),
        "issue_taxonomy_parent_counts": issue_taxonomy_parent_counts.most_common(),
        "visible_taxonomy_parent_counts": visible_taxonomy_parent_counts.most_common(),
        "active_sample_count": len(active),
        "active_sample_case_ids": [row["case_id"] for row in active],
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ").replace("|", "/") for x in row) + " |")
    return "\n".join(out)


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], active: list[dict[str, Any]]) -> None:
    issue_rows = [row for row in rows if row.get("issues")]
    lines = [
        "# AML Language Coverage Audit",
        "",
        "This is a weak-label audit from HML3D captions, AML families, AutoPrompt text, and optional MoMask sanity flags. It is not ground truth.",
        "",
        "## Summary",
        "",
        f"- cases: `{summary['num_cases']}`",
        f"- cases with issues: `{summary['num_cases_with_issues']}`",
        f"- active samples: `{summary['active_sample_count']}`",
        "",
        "## Issue Types",
        "",
        _table(["issue_type", "count"], summary.get("issue_type_counts") or []),
        "",
        "## Issue Labels",
        "",
        _table(["label", "count"], (summary.get("issue_label_counts") or [])[:40]),
        "",
        "## Issue Taxonomy Parents",
        "",
        _table(["taxonomy_parent", "count"], (summary.get("issue_taxonomy_parent_counts") or [])[:40]),
        "",
        "## Visible Taxonomy Parents",
        "",
        _table(["taxonomy_parent", "cases"], (summary.get("visible_taxonomy_parent_counts") or [])[:40]),
        "",
        "## Active Samples",
        "",
        _table(
            ["case", "score", "issue_types", "labels", "auto_prompt"],
            [
                [
                    row["case_id"],
                    row.get("issue_score"),
                    ", ".join(row.get("issue_types") or []),
                    ", ".join(row.get("labels") or []),
                    str(row.get("auto_prompt") or "")[:120],
                ]
                for row in active
            ],
        ),
        "",
        "## Cases With Issues",
        "",
        _table(
            ["case", "score", "issues", "families", "caption"],
            [
                [
                    row["case_id"],
                    row.get("issue_score"),
                    ", ".join(f"{issue['label']}:{issue['issue_type']}" for issue in row.get("issues") or []),
                    ", ".join((row.get("visible_family_ids") or [])[:10]),
                    " / ".join(row.get("captions") or [])[:160],
                ]
                for row in sorted(issue_rows, key=lambda item: (-int(item.get("issue_score") or 0), str(item.get("case_id") or "")))
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- `missing_composed_family`: AML should probably gain a composed family or dominance rule.",
        "- `object_or_intent_ambiguous`: skeleton-only motion can usually support only a candidate/proxy family, not the object name itself.",
        "- `prompt_priority_error`: a reasonable family exists, but lower-level wording is dominating the AutoPrompt.",
        "- `momask_realization_or_scale_review`: do not change AML solely from this flag; inspect generation behavior separately.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", action="append", default=[])
    parser.add_argument("--condition-jsonl", action="append", default=[])
    parser.add_argument("--kinematic-json", action="append", default=[])
    parser.add_argument("--hml-root", default=str(HML_ROOT))
    parser.add_argument("--spec-json", default=str(SPEC_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--active-samples-per-type", type=int, default=12)
    args = parser.parse_args()

    cases = _load_summary_cases(args.summary_json) + _load_condition_cases(args.condition_jsonl)
    if not cases:
        raise SystemExit("No input cases. Provide --summary-json or --condition-jsonl.")
    specs = _load_specs(Path(args.spec_json))
    flags = _load_kinematic_flags(args.kinematic_json)
    rows = audit_cases(cases, flags, Path(args.hml_root), specs)
    active = _active_samples(rows, args.active_samples_per_type)
    summary = summarize(rows, active, args)

    out_dir = Path(args.output_dir)
    _write_jsonl(out_dir / "coverage_cases.jsonl", rows)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "active_samples.json", active)
    (out_dir / "active_sample_case_ids.txt").write_text(
        "\n".join(row["case_id"] for row in active) + ("\n" if active else ""),
        encoding="utf-8",
    )
    write_report(out_dir / "coverage_report.md", summary, rows, active)
    print(f"saved_cases={out_dir / 'coverage_cases.jsonl'}")
    print(f"saved_summary={out_dir / 'summary.json'}")
    print(f"saved_report={out_dir / 'coverage_report.md'}")
    print(f"saved_active_samples={out_dir / 'active_sample_case_ids.txt'}")


if __name__ == "__main__":
    main()
