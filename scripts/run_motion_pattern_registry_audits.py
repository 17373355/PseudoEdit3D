"""Run review-only motion-pattern audits for every manual text target.

This is a v0 AML diagnostic runner. It expands the same audit used for the
jumping-jack smoke test to all targets in
`configs/motion_pattern_text_targets.json`.

The registry is used only as text pseudo-GT for evaluation. It does not affect
Motion-BPE learning, motion-family construction, or runtime AML rules.

Example:
    python scripts/run_motion_pattern_registry_audits.py \
      --output-dir outputs/aml_regression_testset_v2/manual_text_target_audits_v0

Quick check:
    python scripts/run_motion_pattern_registry_audits.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_motion_pattern_pseudo_gt as pseudo_gt_audit  # noqa: E402
import audit_motion_pattern_recall_candidates as recall_audit  # noqa: E402
import build_motion_pattern_family_proposals as family_proposals  # noqa: E402


DEFAULT_TARGET_REGISTRY = Path("configs/motion_pattern_text_targets.json")
DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_BPE_SEQUENCES = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/case_multichannel_bpe_sequences.jsonl")
DEFAULT_CANDIDATES = Path("outputs/aml_regression_testset_v2/coordination_pattern_promotion_candidates_loose_v1/coordination_pattern_promotion_candidates.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/manual_text_target_audits_v0")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _target_rows(registry_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(registry_path)
    rows = payload.get("targets") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"target registry must contain a list of targets: {registry_path}")
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and row.get("id"):
            out.append(row)
    return out


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _target_ids(args: argparse.Namespace, registry_rows: list[dict[str, Any]]) -> list[str]:
    requested = _split_csv(args.target_aliases)
    if requested:
        available = {str(row["id"]) for row in registry_rows}
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"requested target aliases are not in registry: {missing}")
        return requested
    return [str(row["id"]) for row in registry_rows]


def _caption_text(record: dict[str, Any]) -> str:
    return " ".join(str(item) for item in record.get("caption_texts") or [])


def _safe_div(num: int, denom: int) -> float:
    return 0.0 if denom == 0 else num / denom


def _top_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _compile_target_patterns(targets: list[dict[str, Any]]) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for target in targets:
        target_id = str(target["id"])
        pattern = str(target.get("regex") or re.escape(target_id.replace("_", " ")))
        patterns[target_id] = re.compile(pattern, re.IGNORECASE)
    return patterns


def _load_case_text_and_pseudo_gt(source_corpus: Path, targets: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    patterns = _compile_target_patterns(targets)
    target_ids = [str(target["id"]) for target in targets]
    case_text: dict[str, dict[str, Any]] = {}
    pseudo_gt: dict[str, set[str]] = {target_id: set() for target_id in target_ids}
    with source_corpus.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record.get("case_id") or "")
            aliases = {str(item) for item in record.get("caption_alias_ids") or []}
            text = _caption_text(record)
            case_text[case_id] = {
                "case_id": case_id,
                "caption_texts": record.get("caption_texts") or [],
                "caption_alias_ids": sorted(aliases),
            }
            for target_id in target_ids:
                if target_id in aliases or patterns[target_id].search(text):
                    pseudo_gt[target_id].add(case_id)
    return case_text, pseudo_gt


def _select_seed_motif_ids_by_target(args: argparse.Namespace, target_ids: list[str]) -> dict[str, list[str]]:
    statuses = set(_split_csv(args.candidate_statuses))
    out: dict[str, set[str]] = {target_id: set() for target_id in target_ids}
    payload = _read_json(Path(args.candidates))
    for candidate in payload.get("candidates") or []:
        naming = candidate.get("naming_diagnostics") or {}
        target_id = str(naming.get("top_caption_alias") or "")
        if target_id not in out:
            continue
        if statuses and str(candidate.get("status") or "") not in statuses:
            continue
        motif_id = str(candidate.get("source_motif_id") or "")
        if motif_id:
            out[target_id].add(motif_id)
    return {key: sorted(value) for key, value in out.items()}


def _is_candidate_token(token: dict[str, Any], unit_types: set[str], min_channels: int) -> bool:
    symbol = str(token.get("symbol") or "")
    if not symbol:
        return False
    if str(token.get("unit_type") or "") not in unit_types:
        return False
    if len({str(item) for item in token.get("channels") or []}) < min_channels:
        return False
    return True


def _token_example(token: dict[str, Any]) -> dict[str, Any]:
    return {
        "span": token.get("span"),
        "unit_type": token.get("unit_type"),
        "channels": token.get("channels") or [],
        "geometry_clusters": token.get("geometry_clusters") or [],
    }


def _scan_symbol_index(bpe_sequences: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    unit_types = set(_split_csv(args.unit_types))
    min_channels = int(args.min_channels)
    symbol_rows: dict[str, dict[str, Any]] = {}
    with bpe_sequences.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            for token in row.get("tokens") or []:
                if not _is_candidate_token(token, unit_types, min_channels):
                    continue
                symbol = str(token.get("symbol") or "")
                entry = symbol_rows.setdefault(
                    symbol,
                    {
                        "symbol": symbol,
                        "unit_types": Counter(),
                        "case_ids": set(),
                        "occurrences": 0,
                        "channels": Counter(),
                        "geometry_clusters": Counter(),
                        "examples_by_case": defaultdict(list),
                    },
                )
                entry["unit_types"][str(token.get("unit_type") or "")] += 1
                entry["case_ids"].add(case_id)
                entry["occurrences"] += 1
                entry["channels"].update(str(item) for item in token.get("channels") or [])
                entry["geometry_clusters"].update(str(item) for item in token.get("geometry_clusters") or [])
                if len(entry["examples_by_case"][case_id]) < 2:
                    entry["examples_by_case"][case_id].append(_token_example(token))
    return symbol_rows


def _case_examples(case_ids: list[str], case_text: dict[str, dict[str, Any]], predictions: dict[str, list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_id in case_ids[:limit]:
        info = case_text.get(case_id, {"case_id": case_id, "caption_texts": [], "caption_alias_ids": []})
        out.append(
            {
                "case_id": case_id,
                "caption_texts": info.get("caption_texts") or [],
                "caption_alias_ids": info.get("caption_alias_ids") or [],
                "predictions": predictions.get(case_id, []),
            }
        )
    return out


def _examples(
    case_ids: list[str],
    case_text: dict[str, dict[str, Any]],
    examples_by_case: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_id in case_ids[:limit]:
        text = case_text.get(case_id, {})
        out.append(
            {
                "case_id": case_id,
                "caption_texts": text.get("caption_texts") or [],
                "caption_alias_ids": text.get("caption_alias_ids") or [],
                "tokens": examples_by_case.get(case_id, []),
            }
        )
    return out


def _seed_predictions(symbol_rows: dict[str, dict[str, Any]], seed_motif_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for motif_id in seed_motif_ids:
        entry = symbol_rows.get(motif_id)
        if not entry:
            continue
        for case_id, examples in entry["examples_by_case"].items():
            for example in examples:
                predictions[case_id].append({"motif_id": motif_id, **example})
    return dict(predictions)


def _build_pseudo_payload(
    target: dict[str, Any],
    *,
    args: argparse.Namespace,
    case_text: dict[str, dict[str, Any]],
    pseudo_gt_cases: set[str],
    seed_motif_ids: list[str],
    symbol_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    predictions = _seed_predictions(symbol_rows, set(seed_motif_ids))
    gt_cases = set(pseudo_gt_cases)
    pred_cases = set(predictions)
    tp = sorted(gt_cases & pred_cases)
    fp = sorted(pred_cases - gt_cases)
    fn = sorted(gt_cases - pred_cases)
    return {
        "schema_version": "motion_pattern_pseudo_gt_audit_v1",
        "target_alias": target["id"],
        "target_regex": target.get("regex") or "",
        "policy": {
            "purpose": "evaluation and naming diagnostic only",
            "pseudo_gt": "HumanML3D caption_alias_ids plus target regex over caption_texts",
            "prediction": "case contains one of the selected learned motif ids in BPE sequences",
        },
        "inputs": {
            "source_corpus": str(args.source_corpus),
            "bpe_sequences": str(args.bpe_sequences),
            "candidates": str(args.candidates or ""),
            "candidate_statuses": _split_csv(args.candidate_statuses),
        },
        "selected_motif_ids": seed_motif_ids,
        "metrics": {
            "pseudo_gt_case_count": len(gt_cases),
            "predicted_case_count": len(pred_cases),
            "true_positive_count": len(tp),
            "false_positive_count": len(fp),
            "false_negative_count": len(fn),
            "precision_subset_accuracy": round(_safe_div(len(tp), len(pred_cases)), 6),
            "recall_against_text_pseudo_gt": round(_safe_div(len(tp), len(gt_cases)), 6),
            "f1": round(_safe_div(2 * len(tp), 2 * len(tp) + len(fp) + len(fn)), 6),
        },
        "case_ids": {"true_positive": tp, "false_positive": fp, "false_negative": fn},
        "examples": {
            "true_positive": _case_examples(tp, case_text, predictions, int(args.example_limit)),
            "false_positive": _case_examples(fp, case_text, predictions, int(args.example_limit)),
            "false_negative": _case_examples(fn, case_text, predictions, int(args.example_limit)),
        },
    }


def _greedy_precision_preserving_expansion(
    rows: list[dict[str, Any]],
    candidate_case_sets: dict[str, set[str]],
    *,
    seed_pred_cases: set[str],
    pseudo_gt_cases: set[str],
    precision_floor: float,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    active_pred_cases = set(seed_pred_cases)
    active_tp = active_pred_cases & pseudo_gt_cases
    active_fp = active_pred_cases - pseudo_gt_cases
    for row in rows:
        symbol = str(row.get("symbol") or "")
        case_ids = candidate_case_sets.get(symbol) or set()
        next_pred_cases = active_pred_cases | case_ids
        next_tp = next_pred_cases & pseudo_gt_cases
        next_fp = next_pred_cases - pseudo_gt_cases
        incremental_tp = len(next_tp) - len(active_tp)
        incremental_fp = len(next_fp) - len(active_fp)
        if incremental_tp <= 0:
            continue
        next_precision = _safe_div(len(next_tp), len(next_pred_cases))
        if next_precision < precision_floor:
            continue
        active_pred_cases = next_pred_cases
        active_tp = next_tp
        active_fp = next_fp
        selected.append(
            {
                "symbol": symbol,
                "incremental_true_positive": incremental_tp,
                "incremental_false_positive": incremental_fp,
                "precision_after_add": round(next_precision, 6),
                "recall_after_add": round(_safe_div(len(active_tp), len(pseudo_gt_cases)), 6),
                "predicted_case_count_after_add": len(active_pred_cases),
            }
        )
    return {
        "precision_floor": precision_floor,
        "selected_count": len(selected),
        "selected": selected,
        "predicted_case_count": len(active_pred_cases),
        "true_positive_count": len(active_tp),
        "false_positive_count": len(active_fp),
        "precision_subset_accuracy": round(_safe_div(len(active_tp), len(active_pred_cases)), 6),
        "recall_against_text_pseudo_gt": round(_safe_div(len(active_tp), len(pseudo_gt_cases)), 6),
    }


def _build_recall_payload(
    target: dict[str, Any],
    *,
    args: argparse.Namespace,
    case_text: dict[str, dict[str, Any]],
    pseudo_gt_cases: set[str],
    seed_motif_ids: list[str],
    symbol_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    seed_predictions = _seed_predictions(symbol_rows, set(seed_motif_ids))
    seed_pred_cases = set(seed_predictions)
    seed_tp_cases = pseudo_gt_cases & seed_pred_cases
    seed_fp_cases = seed_pred_cases - pseudo_gt_cases
    false_negative_cases = pseudo_gt_cases - seed_pred_cases

    rows: list[dict[str, Any]] = []
    candidate_case_sets: dict[str, set[str]] = {}
    for symbol, entry in symbol_rows.items():
        if symbol in set(seed_motif_ids):
            continue
        case_ids = set(entry["case_ids"])
        fn_hits = sorted(case_ids & false_negative_cases)
        if len(fn_hits) < int(args.min_false_negative_support):
            continue
        candidate_case_sets[symbol] = case_ids
        gt_hits = case_ids & pseudo_gt_cases
        non_gt_hits = case_ids - pseudo_gt_cases
        union_pred = seed_pred_cases | case_ids
        union_tp = union_pred & pseudo_gt_cases
        union_fp = union_pred - pseudo_gt_cases
        rows.append(
            {
                "symbol": symbol,
                "support_cases": len(case_ids),
                "occurrences": int(entry["occurrences"]),
                "pseudo_gt_support_cases": len(gt_hits),
                "non_pseudo_gt_support_cases": len(non_gt_hits),
                "false_negative_support_cases": len(fn_hits),
                "candidate_precision": round(_safe_div(len(gt_hits), len(case_ids)), 6),
                "candidate_recall": round(_safe_div(len(gt_hits), len(pseudo_gt_cases)), 6),
                "union_precision_with_seed": round(_safe_div(len(union_tp), len(union_pred)), 6),
                "union_recall_with_seed": round(_safe_div(len(union_tp), len(pseudo_gt_cases)), 6),
                "incremental_true_positive": len(fn_hits),
                "incremental_false_positive": len(non_gt_hits - seed_fp_cases),
                "unit_types": _top_counter(entry["unit_types"], 6),
                "channels": _top_counter(entry["channels"], 8),
                "geometry_clusters": _top_counter(entry["geometry_clusters"], 12),
                "false_negative_examples": _examples(fn_hits, case_text, entry["examples_by_case"], int(args.example_limit)),
                "false_positive_examples": _examples(
                    sorted(non_gt_hits),
                    case_text,
                    entry["examples_by_case"],
                    max(0, min(int(args.example_limit), 6)),
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["incremental_true_positive"]),
            -float(row["union_precision_with_seed"]),
            -float(row["candidate_precision"]),
            -int(row["support_cases"]),
            str(row["symbol"]),
        )
    )
    top_rows = rows[: int(args.max_candidates)]
    top_symbols = {str(row["symbol"]) for row in top_rows}
    candidate_case_sets = {key: value for key, value in candidate_case_sets.items() if key in top_symbols}
    greedy = _greedy_precision_preserving_expansion(
        top_rows,
        candidate_case_sets,
        seed_pred_cases=seed_pred_cases,
        pseudo_gt_cases=pseudo_gt_cases,
        precision_floor=float(args.precision_floor),
    )
    return {
        "schema_version": "motion_pattern_recall_candidates_v1",
        "target_alias": target["id"],
        "target_regex": target.get("regex") or "",
        "policy": {
            "purpose": "diagnostic only; candidates do not change Motion-BPE or AML runtime",
            "seed_prediction": "cases containing selected seed motif ids",
            "candidate_prediction": "cases containing candidate Motion-BPE symbols",
        },
        "inputs": {
            "source_corpus": str(args.source_corpus),
            "bpe_sequences": str(args.bpe_sequences),
            "candidates": str(args.candidates),
            "unit_types": _split_csv(args.unit_types),
            "min_channels": int(args.min_channels),
            "min_false_negative_support": int(args.min_false_negative_support),
        },
        "seed": {
            "selected_motif_ids": sorted(seed_motif_ids),
            "pseudo_gt_case_count": len(pseudo_gt_cases),
            "predicted_case_count": len(seed_pred_cases),
            "true_positive_count": len(seed_tp_cases),
            "false_positive_count": len(seed_fp_cases),
            "false_negative_count": len(false_negative_cases),
            "precision_subset_accuracy": round(_safe_div(len(seed_tp_cases), len(seed_pred_cases)), 6),
            "recall_against_text_pseudo_gt": round(_safe_div(len(seed_tp_cases), len(pseudo_gt_cases)), 6),
        },
        "summary": {
            "candidate_symbol_count": len(rows),
            "reported_candidate_count": len(top_rows),
        },
        "greedy_precision_preserving_expansion": greedy,
        "recall_candidates": top_rows,
    }


def _summary_row(
    *,
    target: dict[str, Any],
    target_dir: Path,
    pseudo_payload: dict[str, Any],
    recall_payload: dict[str, Any],
    proposal_payload: dict[str, Any],
) -> dict[str, Any]:
    pseudo_metrics = pseudo_payload.get("metrics") or {}
    seed = recall_payload.get("seed") or {}
    greedy = recall_payload.get("greedy_precision_preserving_expansion") or {}
    proposal_summary = proposal_payload.get("summary") or {}
    return {
        "target_alias": target.get("id"),
        "display_name": target.get("display_name") or target.get("id"),
        "regex": target.get("regex"),
        "pseudo_gt_case_count": pseudo_metrics.get("pseudo_gt_case_count"),
        "seed_selected_motif_count": len(pseudo_payload.get("selected_motif_ids") or []),
        "seed_precision": seed.get("precision_subset_accuracy"),
        "seed_recall": seed.get("recall_against_text_pseudo_gt"),
        "candidate_symbol_count": (recall_payload.get("summary") or {}).get("candidate_symbol_count"),
        "greedy_selected_count": greedy.get("selected_count"),
        "expanded_precision": greedy.get("precision_subset_accuracy"),
        "expanded_recall": greedy.get("recall_against_text_pseudo_gt"),
        "proposal_variant_count": proposal_summary.get("variant_count"),
        "proposal_status_counts": proposal_summary.get("status_counts"),
        "outputs": {
            "dir": str(target_dir),
            "pseudo_gt_audit": str(target_dir / "pattern_pseudo_gt_audit.json"),
            "recall_candidates": str(target_dir / "recall_candidate_symbols.json"),
            "family_proposal": str(target_dir / "pattern_family_proposal.json"),
            "review": str(target_dir / "review.md"),
        },
    }


def _write_target_review(path: Path, target: dict[str, Any], row: dict[str, Any], recall_payload: dict[str, Any], proposal_payload: dict[str, Any]) -> None:
    lines = [
        f"# Manual Text Target Audit: {target.get('id')}",
        "",
        "This is a review-only v0 AML diagnostic. The target registry supplies text pseudo-GT only.",
        "",
        f"- display name: `{target.get('display_name') or target.get('id')}`",
        f"- regex: `{target.get('regex')}`",
        f"- pseudo-GT cases: `{row.get('pseudo_gt_case_count')}`",
        f"- seed motif count: `{row.get('seed_selected_motif_count')}`",
        f"- seed precision / recall: `{row.get('seed_precision')}` / `{row.get('seed_recall')}`",
        f"- candidate symbols: `{row.get('candidate_symbol_count')}`",
        f"- expanded precision / recall: `{row.get('expanded_precision')}` / `{row.get('expanded_recall')}`",
        f"- proposal variants: `{row.get('proposal_variant_count')}`",
        "",
        "## Greedy Selected Candidates",
        "",
        "| step | symbol | +TP | +FP | precision | recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    greedy = recall_payload.get("greedy_precision_preserving_expansion") or {}
    for idx, item in enumerate(greedy.get("selected") or [], start=1):
        symbol = str(item.get("symbol") or "").replace("|", "\\|")
        lines.append(
            f"| {idx} | `{symbol}` | {item.get('incremental_true_positive')} | {item.get('incremental_false_positive')} | "
            f"{item.get('precision_after_add')} | {item.get('recall_after_add')} |"
        )
    lines.extend(["", "## Promoted / Review Variants", ""])
    for variant in proposal_payload.get("variants") or []:
        status = str(variant.get("status") or "")
        if status.startswith("reject"):
            continue
        metrics = variant.get("metrics") or {}
        motion = variant.get("motion_signature") or {}
        lines.append(f"### {variant.get('variant_id')}")
        lines.append("")
        lines.append(f"- status: `{status}`")
        lines.append(f"- symbol: `{variant.get('symbol')}`")
        lines.append(f"- metrics: `{metrics}`")
        lines.append(f"- channels: `{motion.get('channels')}`")
        lines.append(f"- geometry: `{motion.get('geometry_clusters')}`")
        examples = variant.get("examples") or []
        if isinstance(examples, dict):
            examples = examples.get("false_negative") or examples.get("false_positive") or []
        if examples:
            lines.append("- examples:")
            for example in examples[:5]:
                captions = " / ".join(str(text).replace("\n", " ") for text in (example.get("caption_texts") or [])[:3])
                lines.append(f"  - `{example.get('case_id')}`: {captions}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_target(
    target: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    *,
    case_text: dict[str, dict[str, Any]],
    pseudo_gt_cases: set[str],
    seed_motif_ids: list[str],
    symbol_rows: dict[str, dict[str, Any]],
    promotion_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_alias = str(target["id"])
    target_dir = output_dir / family_proposals._safe_id(target_alias)
    target_dir.mkdir(parents=True, exist_ok=True)

    pseudo_payload = _build_pseudo_payload(
        target,
        args=args,
        case_text=case_text,
        pseudo_gt_cases=pseudo_gt_cases,
        seed_motif_ids=seed_motif_ids,
        symbol_rows=symbol_rows,
    )
    pseudo_path = target_dir / "pattern_pseudo_gt_audit.json"
    _write_json(pseudo_path, pseudo_payload)
    pseudo_gt_audit.write_report(target_dir / "pattern_pseudo_gt_audit.md", pseudo_payload)

    recall_payload = _build_recall_payload(
        target,
        args=args,
        case_text=case_text,
        pseudo_gt_cases=pseudo_gt_cases,
        seed_motif_ids=seed_motif_ids,
        symbol_rows=symbol_rows,
    )
    recall_payload["report_detail_limit"] = int(args.report_detail_limit)
    recall_path = target_dir / "recall_candidate_symbols.json"
    _write_json(recall_path, recall_payload)
    recall_audit.write_report(target_dir / "recall_candidate_symbols.md", recall_payload)

    proposal_args = argparse.Namespace(
        target_alias=target_alias,
        pseudo_gt_audit=str(pseudo_path),
        recall_candidates=str(recall_path),
        promotion_candidates=str(args.candidates),
        promote_precision=float(args.promote_precision),
        promote_min_tp=int(args.promote_min_tp),
        review_precision=float(args.review_precision),
        review_min_tp=int(args.review_min_tp),
        reject_precision=float(args.reject_precision),
    )
    proposal_payload = family_proposals.build_proposal(
        pseudo_payload,
        recall_payload,
        promotion_index,
        proposal_args,
    )
    _write_json(target_dir / "pattern_family_proposal.json", proposal_payload)
    family_proposals.write_report(target_dir / "pattern_family_proposal.md", proposal_payload)

    row = _summary_row(
        target=target,
        target_dir=target_dir,
        pseudo_payload=pseudo_payload,
        recall_payload=recall_payload,
        proposal_payload=proposal_payload,
    )
    _write_json(target_dir / "summary.json", {"schema_version": "manual_text_target_audit_summary_v0", **row})
    _write_target_review(target_dir / "review.md", target, row, recall_payload, proposal_payload)
    return row


def write_summary_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Manual Text Target Audits v0",
        "",
        "Registry targets are used only as HumanML3D text pseudo-GT for audit and naming diagnostics.",
        "They do not create Motion-BPE merges, motion families, or runtime AML rules.",
        "",
        f"- targets: `{payload['summary']['target_count']}`",
        f"- total pseudo-GT cases: `{payload['summary']['total_pseudo_gt_cases']}`",
        "",
        "| target | pseudo-GT | seed motifs | candidate symbols | expanded precision | expanded recall | variants |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("targets") or []:
        lines.append(
            "| `{target}` | {gt} | {seed} | {cand} | {prec} | {rec} | {vars} |".format(
                target=row.get("target_alias"),
                gt=row.get("pseudo_gt_case_count"),
                seed=row.get("seed_selected_motif_count"),
                cand=row.get("candidate_symbol_count"),
                prec=row.get("expanded_precision"),
                rec=row.get("expanded_recall"),
                vars=row.get("proposal_variant_count"),
            )
        )
    lines.extend(["", "## Output Directories", ""])
    for row in payload.get("targets") or []:
        lines.append(f"- `{row.get('target_alias')}`: `{(row.get('outputs') or {}).get('dir')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_registry_audits(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_rows = _target_rows(Path(args.target_registry))
    requested = set(_target_ids(args, registry_rows))
    active_targets = [target for target in registry_rows if str(target["id"]) in requested]
    case_text, pseudo_gt_by_target = _load_case_text_and_pseudo_gt(Path(args.source_corpus), active_targets)
    seed_motifs_by_target = _select_seed_motif_ids_by_target(args, [str(target["id"]) for target in active_targets])
    symbol_rows = _scan_symbol_index(Path(args.bpe_sequences), args)
    promotion_index = family_proposals._index_promotion_candidates(str(args.candidates))
    rows: list[dict[str, Any]] = []
    for target in active_targets:
        target_id = str(target["id"])
        row = run_target(
            target,
            args,
            output_dir,
            case_text=case_text,
            pseudo_gt_cases=pseudo_gt_by_target[target_id],
            seed_motif_ids=seed_motifs_by_target[target_id],
            symbol_rows=symbol_rows,
            promotion_index=promotion_index,
        )
        rows.append(row)

    summary = {
        "schema_version": "manual_text_target_registry_audits_v0",
        "runtime_policy": "review-only; registry targets are text pseudo-GT for audit and naming diagnostics",
        "inputs": {
            "target_registry": str(args.target_registry),
            "source_corpus": str(args.source_corpus),
            "bpe_sequences": str(args.bpe_sequences),
            "candidates": str(args.candidates),
            "unit_types": _split_csv(args.unit_types),
            "min_channels": int(args.min_channels),
        },
        "summary": {
            "target_count": len(rows),
            "total_pseudo_gt_cases": sum(int(row.get("pseudo_gt_case_count") or 0) for row in rows),
            "indexed_case_count": len(case_text),
            "indexed_symbol_count": len(symbol_rows),
        },
        "targets": rows,
    }
    _write_json(output_dir / "manual_text_target_audit_summary.json", summary)
    write_summary_report(output_dir / "manual_text_target_audit_summary.md", summary)
    return summary


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.jsonl"
        seq = root / "seq.jsonl"
        candidates = root / "candidates.json"
        registry = root / "registry.json"
        out = root / "out"
        registry.write_text(
            json.dumps(
                {
                    "targets": [
                        {"id": "jumping_jack", "display_name": "jumping jack", "regex": r"\bjumping\s+jacks?\b"},
                        {"id": "sit", "display_name": "sit", "regex": r"\bsits?\s+down\b"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        source.write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "a", "caption_texts": ["does jumping jacks"], "caption_alias_ids": []}),
                    json.dumps({"case_id": "b", "caption_texts": ["sits down"], "caption_alias_ids": []}),
                    json.dumps({"case_id": "c", "caption_texts": ["walks"], "caption_alias_ids": []}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        seq.write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "a", "tokens": [{"symbol": "<COM_X>", "unit_type": "coordination_motif", "channels": ["a", "b"], "span": [0, 1]}]}),
                    json.dumps({"case_id": "b", "tokens": [{"symbol": "whole_body_state/SIT_LOW", "unit_type": "channel_event", "channels": ["whole_body_state"], "span": [0, 1]}]}),
                    json.dumps({"case_id": "c", "tokens": [{"symbol": "whole_body_state/SIT_LOW", "unit_type": "channel_event", "channels": ["whole_body_state"], "span": [0, 1]}]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        candidates.write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "source_motif_id": "<COM_X>",
                            "status": "promote_named_coordination_candidate",
                            "support": {"support_cases": 1, "occurrences": 1},
                            "motion_definition": {"required_channels": ["a", "b"], "required_geometry_clusters": ["g"]},
                            "naming_diagnostics": {"top_caption_alias": "jumping_jack"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            target_registry=str(registry),
            target_aliases="",
            source_corpus=str(source),
            bpe_sequences=str(seq),
            candidates=str(candidates),
            output_dir=str(out),
            candidate_statuses="promote_named_coordination_candidate,review_structural_coordination_candidate,review_named_low_support_candidate,diagnostic_coordination_motif",
            unit_types="channel_event,motif,coactivation_packet,coordination_motif",
            min_channels=1,
            min_false_negative_support=1,
            max_candidates=10,
            precision_floor=0.5,
            example_limit=2,
            report_detail_limit=2,
            promote_precision=0.8,
            promote_min_tp=1,
            review_precision=0.5,
            review_min_tp=1,
            reject_precision=0.2,
        )
        payload = build_registry_audits(args)
        assert payload["summary"]["target_count"] == 2
        assert (out / "jumping_jack" / "pattern_family_proposal.json").exists()
        assert (out / "sit" / "review.md").exists()
    print(json.dumps({"ok": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v0 manual text-target audits over learned Motion-BPE symbols.")
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--target-aliases", default="", help="Optional comma-separated subset of registry ids.")
    parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
    parser.add_argument("--bpe-sequences", default=str(DEFAULT_BPE_SEQUENCES))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate-statuses", default="promote_named_coordination_candidate,review_structural_coordination_candidate,review_named_low_support_candidate,diagnostic_coordination_motif")
    parser.add_argument("--unit-types", default="channel_event,motif,coactivation_packet,coordination_motif")
    parser.add_argument("--min-channels", type=int, default=1)
    parser.add_argument("--min-false-negative-support", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--precision-floor", type=float, default=0.65)
    parser.add_argument("--example-limit", type=int, default=8)
    parser.add_argument("--report-detail-limit", type=int, default=8)
    parser.add_argument("--promote-precision", type=float, default=0.85)
    parser.add_argument("--promote-min-tp", type=int, default=3)
    parser.add_argument("--review-precision", type=float, default=0.65)
    parser.add_argument("--review-min-tp", type=int, default=2)
    parser.add_argument("--reject-precision", type=float, default=0.25)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    payload = build_registry_audits(args)
    output_dir = Path(args.output_dir)
    print(output_dir / "manual_text_target_audit_summary.json")
    print(output_dir / "manual_text_target_audit_summary.md")
    print(json.dumps(payload["summary"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
