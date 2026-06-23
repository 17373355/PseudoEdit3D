"""Build review-only motion pattern family proposals.

The script is target-agnostic. Compact action names are input labels from
text-pseudo-GT audits; no action-specific motion logic is hard-coded here.

Inputs:
  - pattern pseudo-GT audit JSON
  - recall-candidate diagnostic JSON
  - optional promotion candidates JSON for seed motif details

Output:
  - pattern_family_proposal.json
  - pattern_family_proposal.md
  - summary.json

Quick check:
    python scripts/build_motion_pattern_family_proposals.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/pattern_family_proposals_v1")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_id(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    compact = "".join(out).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or "unnamed"


def _stable_short_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _top_counter(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _index_promotion_candidates(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = _read_json(p)
    out: dict[str, dict[str, Any]] = {}
    for item in payload.get("candidates") or []:
        motif_id = str(item.get("source_motif_id") or "")
        if motif_id:
            out[motif_id] = item
    return out


def _counts_from_variant_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for item in row.get(key) or []:
            counter[str(item.get("id") or "")] += int(item.get("count") or 0)
    if "" in counter:
        del counter[""]
    return _top_counter(counter)


def _variant_status(row: dict[str, Any], greedy_symbols: set[str], args: argparse.Namespace) -> str:
    precision = float(row.get("candidate_precision") or 0.0)
    incremental_tp = int(row.get("incremental_true_positive") or row.get("false_negative_support_cases") or 0)
    incremental_fp = int(row.get("incremental_false_positive") or 0)
    symbol = str(row.get("symbol") or "")
    if symbol in greedy_symbols and precision >= float(args.promote_precision) and incremental_tp >= int(args.promote_min_tp):
        return "promote_family_variant_candidate"
    if symbol in greedy_symbols:
        return "review_family_variant_candidate"
    if precision >= float(args.review_precision) and incremental_tp >= int(args.review_min_tp):
        return "review_family_variant_candidate"
    if incremental_fp > incremental_tp or precision < float(args.reject_precision):
        return "reject_noisy_variant_candidate"
    return "diagnostic_family_variant_candidate"


def _seed_variants(pseudo_gt_audit: dict[str, Any], promotion_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    motif_ids = [str(item) for item in pseudo_gt_audit.get("selected_motif_ids") or []]
    variants = []
    metrics = pseudo_gt_audit.get("metrics") or {}
    for motif_id in motif_ids:
        promotion = promotion_index.get(motif_id) or {}
        motion = promotion.get("motion_definition") or {}
        support = promotion.get("support") or {}
        naming = promotion.get("naming_diagnostics") or {}
        variants.append(
            {
                "variant_id": "seed_" + _safe_id(motif_id),
                "variant_kind": "seed_motif",
                "status": "seed_promoted_motif",
                "symbol": motif_id,
                "source_candidate_id": promotion.get("candidate_id"),
                "support": {
                    "support_cases": support.get("support_cases"),
                    "occurrences": support.get("occurrences"),
                },
                "metrics": {
                    "precision": metrics.get("precision_subset_accuracy"),
                    "recall": metrics.get("recall_against_text_pseudo_gt"),
                    "true_positive_count": metrics.get("true_positive_count"),
                    "false_positive_count": metrics.get("false_positive_count"),
                    "predicted_case_count": metrics.get("predicted_case_count"),
                },
                "motion_signature": {
                    "parent_signature": motion.get("parent_signature"),
                    "channels": motion.get("required_channels") or [],
                    "geometry_clusters": motion.get("required_geometry_clusters") or [],
                    "top_channels": motion.get("top_channels") or [],
                    "top_geometry_clusters": motion.get("top_geometry_clusters") or [],
                },
                "naming_diagnostics": naming,
                "examples": promotion.get("review_examples") or [],
            }
        )
    return variants


def _candidate_variants(recall_payload: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    greedy = recall_payload.get("greedy_precision_preserving_expansion") or {}
    greedy_by_symbol = {str(item.get("symbol") or ""): item for item in greedy.get("selected") or []}
    greedy_symbols = set(greedy_by_symbol)
    variants = []
    for row in recall_payload.get("recall_candidates") or []:
        symbol = str(row.get("symbol") or "")
        greedy_row = greedy_by_symbol.get(symbol) or {}
        metrics = {
            "candidate_precision": row.get("candidate_precision"),
            "candidate_recall": row.get("candidate_recall"),
            "union_precision_with_seed": row.get("union_precision_with_seed"),
            "union_recall_with_seed": row.get("union_recall_with_seed"),
            "incremental_true_positive": greedy_row.get("incremental_true_positive", row.get("incremental_true_positive")),
            "incremental_false_positive": greedy_row.get("incremental_false_positive", row.get("incremental_false_positive")),
            "precision_after_add": greedy_row.get("precision_after_add"),
            "recall_after_add": greedy_row.get("recall_after_add"),
        }
        variants.append(
            {
                "variant_id": "variant_" + _safe_id(symbol)[:80] + "_" + _stable_short_hash(symbol),
                "variant_kind": "recall_candidate_symbol",
                "status": _variant_status(row, greedy_symbols, args),
                "symbol": symbol,
                "support": {
                    "support_cases": row.get("support_cases"),
                    "occurrences": row.get("occurrences"),
                    "pseudo_gt_support_cases": row.get("pseudo_gt_support_cases"),
                    "non_pseudo_gt_support_cases": row.get("non_pseudo_gt_support_cases"),
                    "false_negative_support_cases": row.get("false_negative_support_cases"),
                },
                "metrics": metrics,
                "motion_signature": {
                    "unit_types": row.get("unit_types") or [],
                    "channels": [item.get("id") for item in row.get("channels") or [] if item.get("id")],
                    "geometry_clusters": [item.get("id") for item in row.get("geometry_clusters") or [] if item.get("id")],
                    "top_channels": row.get("channels") or [],
                    "top_geometry_clusters": row.get("geometry_clusters") or [],
                },
                "examples": {
                    "false_negative": row.get("false_negative_examples") or [],
                    "false_positive": row.get("false_positive_examples") or [],
                },
            }
        )
    return variants


def build_proposal(
    pseudo_gt_audit: dict[str, Any],
    recall_payload: dict[str, Any],
    promotion_index: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_alias = str(recall_payload.get("target_alias") or pseudo_gt_audit.get("target_alias") or args.target_alias)
    seed = recall_payload.get("seed") or {}
    greedy = recall_payload.get("greedy_precision_preserving_expansion") or {}
    seed_variants = _seed_variants(pseudo_gt_audit, promotion_index)
    candidate_variants = _candidate_variants(recall_payload, args)
    variants = seed_variants + candidate_variants
    status_counts = Counter(str(item.get("status") or "") for item in variants)
    promoted_or_review = [
        item
        for item in variants
        if str(item.get("status") or "") in {"seed_promoted_motif", "promote_family_variant_candidate", "review_family_variant_candidate"}
    ]
    family_channels = _counts_from_variant_rows([item.get("motion_signature") or {} for item in promoted_or_review], "top_channels")
    family_geometry = _counts_from_variant_rows([item.get("motion_signature") or {} for item in promoted_or_review], "top_geometry_clusters")
    return {
        "schema_version": "motion_pattern_family_proposal_v1",
        "proposal_id": "pattern_family_" + _safe_id(target_alias),
        "target_alias": target_alias,
        "runtime_policy": "review-only family proposal; do not modify the AML runtime tree automatically",
        "inputs": {
            "pseudo_gt_audit": str(args.pseudo_gt_audit),
            "recall_candidates": str(args.recall_candidates),
            "promotion_candidates": str(args.promotion_candidates or ""),
        },
        "pseudo_gt_policy": pseudo_gt_audit.get("policy") or {},
        "seed_metrics": seed,
        "expanded_metrics": greedy,
        "summary": {
            "variant_count": len(variants),
            "seed_variant_count": len(seed_variants),
            "recall_candidate_count": len(candidate_variants),
            "status_counts": dict(sorted(status_counts.items())),
            "recommended_family_precision": greedy.get("precision_subset_accuracy", seed.get("precision_subset_accuracy")),
            "recommended_family_recall": greedy.get("recall_against_text_pseudo_gt", seed.get("recall_against_text_pseudo_gt")),
        },
        "family_motion_summary": {
            "top_channels": family_channels,
            "top_geometry_clusters": family_geometry,
        },
        "variants": variants,
    }


def write_report(path: Path, proposal: dict[str, Any]) -> None:
    summary = proposal.get("summary") or {}
    seed = proposal.get("seed_metrics") or {}
    expanded = proposal.get("expanded_metrics") or {}
    family_motion = proposal.get("family_motion_summary") or {}
    lines = [
        f"# Motion Pattern Family Proposal: {proposal.get('target_alias')}",
        "",
        "Review-only artifact. This does not modify the AML runtime tree.",
        "",
        "## Summary",
        "",
        f"- proposal id: `{proposal.get('proposal_id')}`",
        f"- variants: `{summary.get('variant_count')}`",
        f"- status counts: `{summary.get('status_counts')}`",
        f"- seed precision: `{seed.get('precision_subset_accuracy')}`",
        f"- seed recall: `{seed.get('recall_against_text_pseudo_gt')}`",
        f"- expanded precision: `{expanded.get('precision_subset_accuracy')}`",
        f"- expanded recall: `{expanded.get('recall_against_text_pseudo_gt')}`",
        "",
        "## Family Motion Summary",
        "",
        f"- top channels: `{family_motion.get('top_channels')}`",
        f"- top geometry: `{family_motion.get('top_geometry_clusters')}`",
        "",
        "## Variants",
        "",
        "| status | symbol | precision | +TP | +FP | channels | top geometry |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for variant in proposal.get("variants") or []:
        metrics = variant.get("metrics") or {}
        motion = variant.get("motion_signature") or {}
        symbol = str(variant.get("symbol") or "").replace("|", "\\|")
        precision = metrics.get("candidate_precision", metrics.get("precision"))
        inc_tp = metrics.get("incremental_true_positive", metrics.get("true_positive_count"))
        inc_fp = metrics.get("incremental_false_positive", metrics.get("false_positive_count"))
        channels = ", ".join(str(item) for item in (motion.get("channels") or [])[:4])
        geometry = ", ".join(str(item) for item in (motion.get("geometry_clusters") or [])[:5])
        lines.append(
            f"| `{variant.get('status')}` | `{symbol}` | {precision} | {inc_tp} | {inc_fp} | {channels} | {geometry} |"
        )
    lines.append("")
    for variant in proposal.get("variants") or []:
        if str(variant.get("status") or "").startswith("reject"):
            continue
        lines.append(f"### {variant.get('variant_id')}")
        lines.append("")
        lines.append(f"- status: `{variant.get('status')}`")
        lines.append(f"- symbol: `{variant.get('symbol')}`")
        lines.append(f"- metrics: `{variant.get('metrics')}`")
        lines.append(f"- motion signature: `{variant.get('motion_signature')}`")
        examples = variant.get("examples") or []
        if isinstance(examples, dict):
            examples = examples.get("false_negative") or examples.get("false_positive") or []
        lines.append("- examples:")
        for example in examples[:6]:
            captions = example.get("caption_texts")
            if captions:
                caption = " / ".join(str(text).replace("\n", " ") for text in captions[:3])
            else:
                caption = str(example.get("caption") or "")
            lines.append(f"  - `{example.get('case_id')}`: {caption}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pseudo = {
            "target_alias": "toy_action",
            "policy": {},
            "selected_motif_ids": ["<COM_A>"],
            "metrics": {
                "precision_subset_accuracy": 0.8,
                "recall_against_text_pseudo_gt": 0.2,
                "true_positive_count": 4,
                "false_positive_count": 1,
                "predicted_case_count": 5,
            },
        }
        recall = {
            "target_alias": "toy_action",
            "seed": {"selected_motif_ids": ["<COM_A>"], "precision_subset_accuracy": 0.8, "recall_against_text_pseudo_gt": 0.2},
            "greedy_precision_preserving_expansion": {
                "precision_subset_accuracy": 0.85,
                "recall_against_text_pseudo_gt": 0.4,
                "selected": [{"symbol": "COORD_SIG[x]", "incremental_true_positive": 3, "incremental_false_positive": 0}],
            },
            "recall_candidates": [
                {
                    "symbol": "COORD_SIG[x]",
                    "candidate_precision": 1.0,
                    "incremental_true_positive": 3,
                    "incremental_false_positive": 0,
                    "channels": [{"id": "a", "count": 3}],
                    "geometry_clusters": [{"id": "G", "count": 3}],
                    "false_negative_examples": [],
                    "false_positive_examples": [],
                }
            ],
        }
        promotion = {
            "candidates": [
                {
                    "source_motif_id": "<COM_A>",
                    "candidate_id": "seed",
                    "support": {"support_cases": 5, "occurrences": 5},
                    "motion_definition": {"required_channels": ["a"], "required_geometry_clusters": ["G"]},
                    "naming_diagnostics": {},
                    "review_examples": [],
                }
            ]
        }
        pseudo_path = root / "pseudo.json"
        recall_path = root / "recall.json"
        promotion_path = root / "promotion.json"
        pseudo_path.write_text(json.dumps(pseudo), encoding="utf-8")
        recall_path.write_text(json.dumps(recall), encoding="utf-8")
        promotion_path.write_text(json.dumps(promotion), encoding="utf-8")
        args = argparse.Namespace(
            target_alias="toy_action",
            pseudo_gt_audit=str(pseudo_path),
            recall_candidates=str(recall_path),
            promotion_candidates=str(promotion_path),
            promote_precision=0.85,
            promote_min_tp=2,
            review_precision=0.65,
            review_min_tp=1,
            reject_precision=0.25,
        )
        proposal = build_proposal(pseudo, recall, _index_promotion_candidates(str(promotion_path)), args)
        assert proposal["target_alias"] == "toy_action"
        assert proposal["summary"]["variant_count"] == 2
        assert proposal["summary"]["status_counts"]["promote_family_variant_candidate"] == 1
    print(json.dumps({"ok": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build review-only motion pattern family proposals from generic audits.")
    parser.add_argument("--target-alias", default="")
    parser.add_argument("--pseudo-gt-audit", default="")
    parser.add_argument("--recall-candidates", default="")
    parser.add_argument("--promotion-candidates", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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
    if not args.pseudo_gt_audit or not args.recall_candidates:
        parser.error("--pseudo-gt-audit and --recall-candidates are required")

    pseudo_gt_audit = _read_json(Path(args.pseudo_gt_audit))
    recall_payload = _read_json(Path(args.recall_candidates))
    if not args.target_alias:
        args.target_alias = str(recall_payload.get("target_alias") or pseudo_gt_audit.get("target_alias") or "")
    output_dir = Path(args.output_dir) / _safe_id(args.target_alias)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal = build_proposal(pseudo_gt_audit, recall_payload, _index_promotion_candidates(args.promotion_candidates), args)
    _write_json(output_dir / "pattern_family_proposal.json", proposal)
    write_report(output_dir / "pattern_family_proposal.md", proposal)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "motion_pattern_family_proposal_summary_v1",
            "proposal_id": proposal["proposal_id"],
            "target_alias": proposal["target_alias"],
            **(proposal.get("summary") or {}),
            "outputs": {
                "proposal": str(output_dir / "pattern_family_proposal.json"),
                "review": str(output_dir / "pattern_family_proposal.md"),
                "summary": str(output_dir / "summary.json"),
            },
        },
    )
    print(output_dir / "summary.json")
    print(output_dir / "pattern_family_proposal.md")


if __name__ == "__main__":
    main()
