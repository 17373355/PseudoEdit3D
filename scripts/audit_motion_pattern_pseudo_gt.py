"""Audit a learned motion-pattern motif against text-derived pseudo GT.

This is an evaluation/naming diagnostic only. It does not affect Motion-BPE
tokens, merge selection, or the runtime AML tree.

Example target audit:
    python scripts/audit_motion_pattern_pseudo_gt.py \
      --target-alias jumping_jack \
      --source-corpus outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl \
      --bpe-sequences outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/case_multichannel_bpe_sequences.jsonl \
      --candidates outputs/aml_regression_testset_v2/coordination_pattern_promotion_candidates_loose_v1/coordination_pattern_promotion_candidates.json \
      --output-dir outputs/aml_regression_testset_v2/jumping_jack_pseudo_gt_audit_loose_v1

Quick check:
    python scripts/audit_motion_pattern_pseudo_gt.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_BPE_SEQUENCES = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/case_multichannel_bpe_sequences.jsonl")
DEFAULT_CANDIDATES = Path("outputs/aml_regression_testset_v2/coordination_pattern_promotion_candidates_loose_v1/coordination_pattern_promotion_candidates.json")
DEFAULT_TARGET_REGISTRY = Path("configs/motion_pattern_text_targets.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/motion_pattern_pseudo_gt_audit_v1")


def _load_target_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    targets = payload.get("targets") if isinstance(payload, dict) else payload
    if not isinstance(targets, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in targets:
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = item
    return out


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _target_re(args: argparse.Namespace) -> re.Pattern[str]:
    target_alias = str(args.target_alias or "")
    if not target_alias and not args.target_regex:
        raise ValueError("--target-alias or --target-regex is required outside --self-test")
    registry = _load_target_registry(Path(args.target_registry)) if getattr(args, "target_registry", "") else {}
    registry_pattern = str((registry.get(target_alias) or {}).get("regex") or "")
    pattern = str(args.target_regex or registry_pattern or re.escape(target_alias.replace("_", " ")))
    return re.compile(pattern, re.IGNORECASE)


def _caption_text(record: dict[str, Any]) -> str:
    return " ".join(str(item) for item in record.get("caption_texts") or [])


def load_pseudo_gt_cases(source_corpus: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    target_re = _target_re(args)
    alias = str(args.target_alias)
    positives: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(source_corpus):
        case_id = str(record.get("case_id") or "")
        aliases = {str(item) for item in record.get("caption_alias_ids") or []}
        text = _caption_text(record)
        alias_hit = alias in aliases
        regex_hit = bool(target_re.search(text))
        if alias_hit or regex_hit:
            positives[case_id] = {
                "case_id": case_id,
                "caption_texts": record.get("caption_texts") or [],
                "caption_alias_ids": sorted(aliases),
                "pseudo_gt_reason": {
                    "caption_alias_hit": alias_hit,
                    "regex_hit": regex_hit,
                    "regex": target_re.pattern,
                },
            }
    return positives


def load_all_case_captions(source_corpus: Path) -> dict[str, dict[str, Any]]:
    captions: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(source_corpus):
        case_id = str(record.get("case_id") or "")
        captions[case_id] = {
            "case_id": case_id,
            "caption_texts": record.get("caption_texts") or [],
            "caption_alias_ids": record.get("caption_alias_ids") or [],
        }
    return captions


def select_target_motif_ids(args: argparse.Namespace) -> list[str]:
    explicit = _split_csv(args.motif_ids)
    if explicit:
        return explicit
    statuses = set(_split_csv(args.candidate_statuses))
    if args.candidates:
        payload = _read_json(Path(args.candidates))
        motifs = []
        for candidate in payload.get("candidates") or []:
            naming = candidate.get("naming_diagnostics") or {}
            if str(naming.get("top_caption_alias") or "") != str(args.target_alias):
                continue
            if statuses and str(candidate.get("status") or "") not in statuses:
                continue
            motif_id = str(candidate.get("source_motif_id") or "")
            if motif_id:
                motifs.append(motif_id)
        return sorted(set(motifs))
    return []


def load_predictions(bpe_sequences: Path, motif_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not motif_ids:
        return {}
    with bpe_sequences.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            for token in row.get("tokens") or []:
                symbol = str(token.get("symbol") or "")
                if symbol not in motif_ids:
                    continue
                predictions[case_id].append(
                    {
                        "motif_id": symbol,
                        "span": token.get("span"),
                        "channels": token.get("channels") or [],
                        "geometry_clusters": token.get("geometry_clusters") or [],
                    }
                )
    return dict(predictions)


def _safe_div(num: int, denom: int) -> float:
    return 0.0 if denom == 0 else num / denom


def _case_examples(case_ids: list[str], captions: dict[str, dict[str, Any]], predictions: dict[str, list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_id in case_ids[:limit]:
        info = captions.get(case_id, {"case_id": case_id, "caption_texts": [], "caption_alias_ids": []})
        out.append(
            {
                "case_id": case_id,
                "caption_texts": info.get("caption_texts") or [],
                "caption_alias_ids": info.get("caption_alias_ids") or [],
                "predictions": predictions.get(case_id, []),
            }
        )
    return out


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    source_corpus = Path(args.source_corpus)
    bpe_sequences = Path(args.bpe_sequences)
    pseudo_gt = load_pseudo_gt_cases(source_corpus, args)
    captions = load_all_case_captions(source_corpus)
    motif_ids = select_target_motif_ids(args)
    predictions = load_predictions(bpe_sequences, set(motif_ids))

    gt_cases = set(pseudo_gt)
    pred_cases = set(predictions)
    tp = sorted(gt_cases & pred_cases)
    fp = sorted(pred_cases - gt_cases)
    fn = sorted(gt_cases - pred_cases)
    precision = _safe_div(len(tp), len(pred_cases))
    recall = _safe_div(len(tp), len(gt_cases))
    f1 = _safe_div(2 * len(tp), 2 * len(tp) + len(fp) + len(fn))
    return {
        "schema_version": "motion_pattern_pseudo_gt_audit_v1",
        "target_alias": args.target_alias,
        "target_regex": _target_re(args).pattern,
        "policy": {
            "purpose": "evaluation and naming diagnostic only",
            "pseudo_gt": "HumanML3D caption_alias_ids plus target regex over caption_texts",
            "prediction": "case contains one of the selected learned motif ids in BPE sequences",
        },
        "inputs": {
            "source_corpus": str(source_corpus),
            "bpe_sequences": str(bpe_sequences),
            "candidates": str(args.candidates or ""),
            "candidate_statuses": _split_csv(args.candidate_statuses),
        },
        "selected_motif_ids": motif_ids,
        "metrics": {
            "pseudo_gt_case_count": len(gt_cases),
            "predicted_case_count": len(pred_cases),
            "true_positive_count": len(tp),
            "false_positive_count": len(fp),
            "false_negative_count": len(fn),
            "precision_subset_accuracy": round(precision, 6),
            "recall_against_text_pseudo_gt": round(recall, 6),
            "f1": round(f1, 6),
        },
        "case_ids": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        },
        "examples": {
            "true_positive": _case_examples(tp, captions, predictions, int(args.example_limit)),
            "false_positive": _case_examples(fp, captions, predictions, int(args.example_limit)),
            "false_negative": _case_examples(fn, captions, predictions, int(args.example_limit)),
        },
    }


def write_report(path: Path, audit: dict[str, Any]) -> None:
    metrics = audit.get("metrics") or {}
    lines = [
        f"# Pattern Pseudo-GT Audit: {audit.get('target_alias')}",
        "",
        "This is an evaluation/naming diagnostic only. It does not affect Motion-BPE or AML runtime rules.",
        "",
        f"- selected motifs: `{audit.get('selected_motif_ids')}`",
        f"- pseudo-GT cases: `{metrics.get('pseudo_gt_case_count')}`",
        f"- predicted cases: `{metrics.get('predicted_case_count')}`",
        f"- true positives: `{metrics.get('true_positive_count')}`",
        f"- false positives: `{metrics.get('false_positive_count')}`",
        f"- false negatives: `{metrics.get('false_negative_count')}`",
        f"- precision / subset accuracy: `{metrics.get('precision_subset_accuracy')}`",
        f"- recall against text pseudo-GT: `{metrics.get('recall_against_text_pseudo_gt')}`",
        f"- f1: `{metrics.get('f1')}`",
        "",
    ]
    for bucket in ["true_positive", "false_positive", "false_negative"]:
        lines.append(f"## {bucket}")
        lines.append("")
        for item in (audit.get("examples") or {}).get(bucket) or []:
            captions = " / ".join(str(text).replace("\n", " ") for text in item.get("caption_texts", [])[:3])
            lines.append(f"- `{item.get('case_id')}`: {captions}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.jsonl"
        seq = root / "seq.jsonl"
        candidates = root / "candidates.json"
        source.write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "a", "caption_texts": ["does jumping jacks"], "caption_alias_ids": []}),
                    json.dumps({"case_id": "b", "caption_texts": ["walks"], "caption_alias_ids": []}),
                    json.dumps({"case_id": "c", "caption_texts": ["star jumps"], "caption_alias_ids": []}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        seq.write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "a", "tokens": [{"symbol": "<COM_X>", "span": [0, 1]}]}),
                    json.dumps({"case_id": "b", "tokens": [{"symbol": "<COM_X>", "span": [0, 1]}]}),
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
                            "naming_diagnostics": {"top_caption_alias": "jumping_jack"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            target_alias="jumping_jack",
            target_regex=r"\bjumping\s+jacks?\b|\bstar\s*jumps?\b",
            target_registry="",
            source_corpus=str(source),
            bpe_sequences=str(seq),
            candidates=str(candidates),
            motif_ids="",
            candidate_statuses="promote_named_coordination_candidate",
            example_limit=4,
        )
        audit = build_audit(args)
        assert audit["metrics"]["pseudo_gt_case_count"] == 2
        assert audit["metrics"]["predicted_case_count"] == 2
        assert audit["metrics"]["true_positive_count"] == 1
        assert audit["metrics"]["false_positive_count"] == 1
        assert audit["metrics"]["false_negative_count"] == 1
    print(json.dumps({"ok": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a learned motion pattern against text-derived pseudo GT.")
    parser.add_argument("--target-alias", default="")
    parser.add_argument("--target-regex", default="")
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
    parser.add_argument("--bpe-sequences", default=str(DEFAULT_BPE_SEQUENCES))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--motif-ids", default="", help="Optional comma-separated motif ids; overrides candidate selection.")
    parser.add_argument("--candidate-statuses", default="promote_named_coordination_candidate,review_named_low_support_candidate")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--example-limit", type=int, default=12)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.target_alias and not args.target_regex:
        parser.error("--target-alias or --target-regex is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    _write_json(output_dir / "pattern_pseudo_gt_audit.json", audit)
    write_report(output_dir / "pattern_pseudo_gt_audit.md", audit)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "motion_pattern_pseudo_gt_audit_summary_v1",
            "target_alias": audit["target_alias"],
            "selected_motif_ids": audit["selected_motif_ids"],
            **audit["metrics"],
            "outputs": {
                "audit": str(output_dir / "pattern_pseudo_gt_audit.json"),
                "review": str(output_dir / "pattern_pseudo_gt_audit.md"),
                "summary": str(output_dir / "summary.json"),
            },
        },
    )
    print(output_dir / "summary.json")
    print(output_dir / "pattern_pseudo_gt_audit.md")


if __name__ == "__main__":
    main()
