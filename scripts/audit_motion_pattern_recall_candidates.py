"""Find recall-improvement candidates for a text-pseudo-GT motion pattern.

This is a diagnostic script. It does not change Motion-BPE, promotion queues,
or the AML runtime tree.

Example target audit:
    python scripts/audit_motion_pattern_recall_candidates.py \
      --target-alias jumping_jack \
      --source-corpus outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl \
      --bpe-sequences outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/case_multichannel_bpe_sequences.jsonl \
      --candidates outputs/aml_regression_testset_v2/coordination_pattern_promotion_candidates_loose_v1/coordination_pattern_promotion_candidates.json \
      --output-dir outputs/aml_regression_testset_v2/jumping_jack_recall_candidates_loose_v1

Quick check:
    python scripts/audit_motion_pattern_recall_candidates.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_BPE_SEQUENCES = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/case_multichannel_bpe_sequences.jsonl")
DEFAULT_CANDIDATES = Path("outputs/aml_regression_testset_v2/coordination_pattern_promotion_candidates_loose_v1/coordination_pattern_promotion_candidates.json")
DEFAULT_TARGET_REGISTRY = Path("configs/motion_pattern_text_targets.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/motion_pattern_recall_candidates_v1")


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


def load_case_text(source_corpus: Path, args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], set[str]]:
    target_re = _target_re(args)
    target_alias = str(args.target_alias)
    records: dict[str, dict[str, Any]] = {}
    pseudo_gt: set[str] = set()
    with source_corpus.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record.get("case_id") or "")
            aliases = [str(item) for item in record.get("caption_alias_ids") or []]
            text = _caption_text(record)
            hit = target_alias in set(aliases) or bool(target_re.search(text))
            records[case_id] = {
                "case_id": case_id,
                "caption_texts": record.get("caption_texts") or [],
                "caption_alias_ids": aliases,
                "is_pseudo_gt": hit,
            }
            if hit:
                pseudo_gt.add(case_id)
    return records, pseudo_gt


def select_seed_motif_ids(args: argparse.Namespace) -> list[str]:
    explicit = _split_csv(args.motif_ids)
    if explicit:
        return explicit
    statuses = set(_split_csv(args.candidate_statuses))
    payload = _read_json(Path(args.candidates))
    motif_ids: list[str] = []
    for candidate in payload.get("candidates") or []:
        naming = candidate.get("naming_diagnostics") or {}
        if str(naming.get("top_caption_alias") or "") != str(args.target_alias):
            continue
        if statuses and str(candidate.get("status") or "") not in statuses:
            continue
        motif_id = str(candidate.get("source_motif_id") or "")
        if motif_id:
            motif_ids.append(motif_id)
    return sorted(set(motif_ids))


def _token_signature(token: dict[str, Any]) -> str:
    return str(token.get("symbol") or "")


def _is_candidate_token(token: dict[str, Any], args: argparse.Namespace) -> bool:
    unit_type = str(token.get("unit_type") or "")
    symbol = _token_signature(token)
    if unit_type not in set(_split_csv(args.unit_types)):
        return False
    if not symbol:
        return False
    channels = {str(item) for item in token.get("channels") or []}
    if len(channels) < int(args.min_channels):
        return False
    return True


def scan_bpe_sequences(
    bpe_sequences: Path,
    args: argparse.Namespace,
    *,
    seed_motif_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    seed_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    symbol_rows: dict[str, dict[str, Any]] = {}
    with bpe_sequences.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            for token in row.get("tokens") or []:
                symbol = _token_signature(token)
                if symbol in seed_motif_ids:
                    seed_predictions[case_id].append(token)
                if not _is_candidate_token(token, args):
                    continue
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
                    entry["examples_by_case"][case_id].append(
                        {
                            "span": token.get("span"),
                            "unit_type": token.get("unit_type"),
                            "channels": token.get("channels") or [],
                            "geometry_clusters": token.get("geometry_clusters") or [],
                        }
                    )
    return dict(seed_predictions), symbol_rows


def _safe_div(num: int, denom: int) -> float:
    return 0.0 if denom == 0 else num / denom


def _top_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


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


def build_recall_candidates(args: argparse.Namespace) -> dict[str, Any]:
    case_text, pseudo_gt_cases = load_case_text(Path(args.source_corpus), args)
    seed_motif_ids = set(select_seed_motif_ids(args))
    seed_predictions, symbol_rows = scan_bpe_sequences(Path(args.bpe_sequences), args, seed_motif_ids=seed_motif_ids)
    seed_pred_cases = set(seed_predictions)
    seed_tp_cases = pseudo_gt_cases & seed_pred_cases
    seed_fp_cases = seed_pred_cases - pseudo_gt_cases
    false_negative_cases = pseudo_gt_cases - seed_pred_cases

    rows: list[dict[str, Any]] = []
    candidate_case_sets: dict[str, set[str]] = {}
    for symbol, entry in symbol_rows.items():
        if symbol in seed_motif_ids:
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
        candidate_precision = _safe_div(len(gt_hits), len(case_ids))
        union_precision = _safe_div(len(union_tp), len(union_pred))
        union_recall = _safe_div(len(union_tp), len(pseudo_gt_cases))
        rows.append(
            {
                "symbol": symbol,
                "support_cases": len(case_ids),
                "occurrences": int(entry["occurrences"]),
                "pseudo_gt_support_cases": len(gt_hits),
                "non_pseudo_gt_support_cases": len(non_gt_hits),
                "false_negative_support_cases": len(fn_hits),
                "candidate_precision": round(candidate_precision, 6),
                "candidate_recall": round(_safe_div(len(gt_hits), len(pseudo_gt_cases)), 6),
                "union_precision_with_seed": round(union_precision, 6),
                "union_recall_with_seed": round(union_recall, 6),
                "incremental_true_positive": len(fn_hits),
                "incremental_false_positive": len(non_gt_hits - seed_fp_cases),
                "unit_types": _top_counter(entry["unit_types"], 6),
                "channels": _top_counter(entry["channels"], 8),
                "geometry_clusters": _top_counter(entry["geometry_clusters"], 12),
                "false_negative_examples": _examples(
                    fn_hits,
                    case_text,
                    entry["examples_by_case"],
                    int(args.example_limit),
                ),
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
        "target_alias": args.target_alias,
        "target_regex": _target_re(args).pattern,
        "policy": {
            "purpose": "diagnostic only; candidates do not change Motion-BPE or AML runtime",
            "seed_prediction": "cases containing selected seed motif ids",
            "candidate_prediction": "cases containing candidate coordination symbols",
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
        if not case_ids:
            continue
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


def write_report(path: Path, payload: dict[str, Any]) -> None:
    seed = payload.get("seed") or {}
    summary = payload.get("summary") or {}
    lines = [
        f"# Recall Candidates: {payload.get('target_alias')}",
        "",
        "Diagnostic only. Text pseudo-GT is used only for audit.",
        "",
        f"- seed motifs: `{seed.get('selected_motif_ids')}`",
        f"- pseudo-GT cases: `{seed.get('pseudo_gt_case_count')}`",
        f"- seed predicted cases: `{seed.get('predicted_case_count')}`",
        f"- seed precision: `{seed.get('precision_subset_accuracy')}`",
        f"- seed recall: `{seed.get('recall_against_text_pseudo_gt')}`",
        f"- candidate symbols: `{summary.get('candidate_symbol_count')}`",
        "",
        "## Greedy precision-preserving expansion",
        "",
    ]
    greedy = payload.get("greedy_precision_preserving_expansion") or {}
    lines.extend(
        [
            f"- precision floor: `{greedy.get('precision_floor')}`",
            f"- selected candidates: `{greedy.get('selected_count')}`",
            f"- expanded predicted cases: `{greedy.get('predicted_case_count')}`",
            f"- expanded true positives: `{greedy.get('true_positive_count')}`",
            f"- expanded false positives: `{greedy.get('false_positive_count')}`",
            f"- expanded precision: `{greedy.get('precision_subset_accuracy')}`",
            f"- expanded recall: `{greedy.get('recall_against_text_pseudo_gt')}`",
            "",
            "| step | symbol | +TP | +FP | precision | recall |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(greedy.get("selected") or [], start=1):
        symbol = str(row.get("symbol") or "").replace("|", "\\|")
        lines.append(
            f"| {idx} | `{symbol}` | {row.get('incremental_true_positive')} | {row.get('incremental_false_positive')} | "
            f"{row.get('precision_after_add')} | {row.get('recall_after_add')} |"
        )
    lines.extend(
        [
            "",
            "## Ranked individual candidates",
            "",
        "| rank | symbol | FN hits | support | cand precision | union precision | union recall | top geometry |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx, row in enumerate(payload.get("recall_candidates") or [], start=1):
        geometry = ", ".join(item["id"] for item in row.get("geometry_clusters", [])[:4])
        symbol = str(row.get("symbol") or "").replace("|", "\\|")
        lines.append(
            f"| {idx} | `{symbol}` | {row.get('false_negative_support_cases')} | {row.get('support_cases')} | "
            f"{row.get('candidate_precision')} | {row.get('union_precision_with_seed')} | {row.get('union_recall_with_seed')} | {geometry} |"
        )
    lines.append("")
    for idx, row in enumerate((payload.get("recall_candidates") or [])[: int(payload.get("report_detail_limit") or 8)], start=1):
        lines.append(f"## {idx}. `{row.get('symbol')}`")
        lines.append("")
        lines.append(f"- false-negative hits: `{row.get('false_negative_support_cases')}`")
        lines.append(f"- support cases: `{row.get('support_cases')}`")
        lines.append(f"- candidate precision: `{row.get('candidate_precision')}`")
        lines.append(f"- union precision with seed: `{row.get('union_precision_with_seed')}`")
        lines.append(f"- union recall with seed: `{row.get('union_recall_with_seed')}`")
        lines.append(f"- channels: `{row.get('channels')}`")
        lines.append(f"- geometry: `{row.get('geometry_clusters')}`")
        lines.append("- false-negative examples:")
        for example in row.get("false_negative_examples") or []:
            captions = " / ".join(str(text).replace("\n", " ") for text in example.get("caption_texts", [])[:3])
            lines.append(f"  - `{example.get('case_id')}`: {captions}")
        lines.append("- false-positive examples:")
        for example in row.get("false_positive_examples") or []:
            captions = " / ".join(str(text).replace("\n", " ") for text in example.get("caption_texts", [])[:2])
            lines.append(f"  - `{example.get('case_id')}`: {captions}")
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
                    json.dumps({"case_id": "b", "caption_texts": ["does star jumps"], "caption_alias_ids": []}),
                    json.dumps({"case_id": "c", "caption_texts": ["walks"], "caption_alias_ids": []}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        seq.write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "a", "tokens": [{"symbol": "<COM_SEED>", "unit_type": "coordination_motif", "channels": ["a", "b"], "span": [0, 1]}]}),
                    json.dumps({"case_id": "b", "tokens": [{"symbol": "COORD_SIG[x]", "unit_type": "coactivation_packet", "channels": ["a", "b"], "geometry_clusters": ["g"], "span": [0, 1]}]}),
                    json.dumps({"case_id": "c", "tokens": [{"symbol": "COORD_SIG[x]", "unit_type": "coactivation_packet", "channels": ["a", "b"], "geometry_clusters": ["g"], "span": [0, 1]}]}),
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
                            "source_motif_id": "<COM_SEED>",
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
            unit_types="coordination_motif,coactivation_packet",
            min_channels=2,
            min_false_negative_support=1,
            max_candidates=5,
            precision_floor=0.6,
            example_limit=2,
            report_detail_limit=2,
        )
        payload = build_recall_candidates(args)
        assert payload["seed"]["true_positive_count"] == 1
        assert payload["recall_candidates"][0]["symbol"] == "COORD_SIG[x]"
        assert payload["recall_candidates"][0]["incremental_true_positive"] == 1
    print(json.dumps({"ok": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Find recall-improvement candidates for a text-pseudo-GT motion pattern.")
    parser.add_argument("--target-alias", default="")
    parser.add_argument("--target-regex", default="")
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
    parser.add_argument("--bpe-sequences", default=str(DEFAULT_BPE_SEQUENCES))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--motif-ids", default="", help="Optional comma-separated seed motif ids; overrides candidate selection.")
    parser.add_argument("--candidate-statuses", default="promote_named_coordination_candidate,review_named_low_support_candidate")
    parser.add_argument("--unit-types", default="coordination_motif,coactivation_packet")
    parser.add_argument("--min-channels", type=int, default=2)
    parser.add_argument("--min-false-negative-support", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--precision-floor", type=float, default=0.80)
    parser.add_argument("--example-limit", type=int, default=8)
    parser.add_argument("--report-detail-limit", type=int, default=10)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.target_alias and not args.target_regex:
        parser.error("--target-alias or --target-regex is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_recall_candidates(args)
    payload["report_detail_limit"] = int(args.report_detail_limit)
    _write_json(output_dir / "recall_candidate_symbols.json", payload)
    write_report(output_dir / "recall_candidate_symbols.md", payload)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "motion_pattern_recall_candidates_summary_v1",
            "target_alias": payload["target_alias"],
            **(payload.get("seed") or {}),
            **(payload.get("summary") or {}),
            "greedy_precision_preserving_expansion": payload.get("greedy_precision_preserving_expansion"),
            "outputs": {
                "candidates": str(output_dir / "recall_candidate_symbols.json"),
                "review": str(output_dir / "recall_candidate_symbols.md"),
                "summary": str(output_dir / "summary.json"),
            },
        },
    )
    print(output_dir / "summary.json")
    print(output_dir / "recall_candidate_symbols.md")


if __name__ == "__main__":
    main()
