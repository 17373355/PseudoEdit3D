"""Audit why full motion patterns did not emerge from Motion-BPE.

This script is diagnostic only. It uses HumanML3D text targets as pseudo-GT
for auditing, but text targets do not create motion tokens, BPE merges, or AML
runtime rules.

Pipeline:
1. Load text pseudo-GT targets from configs/motion_pattern_text_targets.json.
2. Load channel-BPE sequences from a multichannel Motion-BPE run.
3. Rebuild potential overlapping channel coactivations. The default `all_units`
   view uses both base channel events and learned channel motifs; the optional
   `channel_motifs` view reproduces the current Motion-BPE coordination stage
   and uses learned <CHM_*> motifs only.
4. For each text target, rank coactivation symbols by target precision/recall.
5. Write a concise JSON/MD report explaining whether the missing full pattern
   is due to component-only structure, fragmented coactivation, text diversity,
   or selection threshold/budget.

Default full audit:
    python scripts/audit_hml3d_coactivation_recall_v0.py

Current-stage-only comparison:
    python scripts/audit_hml3d_coactivation_recall_v0.py \
      --coactivation-source channel_motifs \
      --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_coactivation_recall_audit_channel_motifs

Quick check:
    python scripts/audit_hml3d_coactivation_recall_v0.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_hml3d_multichannel_motion_bpe import (  # noqa: E402
    CHANNEL_RANK,
    _build_coactivation_units,
    _coactivation_signature,
    _coactivation_symbol,
    _coactivation_stats,
    _coordination_structure_features,
    _overlap_ratio,
)


DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_BPE_SEQUENCES = Path(
    "outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_composition_score_full/case_multichannel_bpe_sequences.jsonl"
)
DEFAULT_BPE_VOCAB = Path(
    "outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_composition_score_full/multichannel_motion_bpe_vocab.json"
)
DEFAULT_TARGET_REGISTRY = Path("configs/motion_pattern_text_targets.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_coactivation_recall_audit_all_units")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_div(num: int | float, denom: int | float) -> float:
    return 0.0 if not denom else float(num) / float(denom)


def _top_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _caption_text(record: dict[str, Any]) -> str:
    return " ".join(str(item) for item in record.get("caption_texts") or [])


def load_target_registry(path: Path, target_ids: set[str] | None = None) -> list[dict[str, Any]]:
    payload = _read_json(path)
    targets = payload.get("targets") if isinstance(payload, dict) else payload
    if not isinstance(targets, list):
        raise ValueError(f"target registry has no target list: {path}")
    out: list[dict[str, Any]] = []
    for item in targets:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        target_id = str(item["id"])
        if target_ids and target_id not in target_ids:
            continue
        out.append(
            {
                "id": target_id,
                "display_name": str(item.get("display_name") or target_id.replace("_", " ")),
                "regex": str(item.get("regex") or re.escape(target_id.replace("_", " "))),
                "notes": str(item.get("notes") or ""),
            }
        )
    return out


def load_case_text(source_corpus: Path, max_cases: int | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with source_corpus.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record.get("case_id") or "")
            rows[case_id] = {
                "case_id": case_id,
                "num_frames": int(record.get("num_frames") or 0),
                "caption_texts": [str(item) for item in record.get("caption_texts") or []],
                "caption_alias_ids": [str(item) for item in record.get("caption_alias_ids") or []],
            }
            if max_cases is not None and len(rows) >= max_cases:
                break
    return rows


def target_case_sets(case_text: dict[str, dict[str, Any]], targets: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for target in targets:
        target_id = str(target["id"])
        pattern = re.compile(str(target["regex"]), re.IGNORECASE)
        hits: set[str] = set()
        for case_id, record in case_text.items():
            aliases = {str(item) for item in record.get("caption_alias_ids") or []}
            alias_hit = target_id in aliases
            regex_hit = bool(pattern.search(_caption_text(record)))
            if alias_hit or regex_hit:
                hits.add(case_id)
        out[target_id] = hits
    return out


def load_channel_sequences(bpe_sequences: Path, max_cases: int | None = None) -> dict[str, list[dict[str, Any]]]:
    sequences: dict[str, list[dict[str, Any]]] = {}
    seen_cases: set[str] = set()
    with bpe_sequences.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sequence_id = str(row.get("sequence_id") or "")
            view = str(row.get("view") or "")
            case_id = str(row.get("case_id") or sequence_id.split("::", 1)[0])
            if max_cases is not None and case_id not in seen_cases and len(seen_cases) >= max_cases:
                continue
            if not view.startswith("channel::"):
                continue
            seen_cases.add(case_id)
            sequences[sequence_id] = [dict(token) for token in row.get("tokens") or []]
    return sequences


def load_selected_coordination_symbols(bpe_vocab: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(bpe_vocab)
    selected: dict[str, dict[str, Any]] = {}
    for merge in payload.get("merges") or []:
        if str(merge.get("operator") or "") != "COORDINATION_MERGE":
            continue
        parents = [str(item) for item in merge.get("parents") or []]
        if len(parents) != 1:
            continue
        selected[parents[0]] = {
            "merge_id": str(merge.get("merge_id") or ""),
            "support_cases": int(merge.get("support_cases") or 0),
            "count": int(merge.get("count") or 0),
            "selection_score": merge.get("selection_score"),
            "selection_features": merge.get("selection_features") or {},
        }
    return selected


def _unit_channel(unit: dict[str, Any]) -> str:
    return str((unit.get("channels") or ["other"])[0])


def _channel_units_by_case(channel_sequences: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence_id, seq in channel_sequences.items():
        case_id = sequence_id.split("::", 1)[0]
        for unit in seq:
            if not unit.get("span"):
                continue
            by_case[case_id].append(dict(unit))
    for units in by_case.values():
        units.sort(key=lambda unit: (int((unit.get("span") or [0, 0])[0]), int((unit.get("span") or [0, 0])[1]), CHANNEL_RANK.get(_unit_channel(unit), 999)))
    return dict(by_case)


def build_all_unit_coactivations(
    channel_sequences: dict[str, list[dict[str, Any]]],
    *,
    parallel_overlap_min: float,
) -> dict[str, list[dict[str, Any]]]:
    """Build coactivation signatures from every channel unit, not only <CHM_*>.

    This is intentionally audit-only. The current Motion-BPE learner composes
    cross-channel motifs only after per-channel merges, so this view checks
    whether full-pattern evidence was lost before the coordination stage.
    """

    out: dict[str, list[dict[str, Any]]] = {}
    for case_id, candidates in _channel_units_by_case(channel_sequences).items():
        parent: dict[int, int] = {idx: idx for idx in range(len(candidates))}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, left in enumerate(candidates):
            left_channels = set(left.get("channels") or [])
            for j in range(i + 1, len(candidates)):
                right = candidates[j]
                if left_channels & set(right.get("channels") or []):
                    continue
                if _overlap_ratio(left, right) >= parallel_overlap_min:
                    union(i, j)

        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for idx, unit in enumerate(candidates):
            groups[find(idx)].append(unit)

        coacts: list[dict[str, Any]] = []
        for idx, members in enumerate(groups.values(), start=1):
            channels = sorted({channel for unit in members for channel in (unit.get("channels") or [])}, key=lambda ch: CHANNEL_RANK.get(ch, 999))
            if len(channels) < 2:
                continue
            span = [
                min(int((unit.get("span") or [0, 0])[0]) for unit in members),
                max(int((unit.get("span") or [0, 0])[1]) for unit in members),
            ]
            coacts.append(
                {
                    "symbol": _coactivation_signature(members),
                    "unit_type": "coactivation_all_units",
                    "base_symbols": [str(symbol) for unit in members for symbol in (unit.get("base_symbols") or [unit.get("symbol")])],
                    "event_ids": [str(event_id) for unit in members for event_id in (unit.get("event_ids") or [])],
                    "packet_ids": [],
                    "span": span,
                    "channels": channels,
                    "geometry_clusters": sorted({str(cluster) for unit in members for cluster in (unit.get("geometry_clusters") or [])}),
                    "raw_geometry_clusters": sorted({str(cluster) for unit in members for cluster in (unit.get("raw_geometry_clusters") or [])}),
                    "observable_refinement_tags": sorted({str(tag) for unit in members for tag in (unit.get("observable_refinement_tags") or [])}),
                    "relation_types": ["coactivation", "all_unit_audit"],
                    "member_symbols": [str(unit.get("symbol") or "") for unit in members],
                    "member_coactivation_symbol": _coactivation_symbol(members),
                    "coactivation_id": f"{case_id}:allunit{idx:04d}",
                }
            )
        coacts.sort(key=lambda unit: (int(unit["span"][0]), int(unit["span"][1]), str(unit["symbol"])))
        if coacts:
            out[f"{case_id}::coactivation"] = coacts
    return out


def build_coactivation_sequences(
    channel_sequences: dict[str, list[dict[str, Any]]],
    *,
    parallel_overlap_min: float,
    source: str,
) -> dict[str, list[dict[str, Any]]]:
    if source == "channel_motifs":
        return _build_coactivation_units(channel_sequences, parallel_overlap_min=parallel_overlap_min)
    if source == "all_units":
        return build_all_unit_coactivations(channel_sequences, parallel_overlap_min=parallel_overlap_min)
    raise ValueError(f"unknown coactivation source: {source}")


def build_symbol_case_examples(coactivation_sequences: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for sequence_id, seq in coactivation_sequences.items():
        case_id = sequence_id.split("::", 1)[0]
        for unit in seq:
            symbol = str(unit.get("symbol") or "")
            if case_id in out[symbol]:
                continue
            out[symbol][case_id] = {
                "case_id": case_id,
                "sequence_id": sequence_id,
                "span": unit.get("span"),
                "channels": unit.get("channels") or [],
                "geometry_clusters": unit.get("geometry_clusters") or [],
                "observable_refinement_tags": unit.get("observable_refinement_tags") or [],
                "member_symbols": unit.get("member_symbols") or [],
                "member_coactivation_symbol": unit.get("member_coactivation_symbol"),
            }
    return {symbol: dict(rows) for symbol, rows in out.items()}


def build_case_coactivation_profile(coactivation_sequences: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"symbols": set(), "channels": Counter(), "geometry": Counter(), "tags": Counter()})
    for sequence_id, seq in coactivation_sequences.items():
        case_id = sequence_id.split("::", 1)[0]
        for unit in seq:
            symbol = str(unit.get("symbol") or "")
            out[case_id]["symbols"].add(symbol)
            out[case_id]["channels"].update(str(item) for item in unit.get("channels") or [])
            out[case_id]["geometry"].update(str(item) for item in unit.get("geometry_clusters") or [])
            out[case_id]["tags"].update(str(item) for item in unit.get("observable_refinement_tags") or [])
    return dict(out)


def _examples_for_cases(
    case_ids: list[str],
    case_text: dict[str, dict[str, Any]],
    examples_by_case: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for case_id in case_ids[:limit]:
        record = case_text.get(case_id, {"caption_texts": [], "caption_alias_ids": []})
        examples.append(
            {
                "case_id": case_id,
                "caption_texts": record.get("caption_texts") or [],
                "caption_alias_ids": record.get("caption_alias_ids") or [],
                "coactivation": examples_by_case.get(case_id, {}),
            }
        )
    return examples


def _case_text_examples(case_ids: list[str], case_text: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case_id,
            "caption_texts": (case_text.get(case_id) or {}).get("caption_texts") or [],
            "caption_alias_ids": (case_text.get(case_id) or {}).get("caption_alias_ids") or [],
        }
        for case_id in case_ids[:limit]
    ]


def _target_profile(target_cases: set[str], case_profile: dict[str, dict[str, Any]], top_k: int) -> dict[str, Any]:
    channels: Counter[str] = Counter()
    geometry: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    with_coactivation = 0
    for case_id in target_cases:
        profile = case_profile.get(case_id)
        if not profile:
            continue
        with_coactivation += 1
        channels.update(profile.get("channels") or {})
        geometry.update(profile.get("geometry") or {})
        tags.update(profile.get("tags") or {})
    return {
        "target_cases_with_any_coactivation": with_coactivation,
        "any_coactivation_recall": round(_safe_div(with_coactivation, len(target_cases)), 6),
        "top_channels": _top_counter(channels, top_k),
        "top_geometry_clusters": _top_counter(geometry, top_k),
        "top_observable_refinement_tags": _top_counter(tags, top_k),
    }


def _candidate_row(
    symbol: str,
    *,
    counts: Counter[str],
    cases: dict[str, set[str]],
    meta: dict[str, dict[str, Counter[str]]],
    features: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    target_cases: set[str],
    case_text: dict[str, dict[str, Any]],
    symbol_examples: dict[str, dict[str, dict[str, Any]]],
    example_limit: int,
    top_k: int,
) -> dict[str, Any]:
    support_cases = cases.get(symbol, set())
    target_hits = sorted(support_cases & target_cases)
    non_target_hits = sorted(support_cases - target_cases)
    precision = _safe_div(len(target_hits), len(support_cases))
    recall = _safe_div(len(target_hits), len(target_cases))
    f1 = _safe_div(2 * precision * recall, precision + recall)
    selected_info = selected.get(symbol)
    symbol_meta = meta.get(symbol, {})
    row = {
        "symbol": symbol,
        "selected_by_motion_bpe": selected_info is not None,
        "selected_merge_id": (selected_info or {}).get("merge_id"),
        "count": int(counts.get(symbol, 0)),
        "support_cases": len(support_cases),
        "target_support_cases": len(target_hits),
        "non_target_support_cases": len(non_target_hits),
        "target_precision": round(precision, 6),
        "target_recall": round(recall, 6),
        "target_f1": round(f1, 6),
        "structure_score": (features.get(symbol) or {}).get("score"),
        "selection_features": features.get(symbol) or {},
        "channels": _top_counter(symbol_meta.get("channels", Counter()), top_k),
        "geometry_clusters": _top_counter(symbol_meta.get("geometry_clusters", Counter()), top_k),
        "observable_refinement_tags": _top_counter(symbol_meta.get("observable_refinement_tags", Counter()), top_k),
        "target_examples": _examples_for_cases(target_hits, case_text, symbol_examples.get(symbol, {}), example_limit),
        "non_target_examples": _examples_for_cases(non_target_hits, case_text, symbol_examples.get(symbol, {}), min(3, example_limit)),
    }
    if selected_info is not None:
        row["selected_motion_bpe_info"] = selected_info
    return row


def _sort_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("target_support_cases") or 0),
            -float(row.get("target_precision") or 0.0),
            -float(row.get("target_recall") or 0.0),
            -float(row.get("structure_score") or -999.0),
            -int(row.get("support_cases") or 0),
            str(row.get("symbol") or ""),
        ),
    )


def _diagnose_target(
    target_case_count: int,
    profile: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    min_candidate_target_support: int,
    stable_precision: float,
    stable_recall: float,
    structure_score_floor: float,
) -> dict[str, Any]:
    if target_case_count == 0:
        return {
            "category": "no_text_pseudo_gt",
            "reason": "no HumanML3D caption matched this target",
            "next_action": "skip this target or expand the text registry",
        }
    any_recall = float(profile.get("any_coactivation_recall") or 0.0)
    if not rows:
        category = "coactivation_fragmented_below_support" if any_recall >= 0.5 else "channel_motif_or_overlap_gap"
        return {
            "category": category,
            "reason": "target cases have no coactivation symbol reaching the target-support reporting threshold",
            "next_action": "inspect channel motif granularity and overlap grouping before changing pattern names",
        }

    best = rows[0]
    zones = set((best.get("selection_features") or {}).get("zones") or [])
    channels = set((best.get("selection_features") or {}).get("channels") or [])
    precision = float(best.get("target_precision") or 0.0)
    recall = float(best.get("target_recall") or 0.0)
    support = int(best.get("target_support_cases") or 0)
    score = float(best.get("structure_score") or -999.0)
    selected = bool(best.get("selected_by_motion_bpe"))
    upper_only = zones == {"upper"} or channels <= {"left_arm", "right_arm", "bimanual"}

    if selected and upper_only:
        return {
            "category": "selected_component_not_full_pattern",
            "reason": "Motion-BPE selected a stable upper-body coordination component, not a full-body pattern",
            "next_action": "look for second-stage composition between this component and lower/vertical/root motifs",
        }
    if upper_only and support >= min_candidate_target_support:
        return {
            "category": "component_only_best_match",
            "reason": "the strongest target-aligned motion evidence is single-zone or upper-body only",
            "next_action": "treat it as a reusable component unless a stable cross-zone composition appears",
        }
    if precision >= stable_precision and recall >= stable_recall:
        category = "stable_candidate_not_selected"
        if score < structure_score_floor:
            category = "stable_candidate_filtered_by_structure_score"
        return {
            "category": category,
            "reason": "a target-aligned coactivation exists but was not promoted as a coordination motif",
            "next_action": "review selection score, support threshold, and BPE coordination budget",
        }
    if precision < max(0.25, stable_precision * 0.5) and support >= min_candidate_target_support:
        return {
            "category": "text_label_motion_diverse_or_ambiguous",
            "reason": "the best repeated coactivation appears in many non-target captions too",
            "next_action": "use text naming only after motion subfamilies are split",
        }
    if recall < stable_recall and any_recall >= 0.5:
        return {
            "category": "target_fragmented_across_many_coactivations",
            "reason": "target cases contain coactivations, but no single symbol covers enough of them",
            "next_action": "improve channel motifs or add a composition layer above coactivation symbols",
        }
    return {
        "category": "weak_motion_evidence",
        "reason": "target has some coactivation evidence, but it is not stable enough for a full pattern node",
        "next_action": "keep as naming-sidecar or component evidence until more stable structure is found",
    }


def build_recall_audit(args: argparse.Namespace) -> dict[str, Any]:
    target_ids = {item.strip() for item in str(args.target_ids or "").split(",") if item.strip()} or None
    targets = load_target_registry(Path(args.target_registry), target_ids=target_ids)
    case_text = load_case_text(Path(args.source_corpus), max_cases=args.max_cases)
    target_cases = target_case_sets(case_text, targets)
    selected_symbols = load_selected_coordination_symbols(Path(args.bpe_vocab))
    channel_sequences = load_channel_sequences(Path(args.bpe_sequences), max_cases=args.max_cases)
    coactivation_sequences = build_coactivation_sequences(
        channel_sequences,
        parallel_overlap_min=float(args.parallel_overlap_min),
        source=str(args.coactivation_source),
    )
    counts, cases, examples, meta = _coactivation_stats(coactivation_sequences)
    del examples
    features = {
        symbol: _coordination_structure_features(
            symbol,
            count=int(counts[symbol]),
            support_cases=len(cases[symbol]),
            meta=meta.get(symbol, {}),
        )
        for symbol in counts
    }
    symbol_examples = build_symbol_case_examples(coactivation_sequences)
    case_profile = build_case_coactivation_profile(coactivation_sequences)

    targets_out: list[dict[str, Any]] = []
    for target in targets:
        target_id = str(target["id"])
        gt_cases = target_cases.get(target_id, set())
        candidate_rows: list[dict[str, Any]] = []
        for symbol in counts:
            hit_count = len(cases.get(symbol, set()) & gt_cases)
            if hit_count < int(args.min_candidate_target_support):
                continue
            candidate_rows.append(
                _candidate_row(
                    symbol,
                    counts=counts,
                    cases=cases,
                    meta=meta,
                    features=features,
                    selected=selected_symbols,
                    target_cases=gt_cases,
                    case_text=case_text,
                    symbol_examples=symbol_examples,
                    example_limit=int(args.example_limit),
                    top_k=int(args.top_k),
                )
            )
        candidate_rows = _sort_candidate_rows(candidate_rows)
        profile = _target_profile(gt_cases, case_profile, int(args.top_k))
        diagnosis = _diagnose_target(
            len(gt_cases),
            profile,
            candidate_rows,
            min_candidate_target_support=int(args.min_candidate_target_support),
            stable_precision=float(args.stable_precision),
            stable_recall=float(args.stable_recall),
            structure_score_floor=float(args.structure_score_floor),
        )
        no_coactivation_cases = sorted(case_id for case_id in gt_cases if case_id not in case_profile)
        targets_out.append(
            {
                "target_id": target_id,
                "display_name": target.get("display_name"),
                "regex": target.get("regex"),
                "pseudo_gt_case_count": len(gt_cases),
                "diagnosis": diagnosis,
                "coactivation_profile": profile,
                "candidate_symbol_count": len(candidate_rows),
                "reported_candidate_count": min(len(candidate_rows), int(args.max_candidates_per_target)),
                "best_candidate": candidate_rows[0] if candidate_rows else None,
                "top_candidates": candidate_rows[: int(args.max_candidates_per_target)],
                "target_cases_without_any_coactivation_examples": _case_text_examples(
                    no_coactivation_cases,
                    case_text,
                    int(args.example_limit),
                ),
            }
        )

    diagnosis_counts = Counter(str((row.get("diagnosis") or {}).get("category") or "") for row in targets_out)
    return {
        "schema_version": "hml3d_coactivation_recall_audit_v0",
        "policy": {
            "purpose": "diagnose missing full pattern nodes",
            "text_targets": "pseudo-GT only; text is not used for Motion-BPE learning",
            "motion_evidence": "all overlapping channel-BPE motifs rebuilt from channel sequences",
        },
        "inputs": {
            "source_corpus": str(args.source_corpus),
            "bpe_sequences": str(args.bpe_sequences),
            "bpe_vocab": str(args.bpe_vocab),
            "target_registry": str(args.target_registry),
            "parallel_overlap_min": float(args.parallel_overlap_min),
            "coactivation_source": str(args.coactivation_source),
            "max_cases": args.max_cases,
        },
        "thresholds": {
            "min_candidate_target_support": int(args.min_candidate_target_support),
            "stable_precision": float(args.stable_precision),
            "stable_recall": float(args.stable_recall),
            "structure_score_floor": float(args.structure_score_floor),
        },
        "summary": {
            "case_text_count": len(case_text),
            "channel_sequence_count": len(channel_sequences),
            "coactivation_sequence_count": len(coactivation_sequences),
            "coactivation_symbol_count": len(counts),
            "coactivation_source": str(args.coactivation_source),
            "selected_coordination_symbol_count": len(selected_symbols),
            "target_count": len(targets_out),
            "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
        },
        "selected_coordination_symbols": selected_symbols,
        "targets": targets_out,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    lines = [
        "# HML3D Coactivation Recall Audit",
        "",
        "Diagnostic only. Text targets are pseudo-GT for audit and do not affect Motion-BPE learning.",
        "",
        "## Summary",
        "",
        f"- cases: `{summary.get('case_text_count')}`",
        f"- channel sequences: `{summary.get('channel_sequence_count')}`",
        f"- coactivation source: `{summary.get('coactivation_source')}`",
        f"- coactivation sequences: `{summary.get('coactivation_sequence_count')}`",
        f"- distinct coactivation symbols: `{summary.get('coactivation_symbol_count')}`",
        f"- selected coordination symbols: `{summary.get('selected_coordination_symbol_count')}`",
        f"- targets: `{summary.get('target_count')}`",
        f"- diagnosis counts: `{summary.get('diagnosis_counts')}`",
        "",
        "## Target Overview",
        "",
        "| target | pseudo-GT | diagnosis | best support | best precision | best recall | selected | best zones | best geometry |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload.get("targets") or []:
        best = row.get("best_candidate") or {}
        features = best.get("selection_features") or {}
        geometry = ", ".join(item["id"] for item in (best.get("geometry_clusters") or [])[:4])
        zones = ", ".join(str(item) for item in features.get("zones") or [])
        lines.append(
            f"| `{row.get('target_id')}` | {row.get('pseudo_gt_case_count')} | "
            f"{(row.get('diagnosis') or {}).get('category')} | "
            f"{best.get('target_support_cases', 0)} | {best.get('target_precision', 0)} | {best.get('target_recall', 0)} | "
            f"{best.get('selected_by_motion_bpe', False)} | {zones} | {geometry} |"
        )

    for row in payload.get("targets") or []:
        diagnosis = row.get("diagnosis") or {}
        profile = row.get("coactivation_profile") or {}
        lines.extend(
            [
                "",
                f"## {row.get('target_id')}",
                "",
                f"- display name: `{row.get('display_name')}`",
                f"- pseudo-GT cases: `{row.get('pseudo_gt_case_count')}`",
                f"- diagnosis: `{diagnosis.get('category')}`",
                f"- reason: {diagnosis.get('reason')}",
                f"- next action: {diagnosis.get('next_action')}",
                f"- target cases with any coactivation: `{profile.get('target_cases_with_any_coactivation')}`",
                f"- any-coactivation recall: `{profile.get('any_coactivation_recall')}`",
                f"- top target geometry: `{profile.get('top_geometry_clusters')}`",
                "",
                "| rank | symbol | support | target hits | precision | recall | selected | score | geometry |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
            ]
        )
        for idx, cand in enumerate(row.get("top_candidates") or [], start=1):
            symbol = str(cand.get("symbol") or "").replace("|", "\\|")
            geometry = ", ".join(item["id"] for item in (cand.get("geometry_clusters") or [])[:5])
            lines.append(
                f"| {idx} | `{symbol}` | {cand.get('support_cases')} | {cand.get('target_support_cases')} | "
                f"{cand.get('target_precision')} | {cand.get('target_recall')} | {cand.get('selected_by_motion_bpe')} | "
                f"{cand.get('structure_score')} | {geometry} |"
            )
        best = row.get("best_candidate") or {}
        if best:
            lines.append("")
            lines.append("Best-candidate target examples:")
            for example in best.get("target_examples") or []:
                caption = " / ".join(str(text).replace("\n", " ") for text in (example.get("caption_texts") or [])[:2])
                lines.append(f"- `{example.get('case_id')}` span={((example.get('coactivation') or {}).get('span'))}: {caption}")
            lines.append("")
            lines.append("Best-candidate non-target examples:")
            for example in best.get("non_target_examples") or []:
                caption = " / ".join(str(text).replace("\n", " ") for text in (example.get("caption_texts") or [])[:2])
                lines.append(f"- `{example.get('case_id')}` span={((example.get('coactivation') or {}).get('span'))}: {caption}")
        missing = row.get("target_cases_without_any_coactivation_examples") or []
        if missing:
            lines.append("")
            lines.append("Target examples without any rebuilt coactivation:")
            for example in missing:
                caption = " / ".join(str(text).replace("\n", " ") for text in (example.get("caption_texts") or [])[:2])
                lines.append(f"- `{example.get('case_id')}`: {caption}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "target_recall_audit.json", payload)
    _write_json(output_dir / "summary.json", {"schema_version": payload["schema_version"], **payload["summary"], "inputs": payload["inputs"]})
    write_report(output_dir / "target_recall_audit.md", payload)


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.jsonl"
        seq = root / "seq.jsonl"
        vocab = root / "vocab.json"
        registry = root / "targets.json"
        source.write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "a", "caption_texts": ["does jumping jacks"], "caption_alias_ids": []}),
                    json.dumps({"case_id": "b", "caption_texts": ["walks forward"], "caption_alias_ids": []}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        token_a1 = {
            "symbol": "<CHM_LA>",
            "unit_type": "merged_motif",
            "span": [0, 10],
            "channels": ["left_arm"],
            "geometry_clusters": ["LEFT_ARM_PERIODIC/LA_BILATERAL_VERTICAL_ARM_CYCLE"],
            "observable_refinement_tags": ["vertical_coupled"],
        }
        token_a2 = {
            "symbol": "<CHM_RA>",
            "unit_type": "merged_motif",
            "span": [0, 10],
            "channels": ["right_arm"],
            "geometry_clusters": ["RIGHT_ARM_PERIODIC/RA_BILATERAL_VERTICAL_ARM_CYCLE"],
            "observable_refinement_tags": ["vertical_coupled"],
        }
        seq.write_text(
            "\n".join(
                [
                    json.dumps({"sequence_id": "a::channel::left_arm", "case_id": "a", "view": "channel::left_arm", "tokens": [token_a1]}),
                    json.dumps({"sequence_id": "a::channel::right_arm", "case_id": "a", "view": "channel::right_arm", "tokens": [token_a2]}),
                    json.dumps({"sequence_id": "b::channel::left_arm", "case_id": "b", "view": "channel::left_arm", "tokens": [dict(token_a1, span=[20, 30])]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        selected_symbol = "COORD_SIG[left_arm:LEFT_ARM_PERIODIC/LA_BILATERAL_VERTICAL_ARM_CYCLE+right_arm:RIGHT_ARM_PERIODIC/RA_BILATERAL_VERTICAL_ARM_CYCLE]"
        vocab.write_text(
            json.dumps({"merges": [{"operator": "COORDINATION_MERGE", "merge_id": "<COM_0001>", "parents": [selected_symbol]}]}),
            encoding="utf-8",
        )
        registry.write_text(
            json.dumps({"targets": [{"id": "jumping_jack", "display_name": "jumping jack", "regex": r"\bjumping\s+jacks?\b"}]}),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            source_corpus=str(source),
            bpe_sequences=str(seq),
            bpe_vocab=str(vocab),
            target_registry=str(registry),
            output_dir=str(root / "out"),
            target_ids="",
            max_cases=None,
            parallel_overlap_min=0.3,
            min_candidate_target_support=1,
            max_candidates_per_target=5,
            stable_precision=0.5,
            stable_recall=0.2,
            structure_score_floor=5.0,
            coactivation_source="all_units",
            example_limit=2,
            top_k=6,
        )
        payload = build_recall_audit(args)
        assert payload["summary"]["coactivation_symbol_count"] == 1
        assert payload["targets"][0]["best_candidate"]["selected_by_motion_bpe"] is True
        assert payload["targets"][0]["diagnosis"]["category"] == "selected_component_not_full_pattern"
    print(json.dumps({"ok": True}, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit missing full-pattern recall from rebuilt Motion-BPE coactivations.")
    parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
    parser.add_argument("--bpe-sequences", default=str(DEFAULT_BPE_SEQUENCES))
    parser.add_argument("--bpe-vocab", default=str(DEFAULT_BPE_VOCAB))
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-ids", default="", help="Optional comma-separated target ids. Empty means all registry targets.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--parallel-overlap-min", type=float, default=0.30)
    parser.add_argument("--coactivation-source", choices=["channel_motifs", "all_units"], default="all_units")
    parser.add_argument("--min-candidate-target-support", type=int, default=2)
    parser.add_argument("--max-candidates-per-target", type=int, default=10)
    parser.add_argument("--stable-precision", type=float, default=0.45)
    parser.add_argument("--stable-recall", type=float, default=0.20)
    parser.add_argument("--structure-score-floor", type=float, default=5.0)
    parser.add_argument("--example-limit", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--self-test", action="store_true", help="Run a small smoke test and exit.")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    payload = build_recall_audit(args)
    write_outputs(Path(args.output_dir), payload)
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "summary": payload["summary"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
