from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import build_coarse_action_program
from pseudoedit3d.edit.aml_semantic_alias_sidecar import matched_caption_alias_rules
from pseudoedit3d.edit.coarse_event_utils import _duration, _event_sort_key, _magnitude, _span
from pseudoedit3d.edit.geometry_sidecar import geometry_cluster_id
from scripts.run_momask_aml_prompt_probe import extract_aml_program


DEFAULT_HML_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_audit_v1")
DEFAULT_TARGET_REGISTRY = Path("configs/motion_pattern_text_targets.json")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _read_case_ids_from_manifest(path: Path) -> list[str]:
    case_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            case_id = row.get("case_id") or row.get("id") or row.get("motion_id")
        else:
            case_id = line.split()[0]
        if case_id:
            case_ids.append(str(case_id))
    return case_ids


def _case_ids_from_args(args: argparse.Namespace, hml_root: Path) -> list[str]:
    if args.case_ids:
        case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    elif args.manifest:
        case_ids = _read_case_ids_from_manifest(Path(args.manifest))
    else:
        case_ids = [path.stem for path in sorted((hml_root / "texts").glob("*.txt"))]
    if args.max_cases is not None:
        case_ids = case_ids[: int(args.max_cases)]
    return case_ids


def _read_prompts(hml_root: Path, case_id: str) -> list[str]:
    path = hml_root / "texts" / f"{case_id}.txt"
    if not path.exists():
        return []
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        prompts.append(line.split("#")[0].strip())
    return prompts


def _as_numpy_joints(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _duration_bin(duration: int) -> str:
    if duration <= 3:
        return "xs"
    if duration <= 8:
        return "s"
    if duration <= 20:
        return "m"
    if duration <= 60:
        return "l"
    return "xl"


def _magnitude_bin(value: float, unit: str) -> str:
    if not math.isfinite(value) or value <= 0.0:
        return "none"
    if unit == "deg":
        if value < 30.0:
            return "deg_xs"
        if value < 90.0:
            return "deg_s"
        if value < 180.0:
            return "deg_m"
        if value < 360.0:
            return "deg_l"
        return "deg_xl"
    if unit == "m":
        if value < 0.06:
            return "m_xs"
        if value < 0.16:
            return "m_s"
        if value < 0.45:
            return "m_m"
        if value < 1.5:
            return "m_l"
        return "m_xl"
    if unit == "ratio":
        if value < 0.15:
            return "r_s"
        if value < 0.45:
            return "r_m"
        return "r_l"
    if value < 1.0:
        return "v_s"
    if value < 5.0:
        return "v_m"
    return "v_l"


def _count_bin(count: Any) -> str | None:
    try:
        value = int(count)
    except (TypeError, ValueError):
        return None
    if value <= 1:
        return "c1"
    if value <= 3:
        return "c2_3"
    if value <= 6:
        return "c4_6"
    return "c7p"


def _event_token_symbol(event: dict[str, Any], granularity: str) -> str:
    family = str(event.get("super_family") or "UNKNOWN_FAMILY")
    cluster = str(event.get("cluster_id") or "UNKNOWN_CLUSTER")
    if granularity == "cluster":
        return f"{family}/{cluster}"

    direction = str(event.get("direction") or "none")
    duration = _duration_bin(_duration(event))
    magnitude = _magnitude_bin(_magnitude(event), str(event.get("unit") or ""))
    count = _count_bin(event.get("count"))
    parts = [f"{family}/{cluster}", f"dir={direction}", f"dur={duration}", f"mag={magnitude}"]
    if granularity == "detailed":
        role = str(event.get("role") or "unknown")
        motion_sig = event.get("motion_signature") or {}
        phase = str(motion_sig.get("phase_template") or "unknown")
        context = str(motion_sig.get("context_mode") or "unknown")
        parts.extend([f"role={role}", f"phase={phase}", f"ctx={context}"])
    if count:
        parts.append(f"count={count}")
    return "|".join(parts)


def _token_unit(event: dict[str, Any], granularity: str) -> dict[str, Any]:
    start, end = _span(event)
    return {
        "symbol": _event_token_symbol(event, granularity),
        "base_symbols": [_event_token_symbol(event, granularity)],
        "event_indices": [int(event.get("event_index", -1))],
        "span": [start, end],
        "parts": [str(event.get("part") or "")],
        "super_families": [str(event.get("super_family") or "")],
        "clusters": [geometry_cluster_id(event)],
    }


def _compact_event(event: dict[str, Any], granularity: str) -> dict[str, Any]:
    motion_sig = event.get("motion_signature") or {}
    start, end = _span(event)
    return {
        "event_index": int(event.get("event_index", -1)),
        "token": _event_token_symbol(event, granularity),
        "geometry_cluster_id": geometry_cluster_id(event),
        "part": str(event.get("part") or ""),
        "super_family": str(event.get("super_family") or ""),
        "cluster_id": str(event.get("cluster_id") or ""),
        "direction": str(event.get("direction") or ""),
        "role": str(event.get("role") or ""),
        "span": [start, end],
        "duration": _duration(event),
        "magnitude": round(_magnitude(event), 4),
        "unit": event.get("unit"),
        "count": event.get("count"),
        "optional_semantic_name": event.get("optional_semantic_name"),
        "motion_signature": {
            "dominant_axis": motion_sig.get("dominant_axis"),
            "repeat_mode": motion_sig.get("repeat_mode"),
            "phase_template": motion_sig.get("phase_template"),
            "context_mode": motion_sig.get("context_mode"),
            "tempo_bucket": motion_sig.get("tempo_bucket"),
            "coupled_with_locomotion": motion_sig.get("coupled_with_locomotion"),
        },
    }


def _compact_actions(coarse: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for idx, action in enumerate(coarse.get("canonical_actions") or []):
        family = action.get("semantic_family") or {}
        slots = action.get("slots") or {}
        covered = slots.get("covered_event_indices") or action.get("covered_event_indices") or []
        actions.append(
            {
                "action_index": idx,
                "family_id": str(family.get("family_id") or action.get("canonical_id") or ""),
                "status": str(family.get("status") or slots.get("semantic_family_status") or ""),
                "taxonomy_parent_id": family.get("taxonomy_parent_id"),
                "pattern_node_id": family.get("pattern_node_id"),
                "pattern_path": family.get("pattern_path"),
                "probe_alias": action.get("probe_alias"),
                "covered_event_indices": [int(item) for item in covered if item is not None],
            }
        )
    return actions


def _caption_alias_ids(captions: list[str]) -> list[str]:
    return [str(rule.get("alias_id") or "") for rule in matched_caption_alias_rules(captions)]


def _extract_record(case_id: str, item: Any, hml_root: Path, granularity: str, include_coarse: bool) -> dict[str, Any] | None:
    joints = _as_numpy_joints(item["joints3d"])
    if len(joints) <= 1:
        return None
    aml = extract_aml_program(joints)
    layer3 = aml["layer3"]
    raw_events = list(layer3.get("events") or [])
    events: list[dict[str, Any]] = []
    for idx, event in enumerate(raw_events):
        copied = dict(event)
        copied["event_index"] = int(copied.get("event_index", idx))
        events.append(copied)
    events = sorted(events, key=_event_sort_key)
    token_units = [_token_unit(event, granularity) for event in events]
    captions = _read_prompts(hml_root, case_id)
    coarse_actions: list[dict[str, Any]] = []
    if include_coarse:
        coarse = build_coarse_action_program(layer3, total_frames=int(len(joints)), max_residual_events=8)
        coarse_actions = _compact_actions(coarse)
    return {
        "case_id": case_id,
        "num_frames": int(len(joints)),
        "caption_texts": captions,
        "caption_alias_ids": _caption_alias_ids(captions),
        "layer_counts": {
            "layer1": int(aml.get("layer1_count") or 0),
            "layer2": int(aml.get("layer2_count") or 0),
            "layer25": int(aml.get("layer25_count") or 0),
            "layer3": len(events),
        },
        "events": [_compact_event(event, granularity) for event in events],
        "token_units": token_units,
        "coarse_actions": coarse_actions,
    }


def build_corpus(args: argparse.Namespace, corpus_path: Path) -> list[dict[str, Any]]:
    if args.reuse_corpus and corpus_path.exists():
        return _read_jsonl(corpus_path)

    hml_root = Path(args.hml_root)
    case_ids = _case_ids_from_args(args, hml_root)
    packed = torch.load(hml_root / "joints3d.pth", map_location="cpu")
    records: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, case_id in enumerate(case_ids, start=1):
        key = f"{case_id}.npy"
        if key not in packed:
            continue
        try:
            record = _extract_record(
                case_id,
                packed[key],
                hml_root,
                args.token_granularity,
                include_coarse=not args.skip_coarse,
            )
        except Exception as exc:  # pragma: no cover - audit script should continue past bad records.
            print(f"skip_failed case={case_id} error={type(exc).__name__}: {exc}", flush=True)
            continue
        if record is not None:
            records.append(record)
        if args.progress_every and idx % int(args.progress_every) == 0:
            print(
                f"processed {idx}/{len(case_ids)}, valid={len(records)}, elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    _write_jsonl(corpus_path, records)
    return records


def _merge_units(left: dict[str, Any], right: dict[str, Any], symbol: str) -> dict[str, Any]:
    span = [min(int(left["span"][0]), int(right["span"][0])), max(int(left["span"][1]), int(right["span"][1]))]
    return {
        "symbol": symbol,
        "base_symbols": list(left.get("base_symbols") or [left["symbol"]]) + list(right.get("base_symbols") or [right["symbol"]]),
        "event_indices": list(left.get("event_indices") or []) + list(right.get("event_indices") or []),
        "span": span,
        "parts": sorted(set(left.get("parts") or []) | set(right.get("parts") or [])),
        "super_families": sorted(set(left.get("super_families") or []) | set(right.get("super_families") or [])),
        "clusters": sorted(set(left.get("clusters") or []) | set(right.get("clusters") or [])),
    }


def _pair_stats(sequences: dict[str, list[dict[str, Any]]]) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], set[str]], dict[tuple[str, str], list[dict[str, Any]]]]:
    counts: Counter[tuple[str, str]] = Counter()
    cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case_id, seq in sequences.items():
        for idx in range(len(seq) - 1):
            pair = (str(seq[idx]["symbol"]), str(seq[idx + 1]["symbol"]))
            counts[pair] += 1
            cases[pair].add(case_id)
            if len(examples[pair]) < 8:
                left_span = seq[idx].get("span") or [0, 0]
                right_span = seq[idx + 1].get("span") or [0, 0]
                examples[pair].append(
                    {
                        "case_id": case_id,
                        "span": [min(int(left_span[0]), int(right_span[0])), max(int(left_span[1]), int(right_span[1]))],
                        "event_indices": list(seq[idx].get("event_indices") or []) + list(seq[idx + 1].get("event_indices") or []),
                    }
                )
    return counts, cases, examples


def _select_pair(
    counts: Counter[tuple[str, str]],
    cases: dict[tuple[str, str], set[str]],
    *,
    min_pair_count: int,
    min_pair_support: int,
    selection: str,
) -> tuple[str, str] | None:
    candidates: list[tuple[str, str]] = []
    for pair, count in counts.items():
        if count < min_pair_count:
            continue
        if len(cases[pair]) < min_pair_support:
            continue
        candidates.append(pair)
    if not candidates:
        return None
    if selection == "support":
        return max(candidates, key=lambda pair: (len(cases[pair]), counts[pair], pair))
    return max(candidates, key=lambda pair: (counts[pair], len(cases[pair]), pair))


def _apply_merge(seq: list[dict[str, Any]], pair: tuple[str, str], merged_symbol: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = 0
    while idx < len(seq):
        if idx < len(seq) - 1 and seq[idx]["symbol"] == pair[0] and seq[idx + 1]["symbol"] == pair[1]:
            out.append(_merge_units(seq[idx], seq[idx + 1], merged_symbol))
            idx += 2
        else:
            out.append(dict(seq[idx]))
            idx += 1
    return out


def learn_bpe(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    sequences = {
        str(record["case_id"]): [dict(unit) for unit in record.get("token_units") or []]
        for record in records
        if record.get("token_units")
    }
    merges: list[dict[str, Any]] = []
    for step in range(1, int(args.num_merges) + 1):
        counts, cases, examples = _pair_stats(sequences)
        pair = _select_pair(
            counts,
            cases,
            min_pair_count=int(args.min_pair_count),
            min_pair_support=int(args.min_pair_support),
            selection=str(args.selection),
        )
        if pair is None:
            break
        merged_symbol = f"<M{step:04d}>"
        support_cases = sorted(cases[pair])
        merge_record = {
            "merge_id": merged_symbol,
            "step": step,
            "parents": list(pair),
            "count": int(counts[pair]),
            "support_cases": len(support_cases),
            "example_case_ids": support_cases[: int(args.examples_per_motif)],
            "example_occurrences": examples[pair][: int(args.examples_per_motif)],
        }
        merges.append(merge_record)
        for case_id in list(sequences):
            sequences[case_id] = _apply_merge(sequences[case_id], pair, merged_symbol)
    return merges, sequences


def _record_maps(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[int, dict[str, Any]]], dict[str, dict[int, list[dict[str, Any]]]]]:
    by_case = {str(record["case_id"]): record for record in records}
    event_maps: dict[str, dict[int, dict[str, Any]]] = {}
    action_links: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for record in records:
        case_id = str(record["case_id"])
        event_maps[case_id] = {int(event["event_index"]): event for event in record.get("events") or []}
        links: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for action in record.get("coarse_actions") or []:
            for event_index in action.get("covered_event_indices") or []:
                links[int(event_index)].append(action)
        action_links[case_id] = dict(links)
    return by_case, event_maps, action_links


def _load_text_target_patterns(path: Path) -> dict[str, re.Pattern[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("targets") if isinstance(payload, dict) else payload
    if not isinstance(targets, list):
        return {}
    patterns: dict[str, re.Pattern[str]] = {}
    for item in targets:
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("id") or "")
        regex = str(item.get("regex") or "")
        if not target_id or not regex:
            continue
        patterns[target_id] = re.compile(regex, re.IGNORECASE)
    return patterns


def _caption_keyword_tags(captions: list[str], target_patterns: dict[str, re.Pattern[str]]) -> list[str]:
    text = " ".join(captions).lower()
    return [name for name, pattern in target_patterns.items() if pattern.search(text)]


def _top_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _motif_occurrences(sequences: dict[str, list[dict[str, Any]]], motif_symbols: set[str]) -> dict[str, list[dict[str, Any]]]:
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case_id, seq in sequences.items():
        for unit in seq:
            symbol = str(unit.get("symbol") or "")
            if symbol not in motif_symbols:
                continue
            occurrences[symbol].append({"case_id": case_id, **dict(unit)})
    return dict(occurrences)


def _audit_motifs(
    records: list[dict[str, Any]],
    merges: list[dict[str, Any]],
    sequences: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_case, event_maps, action_links = _record_maps(records)
    text_target_patterns = _load_text_target_patterns(Path(args.target_registry))
    merge_map = {str(merge["merge_id"]): merge for merge in merges}
    occurrences = _motif_occurrences(sequences, set(merge_map))
    motif_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for motif_id, motif_occs in occurrences.items():
        support_cases = sorted({str(item["case_id"]) for item in motif_occs})
        caption_alias_counter: Counter[str] = Counter()
        keyword_counter: Counter[str] = Counter()
        family_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        cluster_counter: Counter[str] = Counter()
        base_counter: Counter[str] = Counter()
        example_rows: list[dict[str, Any]] = []
        case_aliases: dict[str, set[str]] = defaultdict(set)
        case_keywords: dict[str, set[str]] = defaultdict(set)
        case_families: dict[str, set[str]] = defaultdict(set)
        case_statuses: dict[str, set[str]] = defaultdict(set)
        for occ in motif_occs:
            case_id = str(occ["case_id"])
            record = by_case.get(case_id, {})
            alias_ids = record.get("caption_alias_ids") or []
            case_aliases[case_id].update(str(item) for item in alias_ids if item)
            case_keywords[case_id].update(_caption_keyword_tags(record.get("caption_texts") or [], text_target_patterns))
            base_counter.update(str(item) for item in occ.get("base_symbols") or [])
            linked_families: set[str] = set()
            linked_statuses: set[str] = set()
            for event_index in occ.get("event_indices") or []:
                event = event_maps.get(case_id, {}).get(int(event_index))
                if event:
                    cluster_counter[str(event.get("geometry_cluster_id") or "")] += 1
                for action in action_links.get(case_id, {}).get(int(event_index), []):
                    family = str(action.get("family_id") or "")
                    status = str(action.get("status") or "")
                    if family:
                        linked_families.add(family)
                    if status:
                        linked_statuses.add(status)
            case_families[case_id].update(linked_families or {"__UNLINKED__"})
            case_statuses[case_id].update(linked_statuses or {"__UNLINKED__"})
            if len(example_rows) < int(args.examples_per_motif):
                example_rows.append(
                    {
                        "case_id": case_id,
                        "span": occ.get("span"),
                        "event_indices": occ.get("event_indices"),
                        "caption": (record.get("caption_texts") or [""])[0],
                        "caption_alias_ids": alias_ids,
                        "linked_families": sorted(linked_families),
                    }
                )
        for case_id in support_cases:
            caption_alias_counter.update(case_aliases.get(case_id) or {"__NO_CAPTION_ALIAS__"})
            keyword_counter.update(case_keywords.get(case_id) or {"__NO_KEYWORD__"})
            family_counter.update(case_families.get(case_id) or {"__UNLINKED__"})
            status_counter.update(case_statuses.get(case_id) or {"__UNLINKED__"})
        top_alias, top_alias_count = caption_alias_counter.most_common(1)[0] if caption_alias_counter else ("", 0)
        top_family, top_family_count = family_counter.most_common(1)[0] if family_counter else ("", 0)
        stable_alias = "" if top_alias == "__NO_CAPTION_ALIAS__" else top_alias
        alias_purity = 0.0 if top_alias == "__NO_CAPTION_ALIAS__" else top_alias_count / max(1, len(support_cases))
        family_purity = top_family_count / max(1, len(support_cases))
        merge = merge_map[motif_id]
        row = {
            "motif_id": motif_id,
            "step": int(merge.get("step") or 0),
            "parents": merge.get("parents") or [],
            "occurrences": len(motif_occs),
            "support_cases": len(support_cases),
            "compression_gain_tokens": len(motif_occs),
            "caption_alias_purity": round(alias_purity, 4),
            "top_caption_alias": stable_alias,
            "top_caption_alias_count": int(top_alias_count),
            "tree_family_purity": round(family_purity, 4),
            "top_tree_family": top_family,
            "top_tree_family_count": int(top_family_count),
            "top_caption_aliases": [
                item for item in _top_counter(caption_alias_counter, 8)
                if item["id"] != "__NO_CAPTION_ALIAS__"
            ],
            "top_caption_keywords": _top_counter(keyword_counter, 8),
            "top_tree_families": _top_counter(family_counter, 8),
            "top_tree_statuses": _top_counter(status_counter, 8),
            "top_geometry_clusters": _top_counter(cluster_counter, 12),
            "top_base_symbols": _top_counter(base_counter, 12),
            "example_occurrences": example_rows,
        }
        motif_rows.append(row)
        if (
            len(support_cases) >= int(args.min_candidate_support)
            and alias_purity >= float(args.stable_alias_purity)
            and stable_alias
        ):
            candidates.append(
                {
                    "motif_id": motif_id,
                    "support_cases": len(support_cases),
                    "stable_caption_alias": stable_alias,
                    "caption_alias_purity": round(alias_purity, 4),
                    "top_tree_family": top_family,
                    "tree_family_purity": round(family_purity, 4),
                    "suggested_use": "inspect_as_pattern_tree_node_candidate",
                    "reason": "high-support BPE motif has stable caption semantics; compare its geometry clusters with current AML tree coverage",
                    "top_geometry_clusters": row["top_geometry_clusters"],
                    "example_occurrences": example_rows,
                }
            )
    motif_rows.sort(key=lambda item: (-int(item["support_cases"]), -int(item["occurrences"]), int(item["step"])))
    candidates.sort(key=lambda item: (-float(item["caption_alias_purity"]), -int(item["support_cases"]), str(item["motif_id"])))
    return motif_rows, candidates


def _alias_alignment(
    records: list[dict[str, Any]],
    sequences: dict[str, list[dict[str, Any]]],
    motif_rows: list[dict[str, Any]],
    *,
    examples_per_motif: int,
) -> dict[str, Any]:
    record_map = {str(record["case_id"]): record for record in records}
    motif_map = {str(row["motif_id"]): row for row in motif_rows}
    alias_cases: dict[str, set[str]] = defaultdict(set)
    for record in records:
        case_id = str(record["case_id"])
        for alias in record.get("caption_alias_ids") or []:
            if alias:
                alias_cases[str(alias)].add(case_id)

    motif_case_occurrences: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for case_id, seq in sequences.items():
        for unit in seq:
            symbol = str(unit.get("symbol") or "")
            if not symbol.startswith("<M"):
                continue
            motif_case_occurrences[symbol][case_id].append(dict(unit))

    alignments: list[dict[str, Any]] = []
    for alias, cases in sorted(alias_cases.items(), key=lambda item: (-len(item[1]), item[0])):
        motif_scores: list[dict[str, Any]] = []
        for motif_id, by_case in motif_case_occurrences.items():
            motif_cases = set(by_case)
            positive_cases = sorted(motif_cases & cases)
            if not positive_cases:
                continue
            total_support = len(motif_cases)
            positive_support = len(positive_cases)
            purity = positive_support / max(1, total_support)
            recall = positive_support / max(1, len(cases))
            row = motif_map.get(motif_id, {})
            examples: list[dict[str, Any]] = []
            for case_id in positive_cases[:examples_per_motif]:
                record = record_map.get(case_id, {})
                for unit in by_case.get(case_id, [])[:1]:
                    examples.append(
                        {
                            "case_id": case_id,
                            "span": unit.get("span"),
                            "event_indices": unit.get("event_indices"),
                            "caption": (record.get("caption_texts") or [""])[0],
                        }
                    )
            motif_scores.append(
                {
                    "motif_id": motif_id,
                    "alias_positive_support": positive_support,
                    "motif_total_support": total_support,
                    "alias_total_cases": len(cases),
                    "alias_purity": round(purity, 4),
                    "alias_recall": round(recall, 4),
                    "top_tree_family": row.get("top_tree_family"),
                    "tree_family_purity": row.get("tree_family_purity"),
                    "top_geometry_clusters": row.get("top_geometry_clusters", [])[:8],
                    "examples": examples,
                }
            )
        motif_scores.sort(
            key=lambda item: (
                -int(item["alias_positive_support"]),
                -float(item["alias_purity"]),
                -float(item["alias_recall"]),
                str(item["motif_id"]),
            )
        )
        alignments.append(
            {
                "caption_alias": alias,
                "alias_case_count": len(cases),
                "top_motifs": motif_scores[:20],
            }
        )
    return {
        "description": "For each caption alias, rank BPE motifs by how often they occur in alias-positive cases. This is for corpus audit only; captions do not create motion evidence.",
        "alignments": alignments,
    }


def _sequence_rows(sequences: dict[str, list[dict[str, Any]]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    record_map = {str(record["case_id"]): record for record in records}
    rows: list[dict[str, Any]] = []
    for case_id, seq in sorted(sequences.items()):
        record = record_map.get(case_id, {})
        rows.append(
            {
                "case_id": case_id,
                "num_frames": record.get("num_frames"),
                "caption": (record.get("caption_texts") or [""])[0],
                "caption_alias_ids": record.get("caption_alias_ids") or [],
                "original_token_count": len(record.get("token_units") or []),
                "bpe_token_count": len(seq),
                "compression_ratio": round(len(seq) / max(1, len(record.get("token_units") or [])), 4),
                "bpe_tokens": [
                    {
                        "symbol": unit.get("symbol"),
                        "span": unit.get("span"),
                        "event_indices": unit.get("event_indices"),
                        "parts": unit.get("parts"),
                        "clusters": unit.get("clusters"),
                    }
                    for unit in seq
                ],
            }
        )
    return rows


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    motif_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    alias_alignment: dict[str, Any],
) -> None:
    lines = [
        "# HML3D Layer3 Event-BPE Audit",
        "",
        "This audit treats HumanML3D motions as a motion corpus. It learns symbolic BPE motifs over AML Layer3 event tokens and reports which motifs have stable caption semantics or weak AML tree coverage.",
        "",
        "## Summary",
        "",
        f"- corpus records: `{summary['num_records']}`",
        f"- original tokens: `{summary['original_token_count']}`",
        f"- BPE tokens: `{summary['bpe_token_count']}`",
        f"- compression ratio: `{summary['compression_ratio']:.4f}`",
        f"- merges learned: `{summary['num_merges']}`",
        f"- token granularity: `{summary['token_granularity']}`",
        f"- coarse tree binding included: `{summary['include_coarse']}`",
        "",
        "## Pattern-Tree Candidates",
        "",
    ]
    if not candidates:
        lines.append("No motif reached the configured stable alias threshold.")
    else:
        lines.extend(["| motif | support | stable caption alias | alias purity | top tree family | tree purity | examples |", "| --- | ---: | --- | ---: | --- | ---: | --- |"])
        for row in candidates[:40]:
            examples = ", ".join(str(item.get("case_id")) for item in row.get("example_occurrences") or [])
            lines.append(
                f"| `{row['motif_id']}` | {row['support_cases']} | `{row['stable_caption_alias']}` | {row['caption_alias_purity']:.3f} | `{row['top_tree_family']}` | {row['tree_family_purity']:.3f} | {examples} |"
            )
    lines.extend(["", "## Top Motifs", ""])
    lines.extend(["| motif | support | occ | top alias | alias purity | top tree family | tree purity | top clusters | examples |", "| --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |"])
    for row in motif_rows[:80]:
        clusters = ", ".join(item["id"] for item in row.get("top_geometry_clusters", [])[:4])
        examples = ", ".join(str(item.get("case_id")) for item in row.get("example_occurrences") or [])
        lines.append(
            f"| `{row['motif_id']}` | {row['support_cases']} | {row['occurrences']} | `{row['top_caption_alias']}` | {row['caption_alias_purity']:.3f} | `{row['top_tree_family']}` | {row['tree_family_purity']:.3f} | {clusters} | {examples} |"
        )
    lines.extend(["", "## Caption Alias Alignment", ""])
    lines.append("This table is the reverse view: for each caption semantic alias, which BPE motifs appear most often in its positive cases.")
    lines.append("")
    lines.extend(["| alias | alias cases | motif | positive support | purity | recall | top tree family | examples |", "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |"])
    for alias_row in (alias_alignment.get("alignments") or [])[:30]:
        top_motifs = alias_row.get("top_motifs") or []
        if not top_motifs:
            lines.append(f"| `{alias_row['caption_alias']}` | {alias_row['alias_case_count']} |  | 0 | 0.000 | 0.000 |  |  |")
            continue
        for idx, motif in enumerate(top_motifs[:3]):
            examples = ", ".join(str(item.get("case_id")) for item in motif.get("examples") or [])
            alias_label = f"`{alias_row['caption_alias']}`" if idx == 0 else ""
            case_count = str(alias_row["alias_case_count"]) if idx == 0 else ""
            lines.append(
                f"| {alias_label} | {case_count} | `{motif['motif_id']}` | {motif['alias_positive_support']} | {motif['alias_purity']:.3f} | {motif['alias_recall']:.3f} | `{motif.get('top_tree_family')}` | {examples} |"
            )
    lines.extend(["", "## Notes", ""])
    lines.append("- Caption aliases are used only for audit and purity measurement; they do not create motion evidence.")
    lines.append("- BPE motifs are learned from Layer3 event token order, so concurrent events are approximated by sorted temporal adjacency.")
    lines.append("- A high-purity motif is a candidate for tree inspection, not an automatic tree edit.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn and audit symbolic BPE motifs over HML3D AML Layer3 event tokens.")
    parser.add_argument("--hml-root", default=str(DEFAULT_HML_ROOT))
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reuse-corpus", action="store_true")
    parser.add_argument("--skip-coarse", action="store_true", help="Skip current AML tree binding during corpus extraction.")
    parser.add_argument("--token-granularity", choices=["cluster", "geometry", "detailed"], default="geometry")
    parser.add_argument("--num-merges", type=int, default=128)
    parser.add_argument("--min-pair-count", type=int, default=20)
    parser.add_argument("--min-pair-support", type=int, default=10)
    parser.add_argument("--selection", choices=["count", "support"], default="count")
    parser.add_argument("--min-candidate-support", type=int, default=8)
    parser.add_argument("--stable-alias-purity", type=float, default=0.45)
    parser.add_argument("--examples-per-motif", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--target-registry", default=str(DEFAULT_TARGET_REGISTRY), help="Text pseudo-GT target registry used only for naming diagnostics.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "layer3_event_bpe_corpus.jsonl"
    sequences_path = out_dir / "case_bpe_sequences.jsonl"
    vocab_path = out_dir / "motion_bpe_vocab.json"
    audit_path = out_dir / "bpe_motif_audit.json"
    candidates_path = out_dir / "bpe_phrase_to_pattern_tree_candidates.json"
    alias_alignment_path = out_dir / "bpe_caption_alias_alignment.json"
    summary_path = out_dir / "summary.json"
    md_path = out_dir / "bpe_motif_audit.md"

    records = build_corpus(args, corpus_path)
    original_token_count = sum(len(record.get("token_units") or []) for record in records)
    base_vocab = Counter(
        str(unit.get("symbol") or "")
        for record in records
        for unit in (record.get("token_units") or [])
    )
    merges, sequences = learn_bpe(records, args)
    sequence_rows = _sequence_rows(sequences, records)
    motif_rows, candidates = _audit_motifs(records, merges, sequences, args)
    alias_alignment = _alias_alignment(
        records,
        sequences,
        motif_rows,
        examples_per_motif=int(args.examples_per_motif),
    )
    bpe_token_count = sum(len(seq) for seq in sequences.values())
    summary = {
        "version": "hml3d_layer3_event_bpe_audit_v1",
        "hml_root": str(Path(args.hml_root)),
        "num_records": len(records),
        "original_token_count": int(original_token_count),
        "bpe_token_count": int(bpe_token_count),
        "compression_ratio": round(bpe_token_count / max(1, original_token_count), 6),
        "num_merges": len(merges),
        "base_vocab_size": len(base_vocab),
        "token_granularity": args.token_granularity,
        "include_coarse": not args.skip_coarse,
        "target_registry": str(args.target_registry),
        "min_pair_count": args.min_pair_count,
        "min_pair_support": args.min_pair_support,
        "selection": args.selection,
        "stable_candidate_count": len(candidates),
        "outputs": {
            "corpus": str(corpus_path),
            "case_bpe_sequences": str(sequences_path),
            "motion_bpe_vocab": str(vocab_path),
            "bpe_motif_audit": str(audit_path),
            "pattern_tree_candidates": str(candidates_path),
            "caption_alias_alignment": str(alias_alignment_path),
            "markdown": str(md_path),
        },
    }
    _write_jsonl(sequences_path, sequence_rows)
    _write_json(
        vocab_path,
        {
            "version": "motion_bpe_vocab_v1",
            "token_granularity": args.token_granularity,
            "base_vocab_size": len(base_vocab),
            "base_vocab_top": _top_counter(base_vocab, 100),
            "merges": merges,
        },
    )
    _write_json(audit_path, {"summary": summary, "motifs": motif_rows})
    _write_json(candidates_path, {"summary": summary, "candidates": candidates})
    _write_json(alias_alignment_path, {"summary": summary, **alias_alignment})
    _write_json(summary_path, summary)
    _write_markdown(md_path, summary, motif_rows, candidates, alias_alignment)
    print(summary_path)
    print(md_path)


if __name__ == "__main__":
    main()
