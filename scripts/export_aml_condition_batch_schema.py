from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

NUMERIC_SLOT_NAMES = [
    "distance_m",
    "path_length_m",
    "angle_deg",
    "magnitude",
    "vertical_amplitude_m",
    "mean_vertical_amplitude_m",
    "root_height_gain_m",
    "curvature_rad",
    "circle_score",
    "count",
    "source_event_count",
    "segment_count",
    "locomotion_segment_count",
    "turn_count",
    "left_arm_count",
    "right_arm_count",
    "bimanual_count",
    "raise_spread_count",
]

CATEGORICAL_SLOT_NAMES = [
    "direction",
    "speed",
    "angle_bin",
    "dominant_side",
]

STATUS_BASE_VOCAB = [PAD_TOKEN, UNK_TOKEN, "stable", "candidate", "proxy", "unknown", "legacy_missing"]


def _load_jsonl(path: Path, max_records: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_records and len(records) >= max_records:
                break
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _vocab(values: list[str], base: list[str] | None = None) -> dict[str, int]:
    tokens = list(base or [PAD_TOKEN, UNK_TOKEN])
    seen = set(tokens)
    for value in sorted(set(str(v) for v in values)):
        if value not in seen:
            tokens.append(value)
            seen.add(value)
    return {token: idx for idx, token in enumerate(tokens)}


def _id(vocab: dict[str, int], value: Any) -> int:
    return vocab.get(str(value), vocab[UNK_TOKEN])


def _span(slot_values: dict[str, Any]) -> tuple[int, int] | None:
    value = slot_values.get("span")
    if isinstance(value, list) and len(value) == 2:
        return _safe_int(value[0], 0), _safe_int(value[1], 0)
    return None


def _numeric_slots(slot_values: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(NUMERIC_SLOT_NAMES),), dtype=np.float32)
    mask = np.zeros((len(NUMERIC_SLOT_NAMES),), dtype=np.float32)
    for idx, name in enumerate(NUMERIC_SLOT_NAMES):
        value = slot_values.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[idx] = float(value)
            mask[idx] = 1.0
    return values, mask


def _categorical_slots(
    slot_values: dict[str, Any],
    slot_vocabs: dict[str, dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(CATEGORICAL_SLOT_NAMES),), dtype=np.int64)
    mask = np.zeros((len(CATEGORICAL_SLOT_NAMES),), dtype=np.float32)
    for idx, name in enumerate(CATEGORICAL_SLOT_NAMES):
        value = slot_values.get(name)
        if value is not None:
            values[idx] = _id(slot_vocabs[name], value)
            mask[idx] = 1.0
    return values, mask


def _collect_vocabs(records: list[dict[str, Any]]) -> dict[str, Any]:
    families = []
    statuses = []
    slot_values: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for cond in record.get("selected_conditions") or []:
            families.append(str(cond.get("family_id") or "UNKNOWN"))
            statuses.append(str(cond.get("status") or "unknown"))
            slots = cond.get("slot_values") or {}
            for name in CATEGORICAL_SLOT_NAMES:
                if slots.get(name) is not None:
                    slot_values[name].append(str(slots[name]))
    return {
        "family_vocab": _vocab(families),
        "status_vocab": _vocab(statuses, base=STATUS_BASE_VOCAB),
        "categorical_slot_vocabs": {
            name: _vocab(slot_values.get(name, []))
            for name in CATEGORICAL_SLOT_NAMES
        },
    }


def export_batch_schema(
    records: list[dict[str, Any]],
    max_conditions: int,
    source_jsonl: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    vocabs = _collect_vocabs(records)
    num_cases = len(records)
    num_numeric = len(NUMERIC_SLOT_NAMES)
    num_categorical = len(CATEGORICAL_SLOT_NAMES)

    case_index = np.arange(num_cases, dtype=np.int64)
    num_frames = np.zeros((num_cases,), dtype=np.int64)
    num_selected = np.zeros((num_cases,), dtype=np.int64)
    condition_mask = np.zeros((num_cases, max_conditions), dtype=np.float32)
    action_index = np.full((num_cases, max_conditions), -1, dtype=np.int64)
    family_id = np.zeros((num_cases, max_conditions), dtype=np.int64)
    status_id = np.zeros((num_cases, max_conditions), dtype=np.int64)
    score = np.zeros((num_cases, max_conditions), dtype=np.float32)
    weight = np.zeros((num_cases, max_conditions), dtype=np.float32)
    span = np.full((num_cases, max_conditions, 2), -1, dtype=np.int64)
    span_mask = np.zeros((num_cases, max_conditions), dtype=np.float32)
    span_norm = np.zeros((num_cases, max_conditions, 4), dtype=np.float32)
    numeric_slots = np.zeros((num_cases, max_conditions, num_numeric), dtype=np.float32)
    numeric_slot_mask = np.zeros((num_cases, max_conditions, num_numeric), dtype=np.float32)
    categorical_slots = np.zeros((num_cases, max_conditions, num_categorical), dtype=np.int64)
    categorical_slot_mask = np.zeros((num_cases, max_conditions, num_categorical), dtype=np.float32)

    rows: list[dict[str, Any]] = []
    truncation_count = 0
    for case_idx, record in enumerate(records):
        conditions = list(record.get("selected_conditions") or [])
        if len(conditions) > max_conditions:
            truncation_count += 1
        kept = conditions[:max_conditions]
        n_frames = max(_safe_int(record.get("num_frames"), 1), 1)
        num_frames[case_idx] = n_frames
        num_selected[case_idx] = len(kept)
        rows.append(
            {
                "case_index": case_idx,
                "case_id": record.get("case_id"),
                "num_frames": n_frames,
                "num_selected": len(kept),
                "reference_prompt": record.get("reference_prompt") or "",
                "selected_families": [cond.get("family_id") for cond in kept],
            }
        )
        for cond_idx, cond in enumerate(kept):
            slots = cond.get("slot_values") or {}
            condition_mask[case_idx, cond_idx] = 1.0
            action_index[case_idx, cond_idx] = _safe_int(cond.get("action_index"), -1)
            family_id[case_idx, cond_idx] = _id(vocabs["family_vocab"], cond.get("family_id") or "UNKNOWN")
            status_id[case_idx, cond_idx] = _id(vocabs["status_vocab"], cond.get("status") or "unknown")
            score[case_idx, cond_idx] = _safe_float(cond.get("screen_score"), 0.0)
            weight[case_idx, cond_idx] = _safe_float(cond.get("condition_weight"), 0.0)

            span_value = _span(slots)
            if span_value is not None:
                start, end = span_value
                end = max(start, end)
                span[case_idx, cond_idx] = np.asarray([start, end], dtype=np.int64)
                span_mask[case_idx, cond_idx] = 1.0
                denom = float(n_frames)
                span_norm[case_idx, cond_idx] = np.asarray(
                    [
                        start / denom,
                        end / denom,
                        ((start + end) * 0.5) / denom,
                        max(0, end - start) / denom,
                    ],
                    dtype=np.float32,
                )

            nums, num_mask = _numeric_slots(slots)
            cats, cat_mask = _categorical_slots(slots, vocabs["categorical_slot_vocabs"])
            numeric_slots[case_idx, cond_idx] = nums
            numeric_slot_mask[case_idx, cond_idx] = num_mask
            categorical_slots[case_idx, cond_idx] = cats
            categorical_slot_mask[case_idx, cond_idx] = cat_mask

    arrays = {
        "case_index": case_index,
        "num_frames": num_frames,
        "num_selected": num_selected,
        "condition_mask": condition_mask,
        "action_index": action_index,
        "family_id": family_id,
        "status_id": status_id,
        "score": score,
        "condition_weight": weight,
        "span": span,
        "span_mask": span_mask,
        "span_norm": span_norm,
        "numeric_slots": numeric_slots,
        "numeric_slot_mask": numeric_slot_mask,
        "categorical_slots": categorical_slots,
        "categorical_slot_mask": categorical_slot_mask,
    }
    schema = {
        "schema_version": "aml_condition_batch_schema_v1",
        "source_jsonl": source_jsonl,
        "num_cases": num_cases,
        "max_conditions": max_conditions,
        "numeric_slot_names": NUMERIC_SLOT_NAMES,
        "categorical_slot_names": CATEGORICAL_SLOT_NAMES,
        "span_norm_columns": ["start_norm", "end_norm", "center_norm", "duration_norm"],
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "family_vocab": vocabs["family_vocab"],
        "status_vocab": vocabs["status_vocab"],
        "categorical_slot_vocabs": vocabs["categorical_slot_vocabs"],
        "truncated_case_count": truncation_count,
        "padding_policy": {
            "condition_mask": "1 for real selected condition, 0 for padding",
            "family_id/status_id/categorical_slots": "0 is <pad>, 1 is <unk>",
            "span": "[-1, -1] when missing or padding",
        },
    }
    return arrays, schema, rows


def summarize(arrays: dict[str, np.ndarray], schema: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    mask = arrays["condition_mask"] > 0.0
    score_values = arrays["score"][mask]
    family_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    inv_family = {idx: token for token, idx in schema["family_vocab"].items()}
    inv_status = {idx: token for token, idx in schema["status_vocab"].items()}
    for family_idx in arrays["family_id"][mask].tolist():
        family_counter[inv_family.get(int(family_idx), UNK_TOKEN)] += 1
    for status_idx in arrays["status_id"][mask].tolist():
        status_counter[inv_status.get(int(status_idx), UNK_TOKEN)] += 1
    return {
        "schema_version": "aml_condition_batch_schema_summary_v1",
        "num_cases": int(schema["num_cases"]),
        "max_conditions": int(schema["max_conditions"]),
        "total_real_conditions": int(mask.sum()),
        "condition_count_min": int(arrays["num_selected"].min()) if len(rows) else 0,
        "condition_count_max": int(arrays["num_selected"].max()) if len(rows) else 0,
        "condition_count_mean": float(np.mean(arrays["num_selected"])) if len(rows) else 0.0,
        "score_min": float(score_values.min()) if score_values.size else 0.0,
        "score_max": float(score_values.max()) if score_values.size else 0.0,
        "score_mean": float(score_values.mean()) if score_values.size else 0.0,
        "span_coverage": float(arrays["span_mask"][mask].mean()) if mask.sum() else 0.0,
        "numeric_slot_fill_mean": float(arrays["numeric_slot_mask"][mask].mean()) if mask.sum() else 0.0,
        "categorical_slot_fill_mean": float(arrays["categorical_slot_mask"][mask].mean()) if mask.sum() else 0.0,
        "family_counts_top30": family_counter.most_common(30),
        "status_counts": status_counter.most_common(),
        "array_shapes": schema["array_shapes"],
        "truncated_case_count": int(schema["truncated_case_count"]),
    }


def write_report(path: Path, summary: dict[str, Any], schema: dict[str, Any]) -> None:
    lines = [
        "# AML Condition Batch Schema v1",
        "",
        "## Source",
        "",
        f"- source JSONL: `{schema['source_jsonl']}`",
        "",
        "## Counts",
        "",
        f"- cases: `{summary['num_cases']}`",
        f"- max conditions per case: `{summary['max_conditions']}`",
        f"- real selected conditions: `{summary['total_real_conditions']}`",
        f"- per-case selected count: min `{summary['condition_count_min']}`, mean `{summary['condition_count_mean']:.4f}`, max `{summary['condition_count_max']}`",
        f"- truncated cases: `{summary['truncated_case_count']}`",
        f"- score range: `{summary['score_min']:.4f}` to `{summary['score_max']:.4f}`, mean `{summary['score_mean']:.4f}`",
        "",
        "## Arrays",
        "",
        "| name | shape | dtype |",
        "| --- | --- | --- |",
    ]
    for name, shape in schema["array_shapes"].items():
        lines.append(f"| {name} | {shape} | {schema['array_dtypes'][name]} |")
    lines.extend(
        [
            "",
            "## Slot Columns",
            "",
            f"- numeric slots: `{', '.join(schema['numeric_slot_names'])}`",
            f"- categorical slots: `{', '.join(schema['categorical_slot_names'])}`",
            f"- span_norm columns: `{', '.join(schema['span_norm_columns'])}`",
            "",
            "## Status Counts",
            "",
            "| status | count |",
            "| --- | --- |",
        ]
    )
    for status, count in summary["status_counts"]:
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Top Families", "", "| family | count |", "| --- | --- |"])
    for family, count in summary["family_counts_top30"][:20]:
        lines.append(f"| {family} | {count} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `condition_batch.npz`: fixed-shape arrays.",
            "- `condition_batch_schema.json`: column names, vocabularies, shapes, and padding policy.",
            "- `condition_batch_index.jsonl`: case index to case id and selected family names.",
            "- `condition_batch_summary.json`: aggregate smoke statistics.",
            "- `condition_batch_report.md`: this report.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-conditions", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_jsonl(input_jsonl, args.max_records)
    arrays, schema, rows = export_batch_schema(records, args.max_conditions, str(input_jsonl))
    summary = summarize(arrays, schema, rows)

    np.savez_compressed(out_dir / "condition_batch.npz", **arrays)
    (out_dir / "condition_batch_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "condition_batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(out_dir / "condition_batch_index.jsonl", rows)
    write_report(out_dir / "condition_batch_report.md", summary, schema)

    print(f"saved={out_dir}")
    print(
        "cases={num_cases} conditions={total_real_conditions} max_conditions={max_conditions} "
        "score_mean={score_mean:.4f} truncated={truncated_case_count}".format(**summary)
    )


if __name__ == "__main__":
    main()
