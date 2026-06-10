from __future__ import annotations

import argparse
import importlib.util
import json
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
from pseudoedit3d.edit.aml_condition_schema import (
    missing_required_slots,
    required_approx_slots,
    slot_requirement_satisfied,
)

SOURCE = ROOT_DIR / 'scripts' / 'run_momask_aml_prompt_probe.py'
HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')


def _load_source_module():
    spec = importlib.util.spec_from_file_location('run_momask_aml_prompt_probe', SOURCE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


src = _load_source_module()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _read_prompts(case_id: str) -> list[str]:
    path = HML_ROOT / 'texts' / f'{case_id}.txt'
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            out.append(line.split('#')[0].strip())
    return out


def _first_prompt(case_id: str) -> str:
    prompts = _read_prompts(case_id)
    return prompts[0] if prompts else ''


def _case_ids_from_args(args: argparse.Namespace) -> list[str]:
    case_ids: list[str] = []
    if args.case_ids:
        case_ids.extend(x.strip() for x in args.case_ids.split(',') if x.strip())
    if args.case_list:
        for line in Path(args.case_list).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                case_ids.append(line)
    if args.manifest:
        with Path(args.manifest).open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                case_ids.append(str(json.loads(line)['case_id']))
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
        if isinstance(items, dict) and 'cases' in items:
            items = items['cases']
        if not isinstance(items, list):
            raise ValueError(f'Unsupported summary format: {path}')
        for item in items:
            copied = dict(item)
            copied['_source_summary'] = str(path)
            out.append(copied)
    return out


def _extract_case(case_id: str, packed: dict[str, Any], max_residual_events: int) -> dict[str, Any] | None:
    key = f'{case_id}.npy'
    if key not in packed:
        return None
    joints = packed[key]['joints3d']
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    joints = np.asarray(joints, dtype=np.float32)
    if len(joints) <= 20:
        return None
    aml = src.extract_aml_program(joints)
    coarse = build_coarse_action_program(aml['layer3'], max_residual_events=max_residual_events)
    return {
        'case_id': case_id,
        'selected_hml3d_prompt': _first_prompt(case_id),
        'auto_prompt': '',
        'canonical_actions': coarse.get('canonical_actions') or [],
        'coarse_action_program': coarse,
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
            print(f'processed {idx}/{len(case_ids)}, valid={len(out)}, elapsed={time.time() - t0:.1f}s', flush=True)
    return out


def _semantic_family(action: dict[str, Any]) -> dict[str, Any]:
    sf = action.get('semantic_family') or {}
    if sf:
        return dict(sf)
    slots = action.get('slots') or {}
    status = slots.get('semantic_family_status') or 'legacy_missing'
    family_id = slots.get('semantic_family_id') or action.get('canonical_id') or action.get('family') or 'UNKNOWN'
    return {
        'family_id': family_id,
        'source_family': action.get('canonical_id') or action.get('family') or family_id,
        'status': status,
        'label_confidence': slots.get('confidence'),
        'source': 'legacy_or_missing',
        'motion_only': True,
    }


def _action_span(action: dict[str, Any]) -> list[int] | None:
    slots = action.get('slots') or {}
    span = action.get('span') or slots.get('span')
    if isinstance(span, list) and len(span) == 2:
        return [int(span[0]), int(span[1])]
    approx = action.get('approx_slots') or slots.get('approx_slots') or {}
    approx_span = approx.get('span') or {}
    value = approx_span.get('value')
    if isinstance(value, list) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    return None


def _counter_from_mapping(value: Any) -> Counter[str]:
    out: Counter[str] = Counter()
    if isinstance(value, dict):
        for key, count in value.items():
            try:
                out[str(key)] += int(count)
            except (TypeError, ValueError):
                continue
    return out


def _required_slot_coverage_rows(totals: Counter[str], present: Counter[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, total in totals.items():
        got = int(present.get(key, 0))
        family_id, slot = key.split('/', 1)
        rows.append({
            'family_id': family_id,
            'slot': slot,
            'present': got,
            'required': int(total),
            'missing': int(total) - got,
            'coverage': round(got / max(1, int(total)), 4),
        })
    rows.sort(key=lambda item: (-int(item['missing']), item['family_id'], item['slot']))
    return rows


def _collect(cases: list[dict[str, Any]], example_limit: int) -> dict[str, Any]:
    status_actions: Counter[str] = Counter()
    status_cases: dict[str, set[str]] = defaultdict(set)
    family_actions: Counter[str] = Counter()
    source_family_actions: Counter[str] = Counter()
    status_family_actions: Counter[str] = Counter()
    approx_slot_keys: Counter[str] = Counter()
    approx_slot_quality: Counter[str] = Counter()
    required_slot_totals: Counter[str] = Counter()
    required_slot_present: Counter[str] = Counter()
    missing_required_slot_counts: Counter[str] = Counter()
    missing_required_family_actions: Counter[str] = Counter()
    unknown_cluster_counts: Counter[str] = Counter()
    unknown_family_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_semantic = 0
    total_actions = 0

    for case in cases:
        case_id = str(case.get('case_id', ''))
        actions = list(case.get('canonical_actions') or [])
        for action in actions:
            total_actions += 1
            sf = _semantic_family(action)
            status = str(sf.get('status') or 'missing')
            family_id = str(sf.get('family_id') or action.get('canonical_id') or 'UNKNOWN')
            source_family = str(sf.get('source_family') or action.get('canonical_id') or 'UNKNOWN')
            if status == 'legacy_missing':
                missing_semantic += 1
            status_actions[status] += 1
            status_cases[status].add(case_id)
            family_actions[family_id] += 1
            source_family_actions[source_family] += 1
            status_family_actions[f'{status}/{family_id}'] += 1

            approx_slots = action.get('approx_slots') or (action.get('slots') or {}).get('approx_slots') or {}
            if isinstance(approx_slots, dict):
                approx_slot_keys.update(str(k) for k in approx_slots)
                for slot in approx_slots.values():
                    if isinstance(slot, dict) and slot.get('quality') is not None:
                        approx_slot_quality[str(slot.get('quality'))] += 1
            else:
                approx_slots = {}

            required = required_approx_slots(family_id)
            missing_required = missing_required_slots(family_id, approx_slots)
            for slot_key in required:
                counter_key = f'{family_id}/{slot_key}'
                required_slot_totals[counter_key] += 1
                if slot_requirement_satisfied(slot_key, approx_slots):
                    required_slot_present[counter_key] += 1
                elif slot_key in missing_required:
                    missing_required_slot_counts[counter_key] += 1
            if missing_required:
                missing_required_family_actions[family_id] += 1
                if len(examples['missing_required_slots']) < example_limit:
                    examples['missing_required_slots'].append({
                        'case_id': case_id,
                        'canonical_id': action.get('canonical_id'),
                        'family_id': family_id,
                        'status': status,
                        'span': _action_span(action),
                        'missing_slots': missing_required,
                        'present_slots': sorted(str(k) for k in approx_slots),
                        'selected_hml3d_prompt': case.get('selected_hml3d_prompt') or case.get('gt_prompt') or case.get('selected_hml3d_prompt_for_reference_only'),
                    })

            if status == 'unknown':
                slots = action.get('slots') or {}
                unknown_cluster_counts.update(_counter_from_mapping(slots.get('source_event_cluster_counts')))
                unknown_family_counts.update(_counter_from_mapping(slots.get('source_event_family_counts')))

            if len(examples[f'status/{status}']) < example_limit:
                examples[f'status/{status}'].append({
                    'case_id': case_id,
                    'canonical_id': action.get('canonical_id'),
                    'family_id': family_id,
                    'span': _action_span(action),
                    'probe_alias': action.get('probe_alias'),
                    'selected_hml3d_prompt': case.get('selected_hml3d_prompt') or case.get('gt_prompt') or case.get('selected_hml3d_prompt_for_reference_only'),
                    'auto_prompt': case.get('auto_prompt'),
                })
            if status == 'unknown' and len(examples['unknown_detail']) < example_limit:
                slots = action.get('slots') or {}
                examples['unknown_detail'].append({
                    'case_id': case_id,
                    'canonical_id': action.get('canonical_id'),
                    'family_id': family_id,
                    'span': _action_span(action),
                    'source_event_family_counts': slots.get('source_event_family_counts'),
                    'source_event_cluster_counts': slots.get('source_event_cluster_counts'),
                    'selected_hml3d_prompt': case.get('selected_hml3d_prompt') or case.get('gt_prompt') or case.get('selected_hml3d_prompt_for_reference_only'),
                })

    return {
        'run': {
            'processed_cases': len(cases),
            'total_canonical_actions': total_actions,
            'missing_semantic_family_actions': missing_semantic,
        },
        'status_action_counts': status_actions.most_common(),
        'status_case_support': {k: len(v) for k, v in sorted(status_cases.items())},
        'family_action_counts': family_actions.most_common(),
        'source_family_action_counts': source_family_actions.most_common(),
        'status_family_action_counts': status_family_actions.most_common(),
        'approx_slot_key_counts': approx_slot_keys.most_common(),
        'approx_slot_quality_counts': approx_slot_quality.most_common(),
        'required_slot_missing_counts': missing_required_slot_counts.most_common(),
        'required_slot_missing_family_action_counts': missing_required_family_actions.most_common(),
        'required_slot_coverage': _required_slot_coverage_rows(required_slot_totals, required_slot_present),
        'unknown_source_event_family_counts': unknown_family_counts.most_common(),
        'unknown_source_event_cluster_counts': unknown_cluster_counts.most_common(),
        'examples': dict(examples),
    }


def _pct(num: int | float, den: int | float) -> str:
    if not den:
        return '0.0%'
    return f'{100.0 * float(num) / float(den):.1f}%'


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(str(x).replace('\n', ' ') for x in row) + ' |')
    return '\n'.join(out)


def _top_rows(counter_rows: list[list[Any]] | list[tuple[Any, Any]], total: int, limit: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key, count in list(counter_rows)[:limit]:
        rows.append([key, count, _pct(count, total)])
    return rows


def _write_markdown(summary: dict[str, Any], output: Path, top_n: int) -> None:
    run = summary['run']
    total = int(run.get('total_canonical_actions') or 0)
    lines: list[str] = []
    lines.append('# AML Semantic Family Status Report')
    lines.append('')
    lines.append(f"- processed cases: {run.get('processed_cases')}")
    lines.append(f"- canonical actions: {total}")
    lines.append(f"- actions missing semantic metadata: {run.get('missing_semantic_family_actions')}")
    lines.append('')
    lines.append('## Status Distribution')
    lines.append('')
    status_rows = []
    support = summary.get('status_case_support') or {}
    for status, count in summary.get('status_action_counts') or []:
        status_rows.append([status, count, _pct(count, total), support.get(status, 0)])
    lines.append(_table(['status', 'actions', 'action share', 'case support'], status_rows))
    lines.append('')
    lines.append('## Top Status/Families')
    lines.append('')
    lines.append(_table(['status/family', 'actions', 'share'], _top_rows(summary.get('status_family_action_counts') or [], total, top_n)))
    lines.append('')
    lines.append('## Approx Slot Keys')
    lines.append('')
    lines.append(_table(['slot', 'actions', 'share'], _top_rows(summary.get('approx_slot_key_counts') or [], total, top_n)))
    lines.append('')
    lines.append('## Missing Required Approx Slots')
    lines.append('')
    missing_rows = []
    for item in (summary.get('required_slot_coverage') or [])[:top_n]:
        if int(item.get('missing') or 0) <= 0:
            continue
        missing_rows.append([
            item.get('family_id'),
            item.get('slot'),
            item.get('present'),
            item.get('required'),
            item.get('missing'),
            f"{100.0 * float(item.get('coverage') or 0.0):.1f}%",
        ])
    lines.append(_table(['family', 'slot', 'present', 'required', 'missing', 'coverage'], missing_rows))
    lines.append('')
    lines.append('## Missing Required Slot Examples')
    lines.append('')
    rows = []
    for item in (summary.get('examples') or {}).get('missing_required_slots', [])[:top_n]:
        rows.append([
            item.get('case_id'),
            item.get('canonical_id'),
            item.get('family_id'),
            item.get('status'),
            item.get('missing_slots'),
            item.get('present_slots'),
        ])
    lines.append(_table(['case', 'canonical', 'family', 'status', 'missing', 'present'], rows))
    lines.append('')
    lines.append('## Unknown Source Clusters')
    lines.append('')
    unknown_total = sum(count for _, count in summary.get('unknown_source_event_cluster_counts') or [])
    lines.append(_table(['source cluster', 'events', 'share'], _top_rows(summary.get('unknown_source_event_cluster_counts') or [], unknown_total, top_n)))
    lines.append('')
    lines.append('## Unknown Examples')
    lines.append('')
    rows = []
    for item in (summary.get('examples') or {}).get('unknown_detail', [])[:top_n]:
        rows.append([
            item.get('case_id'),
            item.get('canonical_id'),
            item.get('span'),
            item.get('source_event_cluster_counts'),
            item.get('selected_hml3d_prompt'),
        ])
    lines.append(_table(['case', 'canonical', 'span', 'source clusters', 'reference prompt'], rows))
    lines.append('')
    output.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-json', action='append', default=[])
    parser.add_argument('--case-ids', default=None)
    parser.add_argument('--case-list', default=None)
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--max-residual-events', type=int, default=8)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', default=None)
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--example-limit', type=int, default=20)
    parser.add_argument('--progress-every', type=int, default=500)
    args = parser.parse_args()

    cases = _load_summary_cases(args.summary_json)
    case_ids = _case_ids_from_args(args)
    if case_ids:
        cases.extend(_extract_cases(case_ids, args.max_residual_events, args.progress_every))
    if not cases:
        raise SystemExit('No cases found. Provide --summary-json, --case-ids, --case-list, or --manifest.')

    out = _collect(cases, args.example_limit)
    out['run']['summary_json_inputs'] = list(args.summary_json)
    out['run']['case_id_inputs'] = case_ids
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    if args.output_md:
        _write_markdown(out, Path(args.output_md), args.top_n)
    print(f'saved_json={out_path}')
    if args.output_md:
        print(f'saved_md={args.output_md}')


if __name__ == '__main__':
    main()
