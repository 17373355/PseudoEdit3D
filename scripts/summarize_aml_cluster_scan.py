from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pct(num: float, den: float) -> str:
    if den <= 0:
        return '0.0%'
    return f'{100.0 * num / den:.1f}%'


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |']
    out.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        out.append('| ' + ' | '.join(str(x).replace('\n', ' ') for x in row) + ' |')
    return '\n'.join(out)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _examples_for(scan: dict[str, Any], cluster_key: str, limit: int = 2) -> str:
    examples = scan.get('cluster_examples', {}).get(cluster_key, [])[:limit]
    parts: list[str] = []
    for item in examples:
        prompt = str(item.get('selected_hml3d_prompt') or '').strip()
        if len(prompt) > 90:
            prompt = prompt[:87] + '...'
        parts.append(f"{item.get('case_id')}:{prompt}")
    return ' ; '.join(parts)


def _family_rows(scan: dict[str, Any], top_n: int) -> list[list[Any]]:
    support = scan.get('super_family_case_support', {})
    total_events = scan.get('global_stats', {}).get('total_layer3_events', 0)
    processed = scan.get('run', {}).get('processed_cases', 0)
    rows = []
    for family, count in scan.get('super_family_counts', [])[:top_n]:
        rows.append([
            family,
            count,
            _pct(count, total_events),
            support.get(family, 0),
            _pct(support.get(family, 0), processed),
        ])
    return rows


def _cluster_rows(scan: dict[str, Any], top_n: int) -> list[list[Any]]:
    support = scan.get('family_cluster_case_support', {})
    total_events = scan.get('global_stats', {}).get('total_layer3_events', 0)
    processed = scan.get('run', {}).get('processed_cases', 0)
    rows = []
    for cluster_key, count in scan.get('family_cluster_counts', [])[:top_n]:
        rows.append([
            cluster_key,
            count,
            _pct(count, total_events),
            support.get(cluster_key, 0),
            _pct(support.get(cluster_key, 0), processed),
            _examples_for(scan, cluster_key),
        ])
    return rows


def _low_event_rows(scan: dict[str, Any], limit: int) -> list[list[Any]]:
    rows = []
    for item in scan.get('low_event_cases_le2', [])[:limit]:
        auto = str(item.get('auto_prompt') or '')
        prompt = str(item.get('selected_hml3d_prompt') or '')
        if len(auto) > 110:
            auto = auto[:107] + '...'
        if len(prompt) > 110:
            prompt = prompt[:107] + '...'
        rows.append([item.get('case_id'), item.get('layer3_count'), auto, prompt])
    return rows


def _keyword_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    groups = {
        'stairs_or_rail': ('stair', 'step', 'rail', 'railing'),
        'support_or_wall': ('support', 'wall', 'surface'),
        'object_or_hold': ('hold', 'holding', 'pick', 'grab', 'carry', 'throw', 'put'),
        'sit_lie_kneel': ('sit', 'sitting', 'lie', 'lying', 'kneel', 'kneeling'),
        'jump_or_hop': ('jump', 'hop'),
        'run_or_walk': ('walk', 'walking', 'run', 'running', 'jog'),
    }
    counts = {k: 0 for k in groups}
    for item in items:
        text = str(item.get('selected_hml3d_prompt') or '').lower()
        for key, words in groups.items():
            if any(word in text for word in words):
                counts[key] += 1
    return counts


def _mechanism_notes(scan: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    notes: list[str] = []
    stats = scan.get('global_stats', {})
    run = scan.get('run', {})
    processed = int(run.get('processed_cases') or 0)
    low = int(stats.get('low_event_case_count_le2') or 0)
    notes.append(
        f"Use this with-phase full scan as the current AML regression baseline: "
        f"{processed} cases, avg Layer3={_fmt_float(stats.get('avg_layer3_count'))}, "
        f"low-event<=2={low} ({_pct(low, processed)})."
    )
    if baseline is not None:
        b_stats = baseline.get('global_stats', {})
        b_low = int(b_stats.get('low_event_case_count_le2') or 0)
        b_avg = float(b_stats.get('avg_layer3_count') or 0.0)
        c_avg = float(stats.get('avg_layer3_count') or 0.0)
        notes.append(
            f"Phase detection should stay in the mainline: low-event cases changed "
            f"from {b_low} to {low}, and avg Layer3 changed from {b_avg:.3f} to {c_avg:.3f}."
        )
    notes.append(
        "High-support vertical clusters are the first salience-control target: "
        "ordinary walking root oscillation can become false jump/squat wording if rendered too eagerly."
    )
    notes.append(
        "High-support bimanual and arm-repeat clusters should be treated as family candidates, "
        "not direct action names: walking arm swing, object/hand interaction, clapping/waving, and support-like poses need subclustering."
    )
    notes.append(
        "Support/contact remains under-observed: motion-only AML can infer hand anchoring or support-like extension, "
        "but should not say wall/rail/surface unless scene evidence is available."
    )
    notes.append(
        "The next mechanism step should add full-scan regression gates: vertical suppression during locomotion, "
        "arm-family abstraction, support/contact proxies, and numeric/repetition rendering checks."
    )
    return notes


def build_report(scan: dict[str, Any], scan_path: Path, baseline: dict[str, Any] | None, baseline_path: Path | None, top_n: int, low_limit: int) -> str:
    run = scan.get('run', {})
    stats = scan.get('global_stats', {})
    processed = int(run.get('processed_cases') or 0)
    low = int(stats.get('low_event_case_count_le2') or 0)
    low_items = scan.get('low_event_cases_le2', [])
    keyword_counts = _keyword_counts(low_items)

    lines: list[str] = []
    lines.append('# AML Full HumanML3D Cluster Scan v1')
    lines.append('')
    lines.append('## Scan')
    lines.append('')
    lines.append(f"- scan: `{scan_path}`")
    if baseline_path is not None:
        lines.append(f"- baseline: `{baseline_path}`")
    lines.append(f"- requested cases: {run.get('requested_cases')}")
    lines.append(f"- processed cases: {processed}")
    lines.append(f"- detect phase: {run.get('detect_phase')}")
    lines.append(f"- elapsed sec: {_fmt_float(run.get('elapsed_sec'), 2)}")
    lines.append(f"- avg Layer1 / Layer2 / Layer2.5 / Layer3: {_fmt_float(stats.get('avg_layer1_count'))} / {_fmt_float(stats.get('avg_layer2_count'))} / {_fmt_float(stats.get('avg_layer25_count'))} / {_fmt_float(stats.get('avg_layer3_count'))}")
    lines.append(f"- low-event cases <=2: {low} ({_pct(low, processed)})")
    lines.append('')
    lines.append('## Mechanism Readout')
    lines.append('')
    for note in _mechanism_notes(scan, baseline):
        lines.append(f"- {note}")
    lines.append('')
    lines.append('## Super Families')
    lines.append('')
    lines.append(_markdown_table(['family', 'event count', 'event share', 'case support', 'case share'], _family_rows(scan, top_n)))
    lines.append('')
    lines.append('## Top Family Clusters')
    lines.append('')
    lines.append(_markdown_table(['family/cluster', 'event count', 'event share', 'case support', 'case share', 'examples'], _cluster_rows(scan, top_n)))
    lines.append('')
    lines.append('## Low-Event Prompt Keywords')
    lines.append('')
    lines.append('This diagnostic only uses the stored low-event examples, not the full hidden low-event set.')
    lines.append('')
    lines.append(_markdown_table(['keyword group', 'count in stored low-event examples'], [[k, v] for k, v in keyword_counts.items()]))
    lines.append('')
    lines.append('## Low-Event Examples')
    lines.append('')
    lines.append(_markdown_table(['case', 'L3 count', 'auto prompt', 'selected HML3D prompt'], _low_event_rows(scan, low_limit)))
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', required=True)
    parser.add_argument('--baseline', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--low-event-examples', type=int, default=30)
    args = parser.parse_args()

    scan_path = Path(args.scan)
    baseline_path = Path(args.baseline) if args.baseline else None
    scan = _load(scan_path)
    baseline = _load(baseline_path) if baseline_path else None
    report = build_report(scan, scan_path, baseline, baseline_path, args.top_n, args.low_event_examples)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding='utf-8')
    print(f'saved={out_path}')


if __name__ == '__main__':
    main()
