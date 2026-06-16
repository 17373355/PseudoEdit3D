from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_HML_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")
DEFAULT_TREE_CANDIDATES = Path("outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_pattern_tree_candidates.json")
DEFAULT_BPE_SEQUENCES = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl")
DEFAULT_ALIAS_SIDECAR = Path("pseudoedit3d/edit/aml_semantic_alias_sidecar.json")
DEFAULT_WORDNET_LEXICON = Path("outputs/aml_lexicon/wordnet_action_terms_v1.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1")

STOP_PHRASES = {
    "a person",
    "the person",
    "a man",
    "the man",
    "a woman",
    "the woman",
    "someone",
    "person is",
    "person does",
    "man is",
    "man does",
}

STOP_TOKENS = {
    "a",
    "an",
    "the",
    "person",
    "man",
    "woman",
    "someone",
    "figure",
    "is",
    "are",
    "was",
    "were",
    "and",
    "then",
    "while",
    "with",
}


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


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _caption_tokens(text: str) -> list[str]:
    return [tok for tok in _normalize_text(text).split() if tok]


def _surface_ngrams(tokens: list[str], max_ngram: int) -> set[str]:
    phrases: set[str] = set()
    n_tokens = len(tokens)
    for n in range(1, max_ngram + 1):
        for idx in range(0, n_tokens - n + 1):
            gram = tokens[idx : idx + n]
            phrase = " ".join(gram)
            if phrase in STOP_PHRASES:
                continue
            if n == 1 and gram[0] in STOP_TOKENS:
                continue
            if all(tok in STOP_TOKENS for tok in gram):
                continue
            phrases.add(phrase)
    return phrases


def _count_caption_phrases(rows: list[dict[str, Any]], *, max_ngram: int, min_case_support: int) -> dict[str, dict[str, Any]]:
    phrase_cases: dict[str, set[str]] = defaultdict(set)
    phrase_caption_count: Counter[str] = Counter()
    phrase_surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    phrase_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id") or "")
        caption = str(row.get("caption") or "")
        tokens = _caption_tokens(caption)
        phrases = _surface_ngrams(tokens, max_ngram)
        for phrase in phrases:
            phrase_cases[phrase].add(case_id)
            phrase_caption_count[phrase] += 1
            phrase_surfaces[phrase][phrase] += 1
            if len(phrase_examples[phrase]) < 6:
                phrase_examples[phrase].append({"case_id": case_id, "caption": caption})

    vocab: dict[str, dict[str, Any]] = {}
    for phrase, cases in phrase_cases.items():
        if len(cases) < min_case_support:
            continue
        vocab[phrase] = {
            "phrase_id": f"txt_phrase_{len(vocab) + 1:06d}",
            "normalized_phrase": phrase,
            "surface_phrase": phrase,
            "tokens": phrase.split(),
            "case_support": len(cases),
            "caption_support": int(phrase_caption_count[phrase]),
            "case_ids": sorted(cases),
            "surface_forms": [
                {"surface": surface, "count": int(count)}
                for surface, count in phrase_surfaces[phrase].most_common(6)
            ],
            "examples": phrase_examples[phrase],
        }
    return vocab


def _compile_alias_rules(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for rule in sidecar.get("rules") or []:
        patterns = []
        for pattern in rule.get("caption_patterns") or []:
            try:
                patterns.append(re.compile(str(pattern), flags=re.IGNORECASE))
            except re.error:
                continue
        rules.append({**rule, "_compiled_patterns": patterns})
    return rules


def _alias_matches_for_phrase(phrase: str, alias_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    padded = f" {phrase} "
    for rule in alias_rules:
        alias_id = str(rule.get("alias_id") or "")
        label = str(rule.get("label") or alias_id.replace("_", " "))
        clause = str(rule.get("clause") or label)
        direct_terms = {alias_id.replace("_", " "), label.lower(), clause.lower()}
        if phrase in direct_terms or any(pattern.search(phrase) or pattern.search(padded) for pattern in rule.get("_compiled_patterns") or []):
            matches.append(
                {
                    "alias_id": alias_id,
                    "label": label,
                    "match_type": "sidecar_caption_pattern_or_label",
                    "confidence": float(rule.get("confidence") or 0.5),
                    "compatible_families": rule.get("compatible_families") or [],
                }
            )
    return sorted(matches, key=lambda item: (-float(item["confidence"]), str(item["alias_id"])))


def _load_wordnet_terms(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    return {str(row.get("term") or ""): row for row in payload.get("terms") or []}


def _wordnet_candidates(phrase: str, wordnet_terms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    checks = [phrase]
    toks = phrase.split()
    if len(toks) > 1:
        checks.append(toks[0])
        checks.append(toks[-1])
    seen: set[str] = set()
    for term in checks:
        if term in seen:
            continue
        seen.add(term)
        entry = wordnet_terms.get(term)
        if not entry:
            continue
        candidates.append(
            {
                "term": term,
                "match_type": "exact_phrase" if term == phrase else "head_or_edge_token",
                "source": entry.get("source") or [],
                "pos": entry.get("pos") or [],
                "taxonomy_parent_candidates": entry.get("taxonomy_parent_candidates") or [],
                "candidate_family_ids": entry.get("candidate_family_ids") or [],
                "sample_synsets": (entry.get("wordnet") or {}).get("sample_synsets") or [],
            }
        )
    return candidates


def _node_motif_ids(node: dict[str, Any]) -> set[str]:
    return {str(row.get("motif_id") or "") for row in node.get("source_motifs") or []}


def _node_cases_from_sequences(rows: list[dict[str, Any]], motif_ids: set[str]) -> set[str]:
    cases: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if any(str(tok.get("symbol") or "") in motif_ids for tok in row.get("bpe_tokens") or []):
            cases.add(case_id)
    return cases


def _score_phrase_for_node(
    phrase: dict[str, Any],
    *,
    node_cases: set[str],
    total_cases: int,
) -> dict[str, Any]:
    phrase_cases = set(phrase.get("case_ids") or [])
    overlap = node_cases & phrase_cases
    phrase_support = len(phrase_cases)
    node_support = len(node_cases)
    precision = len(overlap) / max(1, phrase_support)
    recall = len(overlap) / max(1, node_support)
    background = phrase_support / max(1, total_cases)
    lift = precision / max(background, 1e-9)
    score = (precision * 0.45) + (recall * 0.35) + (min(math.log1p(lift) / 3.0, 1.0) * 0.20)
    return {
        "phrase_id": phrase["phrase_id"],
        "normalized_phrase": phrase["normalized_phrase"],
        "case_overlap": len(overlap),
        "phrase_case_support": phrase_support,
        "node_case_support": node_support,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "lift": round(lift, 4),
        "score": round(score, 4),
        "example_case_ids": sorted(overlap)[:12],
    }


def _candidate_label_status(label: dict[str, Any]) -> str:
    if label["label_type"] == "caption_alias" and float(label.get("confidence", 0.0)) >= 0.65:
        return "strong_alias_name"
    if label["label_type"] == "text_phrase" and float(label.get("score", 0.0)) >= 0.35:
        return "strong_phrase_name"
    if label["label_type"] == "wordnet":
        return "lexical_parent_hint"
    return "diagnostic_name"


def build_naming_layer(
    nodes: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    phrase_vocab: dict[str, dict[str, Any]],
    alias_rules: list[dict[str, Any]],
    wordnet_terms: dict[str, dict[str, Any]],
    *,
    top_phrases_per_node: int,
) -> dict[str, Any]:
    total_cases = len(sequence_rows)
    phrase_list = list(phrase_vocab.values())
    motion_node_labels: list[dict[str, Any]] = []
    alias_clusters: dict[str, dict[str, Any]] = {}

    for phrase in phrase_list:
        phrase["alias_candidates"] = _alias_matches_for_phrase(str(phrase["normalized_phrase"]), alias_rules)
        phrase["wordnet_candidates"] = _wordnet_candidates(str(phrase["normalized_phrase"]), wordnet_terms)
        for alias in phrase["alias_candidates"]:
            row = alias_clusters.setdefault(
                str(alias["alias_id"]),
                {
                    "alias_id": alias["alias_id"],
                    "label": alias.get("label"),
                    "source": ["aml_semantic_alias_sidecar", "text_phrase_match"],
                    "phrase_ids": [],
                    "compatible_families": alias.get("compatible_families") or [],
                },
            )
            row["phrase_ids"].append(phrase["phrase_id"])

    for node in nodes:
        motif_ids = _node_motif_ids(node)
        node_cases = _node_cases_from_sequences(sequence_rows, motif_ids)
        phrase_scores = [
            _score_phrase_for_node(phrase, node_cases=node_cases, total_cases=total_cases)
            for phrase in phrase_list
        ]
        phrase_scores = [
            row for row in phrase_scores
            if row["case_overlap"] > 0 and len(str(row["normalized_phrase"]).split()) <= 4
        ]
        phrase_scores.sort(key=lambda item: (-float(item["score"]), -int(item["case_overlap"]), str(item["normalized_phrase"])))
        top_phrases = phrase_scores[:top_phrases_per_node]
        node_aliases = node.get("naming_evidence", {}).get("top_caption_aliases") or []
        candidate_labels: list[dict[str, Any]] = []
        for alias in node_aliases[:6]:
            alias_id = str(alias.get("id") or "")
            if not alias_id:
                continue
            rule = next((item for item in alias_rules if str(item.get("alias_id") or "") == alias_id), {})
            label = {
                "label_type": "caption_alias",
                "label": str(rule.get("label") or alias_id.replace("_", " ")),
                "alias_id": alias_id,
                "confidence": float(rule.get("confidence") or 0.5),
                "support_count": int(alias.get("count") or 0),
                "compatible_families": rule.get("compatible_families") or [],
                "source": ["motion_node_top_caption_alias", "aml_semantic_alias_sidecar"],
            }
            label["status"] = _candidate_label_status(label)
            candidate_labels.append(label)
        for phrase_score in top_phrases[:8]:
            phrase = phrase_vocab[str(phrase_score["normalized_phrase"])]
            label = {
                "label_type": "text_phrase",
                "label": phrase["normalized_phrase"],
                "phrase_id": phrase["phrase_id"],
                "score": phrase_score["score"],
                "precision": phrase_score["precision"],
                "recall": phrase_score["recall"],
                "lift": phrase_score["lift"],
                "case_overlap": phrase_score["case_overlap"],
                "source": ["humanml3d_text_ngram"],
                "alias_candidates": phrase.get("alias_candidates") or [],
                "wordnet_candidates": phrase.get("wordnet_candidates") or [],
            }
            label["status"] = _candidate_label_status(label)
            candidate_labels.append(label)
        for phrase_score in top_phrases[:5]:
            phrase = phrase_vocab[str(phrase_score["normalized_phrase"])]
            for wn in phrase.get("wordnet_candidates") or []:
                label = {
                    "label_type": "wordnet",
                    "label": wn["term"],
                    "phrase_id": phrase["phrase_id"],
                    "match_type": wn["match_type"],
                    "taxonomy_parent_candidates": wn.get("taxonomy_parent_candidates") or [],
                    "candidate_family_ids": wn.get("candidate_family_ids") or [],
                    "source": wn.get("source") or ["wordnet"],
                }
                label["status"] = _candidate_label_status(label)
                candidate_labels.append(label)

        motion_node_labels.append(
            {
                "motion_node_id": node["node_id"],
                "motion_family_key": node["motion_family_key"],
                "source_motif_ids": sorted(motif_ids),
                "node_case_support": len(node_cases),
                "node_case_support_policy": "unique cases containing any source motif in case_bpe_sequences",
                "motion_evidence": node.get("motion_evidence") or {},
                "candidate_labels": candidate_labels,
                "top_phrase_alignments": top_phrases,
                "policy": "labels are attached as naming evidence only; they do not create or restructure motion nodes",
            }
        )

    return {
        "schema_version": "text_bpe_wordnet_naming_layer_v1",
        "runtime_policy": "language names motion-derived nodes; language does not create or restructure the motion tree",
        "summary": {
            "motion_node_count": len(nodes),
            "caption_case_count": total_cases,
            "phrase_vocab_size": len(phrase_vocab),
            "alias_cluster_count": len(alias_clusters),
            "wordnet_term_count": len(wordnet_terms),
        },
        "phrase_vocab": [
            {
                key: value
                for key, value in phrase.items()
                if key not in {"case_ids"}
            }
            for phrase in sorted(phrase_list, key=lambda item: (-int(item["case_support"]), str(item["normalized_phrase"])))
        ],
        "alias_clusters": sorted(alias_clusters.values(), key=lambda item: str(item["alias_id"])),
        "motion_node_labels": motion_node_labels,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Text-BPE / WordNet Naming Layer")
    lines.append("")
    lines.append("This artifact attaches language-side names to motion-derived nodes.")
    lines.append("It does not create or restructure motion nodes.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in (payload.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Motion Node Labels")
    lines.append("")
    lines.append("| node | motion key | case support | top labels | top phrase alignments |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for row in payload.get("motion_node_labels") or []:
        labels = []
        for item in row.get("candidate_labels") or []:
            if item.get("label_type") in {"caption_alias", "text_phrase"}:
                labels.append(f"{item.get('label')}:{item.get('status')}")
            if len(labels) >= 4:
                break
        phrases = ", ".join(
            f"{item['normalized_phrase']}({item['score']})"
            for item in (row.get("top_phrase_alignments") or [])[:4]
        )
        lines.append(
            "| `{node}` | `{key}` | {support} | {labels} | {phrases} |".format(
                node=row["motion_node_id"],
                key=row["motion_family_key"],
                support=row["node_case_support"],
                labels=", ".join(labels) or "none",
                phrases=phrases or "none",
            )
        )
    lines.append("")
    lines.append("## Details")
    for row in payload.get("motion_node_labels") or []:
        lines.append("")
        lines.append(f"### {row['motion_node_id']}")
        lines.append("")
        lines.append(f"- motion family key: `{row['motion_family_key']}`")
        lines.append(f"- source motifs: `{', '.join(row['source_motif_ids'])}`")
        lines.append(f"- unique case support: `{row['node_case_support']}`")
        lines.append("- candidate labels:")
        for item in (row.get("candidate_labels") or [])[:12]:
            lines.append(
                f"  - `{item.get('label')}` type=`{item.get('label_type')}` status=`{item.get('status')}` source=`{','.join(item.get('source') or [])}`"
            )
        lines.append("- top phrase alignments:")
        for item in (row.get("top_phrase_alignments") or [])[:8]:
            lines.append(
                f"  - `{item['normalized_phrase']}` overlap={item['case_overlap']} precision={item['precision']} recall={item['recall']} lift={item['lift']} score={item['score']}"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build language naming evidence for motion-derived tree candidates.")
    parser.add_argument("--hml-root", default=str(DEFAULT_HML_ROOT))
    parser.add_argument("--tree-candidates", default=str(DEFAULT_TREE_CANDIDATES))
    parser.add_argument("--bpe-sequences", default=str(DEFAULT_BPE_SEQUENCES))
    parser.add_argument("--alias-sidecar", default=str(DEFAULT_ALIAS_SIDECAR))
    parser.add_argument("--wordnet-lexicon", default=str(DEFAULT_WORDNET_LEXICON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--min-phrase-case-support", type=int, default=20)
    parser.add_argument("--top-phrases-per-node", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tree_payload = _read_json(Path(args.tree_candidates))
    sequence_rows = _read_jsonl(Path(args.bpe_sequences))
    alias_sidecar = _read_json(Path(args.alias_sidecar))
    wordnet_terms = _load_wordnet_terms(Path(args.wordnet_lexicon))
    phrase_vocab = _count_caption_phrases(
        sequence_rows,
        max_ngram=int(args.max_ngram),
        min_case_support=int(args.min_phrase_case_support),
    )
    alias_rules = _compile_alias_rules(alias_sidecar)
    payload = build_naming_layer(
        tree_payload.get("candidate_nodes") or [],
        sequence_rows,
        phrase_vocab,
        alias_rules,
        wordnet_terms,
        top_phrases_per_node=int(args.top_phrases_per_node),
    )
    payload["source"] = {
        "tree_candidates": str(args.tree_candidates),
        "bpe_sequences": str(args.bpe_sequences),
        "alias_sidecar": str(args.alias_sidecar),
        "wordnet_lexicon": str(args.wordnet_lexicon),
        "hml_root": str(args.hml_root),
    }
    _write_json(output_dir / "text_bpe_wordnet_naming_layer.json", payload)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "text_bpe_wordnet_naming_layer_summary_v1",
            **payload["summary"],
            "source": payload["source"],
        },
    )
    write_report(output_dir / "text_bpe_wordnet_naming_layer.md", payload)
    print(output_dir)


if __name__ == "__main__":
    main()

