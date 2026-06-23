"""Mine caption/WordNet name candidates for v0 AML motion nodes.

This sidecar mines HumanML3D captions for readable phrase candidates, attaches
cached WordNet/taxonomy hints, and aligns those names to already-existing
motion-derived motif/family nodes by case-support overlap.

It does not create, merge, split, promote, or restructure motion nodes. It also
does not read the manual AML registry.

Outputs:
  - name_candidates.json
  - name_candidates.md
  - summary.json

Quick check:
    python -m py_compile scripts/mine_hml3d_caption_wordnet_name_candidates_v0.py
    python scripts/mine_hml3d_caption_wordnet_name_candidates_v0.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_CORPUS = Path(
    "outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl"
)
DEFAULT_MOTIF_FAMILY_CANDIDATES = Path(
    "outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/motif_family_candidates.json"
)
DEFAULT_COORDINATION_FOREST = Path(
    "outputs/aml_regression_testset_v2/coordination_pattern_forest_loose_v1/coordination_pattern_forest.json"
)
DEFAULT_MULTICHANNEL_SEQUENCES = Path(
    "outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/case_multichannel_bpe_sequences.jsonl"
)
DEFAULT_WORDNET_LEXICON = Path("outputs/aml_lexicon/wordnet_action_terms_v1.json")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/hml3d_caption_wordnet_name_candidates_v0"
)


GENERIC_ACTOR_TOKENS = {
    "person",
    "people",
    "man",
    "men",
    "woman",
    "women",
    "someone",
    "somebody",
    "human",
    "figure",
    "stick",
    "character",
}

PRONOUN_OR_OBJECT_TOKENS = {
    "he",
    "she",
    "they",
    "them",
    "him",
    "her",
    "his",
    "hers",
    "their",
    "theirs",
    "it",
    "its",
    "something",
    "anything",
    "object",
    "objects",
}

EDGE_STOP_TOKENS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "am",
    "do",
    "does",
    "did",
    "done",
    "doing",
    "to",
    "from",
    "for",
    "of",
    "with",
    "without",
    "by",
    "as",
    "and",
    "or",
    "but",
    "while",
    "then",
    "before",
    "after",
    "during",
    "into",
    "onto",
    "at",
    "on",
}

LOW_INFORMATION_TOKENS = (
    GENERIC_ACTOR_TOKENS
    | PRONOUN_OR_OBJECT_TOKENS
    | EDGE_STOP_TOKENS
    | {
        "motion",
        "movement",
        "position",
        "pose",
        "stance",
        "original",
        "starting",
        "start",
        "end",
        "ends",
        "back",
        "front",
        "side",
        "left",
        "right",
        "both",
        "each",
    }
)

BODY_PART_TOKENS = {
    "arm",
    "arms",
    "hand",
    "hands",
    "leg",
    "legs",
    "foot",
    "feet",
    "body",
    "torso",
    "head",
    "hip",
    "hips",
    "knee",
    "knees",
    "shoulder",
    "shoulders",
}

PHRASE_DROP_TOKENS = GENERIC_ACTOR_TOKENS | PRONOUN_OR_OBJECT_TOKENS


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _caption_tokens(text: str) -> list[str]:
    return [tok.strip("'") for tok in _normalize_text(text).split() if tok.strip("'")]


def _safe_id(text: str, *, max_len: int = 80) -> str:
    out: list[str] = []
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
    return (compact or "unnamed")[:max_len].strip("_") or "unnamed"


def _phrase_is_readable(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if any(tok in PHRASE_DROP_TOKENS for tok in tokens):
        return False
    if tokens[0] in EDGE_STOP_TOKENS or tokens[-1] in EDGE_STOP_TOKENS:
        return False
    if all(tok in LOW_INFORMATION_TOKENS for tok in tokens):
        return False
    if all(tok in BODY_PART_TOKENS or tok in {"left", "right", "both"} for tok in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) < 3 or token in LOW_INFORMATION_TOKENS or token in BODY_PART_TOKENS:
            return False
    return True


def _surface_ngrams(tokens: list[str], max_phrase_tokens: int) -> set[str]:
    phrases: set[str] = set()
    max_n = max(1, int(max_phrase_tokens))
    for n in range(1, max_n + 1):
        for start in range(0, len(tokens) - n + 1):
            gram = tokens[start : start + n]
            if _phrase_is_readable(gram):
                phrases.add(" ".join(gram))
    return phrases


def _caption_contains_phrase(caption: str, phrase: str) -> bool:
    caption_tokens = _caption_tokens(caption)
    phrase_tokens = phrase.split()
    if not phrase_tokens or len(phrase_tokens) > len(caption_tokens):
        return False
    width = len(phrase_tokens)
    return any(caption_tokens[idx : idx + width] == phrase_tokens for idx in range(0, len(caption_tokens) - width + 1))


def _top_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _simple_term_variants(term: str) -> set[str]:
    term = _normalize_text(term)
    if not term:
        return set()
    variants = {term}
    if " " in term:
        tokens = term.split()
        lemma_tokens = [_best_simple_lemma(tok) for tok in tokens]
        variants.add(" ".join(lemma_tokens))
        return {item for item in variants if item}

    token = term
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        variants.add(stem)
        variants.add(stem + "e")
        if len(stem) > 2 and stem[-1] == stem[-2]:
            variants.add(stem[:-1])
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        variants.add(stem)
        variants.add(stem + "e")
        if len(stem) > 2 and stem[-1] == stem[-2]:
            variants.add(stem[:-1])
    if token.endswith("es") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        variants.add(token[:-1])
    return {item for item in variants if item}


def _best_simple_lemma(token: str) -> str:
    variants = sorted(_simple_term_variants(token), key=lambda item: (len(item), item))
    return variants[0] if variants else token


def _load_wordnet_terms(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    terms: dict[str, dict[str, Any]] = {}
    for row in payload.get("terms") or []:
        term = str(row.get("term") or "")
        if term:
            terms[term] = row
    return terms


def _wordnet_lookup_order(phrase: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    def add(term: str, match_type: str) -> None:
        term = _normalize_text(term)
        if term:
            candidates.append((term, match_type))

    add(phrase, "exact_phrase")
    for variant in sorted(_simple_term_variants(phrase)):
        if variant != phrase:
            add(variant, "simple_lemma_phrase")

    tokens = phrase.split()
    for width in range(min(3, len(tokens)), 1, -1):
        for start in range(0, len(tokens) - width + 1):
            subphrase = " ".join(tokens[start : start + width])
            if subphrase != phrase:
                add(subphrase, "phrase_subspan")
                for variant in sorted(_simple_term_variants(subphrase)):
                    if variant != subphrase:
                        add(variant, "simple_lemma_subspan")
    for token in tokens:
        for variant in sorted(_simple_term_variants(token)):
            add(variant, "token_or_token_lemma")

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for term, match_type in candidates:
        if term in seen:
            continue
        seen.add(term)
        deduped.append((term, match_type))
    return deduped


def _wordnet_hints_for_phrase(
    phrase: str,
    wordnet_terms: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for term, match_type in _wordnet_lookup_order(phrase):
        entry = wordnet_terms.get(term)
        if not entry:
            continue
        wordnet = entry.get("wordnet") or {}
        hints.append(
            {
                "term": term,
                "match_type": match_type,
                "pos": entry.get("pos") or [],
                "source": entry.get("source") or [],
                "is_mapped": bool(entry.get("is_mapped")),
                "taxonomy_parent_candidates": entry.get("taxonomy_parent_candidates") or [],
                "candidate_family_ids": entry.get("candidate_family_ids") or [],
                "wordnet": {
                    "verb_synset_count": int(wordnet.get("verb_synset_count") or 0),
                    "noun_synset_count": int(wordnet.get("noun_synset_count") or 0),
                    "lexnames": wordnet.get("lexnames") or {},
                    "sample_synsets": (wordnet.get("sample_synsets") or [])[:3],
                },
            }
        )
        if len(hints) >= limit:
            break
    return hints


def mine_caption_phrases(
    corpus_path: Path,
    wordnet_terms: dict[str, dict[str, Any]],
    *,
    max_phrase_tokens: int,
    min_phrase_case_support: int,
    max_examples_per_phrase: int,
    wordnet_hints_per_phrase: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    phrase_cases: dict[str, set[str]] = defaultdict(set)
    phrase_caption_count: Counter[str] = Counter()
    phrase_surface_count: dict[str, Counter[str]] = defaultdict(Counter)
    phrase_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phrase_example_case_ids: dict[str, set[str]] = defaultdict(set)
    case_captions: dict[str, list[str]] = {}
    case_count = 0
    caption_count = 0

    for row in _iter_jsonl(corpus_path):
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        captions = [str(item) for item in (row.get("caption_texts") or []) if str(item).strip()]
        case_captions[case_id] = captions
        case_count += 1
        case_seen_phrases: set[str] = set()
        for caption in captions:
            caption_count += 1
            phrases = _surface_ngrams(_caption_tokens(caption), max_phrase_tokens)
            for phrase in phrases:
                phrase_caption_count[phrase] += 1
                phrase_surface_count[phrase][phrase] += 1
                case_seen_phrases.add(phrase)
                if (
                    len(phrase_examples[phrase]) < max_examples_per_phrase
                    and case_id not in phrase_example_case_ids[phrase]
                ):
                    phrase_examples[phrase].append({"case_id": case_id, "caption": caption})
                    phrase_example_case_ids[phrase].add(case_id)
        for phrase in case_seen_phrases:
            phrase_cases[phrase].add(case_id)

    phrase_vocab: dict[str, dict[str, Any]] = {}
    for phrase, case_ids in sorted(phrase_cases.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(case_ids) < int(min_phrase_case_support):
            continue
        phrase_id = f"caption_phrase_{len(phrase_vocab) + 1:06d}"
        wordnet_hints = _wordnet_hints_for_phrase(phrase, wordnet_terms, limit=wordnet_hints_per_phrase)
        phrase_vocab[phrase] = {
            "phrase_id": phrase_id,
            "normalized_phrase": phrase,
            "tokens": phrase.split(),
            "case_support": len(case_ids),
            "caption_support": int(phrase_caption_count[phrase]),
            "case_ids": set(case_ids),
            "surface_forms": [
                {"surface": surface, "count": int(count)}
                for surface, count in phrase_surface_count[phrase].most_common(6)
            ],
            "wordnet_hints": wordnet_hints,
            "examples": phrase_examples[phrase],
        }

    stats = {
        "caption_case_count": case_count,
        "caption_count": caption_count,
        "raw_phrase_count": len(phrase_cases),
        "retained_phrase_count": len(phrase_vocab),
        "min_phrase_case_support": int(min_phrase_case_support),
    }
    return phrase_vocab, case_captions, stats


def _forest_child_map(forest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    node_by_id = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if parent and child in node_by_id:
            children[parent].append(node_by_id[child])
    return children


def _node_source_symbols_from_forest_node(
    node: dict[str, Any],
    children: dict[str, list[dict[str, Any]]],
) -> list[str]:
    symbols: set[str] = set()
    direct = str(node.get("source_motif_id") or "")
    if direct:
        symbols.add(direct)
    for child in children.get(str(node.get("node_id") or ""), []):
        child_symbol = str(child.get("source_motif_id") or "")
        if child_symbol:
            symbols.add(child_symbol)
    return sorted(symbols)


def load_motion_nodes(
    motif_family_candidates: dict[str, Any],
    coordination_forest: dict[str, Any],
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for family in motif_family_candidates.get("families") or []:
        family_id = str(family.get("family_id") or "")
        if not family_id:
            continue
        source_symbols = sorted(
            {
                str(item.get("motif_id") or "")
                for item in (family.get("source_motifs") or [])
                if str(item.get("motif_id") or "")
            }
        )
        nodes.append(
            {
                "node_id": family_id,
                "node_kind": "multichannel_motif_family_candidate",
                "source_artifact": "motif_family_candidates",
                "source_symbols": source_symbols,
                "motion_key": str(family.get("motion_family_key") or family_id),
                "source_status": str(family.get("status") or ""),
                "artifact_support": {
                    "motif_count": int(family.get("motif_count") or len(source_symbols)),
                    "support_cases_sum": int(family.get("support_cases_sum") or 0),
                    "occurrences_sum": int(family.get("occurrences_sum") or 0),
                },
                "motion_definition": family.get("motion_definition") or {},
            }
        )

    children = _forest_child_map(coordination_forest)
    for node in coordination_forest.get("nodes") or []:
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        source_symbols = _node_source_symbols_from_forest_node(node, children)
        support = node.get("support") or {}
        nodes.append(
            {
                "node_id": node_id,
                "node_kind": str(node.get("node_kind") or "coordination_forest_node"),
                "source_artifact": "coordination_pattern_forest",
                "source_symbols": source_symbols,
                "motion_key": str(node.get("family_key") or node.get("candidate_id") or node.get("source_motif_id") or node_id),
                "source_status": str(node.get("status") or ""),
                "artifact_support": support,
                "motion_definition": node.get("motion_definition") or {},
            }
        )
    return nodes


def index_symbol_cases(
    sequence_path: Path,
    needed_symbols: set[str],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    symbol_cases: dict[str, set[str]] = {symbol: set() for symbol in sorted(needed_symbols)}
    sequence_count = 0
    sequence_case_ids: set[str] = set()
    token_count = 0
    matched_token_count = 0

    for row in _iter_jsonl(sequence_path):
        sequence_count += 1
        case_id = str(row.get("case_id") or "")
        if case_id:
            sequence_case_ids.add(case_id)
        for token in row.get("tokens") or []:
            token_count += 1
            symbol = str(token.get("symbol") or "")
            if symbol in symbol_cases:
                symbol_cases[symbol].add(case_id)
                matched_token_count += 1

    stats = {
        "sequence_count": sequence_count,
        "sequence_case_count": len(sequence_case_ids),
        "token_count": token_count,
        "needed_symbol_count": len(needed_symbols),
        "matched_symbol_count": sum(1 for cases in symbol_cases.values() if cases),
        "matched_token_count": matched_token_count,
    }
    return symbol_cases, stats


def _node_cases(node: dict[str, Any], symbol_cases: dict[str, set[str]]) -> set[str]:
    out: set[str] = set()
    for symbol in node.get("source_symbols") or []:
        out.update(symbol_cases.get(str(symbol), set()))
    return out


def _alignment_score(
    *,
    overlap_count: int,
    phrase_support: int,
    node_support: int,
    total_cases: int,
) -> dict[str, Any]:
    precision = overlap_count / max(1, phrase_support)
    recall = overlap_count / max(1, node_support)
    union = phrase_support + node_support - overlap_count
    jaccard = overlap_count / max(1, union)
    node_background = node_support / max(1, total_cases)
    lift = precision / max(node_background, 1e-9)
    lift_bonus = min(max(math.log2(max(lift, 1.0)) / 4.0, 0.0), 1.0)
    support_bonus = min(math.log1p(overlap_count) / math.log1p(50), 1.0)
    score = (
        precision * 0.35
        + recall * 0.30
        + jaccard * 0.20
        + lift_bonus * 0.10
        + support_bonus * 0.05
    )
    return {
        "overlap_cases": int(overlap_count),
        "phrase_case_support": int(phrase_support),
        "node_case_support": int(node_support),
        "precision_phrase_to_node": round(precision, 4),
        "recall_node_covered": round(recall, 4),
        "jaccard": round(jaccard, 4),
        "lift_over_node_background": round(lift, 4),
        "score": round(score, 4),
    }


def _candidate_status(metrics: dict[str, Any], wordnet_hints: list[dict[str, Any]]) -> str:
    score = float(metrics.get("score") or 0.0)
    overlap = int(metrics.get("overlap_cases") or 0)
    precision = float(metrics.get("precision_phrase_to_node") or 0.0)
    mapped_hint = any(bool(hint.get("is_mapped")) for hint in wordnet_hints)
    if overlap >= 8 and score >= 0.35 and (precision >= 0.25 or mapped_hint):
        return "strong_caption_wordnet_name_candidate"
    if overlap >= 4 and score >= 0.2:
        return "review_caption_name_candidate"
    return "diagnostic_caption_name_candidate"


def _examples_for_overlap(
    overlap_case_ids: set[str],
    phrase: str,
    case_captions: dict[str, list[str]],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for case_id in sorted(overlap_case_ids):
        captions = case_captions.get(case_id) or []
        matched = [caption for caption in captions if _caption_contains_phrase(caption, phrase)]
        if not matched:
            matched = captions[:2]
        examples.append(
            {
                "case_id": case_id,
                "matched_captions": matched[:2],
                "caption_texts": captions[:4],
            }
        )
        if len(examples) >= max_examples:
            break
    return examples


def align_name_candidates(
    motion_nodes: list[dict[str, Any]],
    symbol_cases: dict[str, set[str]],
    phrase_vocab: dict[str, dict[str, Any]],
    case_captions: dict[str, list[str]],
    *,
    total_cases: int,
    min_case_overlap: int,
    top_candidates_per_node: int,
    max_examples_per_candidate: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phrase_rows = list(phrase_vocab.values())
    aligned_nodes: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    nodes_with_candidates = 0
    total_candidate_count = 0

    for node in motion_nodes:
        cases = _node_cases(node, symbol_cases)
        candidates: list[dict[str, Any]] = []
        if cases:
            for phrase_row in phrase_rows:
                phrase_cases = phrase_row.get("case_ids") or set()
                overlap = cases & phrase_cases
                if len(overlap) < int(min_case_overlap):
                    continue
                phrase = str(phrase_row["normalized_phrase"])
                metrics = _alignment_score(
                    overlap_count=len(overlap),
                    phrase_support=len(phrase_cases),
                    node_support=len(cases),
                    total_cases=total_cases,
                )
                wordnet_hints = phrase_row.get("wordnet_hints") or []
                candidate = {
                    "phrase_id": phrase_row["phrase_id"],
                    "name": phrase,
                    "source": ["humanml3d_caption_phrase"],
                    "alignment": metrics,
                    "wordnet_hints": wordnet_hints,
                    "examples": _examples_for_overlap(
                        overlap,
                        phrase,
                        case_captions,
                        max_examples=max_examples_per_candidate,
                    ),
                    "overlap_case_ids_sample": sorted(overlap)[:20],
                }
                candidate["status"] = _candidate_status(metrics, wordnet_hints)
                candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                -float((item.get("alignment") or {}).get("score") or 0.0),
                -int((item.get("alignment") or {}).get("overlap_cases") or 0),
                len(str(item.get("name") or "").split()),
                str(item.get("name") or ""),
            )
        )
        candidates = candidates[: int(top_candidates_per_node)]
        status_counts.update(str(item.get("status") or "") for item in candidates)
        if candidates:
            nodes_with_candidates += 1
            total_candidate_count += len(candidates)
        aligned_nodes.append(
            {
                "node_id": node["node_id"],
                "node_kind": node["node_kind"],
                "source_artifact": node["source_artifact"],
                "motion_key": node["motion_key"],
                "source_status": node.get("source_status") or "",
                "source_symbols": node.get("source_symbols") or [],
                "case_support": len(cases),
                "case_support_policy": "unique cases containing any source motif symbol in case_multichannel_bpe_sequences.jsonl",
                "artifact_support": node.get("artifact_support") or {},
                "motion_definition": node.get("motion_definition") or {},
                "name_candidates": candidates,
                "policy": "caption/WordNet names are sidecar evidence only; they do not create or restructure motion nodes",
            }
        )

    stats = {
        "motion_node_count": len(motion_nodes),
        "nodes_with_name_candidates": nodes_with_candidates,
        "name_candidate_count": total_candidate_count,
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "min_case_overlap": int(min_case_overlap),
        "top_candidates_per_node": int(top_candidates_per_node),
    }
    return aligned_nodes, stats


def _serializable_phrase_candidates(
    phrase_vocab: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = sorted(
        phrase_vocab.values(),
        key=lambda item: (-int(item.get("case_support") or 0), str(item.get("normalized_phrase") or "")),
    )
    out: list[dict[str, Any]] = []
    for row in rows[: max(0, int(limit))]:
        out.append(
            {
                "phrase_id": row["phrase_id"],
                "normalized_phrase": row["normalized_phrase"],
                "tokens": row["tokens"],
                "case_support": int(row["case_support"]),
                "caption_support": int(row["caption_support"]),
                "surface_forms": row.get("surface_forms") or [],
                "wordnet_hints": row.get("wordnet_hints") or [],
                "examples": row.get("examples") or [],
            }
        )
    return out


def build_name_candidates(args: argparse.Namespace) -> dict[str, Any]:
    wordnet_terms = _load_wordnet_terms(Path(args.wordnet_lexicon))
    phrase_vocab, case_captions, phrase_stats = mine_caption_phrases(
        Path(args.source_corpus),
        wordnet_terms,
        max_phrase_tokens=int(args.max_phrase_tokens),
        min_phrase_case_support=int(args.min_phrase_case_support),
        max_examples_per_phrase=int(args.max_examples_per_phrase),
        wordnet_hints_per_phrase=int(args.wordnet_hints_per_phrase),
    )
    motif_payload = _read_json(Path(args.motif_family_candidates))
    forest_payload = _read_json(Path(args.coordination_forest))
    motion_nodes = load_motion_nodes(motif_payload, forest_payload)
    needed_symbols = {str(symbol) for node in motion_nodes for symbol in (node.get("source_symbols") or []) if str(symbol)}
    symbol_cases, sequence_stats = index_symbol_cases(Path(args.multichannel_sequences), needed_symbols)
    aligned_nodes, alignment_stats = align_name_candidates(
        motion_nodes,
        symbol_cases,
        phrase_vocab,
        case_captions,
        total_cases=max(1, len(case_captions)),
        min_case_overlap=int(args.min_case_overlap),
        top_candidates_per_node=int(args.top_candidates_per_node),
        max_examples_per_candidate=int(args.max_examples_per_candidate),
    )
    wordnet_hint_phrase_count = sum(1 for row in phrase_vocab.values() if row.get("wordnet_hints"))
    summary = {
        "schema_version": "hml3d_caption_wordnet_name_candidates_summary_v0",
        **phrase_stats,
        **sequence_stats,
        **alignment_stats,
        "wordnet_term_count": len(wordnet_terms),
        "phrases_with_wordnet_hints": wordnet_hint_phrase_count,
    }
    return {
        "schema_version": "hml3d_caption_wordnet_name_candidates_v0",
        "runtime_policy": (
            "offline naming sidecar only; mined captions and cached WordNet hints attach names "
            "to motion-derived nodes without creating or restructuring those nodes"
        ),
        "manual_registry_policy": "manual AML registry is not read and is not used as family generation logic",
        "inputs": {
            "source_corpus": str(args.source_corpus),
            "motif_family_candidates": str(args.motif_family_candidates),
            "coordination_forest": str(args.coordination_forest),
            "multichannel_sequences": str(args.multichannel_sequences),
            "wordnet_lexicon": str(args.wordnet_lexicon),
        },
        "config": {
            "max_phrase_tokens": int(args.max_phrase_tokens),
            "min_phrase_case_support": int(args.min_phrase_case_support),
            "min_case_overlap": int(args.min_case_overlap),
            "top_candidates_per_node": int(args.top_candidates_per_node),
            "max_examples_per_phrase": int(args.max_examples_per_phrase),
            "max_examples_per_candidate": int(args.max_examples_per_candidate),
            "wordnet_hints_per_phrase": int(args.wordnet_hints_per_phrase),
            "max_phrase_candidates_output": int(args.max_phrase_candidates_output),
        },
        "summary": summary,
        "caption_phrase_candidates": _serializable_phrase_candidates(
            phrase_vocab,
            limit=int(args.max_phrase_candidates_output),
        ),
        "motion_node_name_candidates": aligned_nodes,
    }


def _escape_md(text: Any) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    lines: list[str] = [
        "# HML3D Caption / WordNet Name Candidates v0",
        "",
        "Offline naming sidecar. Names are evidence attached to existing motion-derived nodes only.",
        "The manual AML registry is not read and is not used as family generation logic.",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "caption_case_count",
        "caption_count",
        "retained_phrase_count",
        "wordnet_term_count",
        "phrases_with_wordnet_hints",
        "motion_node_count",
        "nodes_with_name_candidates",
        "name_candidate_count",
    ]:
        lines.append(f"- `{key}`: `{summary.get(key)}`")
    lines.extend(["", "## Node Overview", ""])
    lines.append("| node | kind | cases | top candidates |")
    lines.append("| --- | --- | ---: | --- |")
    for node in payload.get("motion_node_name_candidates") or []:
        candidates = []
        for candidate in (node.get("name_candidates") or [])[:5]:
            metrics = candidate.get("alignment") or {}
            candidates.append(
                f"{_escape_md(candidate.get('name'))} ({metrics.get('score')}, {metrics.get('overlap_cases')})"
            )
        lines.append(
            "| `{}` | `{}` | {} | {} |".format(
                _escape_md(node.get("node_id")),
                _escape_md(node.get("node_kind")),
                int(node.get("case_support") or 0),
                "; ".join(candidates) or "none",
            )
        )
    lines.extend(["", "## Candidate Details", ""])
    for node in payload.get("motion_node_name_candidates") or []:
        candidates = node.get("name_candidates") or []
        if not candidates:
            continue
        lines.append(f"### {node.get('node_id')}")
        lines.append("")
        lines.append(f"- kind: `{node.get('node_kind')}`")
        lines.append(f"- motion key: `{_escape_md(node.get('motion_key'))}`")
        lines.append(f"- support cases: `{node.get('case_support')}`")
        lines.append(f"- source symbols: `{', '.join(str(item) for item in (node.get('source_symbols') or [])[:20])}`")
        lines.append("")
        for candidate in candidates[:8]:
            metrics = candidate.get("alignment") or {}
            hint_terms = [
                f"{hint.get('term')}:{','.join(str(parent.get('parent_id')) for parent in (hint.get('taxonomy_parent_candidates') or [])[:2])}"
                for hint in (candidate.get("wordnet_hints") or [])[:3]
            ]
            lines.append(
                "- `{}` status=`{}` score=`{}` overlap=`{}` precision=`{}` recall=`{}`".format(
                    _escape_md(candidate.get("name")),
                    candidate.get("status"),
                    metrics.get("score"),
                    metrics.get("overlap_cases"),
                    metrics.get("precision_phrase_to_node"),
                    metrics.get("recall_node_covered"),
                )
            )
            if hint_terms:
                lines.append(f"  - WordNet hints: `{_escape_md('; '.join(hint_terms))}`")
            lines.append("  - Examples:")
            for example in (candidate.get("examples") or [])[:3]:
                captions = example.get("matched_captions") or example.get("caption_texts") or []
                caption = " / ".join(_escape_md(item) for item in captions[:2])
                lines.append(f"    - `{example.get('case_id')}`: {caption}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    name_candidates_path = output_dir / "name_candidates.json"
    report_path = output_dir / "name_candidates.md"
    summary_path = output_dir / "summary.json"
    outputs = {
        "name_candidates": str(name_candidates_path),
        "report": str(report_path),
        "summary": str(summary_path),
    }
    payload["outputs"] = outputs
    payload["summary"]["outputs"] = outputs
    _write_json(name_candidates_path, payload)
    write_markdown_report(report_path, payload)
    _write_json(summary_path, payload["summary"])
    return outputs


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corpus = root / "corpus.jsonl"
        motifs = root / "motifs.json"
        forest = root / "forest.json"
        sequences = root / "sequences.jsonl"
        wordnet = root / "wordnet.json"
        output_dir = root / "out"

        corpus_rows = [
            {
                "case_id": "c001",
                "caption_texts": [
                    "A person kicks forward with the left leg.",
                    "The man does a sharp kick.",
                ],
            },
            {
                "case_id": "c002",
                "caption_texts": [
                    "A person jumps up and lands.",
                    "Someone performs a high jump.",
                ],
            },
            {
                "case_id": "c003",
                "caption_texts": [
                    "A woman waves both hands.",
                    "The person does a friendly wave.",
                ],
            },
        ]
        corpus.write_text("\n".join(json.dumps(row) for row in corpus_rows) + "\n", encoding="utf-8")
        motifs.write_text(
            json.dumps(
                {
                    "families": [
                        {
                            "family_id": "motif_family_kick",
                            "motion_family_key": "left_leg_kick",
                            "status": "candidate_family",
                            "motif_count": 1,
                            "support_cases_sum": 1,
                            "occurrences_sum": 1,
                            "motion_definition": {"required_geometry_clusters": ["LEFT_LEG_ACTION/LL_KICK_FORWARD"]},
                            "source_motifs": [{"motif_id": "<CHM_KICK>", "support_cases": 1}],
                        },
                        {
                            "family_id": "motif_family_jump",
                            "motion_family_key": "vertical_jump",
                            "status": "candidate_family",
                            "motif_count": 1,
                            "support_cases_sum": 1,
                            "occurrences_sum": 1,
                            "motion_definition": {"required_geometry_clusters": ["WHOLE_BODY_VERTICAL/WB_VERT_UP"]},
                            "source_motifs": [{"motif_id": "<CHM_JUMP>", "support_cases": 1}],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        forest.write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "node_id": "coord_family_wave",
                            "node_kind": "structural_coordination_family",
                            "status": "review_structural_coordination_candidate",
                            "family_key": "structural:arms_wave",
                            "support": {"candidate_count": 1, "support_cases_sum": 1},
                            "motion_definition": {"required_channels": ["left_arm", "right_arm"]},
                        },
                        {
                            "node_id": "coord_leaf_wave",
                            "node_kind": "coordination_motif_leaf",
                            "status": "review_structural_coordination_candidate",
                            "source_motif_id": "<COM_WAVE>",
                            "support": {"support_cases": 1},
                            "motion_definition": {"required_channels": ["left_arm", "right_arm"]},
                        },
                    ],
                    "edges": [
                        {
                            "parent_node_id": "coord_family_wave",
                            "child_node_id": "coord_leaf_wave",
                            "edge_type": "coordination_family_member",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        seq_rows = [
            {"case_id": "c001", "sequence_id": "c001::left_leg", "tokens": [{"symbol": "<CHM_KICK>"}]},
            {"case_id": "c002", "sequence_id": "c002::vertical", "tokens": [{"symbol": "<CHM_JUMP>"}]},
            {"case_id": "c003", "sequence_id": "c003::coactivation", "tokens": [{"symbol": "<COM_WAVE>"}]},
        ]
        sequences.write_text("\n".join(json.dumps(row) for row in seq_rows) + "\n", encoding="utf-8")
        wordnet.write_text(
            json.dumps(
                {
                    "terms": [
                        {
                            "term": "kick",
                            "pos": ["verb"],
                            "source": ["wordnet"],
                            "is_mapped": True,
                            "taxonomy_parent_candidates": [{"parent_id": "LOWER_LIMB_ACTION", "confidence": 0.9}],
                            "candidate_family_ids": [],
                            "wordnet": {"verb_synset_count": 1, "noun_synset_count": 0, "lexnames": {"verb.contact": 1}},
                        },
                        {
                            "term": "jump",
                            "pos": ["verb"],
                            "source": ["wordnet"],
                            "is_mapped": True,
                            "taxonomy_parent_candidates": [{"parent_id": "VERTICAL_IMPULSE", "confidence": 0.9}],
                            "candidate_family_ids": [],
                            "wordnet": {"verb_synset_count": 1, "noun_synset_count": 0, "lexnames": {"verb.motion": 1}},
                        },
                        {
                            "term": "wave",
                            "pos": ["verb"],
                            "source": ["wordnet"],
                            "is_mapped": True,
                            "taxonomy_parent_candidates": [{"parent_id": "UPPER_LIMB_GESTURE", "confidence": 0.9}],
                            "candidate_family_ids": [],
                            "wordnet": {"verb_synset_count": 1, "noun_synset_count": 0, "lexnames": {"verb.motion": 1}},
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        args = argparse.Namespace(
            source_corpus=str(corpus),
            motif_family_candidates=str(motifs),
            coordination_forest=str(forest),
            multichannel_sequences=str(sequences),
            wordnet_lexicon=str(wordnet),
            output_dir=str(output_dir),
            max_phrase_tokens=4,
            min_phrase_case_support=1,
            min_case_overlap=1,
            top_candidates_per_node=5,
            max_examples_per_phrase=3,
            max_examples_per_candidate=2,
            wordnet_hints_per_phrase=4,
            max_phrase_candidates_output=100,
        )
        payload = build_name_candidates(args)
        outputs = write_outputs(output_dir, payload)
        assert Path(outputs["name_candidates"]).exists()
        assert Path(outputs["report"]).exists()
        assert Path(outputs["summary"]).exists()
        by_node = {str(node["node_id"]): node for node in payload["motion_node_name_candidates"]}
        kick_names = [item["name"] for item in by_node["motif_family_kick"]["name_candidates"]]
        wave_candidates = by_node["coord_leaf_wave"]["name_candidates"]
        assert "kick" in kick_names or "kicks" in kick_names
        assert any(item["name"] == "wave" for item in wave_candidates)
        assert any((hint.get("term") == "wave") for item in wave_candidates for hint in item.get("wordnet_hints", []))
        assert wave_candidates[0]["examples"][0]["caption_texts"]
    print(json.dumps({"ok": True}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine HumanML3D caption/WordNet name candidates for existing v0 AML motion nodes."
    )
    parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
    parser.add_argument("--motif-family-candidates", default=str(DEFAULT_MOTIF_FAMILY_CANDIDATES))
    parser.add_argument("--coordination-forest", default=str(DEFAULT_COORDINATION_FOREST))
    parser.add_argument("--multichannel-sequences", default=str(DEFAULT_MULTICHANNEL_SEQUENCES))
    parser.add_argument("--wordnet-lexicon", default=str(DEFAULT_WORDNET_LEXICON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-phrase-tokens", type=int, default=4)
    parser.add_argument("--min-phrase-case-support", type=int, default=8)
    parser.add_argument("--min-case-overlap", type=int, default=3)
    parser.add_argument("--top-candidates-per-node", type=int, default=12)
    parser.add_argument("--max-examples-per-phrase", type=int, default=5)
    parser.add_argument("--max-examples-per-candidate", type=int, default=4)
    parser.add_argument("--wordnet-hints-per-phrase", type=int, default=5)
    parser.add_argument("--max-phrase-candidates-output", type=int, default=2000)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    payload = build_name_candidates(args)
    outputs = write_outputs(Path(args.output_dir), payload)
    print(outputs["summary"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
