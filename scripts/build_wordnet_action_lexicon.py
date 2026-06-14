from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "aml_lexicon" / "wordnet_action_terms_v1.json"
DEFAULT_TAXONOMY = ROOT_DIR / "pseudoedit3d" / "edit" / "aml_family_taxonomy.json"
DEFAULT_CONFIG = ROOT_DIR / "pseudoedit3d" / "edit" / "aml_wordnet_lexicon_config.json"

ACTION_NOUN_LEXNAMES: set[str] = set()
STOP_TERMS: set[str] = set()
PARENT_TERM_SEEDS: dict[str, set[str]] = {}
PARENT_REGEX_RULES: dict[str, list[tuple[str, str]]] = {}
FAMILY_TERM_SEEDS: dict[str, str] = {}


def _load_builder_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported WordNet lexicon config format: {path}")
    return payload


def _install_builder_config(config: dict[str, Any]) -> None:
    global ACTION_NOUN_LEXNAMES, STOP_TERMS, PARENT_TERM_SEEDS, PARENT_REGEX_RULES, FAMILY_TERM_SEEDS
    ACTION_NOUN_LEXNAMES = {str(item) for item in config.get("action_noun_lexnames") or []}
    STOP_TERMS = {str(item) for item in config.get("stop_terms") or []}
    PARENT_TERM_SEEDS = {
        str(parent_id): {str(item) for item in terms}
        for parent_id, terms in (config.get("parent_term_seeds") or {}).items()
    }
    PARENT_REGEX_RULES = {
        str(parent_id): [
            (str(rule.get("name", "")), str(rule.get("pattern", "")))
            for rule in rules
            if isinstance(rule, dict)
        ]
        for parent_id, rules in (config.get("parent_regex_rules") or {}).items()
    }
    FAMILY_TERM_SEEDS = {
        str(term): str(family_id)
        for term, family_id in (config.get("family_term_seeds") or {}).items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a cached WordNet-derived AML action lexicon. This is an offline "
            "builder; AML extraction should read the output JSON and should not query WordNet."
        )
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--download-wordnet", action="store_true", help="Download the English WordNet corpus via nltk before extraction.")
    parser.add_argument("--download-omw", action="store_true", help="Optionally download omw-1.4. Not needed for the English AML lexicon.")
    parser.add_argument("--include-nouns", action="store_true", default=False, help="Also include activity-like noun synsets. Verbs are exported by default.")
    parser.add_argument("--no-include-nouns", dest="include_nouns", action="store_false")
    parser.add_argument("--include-curated-seeds", action="store_true", default=True, help="Include AML seed phrases that WordNet may not contain, such as jumping jack.")
    parser.add_argument("--no-include-curated-seeds", dest="include_curated_seeds", action="store_false")
    parser.add_argument("--max-sample-synsets", type=int, default=5)
    return parser.parse_args()


def _load_wordnet(*, download: bool, download_omw: bool) -> Any:
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "nltk is not installed in this Python environment. Install/use an env with nltk, "
            "then run this one-shot builder with --download-wordnet if the corpus is missing. "
            "AML runtime does not depend on nltk."
        ) from exc

    if download:
        nltk.download("wordnet")
    if download_omw:
        nltk.download("omw-1.4")

    try:
        wn.synsets("walk", pos=wn.VERB)
    except LookupError as exc:
        raise SystemExit(
            "WordNet corpus is not available. Re-run this builder with --download-wordnet once, "
            "then commit or reuse the exported JSON artifact."
        ) from exc
    return wn


def _normalize_term(term: str) -> str | None:
    normalized = term.replace("_", " ").replace("-", " ").lower()
    normalized = re.sub(r"[^a-z0-9' ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized or normalized in STOP_TERMS:
        return None
    if len(normalized) < 3 and normalized not in {"go"}:
        return None
    return normalized


def _collect_terms(wn: Any, *, include_nouns: bool, max_sample_synsets: int) -> dict[str, dict[str, Any]]:
    terms: dict[str, dict[str, Any]] = {}

    def add_synset(pos_label: str, syn: Any) -> None:
        lexname = str(syn.lexname())
        if pos_label == "noun" and lexname not in ACTION_NOUN_LEXNAMES:
            return
        for lemma in syn.lemmas():
            term = _normalize_term(lemma.name())
            if term is None:
                continue
            entry = terms.setdefault(
                term,
                {
                    "term": term,
                    "pos": set(),
                    "wordnet": {
                        "verb_synset_count": 0,
                        "noun_synset_count": 0,
                        "lexnames": Counter(),
                        "sample_synsets": [],
                    },
                    "source": set(),
                },
            )
            entry["pos"].add(pos_label)
            entry["source"].add("wordnet")
            key = f"{pos_label}_synset_count"
            entry["wordnet"][key] += 1
            entry["wordnet"]["lexnames"][lexname] += 1
            if len(entry["wordnet"]["sample_synsets"]) < max_sample_synsets:
                entry["wordnet"]["sample_synsets"].append(
                    {
                        "name": syn.name(),
                        "pos": pos_label,
                        "lexname": lexname,
                        "definition": syn.definition(),
                    }
                )

    for synset in wn.all_synsets(wn.VERB):
        add_synset("verb", synset)
    if include_nouns:
        for synset in wn.all_synsets(wn.NOUN):
            add_synset("noun", synset)

    return terms


def _curated_seed_terms() -> set[str]:
    terms: set[str] = set()
    for seeds in PARENT_TERM_SEEDS.values():
        terms.update(seeds)
    terms.update(FAMILY_TERM_SEEDS)
    return {term for term in terms if " " in term}


def _add_curated_seed_terms(terms: dict[str, dict[str, Any]]) -> None:
    for raw_term in sorted(_curated_seed_terms()):
        term = _normalize_term(raw_term)
        if term is None:
            continue
        entry = terms.setdefault(
            term,
            {
                "term": term,
                "pos": set(),
                "wordnet": {
                    "verb_synset_count": 0,
                    "noun_synset_count": 0,
                    "lexnames": Counter(),
                    "sample_synsets": [],
                },
                "source": set(),
            },
        )
        if not entry["pos"]:
            entry["pos"].add("curated_phrase")
        entry["source"].add("curated_seed")


def _candidate_boundary(taxonomy: dict[str, Any], parent_id: str) -> str:
    parent = (taxonomy.get("parents") or {}).get(parent_id) or {}
    recoverability = str(parent.get("recoverability") or "")
    if parent_id == "ACTIVITY_INTENT_PROXY":
        return "object_or_intent_ambiguous"
    if recoverability == "geometry_recoverable":
        return "motion_geometry"
    if recoverability == "geometry_candidate":
        return "geometry_candidate"
    return "unknown_family"


def _taxonomy_parent_candidates(term: str, taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for parent_id, seeds in PARENT_TERM_SEEDS.items():
        if term in seeds:
            candidates[parent_id] = {
                "parent_id": parent_id,
                "confidence": 0.9,
                "rule": f"seed:{term}",
                "ambiguity_boundary": _candidate_boundary(taxonomy, parent_id),
            }
    for parent_id, rules in PARENT_REGEX_RULES.items():
        for rule_name, pattern in rules:
            if re.search(pattern, term):
                current = candidates.get(parent_id)
                if current is None or float(current["confidence"]) < 0.65:
                    candidates[parent_id] = {
                        "parent_id": parent_id,
                        "confidence": 0.65,
                        "rule": f"regex:{rule_name}",
                        "ambiguity_boundary": _candidate_boundary(taxonomy, parent_id),
                    }
    return sorted(candidates.values(), key=lambda item: (-float(item["confidence"]), str(item["parent_id"])))


def _taxonomy_parents_for_family(taxonomy: dict[str, Any], family_id: str) -> list[str]:
    parent_ids: list[str] = []
    for parent_id, parent in (taxonomy.get("parents") or {}).items():
        if str(family_id) in {str(item) for item in parent.get("children") or []}:
            parent_ids.append(str(parent_id))
    for parent_id in ((taxonomy.get("family_overrides") or {}).get(str(family_id)) or {}).get("secondary_parents") or []:
        if str(parent_id) not in parent_ids:
            parent_ids.append(str(parent_id))
    return parent_ids


def _family_candidates(term: str, taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    direct = FAMILY_TERM_SEEDS.get(term)
    if not direct:
        return []
    return [
        {
            "family_id": direct,
            "confidence": 0.8,
            "rule": f"seed:{term}",
            "taxonomy_parent_ids": _taxonomy_parents_for_family(taxonomy, direct),
        }
    ]


def _finalize_entries(terms: dict[str, dict[str, Any]], taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for term, entry in terms.items():
        wordnet = entry["wordnet"]
        candidates = _taxonomy_parent_candidates(term, taxonomy)
        family_candidates = _family_candidates(term, taxonomy)
        lexnames = wordnet["lexnames"]
        entries.append(
            {
                "term": term,
                "pos": sorted(entry["pos"]),
                "source": sorted(entry.get("source") or ["wordnet"]),
                "wordnet": {
                    "verb_synset_count": int(wordnet["verb_synset_count"]),
                    "noun_synset_count": int(wordnet["noun_synset_count"]),
                    "lexnames": dict(sorted((str(k), int(v)) for k, v in lexnames.items())),
                    "sample_synsets": wordnet["sample_synsets"],
                },
                "taxonomy_parent_candidates": candidates,
                "candidate_family_ids": family_candidates,
                "is_mapped": bool(candidates or family_candidates),
            }
        )
    return sorted(
        entries,
        key=lambda item: (
            0 if item["is_mapped"] else 1,
            -max([float(c["confidence"]) for c in item["taxonomy_parent_candidates"]] or [0.0]),
            str(item["term"]),
        ),
    )


def _summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    pos_counts: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for entry in entries:
        pos_counts.update(entry["pos"])
        for candidate in entry["taxonomy_parent_candidates"]:
            parent_counts[str(candidate["parent_id"])] += 1
            boundary_counts[str(candidate["ambiguity_boundary"])] += 1
        for candidate in entry["candidate_family_ids"]:
            family_counts[str(candidate["family_id"])] += 1
    return {
        "num_terms": len(entries),
        "num_mapped_terms": sum(1 for entry in entries if entry["is_mapped"]),
        "num_unmapped_terms": sum(1 for entry in entries if not entry["is_mapped"]),
        "pos_counts": dict(sorted(pos_counts.items())),
        "taxonomy_parent_counts": parent_counts.most_common(),
        "ambiguity_boundary_counts": boundary_counts.most_common(),
        "candidate_family_counts": family_counts.most_common(),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = _load_builder_config(config_path)
    _install_builder_config(config)
    taxonomy_path = Path(args.taxonomy)
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy["__path__"] = str(taxonomy_path)
    wn = _load_wordnet(download=bool(args.download_wordnet), download_omw=bool(args.download_omw))
    terms = _collect_terms(wn, include_nouns=bool(args.include_nouns), max_sample_synsets=max(1, int(args.max_sample_synsets)))
    if args.include_curated_seeds:
        _add_curated_seed_terms(terms)
    entries = _finalize_entries(terms, taxonomy)
    output = Path(args.output)
    artifact = {
        "schema_version": "wordnet_action_terms_v1",
        "runtime_policy": "offline_builder_only; AML extraction must read this JSON and must not query WordNet",
        "source": {
            "wordnet": "nltk.corpus.wordnet",
            "taxonomy": str(taxonomy_path),
            "builder": str(Path(__file__).relative_to(ROOT_DIR)),
            "config": str(config_path),
            "include_nouns": bool(args.include_nouns),
            "include_curated_seeds": bool(args.include_curated_seeds),
            "download_omw": bool(args.download_omw),
            "action_noun_lexnames": sorted(ACTION_NOUN_LEXNAMES),
        },
        "summary": _summarize(entries),
        "terms": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], ensure_ascii=True, indent=2))
    print(f"saved_wordnet_action_lexicon={output}")


if __name__ == "__main__":
    main()
