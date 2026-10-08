"""Cascade: empty -> (translate non-English) -> keyword -> embedding -> LLM classify, with exclude/level vetoes
applied to EVERY stage's output. Titles are de-duplicated first, so cost scales with unique titles, not contacts."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import pandas as pd

from persona.config import Persona, PersonaConfig
from persona.embed import Embedder, HashingEmbedder, PersonaIndex
from persona.llm import LLMClient, validate_llm_result
from persona.normalize import looks_english, normalize_title
from persona.rules import detect_function, detect_level, keyword_hit

OUTPUT_COLUMNS = ["contact_id", "title", "personas", "level", "function", "method", "confidence", "reason"]


@dataclass
class _Work:
    raw: str
    norm: str
    text: str = ""            # what the rules/embedder see (translation if we translated)
    translated: bool = False
    level: str = "Unknown"
    function: str = "Unknown"
    personas: list[str] = field(default_factory=list)
    method: str = ""
    confidence: float = 0.0
    reason: str = ""
    needs_llm: bool = False


def _veto(work: _Work, by_id: dict[str, Persona]) -> list[str]:
    """Apply exclude_keywords then the level constraint. Sets method/reason when it empties a non-empty set."""
    candidates = list(work.personas)
    kept = [pid for pid in candidates if not any(keyword_hit(work.text, k) for k in by_id[pid].exclude_keywords)]
    if candidates and not kept:
        work.method, work.confidence = "exclude_rule", 1.0
        work.reason = "matched persona(s) %s but an exclude keyword applies" % ",".join(candidates)
        return []
    by_level = [pid for pid in kept if not by_id[pid].levels or work.level == "Unknown" or work.level in by_id[pid].levels]
    if kept and not by_level:
        work.method, work.confidence = "level_filter", 0.9
        work.reason = f"matched {','.join(kept)} but level {work.level} is not in the persona's levels"
        return []
    return by_level


def _match_local(work: _Work, index: PersonaIndex, scores_row, cfg: PersonaConfig) -> None:
    """Keyword first (cheap, explainable), then embedding similarity."""
    hits = [p.persona_id for p in index.personas if any(keyword_hit(work.text, k) for k in p.include_keywords)]
    if hits:
        work.personas, work.method, work.confidence = hits, "keyword", cfg.keyword_confidence
        work.reason = "include keyword match"
        return
    best = float(scores_row.max()) if len(scores_row) else 0.0
    if best >= cfg.embed_accept:
        work.personas = [p.persona_id for p, s in zip(index.personas, scores_row)
                         if s >= cfg.embed_accept and s >= best - cfg.embed_multi_gap]
        work.method, work.confidence = "embed", round(best, 2)
        work.reason = f"embedding similarity {best:.2f} >= {cfg.embed_accept}"


def classify_titles(
    contacts: pd.DataFrame,
    personas: Sequence[Persona],
    cfg: PersonaConfig | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> pd.DataFrame:
    """One output row per input row, input order preserved. Never raises on bad titles."""
    cfg = cfg or PersonaConfig()
    for col in ("contact_id", "title"):
        if col not in contacts.columns:
            raise ValueError(f"contacts is missing required column {col!r}")
    ids = [p.persona_id for p in personas]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate persona_id in personas")
    by_id = {p.persona_id: p for p in personas}
    index = PersonaIndex(personas, embedder or HashingEmbedder())
    spec = [{"persona_id": p.persona_id, "name": p.name, "include_keywords": list(p.include_keywords)} for p in personas]

    # 1. de-duplicate on the normalised title
    work: dict[str, _Work] = {}
    for raw in contacts["title"]:
        norm = normalize_title(raw)
        if norm not in work:
            work[norm] = _Work(raw=raw.strip() if isinstance(raw, str) else "", norm=norm, text=norm)
    budget = cfg.max_llm_titles

    def take(items: list[_Work]) -> tuple[list[_Work], list[_Work]]:
        nonlocal budget
        items = sorted(items, key=lambda w: w.norm)  # deterministic who-gets-budget
        ok, over = items[:budget], items[budget:]
        budget -= len(ok)
        return ok, over

    # 2. empty input never reaches any model
    live = [w for w in work.values() if w.norm]
    for w in work.values():
        if not w.norm:
            w.method, w.reason = "empty_input", "title is empty or has no usable characters"

    # 3. translate non-English titles in batches
    if llm is not None:
        todo, _ = take([w for w in live if not looks_english(w.norm)])
        for i in range(0, len(todo), cfg.llm_batch_size):
            batch = todo[i:i + cfg.llm_batch_size]
            answers = llm.translate([w.raw for w in batch])
            for w in batch:
                english = normalize_title(answers.get(w.raw))
                if english:
                    w.text, w.translated = english, True

    # 4. rules + embedding (one matrix multiply for all unique titles)
    scores = index.scores([w.text for w in live])
    for w, row in zip(live, scores):
        w.level, w.function = detect_level(w.text), detect_function(w.text)
        _match_local(w, index, row, cfg)
        w.personas = _veto(w, by_id)
        if w.translated and w.method in {"keyword", "embed"}:
            w.method = f"llm_translate+{w.method}"
        if w.personas and w.level == "Unknown" and any(by_id[p].levels for p in w.personas):
            w.confidence = round(max(0.0, w.confidence - cfg.unknown_level_penalty), 2)
        if not w.method:
            w.needs_llm = True

    # 5. LLM classify only what rules + embeddings could not decide
    pending = [w for w in live if w.needs_llm]
    if llm is None:
        for w in pending:
            w.method, w.reason = "no_match", "no keyword/embedding match and no LLM configured"
    else:
        ok, over = take(pending)
        for w in over:
            w.method, w.reason = "budget_exceeded", "LLM budget exhausted; left unclassified"
        valid = set(ids)
        for i in range(0, len(ok), cfg.llm_batch_size):
            batch = ok[i:i + cfg.llm_batch_size]
            answers = llm.classify([w.text for w in batch], spec)
            for w in batch:
                res = validate_llm_result(answers.get(w.text), valid)
                if res is None:
                    w.method, w.reason = "llm_invalid", "LLM returned no usable answer"
                    continue
                w.method, w.confidence = "llm", res["confidence"]
                w.level = w.level if w.level != "Unknown" else res["level"]
                w.function = w.function if w.function != "Unknown" else res["function"]
                w.personas = res["personas"]
                w.reason = "LLM classification"
                w.personas = _veto(w, by_id)
                if w.personas and w.confidence < cfg.min_confidence:
                    w.reason = f"LLM confidence {w.confidence:.2f} < {cfg.min_confidence}; not stamped"
                    w.personas = []
                elif not w.personas and w.confidence < cfg.min_confidence and w.method == "llm":
                    w.reason = f"LLM confidence {w.confidence:.2f} < {cfg.min_confidence}; below threshold"

    rows = []
    for cid, raw in zip(contacts["contact_id"], contacts["title"]):
        w = work[normalize_title(raw)]
        rows.append({"contact_id": cid, "title": raw, "personas": list(w.personas), "level": w.level,
                     "function": w.function, "method": w.method, "confidence": round(w.confidence, 2),
                     "reason": w.reason})
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
