# Problem 2 — Job Title → Persona / Level / Function (rules → embeddings → LLM fallback)

Reference solution for the Demandbase "stamping service": given a tenant's personas and a stream of raw titles, assign
persona(s), level and function — **cheaply, deterministically, multilingual**. Small enough to re-type and explain
(~400 lines, 71 tests, <1 s, no model downloads: the embedder and the LLM are pluggable interfaces with offline stand-ins).

```
persona/
  config.py     Persona, PersonaConfig (frozen, validated), LEVELS / FUNCTIONS vocabularies
  normalize.py  normalize_title (NFKD accent strip, emoji/symbol drop, abbreviations) · looks_english
  rules.py      detect_level · detect_function · keyword_hit (word-boundary / prefix matching)
  embed.py      Embedder Protocol · HashingEmbedder (crc32, offline) · PersonaIndex (anchors, max-cosine)
  llm.py        LLMClient Protocol · validate_llm_result (never trust model output) · ScriptedLLM (tests/demo)
  classify.py   classify_titles: dedupe → translate → keyword → embed → LLM → vetoes
  cli.py        python -m persona.cli --contacts sample_data/contacts.csv --personas sample_data/personas.json
tests/          normalize/rules (traps) · classify (golden sample, budget, dedupe, vetoes, bad LLM output) · embed
```

```bash
pip install -r requirements.txt && python -m pytest -q      # 71 passed
python -m persona.cli --contacts sample_data/contacts.csv --personas sample_data/personas.json
```

Mock interviews for this problem: [INTERVIEW_TRAJECTORIES.md](INTERVIEW_TRAJECTORIES.md)

## The 60-second design pitch

1. **Dedupe first.** Normalise every title, classify each *unique* title once, map back. Cost scales with unique titles (millions of contacts → thousands of titles).
2. **Cascade, cheapest first**, each stage records `method` + `confidence` + `reason`:
   `empty_input` → *(translate if not English)* → `keyword` → `embed` → `llm` (only what is still undecided).
3. **Vetoes apply to every stage's output**, including the LLM: `exclude_keywords` ⇒ `exclude_rule` (conf 1.0), persona `levels` ⇒ `level_filter`. A model cannot stamp a persona the tenant excluded.
4. **Level and function are rule-based** (taxonomy + word-boundary matching) and independent of persona matching; the LLM only fills them in when the rules say Unknown.
5. **LLM is bounded and validated**: batched, hard budget (`max_llm_titles`, rest = `budget_exceeded`), strict JSON validation (hallucinated persona ids dropped, bad confidence ⇒ `llm_invalid`), answers below `min_confidence` are **not stamped**.

## Decisions worth defending

| Decision | Why | Trade-off / what I'd change |
|---|---|---|
| Keyword before embedding | Explainable, free, 0.95 confidence; most real titles hit | Keyword lists need curation → mine them from LLM-labelled titles |
| Keywords ≤3 chars match whole tokens; longer match word *prefix* | `cmo`/`it`/`hr` must not match inside words; `demand gen` should match `demand generation` | Prefix misses `finance`→`financial`; the embedder's 5-char stem covers it |
| Anchors = name + each keyword, score = max cosine | A short keyword anchor isn't diluted by a long description | Real model: add a description anchor, fit `embed_accept` on labelled data (cosine scales differ per model) |
| Translate non-English **first**, then run the same English rules | One taxonomy to maintain; level/function come from the translation (`Directeur Financier` → `Chief Financial Officer` ⇒ C-Level, not "Director") | `looks_english` is a vocabulary heuristic → swap for fastText lid-176; add a cached glossary of frequent foreign titles to skip the LLM |
| Tied functions ⇒ Unknown; unknown level is *neutral* (−0.10 conf, not a rejection) | Precision first, but missing data shouldn't zero a good match | Tenants who want recall can lower `unknown_level_penalty` |
| `crc32` hashing, not builtin `hash()` | `hash(str)` is salted per process ⇒ non-reproducible embeddings | — |
| Low-confidence LLM answer ⇒ personas `[]` with the LLM's function/level kept | Stamping a wrong persona on 1P/3P contacts is worse than leaving it empty | Route those to a review queue / active learning |

## Where AI-generated code typically goes wrong (covered by tests)

| Trap | What the code/tests do instead |
|---|---|
| `"cmo" in title` / `"it" in title` substring matching | word-boundary patterns; `Chief Mission Officer` ≠ CMO, `white collar` ≠ IT |
| `"president" in title` ⇒ C-Level makes *Vice President* C-Level | VP checked before C-Level |
| `"ceo" in title` ⇒ *Assistant to the CEO* is C-Level | `assistant to` ⇒ Staff |
| Python `hash()` for feature hashing | `zlib.crc32` (test: deterministic embeddings) |
| NaN / None / emoji-only / whitespace titles sent to the LLM (and billed) | `empty_input` before any model call; test asserts zero LLM calls |
| Same title classified once per contact | dedupe; test: 100 contacts → 1 LLM item |
| Trusting LLM JSON | `validate_llm_result`; hallucinated ids dropped, `"high"`/NaN/bool confidence rejected |
| Unbounded LLM spend | hard budget, deterministic who-gets-budget (sorted), `budget_exceeded` method |
| LLM overriding tenant exclusions | veto runs after the LLM too (test: "Assistant to the CFO office") |
| `idx`/dict-order nondeterminism | sorted ids, shuffle test with `assert_frame_equal` |

## Sample output vs the brief

| contact | result here | brief | note |
|---|---|---|---|
| 1 Sr. Director, Demand Generation | Director / Marketing, `keyword` 0.95 | same | |
| 2 Directeur Financier | C-Level / Finance, `llm_translate+embed` | 0.88 | confidence differs: our hashing embedder is a lexical stand-in; a real model gives higher cosine |
| 3 Marketing Intern | `[]`, Intern, `exclude_rule` 1.00 | same | |
| 4 Head of InfoSec & Risk | Director / IT-Security, `keyword` 0.95 | `embed` 0.83 | `infosec` is in the include list, so a keyword hit is the right, cheaper answer |
| 5 `"  "` | `[]`, `empty_input` 0.00 | same | |
| 6 Chief Happiness Officer 🎉 | `[]`, C-Level, HR, `llm` 0.40 → below threshold | same | |

## Stretch talking points

* **Scale**: unique-title dedupe → embed once, cache by `(tenant, title)`; ANN index (FAISS/pgvector) when personas × anchors is large; Spark `mapInPandas` for the batch path.
* **Top-25 similar titles per persona** (the Buying Group agent): embed the de-duped title universe once, cosine top-k against persona anchors — same `PersonaIndex`, `argpartition` for k.
* **Evaluation**: labelled titles → precision/recall per `method`, calibrate `embed_accept` and `min_confidence` from the PR curve (precision-weighted), track LLM-vs-rule disagreement as a drift signal.
* **Multilingual**: translate-then-classify keeps one taxonomy; for hot languages a multilingual encoder (e5/LaBSE) as the embedder removes most LLM calls.
* **Cost control**: budget cap, batch size, response cache keyed by normalised title, small model first and escalate on low confidence.
