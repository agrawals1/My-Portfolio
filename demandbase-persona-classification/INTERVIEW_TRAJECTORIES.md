# Mock interview trajectories: Problem 2 (Job Title → Persona / Level / Function)

Five realistic ways a 60–90 minute round on this problem can go. **I** = interviewer, **C** = candidate (you).
`[AI]` lines are what you type into the coding assistant, and `[screen]` lines are what happens on screen.
Every number and behaviour quoted here matches the code in this folder, so you can say it with confidence.

| # | Situation | What it trains |
|---|---|---|
| 1 | Standard, cooperative interviewer | Clarify, pitch the cascade, tests first, build, run the sample, wrap up |
| 2 | Cost and scale interviewer ("why not just call the LLM?") | Back-of-envelope cost, dedupe, caching, top-25 titles, Spark, budget, deployment |
| 3 | The AI (and you) get things wrong | Catching substring matching, `hash()`, threshold set on a knife edge, a wrong test |
| 4 | ML-depth interviewer | Evaluation, threshold calibration, multilingual, LLM output and confidence, distillation |
| 5 | Requirements change and time runs short | Config vs code changes, pushing back on scope, cutting the LLM stage cleanly |

At the end there's a short **answer bank** for questions that come up in every variant.

---

## Trajectory 1: the standard run

**00:00**

**I:** This is the persona stamping problem. We get a tenant's persona definitions and a big stream of raw job titles. For each contact we want personas, level and function. Read it and ask whatever you need.

**C:** *(reads for ~2 minutes)* A few questions first.
1. Can a contact have more than one persona? The sample output uses lists.
2. Which is worse: stamping the wrong persona, or leaving a contact unstamped?
3. Are levels and functions a fixed global taxonomy, with only personas defined per tenant?
4. Volume and latency: is this a nightly batch over all contacts, or online when a contact is created?
5. Languages, and is there an LLM budget I should design around?
6. `exclude_keywords`: is that a veto for that persona only, or for all personas?

**I:** Multi-label is allowed. Wrong is worse, because sales reps stop trusting the data if an intern shows up as a "Marketing Leader". Levels and functions are global. It's mostly batch, tens of millions of contacts across tenants, with an online path later. Mostly English, then French, German, Spanish and some Japanese. Assume the LLM is expensive. Exclude is per persona.

**C:** Then I'd build a cascade. Cheap deterministic steps run first, and the LLM only sees what's left. Before any of that I'd de-duplicate titles, since the same title repeats across many contacts.

**I:** How much do they repeat?

**C:** I don't know your data. CRM titles usually have a heavy head ("Software Engineer", "Account Executive") and a long tail. I'd measure unique/total on one real tenant first. Either way, classifying unique titles is never worse than classifying rows.

**00:05: the pitch**

**C:** Here's the plan in one minute.
1. **Normalise** the title: strip accents, drop emoji and symbols, casefold, expand abbreviations like `Sr.` → `senior`. Empty or junk input becomes `empty_input` and never reaches a model.
2. **De-duplicate** on the normalised title.
3. **Translate** non-English titles with the LLM, then run the same English pipeline on the translation.
4. **Level and function** come from rules, independent of personas.
5. **Persona matching** tries keyword first, then embedding similarity.
6. **Vetoes**: exclude keywords and the persona's allowed levels apply to every stage's output, including the LLM's.
7. **LLM classify** only what's still undecided, in batches, with a hard budget and strict validation of its JSON.

Every output row gets `method`, `confidence` and a human-readable `reason`.

**I:** Why keep level and function separate from persona?

**C:** Level and function are facts about the title. A persona is a tenant's definition layered on top. Keeping them separate means the level rules are written and tested once and every tenant benefits. A persona's level constraint then just reads the detected level.

**00:08: interfaces first**

**C:** I'll write the signatures myself, then use the AI to fill in bodies.

```
[screen] persona/config.py     Persona(persona_id, name, include_keywords, exclude_keywords, levels)  frozen + validated
                               PersonaConfig(embed_accept, min_confidence, keyword_confidence, max_llm_titles, ...)
         persona/normalize.py  normalize_title(value) -> str ; looks_english(norm) -> bool
         persona/rules.py      detect_level(norm) -> str ; detect_function(norm) -> str ; keyword_hit(norm, kw) -> bool
         persona/embed.py      class Embedder(Protocol): embed(texts) -> ndarray
         persona/llm.py        class LLMClient(Protocol): translate(titles) ; classify(titles, personas)
         persona/classify.py   classify_titles(contacts, personas, cfg, embedder, llm) -> DataFrame
```

**I:** Why Protocols for the embedder and the LLM?

**C:** I don't want this session to depend on downloading model weights or having an API key. Tests use a small hashing embedder and a scripted LLM. In production you pass a real model through the same interface without touching the classifier.

**00:11: tests first for the rules**

**C:** Level detection has a few classic traps, so I'll write the table first.

```
[screen] tests/test_normalize_and_rules.py
  ("vice president of sales", "VP"),        # must not become C-Level via "president"
  ("assistant to the ceo",   "Staff"),      # contains "ceo"
  ("marketing intern",       "Intern"),
  ("head of infosec",        "Director"),
  ("cmo",                    "C-Level"),
  ("", "Unknown"), ("wizard", "Unknown"),
```

```
[AI] Implement detect_level(norm: str) -> str in persona/rules.py. Input is already normalised
     (lowercase, abbreviations expanded). Return one of LEVELS. Check in this order: intern words,
     "assistant to", vice/vp, C-level (chief, c?o tokens, founder, owner, president), director/head,
     manager/lead/supervisor, staff words, else Unknown. Work on tokens, not substrings.
```

```
[screen] $ pytest tests/test_normalize_and_rules.py -q   → all pass
```

**C:** I'll read it before moving on. *(scrolls)* It checks tokens with set intersections and the order matches what I asked for. Good.

**I:** Why is "Head of" a Director?

**C:** It's a convention, and it's ambiguous. "Head of Marketing" at a 20-person startup may be the most senior marketer there is. The brief writes it as `Director*`, which I read as "inferred". If it mattered, I'd add a flag saying the level was inferred rather than explicit.

**00:20: keyword matching**

**C:** Keywords are where naive code goes wrong. My rule: keywords of three characters or fewer must match a whole token, so `cmo`, `it` and `hr` don't match inside other words. Longer keywords match as a word *prefix*, so `demand gen` matches "demand generation".

**I:** Why not plain substring?

**C:** `"marketing" in "telemarketing agent"` is true, and a telemarketing agent isn't a marketing leader. `"it" in "white collar"` is also true. There are tests for both.

**00:27: the embedding stage**

**C:** I want to be honest about this part. The built-in `HashingEmbedder` is lexical, not semantic. It hashes 5-character word stems and character trigrams. It catches "financial" vs "finance" and even "financier", but it won't map "Cyber Defence" to "security". Each persona has several anchors (its name plus each include keyword), and a persona's score is the best cosine over its anchors. A short keyword anchor then isn't diluted by a long description.

**I:** What would a real model give you?

**C:** Synonyms with no shared letters, and with a multilingual encoder, most of the translation calls go away. The threshold has to be re-fit per model, because cosine scales differ between models.

**00:35: the classifier**

```
[AI] Implement classify_titles in persona/classify.py using the functions above. Steps:
     dedupe on normalize_title; empty -> method empty_input; translate non-English titles in batches
     (llm may be None); level/function via rules on the (translated) text; keyword -> embed;
     then vetoes (exclude -> exclude_rule conf 1.0, level not allowed -> level_filter);
     remaining -> llm.classify in batches with a budget; validate LLM JSON. One output row per input row, input order.
```

**C:** *(reading the result)* One thing I'm checking specifically is that the veto runs **after** the LLM too.

**I:** Why does that matter?

**C:** The tenant's exclusions are a business rule, and the model doesn't get to overrule them. There's a test where the scripted LLM says "Assistant to the CFO office" is `P_FIN` with 0.95 confidence, and we still drop it because "assistant" is an exclude keyword for that persona.

**00:48: run the sample**

```
[screen] $ python -m persona.cli --contacts sample_data/contacts.csv --personas sample_data/personas.json
 1 Sr. Director, Demand Generation  [P_MKT_LEAD] Director Marketing    keyword              0.95
 2 Directeur Financier              [P_FIN]      C-Level  Finance      llm_translate+embed  0.51
 3 Marketing Intern                 []           Intern   Marketing    exclude_rule         1.00
 4 Head of InfoSec & Risk           [P_IT_SEC]   Director IT/Security  keyword              0.95
 5 "  "                             []           Unknown  Unknown      empty_input          0.00
 6 Chief Happiness Officer 🎉       []           C-Level  HR           llm                  0.40
```

**I:** Row 4 says `embed` in the brief and you have `keyword`. Is yours wrong?

**C:** `infosec` is in `P_IT_SEC`'s include list, so the keyword stage catches it first. I think that's the better answer: it's cheaper and easier to explain. If the brief meant keywords to be matched only as whole titles, I'd change that, but I'd ask before doing it. Row 2's confidence is lower than the brief's 0.88 because my embedder is the lexical stand-in.

**I:** And row 2 is C-Level, not Director, even though "directeur" means director?

**C:** Yes. "Directeur Financier" is the French CFO title. Level and function are detected on the translation, "Chief Financial Officer", so it's C-Level.

**00:55: wrap-up**

**I:** Five minutes. What are the limitations, and how would you deploy it?

**C:** Limitations first:
- `looks_english` is a vocabulary heuristic. "Directeur Marketing" has one known word out of two, so it counts as English and isn't translated. The persona is still right via the keyword, but the level is Unknown. I'd swap in a real language-ID model.
- The embedder is lexical. "Head of Cyber Defence" gets function IT/Security from the rules, but no persona.
- Tied functions return Unknown, which is a precision choice.

Scaling: classify unique titles only. Cache level and function globally by normalised title. Cache persona results per tenant, keyed by a hash of that tenant's persona config. The LLM budget is a hard cap.

Deployment: a nightly batch over new or changed titles, plus a small online endpoint that runs rules and embeddings synchronously and queues LLM work. I'd monitor the method mix, LLM calls per day, how often the budget is exceeded, and how many results are low confidence.

**I:** Thanks, that's time.

---

## Trajectory 2: the cost and scale interviewer

This interviewer lets you build the basics, then spends 30 minutes on cost.

**I:** Honestly, why not send every title to the LLM? It's less code and probably more accurate.

**C:** Four reasons.
1. **Cost**: it scales with unique titles, and the rules handle the heavy head for free.
2. **Determinism**: the same title should get the same persona tomorrow. Even at temperature 0, a model upgrade can change answers. Rules and a fixed embedder don't drift.
3. **Explainability**: "matched keyword `cfo`" is something a rep can understand.
4. **Latency** for the online path.

It's probably more accurate on the long tail, and that's exactly where we use it.

**I:** Give me a rough cost.

**C:** I'll state the assumptions, because I don't have your numbers.
- Say 50M contacts and 3M unique normalised titles.
- Say rules and embeddings decide 80%. That leaves 600k titles for the LLM.
- At 20 titles per call, that's 30k calls for a full backfill.
- After that, it's incremental: only new titles hit the LLM, and those are a small fraction per day.

The real number depends entirely on the 80%, so the first thing I'd measure is the method distribution on a real tenant.

**I:** How does caching work when a tenant edits a persona?

**C:** There are two caches. Level and function don't depend on the tenant, so they're cached globally by normalised title. Persona assignment depends on the tenant's definitions, so its key includes a hash of that tenant's persona config. Editing a persona invalidates that tenant's persona cache only. Title embeddings stay valid; only the persona anchors are re-embedded, which is a handful of vectors.

**I:** The Buying Group agent wants "the top 25 job titles most semantically similar to each persona". How?

**C:** It's the same index used in reverse. Embed the de-duplicated title universe once. For each persona, compute similarity to its anchors and take the top 25 with `argpartition`. With millions of titles, I'd put them in an ANN index (FAISS or pgvector). Two details:
- Apply the exclude veto **before** taking the top 25, otherwise "Marketing Intern" variants eat the slots.
- Normalisation already collapses "Sr Dir Marketing" and "Senior Director, Marketing", so near-duplicates don't take five slots.

**I:** How does this run in Spark?

**C:** `mapInPandas` over the unique titles, with the persona index broadcast because it's tiny. I would **not** call the LLM from 500 executors. That stage goes to a rate-limited service with a cache table: Spark writes the undecided titles, the service fills in answers, and a second pass joins them back.

**I:** What happens when the budget runs out?

**C:** Those titles get `method = budget_exceeded`, which is an honest "we didn't look". Today the code hands out budget in alphabetical order of the normalised title. That's deterministic but not smart. I'd change it to frequency order, so the most common unresolved titles are answered first. That's a one-line change to the sort key.

**I:** Latency for the online path?

**C:** Rules and the embedding lookup take milliseconds in-process. The LLM is asynchronous: stamp with what we have (possibly nothing), enqueue the title, and update the contact when the answer arrives. Since the cache is shared, a new contact with a known title never waits.

**I:** What would you put on a dashboard?

**C:** Share of each method per day, LLM calls and cost, cache hit rate, `budget_exceeded` count, share of `llm` answers below the confidence threshold, and the rate of disagreement between the LLM and the rules on a small overlap sample. A jump in that last one usually means the data or the model changed.

---

## Trajectory 3: the AI gets it wrong (and so do you)

The interviewer watches how you verify. Things break, and that's fine. Narrate.

**00:14: substring matching**

```
[AI] Write keyword_hit(norm, keyword) -> bool.
[screen] def keyword_hit(norm, keyword):
             return keyword.lower() in norm
```

**C:** That's a substring check. "Telemarketing agent" would match "marketing", and "white collar" contains "it". I'll add those as failing tests and ask again with the rule spelled out.

```
[AI] Rewrite: normalise the keyword with normalize_title. Keywords of <=3 chars must match a whole token;
     longer keywords match at a word start (prefix), never mid-word. Use a compiled regex with (?<!\w).
[screen] $ pytest -q -k keyword_hit   → 8 passed
```

**00:18: VP read as C-Level**

```
[screen] FAILED test_detect_level[vice president of sales-VP]  assert 'C-Level' == 'VP'
```

**C:** The AI checks "president" before "vice", so every VP becomes C-Level. The fix is ordering: VP before C-Level. The test caught it, which is why the table had that row.

**00:30: Python's `hash()` in the embedder**

```
[screen] idx = hash(feature) % self.dim
```

**C:** This will pass every test in a single process, and it's still wrong. `hash()` on strings is salted per process (`PYTHONHASHSEED`), so a vector computed in tonight's batch won't match one computed tomorrow or on another executor. I'll show it.

```
[screen] $ python -c "print(hash('w:finan') % 2048)"   → 882
         $ python -c "print(hash('w:finan') % 2048)"   → 100
         $ python -c "print(hash('w:finan') % 2048)"   → 348
```

**C:** I'll switch to `zlib.crc32`, which is stable, and add a test that embeds the same text twice and checks the vectors are equal. The honest caveat is that a single-process test can't really catch the salting. The reason I know is that I've been burned by it before, so I mention it out loud.

**00:40: the threshold sits on a knife edge**

```
[screen] 2 Directeur Financier  [P_FIN]  ...  llm_translate+embed  0.51      (embed_accept = 0.50)
```

**C:** That passes, but 0.51 against a threshold of 0.50 is too close for comfort. Before trusting it, I'll look at what the negatives score.

```
[screen] chief financial officer   [0.02 0.04 0.51]
         chief happiness officer   [0.02 0.04 0.04]
         head of people            [0.03 0.13 0.  ]
         director of cyber defence [0.05 0.07 0.09]
```

**C:** Positives are around 0.5 and the negatives top out at 0.13. I'll put the threshold mid-gap, at 0.35, and add a test that a related title beats an unrelated one by a clear margin.

**I:** Isn't that tuning on your test set?

**C:** Yes. With a handful of examples it's a sanity check, not calibration. The real procedure is a labelled set of titles, a precision/recall curve, and picking the threshold at the precision target. That has to be redone for every embedding model.

**00:46: a test that was wrong, not the code**

```
[screen] FAILED test_detect_level[svp sales-VP]  assert 'Unknown' == 'VP'
```

**C:** *(reads)* `detect_level` takes **normalised** text, and normalisation expands `svp` to `vp`. My test passed the raw abbreviation. The code is right and the test is wrong. I'll change the input to `vp sales` and add `vice president` as its own case.

**I:** How did you decide it was the test?

**C:** I checked the contract. The docstring says the input is normalised, and `normalize_title("SVP Sales")` returns `vp sales`, which passes. If the contract were unclear, I'd have fixed the contract first.

**00:50: a surprise that turned out to be good**

```
[screen] FAILED test_without_llm_unmatched_titles_are_reported_not_guessed
         'Directeur Financier' → [P_FIN], method=embed, confidence 0.50
```

**C:** I expected `no_match` without an LLM. But "financier" shares the `finan` stem with "finance", so the embedder finds `P_FIN` at about 0.6. The level is Unknown, so the 0.10 penalty brings confidence to 0.50. That's reasonable behaviour: cheap cross-lingual recall, flagged with lower confidence. I'll keep the behaviour, document it with its own test, and use a truly unknown title for the `no_match` test.

**00:55: a real bug found by trying Japanese**

**C:** One more check, since the brief says Japanese. *(types `マーケティング部長`)*

```
[screen] normalize_title("マーケティング部長")  → 'マーケティンク部長'
```

**C:** That's a bug. To strip accents I decompose with NFKD and drop combining marks. In Japanese, the voicing mark on グ is also a combining mark, so グ became ク and the word changed. The fix is to drop combining marks only when they follow an ASCII letter, then recompose with NFKC. NFKC also turns full-width "Ｍａｒｋｅｔｉｎｇ" into "marketing". I'll add both as tests.

**I:** Good. What's the lesson?

**C:** Every test passed and the code was still wrong, because no test covered that input. For multilingual code, the test table needs at least one example per script you claim to support.

---

## Trajectory 4: the ML-depth interviewer

This interviewer cares less about the code and more about whether the numbers mean anything.

**I:** How would you evaluate this?

**C:**
1. **Labelled set**: sample titles stratified by method and by tenant, because a uniform sample is mostly easy keyword hits. Two annotators label persona, level and function. Titles are ambiguous, so I'd measure agreement between annotators first. If humans agree only 85% of the time, the model can't be judged against 99%.
2. **Metrics**: precision and recall per persona, and **per method**. A low precision for `embed` tells me exactly which threshold to move. Level and function are multi-class, so I'd look at the confusion matrix rather than just accuracy.
3. **Thresholds**: pick `embed_accept` and `min_confidence` from the precision/recall curve at the precision target the product agrees on, for example 0.95.
4. **Ongoing**: re-label a small random sample every month, and track LLM-vs-rule disagreement on titles where both give an answer.

**I:** The LLM returns a confidence. Do you trust it?

**C:** Not as a probability. Self-reported confidence is poorly calibrated. I treat it as a score: on the labelled set, I check what precision we get above 0.6, 0.7 and 0.8 and set `min_confidence` from that. If it doesn't separate correct from wrong answers at all, I'd stop asking for it and use something like agreement across two prompts instead.

**I:** And the rest of the LLM output?

**C:** Strictly validated. `validate_llm_result`:
- drops persona ids that aren't in the tenant's list
- maps unknown levels and functions to Unknown
- rejects confidences that are strings, booleans or NaN, and clips the rest to [0, 1]

If the payload is unusable, the method is `llm_invalid`, nothing is stamped, and nothing crashes. On the prompt side: temperature 0, a JSON schema or tool-use for structured output, the tenant's persona definitions and a few examples.

**I:** Translate-then-classify, or a multilingual model?

**C:** Translate-then-classify keeps one English taxonomy and one set of rules, which matters with a small team. It costs an LLM call per unique non-English title, which caching makes acceptable. A multilingual encoder such as e5 or LaBSE as the embedder would handle most European titles without translation. Then I'd translate only what the encoder isn't confident about. Language detection also needs to be real. Today a vocabulary heuristic lets "Directeur Marketing" through as English, so its level is Unknown.

**I:** Would you ever train a model?

**C:** Yes, once there's data. After a few months of LLM answers that were spot-checked by humans, I'd train a small classifier for level and function on title embeddings. Logistic regression is enough to start. It's a distillation of the LLM, and it would take most of the tail off the LLM path. Personas stay rule-and-similarity based, because they're per-tenant definitions and change whenever a customer edits them.

**I:** What about a title that genuinely has two functions, like "VP Finance and Security"?

**C:** Persona matching is multi-label, so it gets both `P_IT_SEC` and `P_FIN`. The function field is single-valued, and a tie between functions returns Unknown rather than guessing. If the product wants two functions, the output type changes to a list. That's a contract change, so I'd ask before making it.

---

## Trajectory 5: requirements change and time runs short

**00:35**

**I:** New requirement from a customer: their "Marketing Leader" persona should include managers.

**C:** That's a config change, not a code change. I add `"Manager"` to that persona's `levels` in `personas.json`. Today "Marketing Manager" returns `level_filter`. After the change it returns `P_MKT_LEAD` from the keyword match. I'll change the test to cover both versions of the config.

**I:** They also want "Head of" to count as VP, just for them.

**C:** That's a different kind of change. Level is a global taxonomy today, and making it tenant-specific means a per-tenant override map that runs after `detect_level`, plus cache keys that include it. It's maybe 30 minutes of careful work. I'd rather not start it now and leave the core half-done. Can I note it as a follow-up and show you where it would go?

**I:** Fine, note it.

**00:50**

**I:** You have 20 minutes, and the LLM part isn't there yet.

**C:** Then I'll prioritise. Must have:
- `validate_llm_result` and its tests, since that's the safety boundary
- the classify fallback with a budget
- vetoes applied after the LLM

I'll skip batched translation today. Non-English titles go straight to the classify fallback, which costs a bit of accuracy but keeps one code path. The `LLMClient` Protocol stays the same, so translation can be added later without changing callers.

```
[AI] Add the LLM classify fallback to classify_titles: only titles with no method yet; sorted for
     deterministic budget; batches of cfg.llm_batch_size; validate each answer with validate_llm_result;
     re-run _veto on the result; below cfg.min_confidence -> no personas, reason says so.
```

**C:** *(after tests pass)* Here's what I didn't do and why:
- translation is skipped
- per-tenant level overrides are noted, not built
- the budget is spent in alphabetical order rather than by frequency

None of those affect correctness of what's there. They affect cost and coverage.

**I:** If you had another hour?

**C:** Translation, frequency-ordered budget, a real language detector, and a labelled set to set thresholds. That's the order I'd do them in.

---

## Answer bank

| Question | Short answer |
|---|---|
| Why rules before embeddings before LLM? | Cheapest and most explainable first; each stage only sees what earlier ones couldn't decide. |
| Why de-duplicate first? | Cost scales with unique titles, not contacts; also guarantees the same title gets the same answer. |
| Why apply vetoes after every stage? | Tenant exclusions are business rules; neither embeddings nor the LLM may override them. |
| Why is an unknown level not a rejection? | Missing data shouldn't zero a good match; it costs 0.10 confidence instead (0.95 → 0.85). |
| Why return Unknown on tied functions? | Precision first; a wrong function is worse than none. |
| Why `crc32` instead of `hash()`? | `hash()` on strings is salted per process, so embeddings wouldn't be reproducible. |
| Why translate instead of multilingual rules? | One taxonomy and one rule set; translations are cached per unique title. |
| What does `budget_exceeded` mean? | We didn't look: the LLM budget ran out. It's honest, measurable, and different from "no match". |
| What if the LLM invents a persona id? | It's dropped by `validate_llm_result`; an unusable payload becomes `llm_invalid`. |
| Weakest part of the current code? | Language detection (vocabulary heuristic) and the lexical embedder; both sit behind interfaces and are meant to be swapped. |
