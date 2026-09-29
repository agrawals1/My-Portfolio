# How to use AI in the "AI-assisted coding" round (and how to answer *"how do you use AI for coding?"*)

Distilled from current guidance (sources at the bottom). The interviewers grade **approach, control, verification and
communication** — not typing speed. The failure mode they are looking for is the *prompt → paste → paste error → repeat* loop.

## The one-sentence answer
> "I treat the model as a fast junior pair: I write the spec, the interfaces and the tests (or the verification criteria) first, let it
> implement one small module at a time, read and run everything it produces, and use a second, fresh-context pass to attack it —
> so I'm reviewing evidence, not vibes."

## Workflow: Spec → Tests → Small slice → Verify → Adversarial review → Simplify

1. **Clarify & spec (you, 5–8 min).** Restate the problem, ask 3–4 clarifying questions (precision vs recall, tie policy, data size, 1:1?). Write down signatures, I/O, the edge-case list. The spec is your source of truth — prompts come and go.
2. **Explore / plan before code.** Ask the AI for a plan and the risks *without* writing code ("plan mode"). Edit the plan. Skip planning only when the diff is describable in one sentence.
3. **Tests first (or examples in the prompt).** Highest-leverage habit: give the model a pass/fail signal it can run. Paste a parametrised table of edge cases → it implements against it → you run `pytest`.
4. **One module per prompt, pure functions.** Small context, small diff, easy to review. Reference existing files/patterns ("follow the style of `normalize.py`").
5. **Read the diff like a reviewer.** For every branch ask "which test covers this?" and "what input breaks this?". If you can't explain a line, ask the AI to explain it or delete it.
6. **Adversarial pass in a fresh context.** New chat/subagent: give only the diff + the requirements and ask for *correctness bugs and missed edge cases only* (otherwise you get over-engineering). Writer/Reviewer split.
7. **Simplify.** "Remove anything not needed by the spec; no new dependencies; no speculative abstractions."
8. **Reset when it drifts.** Corrected the same issue twice? Stop, start a fresh context with a better prompt that includes what you learned — don't pile on corrections.

## Prompt anatomy that gets simple, production-shaped code

```
ROLE:      Senior Python/data engineer; prefer stdlib + pandas + rapidfuzz.
CONTEXT:   <2–3 lines of business context; paste the relevant spec section / sample data>
TASK:      Implement `normalize_domain(value) -> str | None` in er/normalize.py only.
CONTRACT:  Never raises (None/NaN/pd.NA/numbers/junk -> None). Deterministic. No I/O. Type hints.
CONSTRAINTS: Simple > clever. No new dependencies. No regex you can't explain in one line. <40 lines. No classes.
VERIFY:    Make these tests pass (paste table). Run `pytest -q tests/test_normalize.py` and show the output.
OUT OF SCOPE: blocking, scoring, CLI.
FORMAT:    Code only, then a 3-bullet list of assumptions and any edge case you did NOT handle.
```

Reusable one-liners:

* **Force the risks up front:** "Before coding, list 8 ways this could be subtly wrong for real CRM data (NaN, unicode, ties, memory). Then code."
* **Force honesty:** "List the assumptions you made and every input you did not handle."
* **Explain-to-verify:** "Explain lines 12–30 as if to a reviewer. Which branches are untested?"
* **Test-gap hunt:** "Write property-based (hypothesis) tests: idempotency, never raises, determinism under shuffling."
* **Performance review:** "Where is this O(n·m)? Any `apply(axis=1)`, cross join, or per-row DataFrame creation?"
* **Minimal fix on failure:** "Test X fails with <paste>. State what the code does now vs what the test expects, then propose the smallest fix. Do not rewrite the function."
* **Root cause, not symptom:** "Fix the root cause; do not special-case the test input or suppress the error."
* **Review a diff, not a file:** "Review only this diff for correctness, edge cases and perf; show minimal patches; suggest 2 tests."
* **Simplify:** "Reduce this by 30 % without changing behaviour; keep all tests green."

## Anti-patterns to name (shows maturity)

* Prompt → paste → paste error → repeat (no spec, no tests, you're the only verification loop).
* "Kitchen-sink" session: unrelated tasks in one context → degraded output. Reset between tasks.
* Correcting over and over in the same context.
* Trust-then-verify gap: plausible code that ignores NaN / ties / unicode. Always ask for evidence (test output).
* Letting the model choose weights/thresholds/priorities silently — those are *your* product decisions; make them explicit in config.
* Accepting a rewrite when a 2-line fix was needed; accepting new dependencies without asking why.
* Asking the same session that wrote the code to review it (it is biased toward its own work).

## What to *say* while working (communication = 1/4 of the grade)

* "Here's my plan and the 3 risks; tell me if precision matters more than recall."
* "I'm writing the tests first so the AI has a target — and so I'm not the verification loop."
* "This generated line looks right on the sample, but let me try `co.in` / NaN / tie."
* "I'm rejecting this suggestion — it adds a cross join / a dependency / speculative abstraction."
* When a test fails: state the hypothesis *before* asking the AI.
* Narrate trade-offs, then stop and let them steer.

## Verification checklist for every AI-generated chunk

- [ ] Can I explain each branch? Which test hits it?
- [ ] Inputs: None / NaN / pd.NA / empty / unicode / huge / duplicates / unsorted?
- [ ] Determinism (row order, ties, sets iteration)?
- [ ] Complexity: any hidden O(n·m), `apply(axis=1)`, repeated DataFrame concat, chained assignment?
- [ ] Errors: clear `ValueError` for bad input, no bare `except`, no swallowed exceptions?
- [ ] Config vs magic numbers; no hard-coded thresholds in logic?
- [ ] Extra dependencies / dead code / speculative abstraction? Delete.
- [ ] Ran it myself on the sample and on one nasty case.

## Sources (searched for this guide)

* Anthropic — [Best practices for Claude Code](https://code.claude.com/docs/en/best-practices): give the model a way to verify its work; explore→plan→implement→commit; be specific, reference patterns; interview-then-spec; reset context after two failed corrections; Writer/Reviewer fresh-context review; the "trust-then-verify gap".
* Microsoft — [Spec-Driven Development: A Spec-First Approach to AI-Native Engineering](https://developer.microsoft.com/blog/spec-driven-development-ai-native-engineering/).
* Augment Code — [Spec + TDD: the combination that produces shippable AI-generated code](https://www.augmentcode.com/guides/spec-tdd-shippable-ai-generated-code) and [Prompt engineering techniques for AI coding](https://www.augmentcode.com/guides/master-prompt-engineering-techniques-for-ai-coding).
* LeadDev — [7 prompting strategies to sharpen your AI-assisted code](https://leaddev.com/software-quality/7-prompting-strategies-to-sharpen-your-ai-assisted-code).
* Google Cloud — [Five best practices for using AI coding assistants](https://cloud.google.com/blog/topics/developers-practitioners/five-best-practices-for-using-ai-coding-assistants).
* Supabase — [Vibe coding: best practices for prompting](https://supabase.com/blog/vibe-coding-best-practices-for-prompting).
* DEV — [Cursor + Claude: my AI code review checklist](https://dev.to/sathish_daggula/cursor-claude-my-ai-code-review-checklist-hm5) (review diffs, not files; ask for 2 tests).
* PracHub — [The AI coding interview: a complete 2026 guide](https://prachub.com/resources/ai-coding-interview-guide) (only the search summary was readable; the page itself was blocked from this environment).
