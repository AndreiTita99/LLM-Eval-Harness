# Developer / Interview Prep — LLM Eval Harness

> Private notes (not for the client). This is the "why" behind every decision, plus
> rehearsed answers to the questions an interviewer will probe. Grows each phase.

## The 30-second pitch

> "I built an automated evaluation harness for LLM prompts — essentially CI for
> prompts. It runs a golden dataset through a prompt, scores the outputs with three
> families of scorers including an LLM-as-judge, handles non-determinism by running
> each case N times and reporting pass-rate, and compares results to a known-good
> baseline. If a prompt or model change regresses quality, latency, or cost beyond a
> tolerance, CI fails and the change can't merge."

## Why this project matters (what it proves)

Most people who call themselves "AI testers" have never built the machinery for
testing non-deterministic systems. This project *is* that machinery: golden datasets,
multiple scorer types, variance handling, baseline regression gating, and a CI hook.
It demonstrates the "testing AI systems" skill set directly.

## Core mental model

**Gate on regression, not on perfection.** The harness never demands 100% accuracy —
real LLM tasks aren't 100%. It demands that a change doesn't make things *worse* than
the last known-good baseline, beyond a defined tolerance. That reframes "how do you
test something that's never perfect?" into a tractable, CI-friendly question.

## Rehearsed answers to the likely questions

**Q: How do you test something non-deterministic?**
Run each case N times (`EVAL_REPEATS`, default 3), report pass-rate per case, and flag
*flaky* cases — ones that neither always pass nor always fail (`0 < passes < N`). A case
that passes 3/5 is a different signal than 5/5, and the harness surfaces that instead of
trusting one sample. Temperature is recorded; the most capable models (Opus 4.8/4.7) no
longer expose `temperature` at all, so the harness leans on N-repeats rather than pinning
temperature for determinism. (Built in Phase 4. The mock is deterministic by default for
reproducible CI, with an `EVAL_MOCK_FLAKINESS` knob — seeded by (input, repeat) so even
the "non-deterministic" demo is reproducible — to show flaky detection live.)

**Q: What non-accuracy metrics do you track, and why?**
Property metrics, treated as first-class, not afterthoughts: **format validity** (did we
get parseable JSON), **refusal** (did the model decline — a triage bot essentially never
should), **latency** (mean + p95), **token usage**, and **estimated cost**. A prompt or
model change can regress latency or cost without touching accuracy — and Phase 5 gates on
p95 latency and cost, not just pass-rate. Cost uses a static price table; offline it falls
back to the configured model's rates so the number is illustrative.

**Q: How do you trust an LLM judge?**
Four mitigations: (1) score against an explicit **rubric** with defined levels, not a
vague 1–10; (2) guard against known biases — verbosity (longer ≠ better),
position (pairwise), self-preference; (3) use a **different, cheaper model** as the
judge than the SUT to reduce self-preference and cost; (4) **validate against humans** —
hand-label ~12 cases, run the judge, report agreement. A judge you haven't validated
is just vibes. **Position bias doesn't apply here** — that's a pairwise-comparison
artifact, and this judge grades a single output pointwise. (Built in Phase 3:
`eval judge-validate` reports exact agreement, pass/fail agreement, and Cohen's kappa.)

**Q: How did you actually measure judge reliability — and what did you find?**
On the 12-case hand-labelled set, the mock judge gets 75% exact agreement, 83%
pass/fail agreement, kappa 0.56 (moderate). More important than the numbers is
*where* it disagrees: it scored a hallucinated summary as adequate (missed invented
detail — j07), down-scored a verbose-but-correct summary (verbosity bias — j12), and
was over-harsh on a vague one (j06). Those are the textbook judge failure modes, and
the validation surfaces them instead of hiding them. With a real judge model I'd
expect higher agreement; the methodology is identical either way.

**Q: What's a regression here, and how does CI catch it?**
`baseline.json` stores last-known-good aggregate scores. A run compares new scores
within a tolerance (e.g. accuracy may not drop >2 points; p95 latency may not grow
>X%). On regression we write the report **and exit non-zero**, which fails the PR
check. `eval baseline update` promotes current scores — a deliberate, reviewed action.
*(Lands in Phase 5.)*

**Q: Why hold out part of the dataset?**
Same reason you don't evaluate an ML model on its training data. Cases used while
iterating on the prompt are "training"; held-out cases give an honest read on
generalisation. The dataset marks held-out cases with `held_out: true`.

**Q: How do you keep eval costs sane?**
Two paths: a fast **synchronous** path for the small PR-gate subset, and the
**Message Batches API** (~50% cheaper, async, results within 24h) for large nightly
sweeps. Report estimated cost per run. *(Batch path noted in Phase 6.)*

**Q: Build vs promptfoo / DeepEval — when would you reach for each?**
Those frameworks exist and are great. I built a minimal core myself to prove I
understand the mechanics — judging, variance, gating — under the hood. In a real job
I'd reach for an established framework for breadth, but knowing what they do
internally means I can debug them, extend them, and judge when they're wrong.

**Q: What's the failure mode of this system?**
A bad or biased judge silently blessing regressions. If the judge is mis-calibrated,
the gate passes garbage. That's exactly why judge↔human agreement is measured and
reported — the judge is itself under test.

## Design decisions log

- **Single provider behind a narrow interface.** `complete(system, user) -> LLMResponse`.
  Keeps call sites provider-agnostic and makes the mock a drop-in. Multi-provider was
  deliberately cut.
- **Offline mock provider.** Lets `eval run` work with zero setup (and keeps CI from
  making live calls). The mock uses keyword heuristics so it's intentionally imperfect
  — scoring output stays meaningful. Live tests sit behind a `live` pytest marker.
- **`temperature` left unset by default.** Opus 4.8/4.7 reject sampling params; sending
  temperature only when configured keeps the harness model-agnostic. Good, concrete
  talking point about knowing the current model surface.
- **Typed everything with pydantic.** `EvalCase`, `RunResult`, `CaseScore`,
  `RunSummary`. Schema validation comes for free and the data contracts are explicit.
- **`category` exact-match is hardcoded in Phase 1**, then generalised into a scorer
  registry in Phase 2 — shows the framework evolving from a script into a framework.
- **Scorer primitives are generic; SUT-specifics live in the registry.** `ExactMatch`,
  `EnumValid`, `Contains`, `SchemaValid` take no knowledge of triage; `registry.py`
  holds the allowed categories/urgencies and the response schema and wires up the
  named instances. Adding a metric = registering one entry.
- **Enum validity ≠ correctness — and that distinction is deliberate.** `urgency_schema`
  checks the value is a *valid* enum member (a format/schema check); `category_exact`
  checks the value is *correct* vs expected. Good talking point: structural scorers
  split "is the output well-formed?" from "is the output right?".
- **Unregistered scorers are skipped, not failed.** The dataset declares
  `summary_judge` from the start; before Phase 3 it's skipped and surfaced, so the
  intended metric set is visible without breaking the run. Avoids fail-by-omission.

## Talking point: why a registry at all?

A naive harness hardcodes scorers in the runner. The registry decouples *what to
measure* (declared per-case in YAML) from *how to measure it* (scorer classes) and
*which metrics exist* (registration). That's the seam that lets non-engineers add
cases, lets me add scorers without touching the runner, and makes the
"skipped vs failed" behaviour clean.

## Phase-by-phase status

- **Phase 1 (done):** Skeleton end to end — config, models, client + mock, dataset
  loader, one hardcoded exact-match scorer, CLI. `eval run` prints a pass/fail table.
- **Phase 2 (done):** Scorer registry + structural scorer primitives (exact / enum /
  contains / whole-response schema). Cases declare scorers by name; per-scorer and
  overall pass-rate aggregation; unknown scorers skipped and surfaced.
- **Phase 3 (done):** LLM-as-judge (`summary_judge`) grading the summary 1–3 against a
  rubric, on a cheaper model, with verbosity/self-preference guards. Offline mock
  judge. `eval judge-validate` reports judge↔human agreement + Cohen's kappa on a
  hand-labelled set (75% / 83% / 0.56 with the mock).
- **Phase 4 (done):** N repeats with per-case pass-rate + flaky detection; universal
  property scorers (`format_valid`, `no_refusal`); latency mean/p95, token totals,
  estimated cost. Reproducible flakiness knob for the mock to demo variance.
- Phases 5–8: see README roadmap.

## Design decisions log (Phase 3 additions)

- **Judge engine vs judge scorer are separate.** `src/llm/judge.py` owns the
  rubric, prompt, bias guards, and parsing → `JudgeResult`. `src/scorers/judge.py`
  is a thin adapter to the `Scorer` interface. Keeps the grading logic testable
  without the scorer plumbing, and the rubric reusable.
- **Judge reuses the SUT client with a model override.** `complete(..., model=)`
  rather than a second SDK wrapper — one place for latency/usage/error handling.
- **Mock judge is intentionally biased.** It over-penalises length and can't see
  hallucinations, so the human-agreement check has something real to catch. Honest
  demo > flattering demo.
- **Cohen's kappa, hand-rolled and dependency-free** (`_cohen_kappa_binary`). Being
  able to explain *why* raw agreement is misleading (chance agreement when classes
  are imbalanced) is a strong signal — kappa corrects for it.

## Design decisions log (Phase 4 additions)

- **Declared vs universal scorers.** Structural/judge scorers are declared per case
  (they need expected values); property scorers (`format_valid`, `no_refusal`) are
  intrinsic to any call, so the runner applies them to every result automatically.
  That split is a clean way to explain why some scorers live in the YAML and some don't.
- **Pass-rate, not a single sample.** The unit of truth is N repeats per case. `is_flaky`
  = `0 < passes < N`. Flaky ≠ failed — it's the signal that the prompt is unstable on
  that input, which is often more actionable than a hard fail.
- **Reproducible non-determinism (the mock flakiness trick).** Seeding the mock RNG by
  `(input, repeat)` gives output that varies across repeats yet is identical run-to-run.
  Lets the demo *show* flaky detection while CI stays deterministic. Good "how do you
  test the thing that tests non-deterministic systems?" answer.
- **Latency/cost are metrics, not pass/fail (yet).** They're continuous; turning them
  into gates needs a baseline + tolerance, which is Phase 5. p95 (nearest-rank) over
  mean because tail latency is what hurts in production.
- **Cost is an honest estimate.** Static price table; unknown/mock model falls back to
  the configured SUT rates. I can say plainly "offline this is illustrative, against a
  real model it's exact, and judge-call cost isn't included yet."

## Things to be able to show live

- A green run (`eval run`) with the per-scorer breakdown.
- *(Later)* a deliberately regressed prompt that makes CI exit non-zero — the money shot.
