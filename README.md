# LLM Evaluation Harness — "CI for Prompts"

> Automated evals for an LLM prompt, with **regression gating wired into CI** — so a
> prompt or model change that quietly makes quality worse **cannot be merged**.
> Think *unit tests + CI, but for non-deterministic AI behaviour.*

## The problem

The moment a team ships an LLM feature, they hit a wall: how do you change a prompt
without silently breaking 30% of cases? Outputs aren't deterministic, so you can't
just `assert response == expected`. This harness is the machinery that makes prompt
changes safe: golden datasets, multiple scorer families (including LLM-as-judge),
variance handling, and a baseline comparison that blocks regressions in CI.

The demo **system under test (SUT)** is a support-ticket triage prompt: given a
customer message, the model returns a `category` (exact-match scoring), an `urgency`
enum (schema validation), and a one-line `summary` (graded by an LLM judge). The
harness itself is prompt-agnostic — the triage task is just the showcase.

## Architecture

```
   datasets/triage.yaml            prompts/triage_v1.txt
   (golden input cases)            (the system under test)
            \                            /
             v                          v
        +-------------------------------------+
        |              Runner                 |
        |  for each case x N repeats:         |
        |    call SUT prompt -> output        |
        |    capture latency + token cost     |
        +------------------+------------------+
                           |
                           v
        +-------------------------------------+
        |             Scorers                 |
        |  structural | llm-judge | property  |
        +------------------+------------------+
                           |
                           v
        +-------------------------------------+
        |   Aggregator: per-metric scores,    |
        |   pass-rate, variance, cost/latency |
        +------------------+------------------+
                           |
              compare to baseline.json
                           |
              pass? -> report + exit 0
              regression? -> report + exit 1  (blocks the PR)
```

**Core principle — gate on regression, not on perfection.** The harness doesn't demand
100% accuracy. It demands that a change doesn't make things *worse* than the last
known-good baseline beyond a tolerance. That's what makes it a realistic CI gate
rather than a vanity metric.

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"

# Run the golden set through the SUT prompt:
eval run
# or, without installing:  python -m src.cli run
```

With **no API key set**, the harness runs against a built-in **mock provider** so the
full pipeline is runnable out of the box. To run against the real Anthropic API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # set EVAL_PROVIDER=anthropic to force it
eval run
```

### Configuration

All run settings are environment-driven (see `src/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_PROVIDER` | auto (`anthropic` if key present, else `mock`) | Which client to use |
| `EVAL_SUT_MODEL` | `claude-opus-4-8` | Model whose prompt is under test |
| `EVAL_JUDGE_MODEL` | `claude-haiku-4-5` | Cheaper, different model for the judge |
| `EVAL_REPEATS` | `3` | How many times to run each case (variance handling) |
| `EVAL_SUT_TEMPERATURE` | unset | Only sent to models that accept it |
| `EVAL_JUDGE_THRESHOLD` | `2` | Minimum rubric score (1–3) for the summary judge to pass |
| `EVAL_MOCK_FLAKINESS` | `0.0` | Mock-only: probability the mock perturbs its output, to demo variance |

## Scorers

Cases declare which scorers apply to them by name; a registry resolves those names
to scorer instances. Three families:

| Family | Implemented | Examples |
|---|---|---|
| **Structural** (deterministic, cheap) | ✅ Phase 2 | `category_exact` (exact match), `urgency_schema` (enum validity), `response_schema` (whole-response validation); `contains` primitive available |
| **LLM-as-judge** (rubric-graded free text) | ✅ Phase 3 | `summary_judge` — grades the one-line summary 1–3 against a rubric, on a cheaper model |
| **Property** (latency, cost, format, refusal) | ✅ Phase 4 | `format_valid` + `no_refusal` (applied to *every* call); latency p95 and token cost reported as metrics |

Scorers that aren't registered yet are **skipped, not failed**, and reported as such —
so the dataset can declare the full intended set from day one. Structural and judge
scorers are **declared per case** (they need expected values); property scorers are
**universal** (intrinsic to any call) and run on every result automatically.

### Variance handling

LLM outputs aren't deterministic, so a single sample is a weak signal. Each case is
run **N times** (`EVAL_REPEATS`, default 3) and the harness reports **pass-rate per
case** and flags **flaky** cases — ones that neither always pass nor always fail. A
case that passes 3/5 is a different signal than 5/5, and the harness surfaces that
instead of hiding it behind one lucky (or unlucky) run.

The mock provider is deterministic by default (so tests are reproducible). Set
`EVAL_MOCK_FLAKINESS` to make it genuinely vary across repeats — seeded by
`(input, repeat)`, so a run is still reproducible — to see variance handling in action:

```bash
EVAL_MOCK_FLAKINESS=0.34 EVAL_REPEATS=5 eval run    # surfaces flaky cases
```

### The judge, done responsibly

The LLM-as-judge is the part most teams get wrong, so the mitigations are explicit:

- **Explicit 1–3 rubric** with defined level anchors, not a vague "rate 1–10".
- **Verbosity-bias guard** — the prompt states that a concise correct summary scores
  as high as a verbose one.
- **Self-preference guard** — the judge runs on a different, cheaper model
  (`judge_model`, default Haiku) than the SUT (default Opus).
- **Validated against humans.** `eval judge-validate` runs the judge over a
  hand-labelled set and reports exact agreement, pass/fail agreement, and **Cohen's
  kappa** (chance-corrected). A judge you haven't validated is just vibes.

```bash
eval judge-validate     # prints judge↔human agreement on datasets/judge_labeled.yaml
```

## Project status

Built in phases; each phase ends with something runnable.

- [x] **Phase 1 — Skeleton + one case end to end.** Config, typed models, LLM client
      (+ offline mock), YAML dataset loader, one exact-match scorer, CLI.
- [x] **Phase 2 — Scorer registry + structural scorers.** Generic primitives
      (exact / enum / contains / whole-response schema), a name→scorer registry that
      cases opt into, and per-scorer + overall pass-rate aggregation.
- [x] **Phase 3 — LLM-as-judge.** Rubric-graded `summary_judge` with verbosity/
      self-preference bias guards, an offline mock judge, and `eval judge-validate`
      reporting judge↔human agreement + Cohen's kappa on a hand-labelled set.
- [x] **Phase 4 — Variance + properties.** N repeats with per-case pass-rate and
      flaky-case detection; universal `format_valid`/`no_refusal` property scorers;
      latency mean/p95, token totals, and estimated cost reported per run.
- [ ] Phase 5 — Baseline tracking + regression gating + non-zero exit codes.
- [ ] Phase 6 — HTML + JSON reports with a diff-vs-baseline section.
- [ ] Phase 7 — GitHub Actions workflow that gates PRs touching prompts/datasets.
- [ ] Phase 8 — Polish: judge-validation writeup, batch-API cost note, screenshots.

## Deliberately out of scope

These were bounded deliberately, not forgotten:

- **No web dashboard / hosted service.** A static HTML report is enough.
- **No data-labelling UI.** Datasets are hand-curated YAML in the repo.
- **No multi-provider support.** One provider (Anthropic) behind one interface.
- **One task under test, done well** rather than evaluating "everything."

## Tech stack

Python 3.11+ · `anthropic` SDK · `pydantic` (typed cases/results/config) ·
`jinja2` (HTML report) · `pyyaml` (datasets) · `pytest` (testing the harness) ·
GitHub Actions (CI gating).
