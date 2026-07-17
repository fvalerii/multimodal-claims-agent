# Multi-Modal Evidence Review — Solution Guide

Technical how-to for the pipeline in this folder. For a portfolio-style overview
(problem, architecture highlights, recruiter summary), see the [root README](../README.md).

A LangGraph-orchestrated pipeline that verifies damage claims (cars, laptops,
packages) against their submitted images. For every row in a claims CSV it
produces a structured decision: whether the evidence is usable, what the issue
is, where it is, whether the claim is supported, and how severe it is.

---

## 1. How it works

The system is a [LangGraph](https://github.com/langchain-ai/langgraph)
`StateGraph` with one shared `AgentState`. Flow per claim:

```text
START
  │
  ▼
semantic_guardrail  (safety / relevance / scope screen)
  │  route_after_guardrail
  ├── block → guardrail_block_node ───────────────────── END
  └── allow → route_by_object  (reads claim.claim_object)
        ├── car      → evaluate_car_node ┐
        ├── laptop   → evaluate_laptop_node ├─ (shared _evaluate_claim_images)
        └── package  → evaluate_package_node ┘
                              │  route_post_evaluation (findings["valid_image"])
                              ├── valid   → posterior_risk_node ── END
                              └── invalid → fast_fail_node ────── END
```

- **Semantic guardrail** (`semantic_guardrail_node`) is the first hop. It
  screens the incoming claim text and retrieved history context for prompt
  injection / claim-manipulation, relevance, and scope, then routes unsafe or
  out-of-bounds requests to the safe fallback `guardrail_block_node` instead of
  the VLM. Layered as deterministic rules + an optional LLM pass, controlled by
  `GUARDRAIL_MODE` (`hybrid` default · `rules` · `off`); the LLM layer fails
  open so a transient outage never drops a legitimate claim.
- **Routing** (`route_by_object`) sends each allowed claim to the
  object-specific vision node so prompts can use the correct `object_part`
  vocabulary.
- **Vision nodes** all delegate to one helper (`_evaluate_claim_images`) that
  loads + downscales images, builds the prompt, and calls the VLM with a
  Pydantic-typed structured-output schema.
- **`route_post_evaluation`** short-circuits unusable evidence (no/corrupt/
  blurry images) to `fast_fail_node`, which emits a safe
  `not_enough_information` row instead of spending more reasoning.
- **`posterior_risk_node`** produces the final `ClaimOutput`: it trusts the
  VLM's `claim_status`/`evidence_standard_met` (gated on `valid_image`),
  cross-references user history and evidence requirements, and assigns
  `severity` deterministically from the predicted `issue_type`.

### Provider abstraction

A single dispatcher (`_call_vision_model`) supports two backends:

| Provider | Model (default) | Structured output |
|----------|-----------------|-------------------|
| Anthropic (default) | `claude-sonnet-4-5` | tool-use with JSON schema + enum constraints |
| Google (fallback)   | `gemini-2.5-flash`  | `response_schema` (Pydantic) |

The provider is auto-selected: Anthropic if `ANTHROPIC_API_KEY` is present,
otherwise Google. Override with `VISION_PROVIDER`.

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
cp code/.env.example .env   # then add your API key
```

Required environment (see `.env.example`): `ANTHROPIC_API_KEY` **or**
`GEMINI_API_KEY`. Keys are read from the environment only (via `python-dotenv`)
and never hardcoded.

---

## 3. Run the pipeline

```bash
# Development run against the labelled sample
python code/main.py --claims dataset/sample_claims.csv --output output.csv

# Final run against the test set (this is the submission output.csv)
python code/main.py --claims dataset/claims.csv --output output.csv
```

Each run writes `output.csv` (14 columns, exact schema/order from
`problem_statement.md`) and a sidecar `output.run_stats.json` capturing the
provider, model, runtime, row count, and per-category failure counts.

Robustness guarantees:

- **One output row per input row, always** — malformed input, corrupt images,
  API/quota/network errors, and graph errors each fall back to a valid
  `not_enough_information` row (categorised in the end-of-run summary).
- **Retry with backoff** on transient API errors; daily-quota exhaustion
  fast-fails rather than spinning.
- **Deterministic** decoding (`temperature=0`) for reproducible output.

---

## 4. Evaluate

There are two complementary evaluators.

### 4.1 Pytest suite (`code/evaluation/main.py`) — mocked, offline, no cost

```bash
pytest code/evaluation           # picks up pytest.ini and collects main.py
# or:  python code/evaluation/main.py    # convenience wrapper (pytest -s -v)
```

Runs the **real** LangGraph pipeline end-to-end against every row of
`dataset/sample_claims.csv`, but with all LLM/VLM API calls mocked via pytest
fixtures (a ground-truth *oracle* vision mock), so the whole suite finishes in
~2 s with no network and no cost. It uses `pytest.mark.parametrize` to iterate
the dataset and computes strict, exact-match **answer-correctness** metrics plus
set-based **retrieval precision / recall / F1** for `supporting_image_ids` and
`risk_flags`. It also covers the semantic guardrail (injection, RAG-context
poisoning, hybrid-LLM block, fail-open, schema strictness).

At the end it writes `code/evaluation/pytest_eval_report.md` — a statistical
breakdown with per-field accuracy + error variance, precision/recall
distributions (mean/std/quartiles), and an enumerated list of edge-case
failures — and enforces aggregate quality floors as regression guards. Because
the vision layer is a faithful oracle, any divergence is attributable to the
pipeline's own deterministic post-processing (severity map, evidence gating,
history-derived flags), which the report makes explicit.

### 4.2 Offline scorer (`code/evaluation/report.py`) — scores a real run

```bash
python code/evaluation/report.py
```

Joins a previously generated `output.csv` against `dataset/sample_claims.csv` on
`user_id` and writes `code/evaluation/evaluation_report.md` with per-field
accuracy plus an operational analysis (model calls, tokens, cost, latency,
RPM/TPM) and a projection to the full test set. It auto-reads
`output.run_stats.json` for the real model/runtime; `--model`,
`--runtime-seconds`, and `--test-rows` override it.

---

## 5. Key design decisions

- **Structured output over free text.** Both providers are forced to emit a
  typed object; enum constraints + a `_safe_enum` normaliser guarantee every
  field is a legal value, so the output CSV never contains hallucinated labels.
- **Deterministic severity.** Ground-truth severity is tightly correlated with
  `issue_type`, so severity is mapped from the predicted issue type
  (`_SEVERITY_BY_ISSUE`) rather than asked of the model — this removed a major
  source of variance.
- **`glass_shatter → crack` remap.** The label set distinguishes them but the
  ground truth labels all glass/screen cracking as `crack`; a post-processing
  remap aligns with that convention.
- **Cost / latency awareness.** One VLM call per claim (no redundant calls),
  images downscaled to ≤1568px before upload, and a fast-fail path that avoids
  full reasoning on unusable evidence. See the operational analysis in the
  evaluation report for numbers.

---

## 6. Files

```text
code/
├── main.py              # pipeline: models, guardrail, ingestion, graph, loop
├── requirements.txt     # pinned dependencies (incl. pytest for the suite)
├── README.md            # this file
└── evaluation/
    ├── main.py                 # pytest suite (mocked, offline) + stats report
    ├── report.py               # offline scorer for a real output.csv
    ├── pytest.ini              # lets pytest collect main.py
    ├── pytest_eval_report.md   # generated: statistical breakdown (suite)
    └── evaluation_report.md    # generated: accuracy + ops analysis (report.py)
```
