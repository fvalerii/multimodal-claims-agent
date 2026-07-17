# Multi-Modal Evidence Review — Solution

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
  │  route_by_object  (reads claim.claim_object)
  ├── car      → evaluate_car_node ┐
  ├── laptop   → evaluate_laptop_node ├─ (shared _evaluate_claim_images)
  └── package  → evaluate_package_node ┘
                        │  route_post_evaluation (findings["valid_image"])
                        ├── valid   → posterior_risk_node ── END
                        └── invalid → fast_fail_node ────── END
```

- **Routing** (`route_by_object`) sends each claim to the object-specific
  vision node so prompts can use the correct `object_part` vocabulary.
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

```bash
python code/evaluation/main.py
```

Joins `output.csv` against `dataset/sample_claims.csv` on `user_id` and writes
`code/evaluation/evaluation_report.md` with per-field accuracy (claim_status,
issue_type, object_part, evidence_standard_met, valid_image, severity) plus an
operational analysis (model calls, tokens, cost, latency, RPM/TPM) including a
projection to the full test set. The evaluator auto-reads
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
├── main.py              # pipeline: models, ingestion, graph, production loop
├── requirements.txt     # pinned dependencies
├── README.md            # this file
└── evaluation/
    ├── main.py              # accuracy + operational analysis
    └── evaluation_report.md # generated report
```
