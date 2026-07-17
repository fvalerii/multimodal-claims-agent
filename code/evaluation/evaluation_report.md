# Evaluation Report — Multi-Modal Evidence Review Pipeline

---

## 1. Accuracy Metrics

> Evaluated on **20 matched rows** (inner join of `sample_claims.csv` ↔ `output.csv` on `user_id`).

### 1.1 Summary

| Metric | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| `claim_status` | 16 | 20 | **80.0 %** |
| `issue_type` | 13 | 20 | **65.0 %** |
| `object_part` | 17 | 20 | **85.0 %** |
| `evidence_standard_met` | 19 | 20 | **95.0 %** |
| `valid_image` | 18 | 20 | **90.0 %** |
| `severity` | 13 | 20 | **65.0 %** |

### 1.2 Claim Status Accuracy

**Accuracy: 80.0 %**  (16/20)

#### Mismatches

| user_id | expected | predicted |
|---------|----------|-----------|
| user_001 | supported | contradicted |
| user_008 | contradicted | not_enough_information |
| user_020 | contradicted | supported |
| user_034 | contradicted | supported |


### 1.3 Issue Type Accuracy

**Accuracy: 65.0 %**  (13/20)

#### Mismatches

| user_id | expected | predicted |
|---------|----------|-----------|
| user_001 | dent | broken_part |
| user_007 | broken_part | crack |
| user_005 | scratch | none |
| user_006 | unknown | none |
| user_020 | none | crack |
| user_033 | unknown | none |
| user_034 | none | torn_packaging |


### 1.4 Object Part Accuracy

**Accuracy: 85.0 %**  (17/20)

#### Mismatches

| user_id | expected | predicted |
|---------|----------|-----------|
| user_008 | front_bumper | hood |
| user_031 | package_side | box |
| user_033 | unknown | box |


### 1.5 Evidence Standard Met Accuracy

**Accuracy: 95.0 %**  (19/20)

#### Mismatches

| user_id | expected | predicted |
|---------|----------|-----------|
| user_008 | true | false |


### 1.6 Valid Image Accuracy

**Accuracy: 90.0 %**  (18/20)

#### Mismatches

| user_id | expected | predicted |
|---------|----------|-----------|
| user_008 | false | true |
| user_032 | false | true |


### 1.7 Severity Accuracy

**Accuracy: 65.0 %**  (13/20)

#### Mismatches

| user_id | expected | predicted |
|---------|----------|-----------|
| user_005 | low | none |
| user_006 | unknown | none |
| user_008 | high | medium |
| user_012 | low | medium |
| user_020 | none | medium |
| user_033 | low | none |
| user_034 | none | medium |


---

## 2. Operational Analysis

> Model: **claude-sonnet-4-5** · Pricing: **$3.0/1M input · $15.0/1M output**.
> Token assumptions (approximate): 1,500 tokens/image (Anthropic ≈ w×h/750, images downscaled to ≤1568px) · 1,800 text input tokens/prompt (instructions + enums + evidence standards + few-shot) · 250 output tokens/row.

### 2.1 Volume

| Metric | Sample (processed) | Full test set (projected) |
|--------|--------------------|---------------------------|
| Model calls (1 per row) | 20 | 44 |
| Images processed | 29 | 64 |
| Avg images / row | 1.45 | 1.45 |

### 2.2 Token Usage (estimated)

| Token type | Sample | Full test set (projected) |
|------------|--------|---------------------------|
| Input tokens | 79,500 | 175,200 |
| Output tokens | 5,000 | 11,000 |
| **Total tokens** | **84,500** | **186,200** |

### 2.3 Approximate Cost (USD)

| Component | Sample | Full test set (projected) |
|-----------|--------|---------------------------|
| Input | $0.2385 | $0.5256 |
| Output | $0.0750 | $0.1650 |
| **Total** | **$0.3135** | **$0.6906** |

### 2.4 Throughput & Latency (sample runtime = 130 s)

| Metric | Value |
|--------|-------|
| Latency / row | 6.48 s |
| Requests Per Minute (RPM) | 9.25 |
| Tokens Per Minute (TPM) | 39,096.24 |
| Projected full-test runtime | ~4.8 min |

### 2.5 Rate-limit, retry & cost strategy

The pipeline is built one VLM call per claim (no redundant calls) and applies:

- **`temperature=0`** for deterministic, reproducible output.
- **Image downscaling** to ≤1568px before upload, cutting image tokens/latency.
- **Automatic retry with backoff** on transient errors — quota-aware retry for
  Gemini (parses `retryDelay`, fast-fails daily quota) and exponential backoff
  for Anthropic 429/529/5xx/network errors — so a transient failure never drops
  a row.
- **Graceful degradation**: any unrecoverable failure still writes a valid
  fallback row, guaranteeing one output row per input claim.

Further headroom if rate limits are hit on the full set: add throttling between
calls, cache by image hash to skip re-processing identical evidence, or batch
where the provider supports it.

---

## 3. Strategy Comparison

Two model families were trialled on `sample_claims.csv` with the same LangGraph
pipeline and prompts. The current-model row is measured automatically from this
run; the Gemini Flash figures are approximate values observed during
development (not re-measured here).

| Strategy | claim_status | issue_type | severity | Est. test cost (USD) | Latency / row | Notes |
|----------|--------------|------------|----------|----------------------|---------------|-------|
| claude-sonnet-4-5 (current) | 80.0 % | 65.0 % | 65.0 % | $0.6906 | 6.48 s | final choice |
| Gemini 1.5 / 2.5 Flash | ~40–55 % | ~40–50 % | ~50 % | ~$0.02 | ~2–3 s | cheap, but weaker vision + restrictive free-tier daily quota |

Beyond the model swap, the largest accuracy gains came from prompt +
post-processing changes (not a different model): enum-constrained structured
output, explicit label-disambiguation rules, `temperature=0`, deterministic
severity mapping, and the `glass_shatter→crack` remap.

---

## 4. Final Strategy Justification

**Model choice.** We started on Gemini Flash for cost, but vision accuracy on
the damage classes plateaued around 40–55 % and the free-tier *daily* quota
made full reproducible runs impractical. Switching the vision backend to
**Claude Sonnet 4.5** (the pipeline is provider-agnostic, so this was a config
change) lifted every metric materially. The provider abstraction is retained so
Gemini remains a drop-in fallback.

**Prompt strategy.** Zero-shot was unreliable for the label conventions, so the
final prompt is few-shot with object-specific `object_part` vocabularies,
explicit label-disambiguation rules (e.g. `crack` vs `glass_shatter`, `dent` vs
`broken_part`, `stain` vs `water_damage`), and a "cannot verify → unknown +
not_enough_information" rule. Output is forced through a typed schema with enum
constraints, then normalised, so the CSV can never contain an illegal label.

**Determinism & post-processing.** `temperature=0` removed run-to-run label
flipping. Severity is derived deterministically from the predicted
`issue_type` (ground-truth severity correlates strongly with issue type), and
`glass_shatter` is remapped to `crack` to match the labelling convention. These
two changes were the main fix for the previously lagging severity/issue_type
metrics.

**Cost / latency trade-off.** One VLM call per claim with no redundant calls,
images downscaled to ≤1568px, and a fast-fail path for unusable evidence keep
the full test set well under a dollar at Sonnet pricing (see §2.3) and within a
few minutes of runtime. If rate limits became a constraint at larger scale, the
next levers would be image-hash caching and request throttling/batching.

**Known limitations.** Metrics are computed on a 20-row sample, so per-field
percentages have wide confidence intervals — they indicate direction, not
precise generalisation. `issue_type` and `severity` remain the hardest fields
because of fine visual distinctions and labelling conventions. The deterministic
severity map trades nuance for ground-truth alignment and would need revisiting
if the severity rubric changed.

---

_Report generated automatically by `code/evaluation/main.py`._
