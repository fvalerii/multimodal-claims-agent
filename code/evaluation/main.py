"""
Evaluation script for the HackerRank Orchestrate pipeline.

Usage:
    python code/evaluation/main.py [--runtime-seconds N] [--model NAME] [--test-rows N]

Reads:
    dataset/sample_claims.csv   — ground-truth labels
    output.csv                  — pipeline predictions (run code/main.py first)
    output.run_stats.json       — optional run-stats sidecar written by main.py
                                  (model + runtime auto-detected from it)

Settings precedence: explicit CLI arg > run-stats sidecar > built-in default.

Writes:
    code/evaluation/evaluation_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT         = Path(__file__).parent.parent.parent
GROUND_TRUTH_PATH = REPO_ROOT / "dataset" / "sample_claims.csv"
PREDICTIONS_PATH  = REPO_ROOT / "output.csv"
RUN_STATS_PATH    = PREDICTIONS_PATH.with_name(PREDICTIONS_PATH.stem + ".run_stats.json")
REPORT_PATH       = Path(__file__).parent / "evaluation_report.md"

# ---------------------------------------------------------------------------
# Pricing & token assumptions
# ---------------------------------------------------------------------------

#: USD per 1M tokens, keyed by model. Matches the providers main.py can use.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5":         {"input": 3.00,  "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80,  "output": 4.00},
    "gemini-2.5-flash":          {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash":          {"input": 0.075, "output": 0.30},
}

#: Default must match main.py's default ANTHROPIC_MODEL.
DEFAULT_MODEL = "claude-sonnet-4-5"

# Token assumptions (documented approximations used for the cost estimate).
#   - Anthropic image tokens ≈ (width × height) / 750; images are downscaled to
#     ≤1568px in main.py, giving ~1,500 tokens for a typical image.
#   - The text prompt now carries base instructions + full enum lists + the
#     evidence-standard text + the few-shot block (~1,800 tokens).
#   - Output is a structured tool-use JSON object plus a short justification.
TOKENS_PER_IMAGE         = 1500
TEXT_INPUT_TOKENS        = 1800
OUTPUT_TOKENS_PER_ROW    = 250

#: Number of rows in the final test set (dataset/claims.csv). Auto-detected at
#: runtime when the file is present; this is only the fallback.
DEFAULT_TEST_ROWS = 45

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def load_data(ground_truth_path: Path, predictions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs, normalise dtypes and enum-like columns, return (ground_truth, predictions)."""
    gt   = pd.read_csv(ground_truth_path, dtype=str).fillna("")
    pred = pd.read_csv(predictions_path,  dtype=str).fillna("")

    # Normalise columns that hold enum values so comparison is case- and
    # whitespace-insensitive (guards against "Supported" vs "supported", etc.).
    _ENUM_COLS = ["claim_status", "issue_type", "severity", "claim_object",
                  "valid_image", "evidence_standard_met", "object_part"]
    for col in _ENUM_COLS:
        for df in (gt, pred):
            if col in df.columns:
                df[col] = df[col].str.strip().str.lower()

    return gt, pred


_REQUIRED_GT_COLS   = {"user_id", "claim_status", "issue_type"}
_REQUIRED_PRED_COLS = {"user_id", "claim_status", "issue_type", "image_paths"}


def validate_columns(gt: pd.DataFrame, pred: pd.DataFrame) -> None:
    """
    Raise ``ValueError`` if either DataFrame is missing columns required for
    accuracy or operational analysis, so errors surface before any computation.
    """
    missing_gt   = _REQUIRED_GT_COLS   - set(gt.columns)
    missing_pred = _REQUIRED_PRED_COLS - set(pred.columns)
    errors: list[str] = []
    if missing_gt:
        errors.append(f"Ground-truth CSV is missing columns: {sorted(missing_gt)}")
    if missing_pred:
        errors.append(f"Predictions CSV is missing columns: {sorted(missing_pred)}")
    if errors:
        raise ValueError("\n".join(errors))


def join_on_user_id(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join ground truth and predictions on ``user_id``.
    Suffixes: ``_gt`` for ground truth, ``_pred`` for predictions.

    Warns if predictions contain user_ids not present in the ground truth
    (extra rows that cannot be evaluated) or if predictions are duplicated.
    """
    gt_ids   = set(gt["user_id"])
    pred_ids = pred["user_id"]

    extra = set(pred_ids) - gt_ids
    if extra:
        print(
            f"[WARNING] {len(extra)} prediction user_id(s) not in ground truth "
            f"— they will be excluded from accuracy metrics: {sorted(extra)}",
            file=sys.stderr,
        )

    dupes = pred_ids[pred_ids.duplicated()].unique()
    if len(dupes):
        print(
            f"[WARNING] {len(dupes)} duplicate user_id(s) found in predictions "
            f"— this may inflate or skew metrics: {sorted(dupes)}",
            file=sys.stderr,
        )

    merged = gt.merge(pred, on="user_id", suffixes=("_gt", "_pred"), how="inner")
    return merged


def accuracy(merged: pd.DataFrame, column: str) -> dict:
    """
    Compute exact-match accuracy for *column* between ground truth and
    predictions.  Returns a dict with total, correct, accuracy_pct, and a
    DataFrame of mismatches.

    Both sides are already lower-cased by ``load_data``; comparison is exact
    after that normalisation.
    """
    gt_col   = f"{column}_gt"
    pred_col = f"{column}_pred"

    if gt_col not in merged.columns or pred_col not in merged.columns:
        return {"total": 0, "correct": 0, "accuracy_pct": 0.0, "mismatches": pd.DataFrame()}

    total   = len(merged)
    correct = (merged[gt_col].str.strip() == merged[pred_col].str.strip()).sum()
    pct     = round(correct / total * 100, 2) if total else 0.0

    mismatches = merged.loc[
        merged[gt_col].str.strip() != merged[pred_col].str.strip(),
        ["user_id", gt_col, pred_col],
    ].rename(columns={gt_col: "expected", pred_col: "predicted"})

    return {
        "total":        total,
        "correct":      int(correct),
        "accuracy_pct": pct,
        "mismatches":   mismatches,
    }


# ---------------------------------------------------------------------------
# Operational analysis helpers
# ---------------------------------------------------------------------------


def count_images(pred: pd.DataFrame) -> int:
    """Count total image paths across all rows (semicolon-separated)."""
    return pred["image_paths"].apply(
        lambda x: len([p for p in x.split(";") if p.strip()])
    ).sum()


def _token_cost(rows: int, images: int, price_in: float, price_out: float) -> dict:
    """Estimate token usage and cost for a given row/image volume."""
    input_tokens  = int(images * TOKENS_PER_IMAGE + rows * TEXT_INPUT_TOKENS)
    output_tokens = int(rows * OUTPUT_TOKENS_PER_ROW)
    total_tokens  = input_tokens + output_tokens
    input_cost    = input_tokens  / 1_000_000 * price_in
    output_cost   = output_tokens / 1_000_000 * price_out
    return {
        "rows":           rows,
        "images":         images,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "total_tokens":   total_tokens,
        "input_cost_usd":  round(input_cost,  6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd":  round(input_cost + output_cost, 6),
    }


def operational_stats(
    pred: pd.DataFrame,
    runtime_seconds: float,
    model: str = DEFAULT_MODEL,
    test_rows: int = DEFAULT_TEST_ROWS,
) -> dict:
    """
    Derive token usage, cost, throughput, and a full-test-set projection.

    Args:
        pred:            Predictions DataFrame (the processed sample).
        runtime_seconds: Wall-clock seconds the pipeline took to run.
        model:           Model name used, to pick the right pricing.
        test_rows:       Number of rows in the final test set (claims.csv).
    """
    price = PRICING.get(model, PRICING[DEFAULT_MODEL])
    price_in, price_out = price["input"], price["output"]

    total_rows   = len(pred)
    total_images = int(count_images(pred))
    avg_images   = total_images / max(total_rows, 1)

    # Observed (the sample that was actually processed)
    observed = _token_cost(total_rows, total_images, price_in, price_out)

    # Projection to the full test set, scaling images by the observed average
    projected_images = int(round(avg_images * test_rows))
    projected = _token_cost(test_rows, projected_images, price_in, price_out)

    # Throughput / latency
    runtime_minutes = runtime_seconds / 60
    rpm = round(total_rows            / runtime_minutes, 2) if runtime_minutes else 0.0
    tpm = round(observed["total_tokens"] / runtime_minutes, 2) if runtime_minutes else 0.0
    sec_per_row = round(runtime_seconds / total_rows, 2) if total_rows else 0.0
    projected_runtime_min = round(sec_per_row * test_rows / 60, 1)

    return {
        "model":            model,
        "price_in":         price_in,
        "price_out":        price_out,
        # observed sample
        "total_rows":           total_rows,
        "total_images":         total_images,
        "avg_images_per_row":   round(avg_images, 2),
        "total_input_tokens":   observed["input_tokens"],
        "total_output_tokens":  observed["output_tokens"],
        "total_tokens":         observed["total_tokens"],
        "input_cost_usd":       observed["input_cost_usd"],
        "output_cost_usd":      observed["output_cost_usd"],
        "total_cost_usd":       observed["total_cost_usd"],
        # projected full test set
        "projected": projected,
        # throughput
        "runtime_seconds":      runtime_seconds,
        "sec_per_row":          sec_per_row,
        "projected_runtime_min": projected_runtime_min,
        "rpm":                  rpm,
        "tpm":                  tpm,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _escape_md(text: str) -> str:
    """Escape characters that would break a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", "")


def _mismatch_table(mismatches: pd.DataFrame) -> str:
    """Render a mismatch DataFrame as a Markdown table, or a short message."""
    if mismatches.empty:
        return "_No mismatches — perfect score on this metric._\n"
    rows = ["| user_id | expected | predicted |",
            "|---------|----------|-----------|"]
    for _, row in mismatches.iterrows():
        rows.append(
            f"| {_escape_md(row['user_id'])} "
            f"| {_escape_md(row['expected'])} "
            f"| {_escape_md(row['predicted'])} |"
        )
    return "\n".join(rows) + "\n"


#: Display order and human-readable titles for each accuracy metric.
_METRIC_TITLES: list[tuple[str, str]] = [
    ("claim_status",          "Claim Status Accuracy"),
    ("issue_type",            "Issue Type Accuracy"),
    ("object_part",           "Object Part Accuracy"),
    ("evidence_standard_met", "Evidence Standard Met Accuracy"),
    ("valid_image",           "Valid Image Accuracy"),
    ("severity",              "Severity Accuracy"),
]


def _accuracy_section(metrics: dict[str, dict]) -> str:
    """Render the full accuracy section (summary table + per-metric details)."""
    # Summary table across all metrics
    lines = [
        "### 1.1 Summary",
        "",
        "| Metric | Correct | Total | Accuracy |",
        "|--------|---------|-------|----------|",
    ]
    for key, title in _METRIC_TITLES:
        m = metrics.get(key)
        if m is None:
            continue
        lines.append(
            f"| `{key}` | {m['correct']} | {m['total']} | **{m['accuracy_pct']} %** |"
        )

    # Per-metric mismatch detail
    detail: list[str] = []
    for idx, (key, title) in enumerate(_METRIC_TITLES, start=2):
        m = metrics.get(key)
        if m is None:
            continue
        detail.append(f"\n### 1.{idx} {title}\n")
        detail.append(f"**Accuracy: {m['accuracy_pct']} %**  ({m['correct']}/{m['total']})\n")
        detail.append("#### Mismatches\n")
        detail.append(_mismatch_table(m["mismatches"]))

    return "\n".join(lines) + "\n" + "\n".join(detail)


def write_report(
    report_path: Path,
    metrics: dict[str, dict],
    ops: dict,
) -> None:
    """Write the full evaluation report to *report_path* as Markdown.

    Args:
        metrics: Mapping of column name → accuracy dict (from ``accuracy()``).
        ops:     Operational-stats dict.
    """
    cs = metrics["claim_status"]
    it = metrics["issue_type"]
    sev = metrics.get("severity", {"accuracy_pct": 0.0})
    pr = ops["projected"]
    accuracy_section = _accuracy_section(metrics)

    report = f"""\
# Evaluation Report — Multi-Modal Evidence Review Pipeline

---

## 1. Accuracy Metrics

> Evaluated on **{cs['total']} matched rows** (inner join of `sample_claims.csv` ↔ `output.csv` on `user_id`).

{accuracy_section}

---

## 2. Operational Analysis

> Model: **{ops['model']}** · Pricing: **${ops['price_in']}/1M input · ${ops['price_out']}/1M output**.
> Token assumptions (approximate): {TOKENS_PER_IMAGE:,} tokens/image (Anthropic ≈ w×h/750, images downscaled to ≤1568px) · {TEXT_INPUT_TOKENS:,} text input tokens/prompt (instructions + enums + evidence standards + few-shot) · {OUTPUT_TOKENS_PER_ROW} output tokens/row.

### 2.1 Volume

| Metric | Sample (processed) | Full test set (projected) |
|--------|--------------------|---------------------------|
| Model calls (1 per row) | {ops['total_rows']} | {pr['rows']} |
| Images processed | {ops['total_images']} | {pr['images']} |
| Avg images / row | {ops['avg_images_per_row']} | {ops['avg_images_per_row']} |

### 2.2 Token Usage (estimated)

| Token type | Sample | Full test set (projected) |
|------------|--------|---------------------------|
| Input tokens | {ops['total_input_tokens']:,} | {pr['input_tokens']:,} |
| Output tokens | {ops['total_output_tokens']:,} | {pr['output_tokens']:,} |
| **Total tokens** | **{ops['total_tokens']:,}** | **{pr['total_tokens']:,}** |

### 2.3 Approximate Cost (USD)

| Component | Sample | Full test set (projected) |
|-----------|--------|---------------------------|
| Input | ${ops['input_cost_usd']:.4f} | ${pr['input_cost_usd']:.4f} |
| Output | ${ops['output_cost_usd']:.4f} | ${pr['output_cost_usd']:.4f} |
| **Total** | **${ops['total_cost_usd']:.4f}** | **${pr['total_cost_usd']:.4f}** |

### 2.4 Throughput & Latency (sample runtime = {ops['runtime_seconds']:.0f} s)

| Metric | Value |
|--------|-------|
| Latency / row | {ops['sec_per_row']} s |
| Requests Per Minute (RPM) | {ops['rpm']} |
| Tokens Per Minute (TPM) | {ops['tpm']:,} |
| Projected full-test runtime | ~{ops['projected_runtime_min']} min |

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
| {ops['model']} (current) | {cs['accuracy_pct']} % | {it['accuracy_pct']} % | {sev['accuracy_pct']} % | ${pr['total_cost_usd']:.4f} | {ops['sec_per_row']} s | final choice |
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
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written → {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_run_stats() -> dict:
    """Load the run-stats sidecar written by main.py, if present.

    Returns an empty dict when the file is missing or unreadable, so CLI
    arguments / defaults remain the source of truth.
    """
    if not RUN_STATS_PATH.exists():
        return {}
    try:
        with open(RUN_STATS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Could not read run-stats sidecar: {exc}", file=sys.stderr)
        return {}


def _detect_test_rows() -> int:
    """Count data rows in dataset/claims.csv, falling back to the default."""
    claims_path = REPO_ROOT / "dataset" / "claims.csv"
    if claims_path.exists():
        try:
            with open(claims_path, encoding="utf-8") as f:
                return max(sum(1 for _ in f) - 1, 0)  # minus header
        except OSError:
            pass
    return DEFAULT_TEST_ROWS


def main(
    runtime_seconds: float | None = None,
    model: str | None = None,
    test_rows: int | None = None,
) -> None:
    # Precedence for each setting: explicit CLI arg > run-stats sidecar > default.
    stats = load_run_stats()
    if stats:
        print(f"Using run-stats sidecar: {RUN_STATS_PATH}")

    if runtime_seconds is None:
        runtime_seconds = float(stats.get("runtime_seconds", 60.0))
    if model is None:
        model = stats.get("model", DEFAULT_MODEL)
    if test_rows is None:
        test_rows = _detect_test_rows()

    # ---- load data ----------------------------------------------------------
    if not GROUND_TRUTH_PATH.exists():
        print(f"[ERROR] Ground-truth file not found: {GROUND_TRUTH_PATH}", file=sys.stderr)
        sys.exit(1)

    if not PREDICTIONS_PATH.exists():
        print(
            f"[ERROR] Predictions file not found: {PREDICTIONS_PATH}\n"
            "        Run `python code/main.py` first to generate output.csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    gt, pred = load_data(GROUND_TRUTH_PATH, PREDICTIONS_PATH)

    if pred.empty:
        print("[ERROR] output.csv is empty — run the pipeline first.", file=sys.stderr)
        sys.exit(1)

    try:
        validate_columns(gt, pred)
    except ValueError as exc:
        print(f"[ERROR] Column validation failed:\n{exc}", file=sys.stderr)
        sys.exit(1)

    # ---- accuracy -----------------------------------------------------------
    merged = join_on_user_id(gt, pred)

    if merged.empty:
        print(
            "[WARNING] No rows matched between ground truth and predictions on user_id. "
            "Accuracy metrics will be zero.",
            file=sys.stderr,
        )

    metrics = {key: accuracy(merged, key) for key, _ in _METRIC_TITLES}

    # ---- operational stats --------------------------------------------------
    ops = operational_stats(pred, runtime_seconds, model=model, test_rows=test_rows)

    # ---- console summary ----------------------------------------------------
    print(f"\n{'='*60}")
    for key, _ in _METRIC_TITLES:
        m = metrics[key]
        print(f"  {key:<22} accuracy : {m['accuracy_pct']:>5} %  ({m['correct']}/{m['total']})")
    print(f"  {'-'*48}")
    print(f"  model                  : {ops['model']}")
    print(f"  rows processed         : {ops['total_rows']}  (images: {ops['total_images']})")
    print(f"  sample cost (USD)      : ${ops['total_cost_usd']:.4f}")
    print(f"  projected test cost    : ${ops['projected']['total_cost_usd']:.4f}  ({ops['projected']['rows']} rows)")
    print(f"  latency / RPM / TPM    : {ops['sec_per_row']}s / {ops['rpm']} / {ops['tpm']:,}")
    print(f"{'='*60}\n")

    # ---- write report -------------------------------------------------------
    write_report(REPORT_PATH, metrics, ops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the claims pipeline.")
    parser.add_argument(
        "--runtime-seconds",
        type=float,
        default=None,
        help="Wall-clock seconds the pipeline took to run (used for RPM/TPM). "
             "Default: read from output.run_stats.json, else 60.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model used by the pipeline, selects pricing. Known: {', '.join(PRICING)}. "
             f"Default: read from output.run_stats.json, else {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--test-rows",
        type=int,
        default=None,
        help="Rows in the full test set for cost projection. Default: auto-detect from "
             "dataset/claims.csv.",
    )
    args = parser.parse_args()
    main(runtime_seconds=args.runtime_seconds, model=args.model, test_rows=args.test_rows)
