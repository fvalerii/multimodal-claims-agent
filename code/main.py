"""
HackerRank Orchestrate — Multi-Modal Evidence Review
Entry point: reads dataset/claims.csv and writes output.csv.
"""

from __future__ import annotations

import base64
import enum
import io
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Generator, Literal, Optional, TypedDict

import anthropic as _anthropic_sdk
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from PIL import Image
from pydantic import BaseModel
from langgraph.graph import END, START, StateGraph

# Load .env before any code that reads env vars (genai client, etc.).
# override=False lets shell-level secrets (e.g. CI/CD) take precedence
# over values in the local .env file.
load_dotenv(override=False)

_has_any_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")
if not _has_any_key:
    print("[ERROR] No API key found. Set ANTHROPIC_API_KEY or GEMINI_API_KEY.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core categorical enums (strict allowed values per problem_statement.md)
# ---------------------------------------------------------------------------


class ClaimObject(str, enum.Enum):
    car = "car"
    laptop = "laptop"
    package = "package"


class ClaimStatus(str, enum.Enum):
    supported = "supported"
    contradicted = "contradicted"
    not_enough_information = "not_enough_information"


class Severity(str, enum.Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    unknown = "unknown"


class IssueType(str, enum.Enum):
    dent = "dent"
    scratch = "scratch"
    crack = "crack"
    glass_shatter = "glass_shatter"
    broken_part = "broken_part"
    missing_part = "missing_part"
    torn_packaging = "torn_packaging"
    crushed_packaging = "crushed_packaging"
    water_damage = "water_damage"
    stain = "stain"
    none = "none"
    unknown = "unknown"


#: Deterministic severity per issue type. Ground-truth severity tracks the
#: issue type closely: cosmetic marks are "low", real/functional damage is
#: "medium", and "high" is reserved for catastrophic structural loss. Used by
#: posterior_risk_node so severity stays consistent and aligned with the labels.
_SEVERITY_BY_ISSUE: dict[str, str] = {
    "none": "none",
    "unknown": "unknown",
    "scratch": "low",
    "dent": "medium",
    "crack": "medium",
    "glass_shatter": "high",
    "broken_part": "medium",
    "missing_part": "medium",
    "torn_packaging": "medium",
    "crushed_packaging": "medium",
    "water_damage": "medium",
    "stain": "medium",
}


class CarObjectPart(str, enum.Enum):
    front_bumper = "front_bumper"
    rear_bumper = "rear_bumper"
    door = "door"
    hood = "hood"
    windshield = "windshield"
    side_mirror = "side_mirror"
    headlight = "headlight"
    taillight = "taillight"
    fender = "fender"
    quarter_panel = "quarter_panel"
    body = "body"
    unknown = "unknown"


class LaptopObjectPart(str, enum.Enum):
    screen = "screen"
    keyboard = "keyboard"
    trackpad = "trackpad"
    hinge = "hinge"
    lid = "lid"
    corner = "corner"
    port = "port"
    base = "base"
    body = "body"
    unknown = "unknown"


class PackageObjectPart(str, enum.Enum):
    box = "box"
    package_corner = "package_corner"
    package_side = "package_side"
    seal = "seal"
    label = "label"
    contents = "contents"
    item = "item"
    unknown = "unknown"


class RiskFlag(str, enum.Enum):
    none = "none"
    blurry_image = "blurry_image"
    cropped_or_obstructed = "cropped_or_obstructed"
    low_light_or_glare = "low_light_or_glare"
    wrong_angle = "wrong_angle"
    wrong_object = "wrong_object"
    wrong_object_part = "wrong_object_part"
    damage_not_visible = "damage_not_visible"
    claim_mismatch = "claim_mismatch"
    possible_manipulation = "possible_manipulation"
    non_original_image = "non_original_image"
    text_instruction_present = "text_instruction_present"
    user_history_risk = "user_history_risk"
    manual_review_required = "manual_review_required"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class ClaimInput(BaseModel):
    """One row from claims.csv or sample_claims.csv (input columns only)."""

    user_id: str
    image_paths: str  # semicolon-separated paths, e.g. "images/test/case_001/img_1.jpg;..."
    user_claim: str
    claim_object: ClaimObject

    @property
    def image_path_list(self) -> list[str]:
        return [p.strip() for p in self.image_paths.split(";") if p.strip()]

    @property
    def image_id_list(self) -> list[str]:
        """Filename without extension for each path."""
        import os
        return [os.path.splitext(os.path.basename(p))[0] for p in self.image_path_list]


class UserHistory(BaseModel):
    """One row from user_history.csv."""

    user_id: str
    past_claim_count: int = 0
    accept_claim: int = 0
    manual_review_claim: int = 0
    rejected_claim: int = 0
    last_90_days_claim_count: int = 0
    history_flags: Optional[str] = None
    history_summary: Optional[str] = None


class EvidenceRequirement(BaseModel):
    """One row from evidence_requirements.csv."""

    requirement_id: str
    claim_object: str  # "car", "laptop", "package", or "all"
    applies_to: str    # issue family, e.g. "dent or scratch"
    minimum_image_evidence: str


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class ClaimOutput(BaseModel):
    """
    One row written to output.csv.
    Column order matches the required output schema exactly.
    """

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: ClaimObject

    evidence_standard_met: bool
    evidence_standard_met_reason: str

    # Semicolon-separated RiskFlag values, or "none"
    risk_flags: str = "none"

    issue_type: IssueType
    object_part: str  # one of Car/Laptop/PackageObjectPart — kept str for cross-object flexibility

    claim_status: ClaimStatus
    claim_status_justification: str

    # Semicolon-separated image IDs, or "none"
    supporting_image_ids: str = "none"

    valid_image: bool
    severity: Severity

    def to_csv_row(self) -> dict[str, str]:
        """Serialise to a flat dict of strings ready for csv.DictWriter."""
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object.value,
            "evidence_standard_met": str(self.evidence_standard_met).lower(),
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": self.risk_flags,
            "issue_type": self.issue_type.value,
            "object_part": self.object_part,
            "claim_status": self.claim_status.value,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": self.supporting_image_ids,
            "valid_image": str(self.valid_image).lower(),
            "severity": self.severity.value,
        }


# Required column order for output.csv
OUTPUT_COLUMNS: list[str] = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


# ---------------------------------------------------------------------------
# Data ingestion helpers
# ---------------------------------------------------------------------------

#: Column → fill value for user_history.csv (all numeric columns default to 0,
#: string columns to "none" so Pydantic Optional[str] fields accept them cleanly).
_HISTORY_FILL: dict[str, object] = {
    "past_claim_count": 0,
    "accept_claim": 0,
    "manual_review_claim": 0,
    "rejected_claim": 0,
    "last_90_days_claim_count": 0,
    "history_flags": "none",
    "history_summary": "none",
}


def _load_dataframes(
    claims_path: str | Path,
    history_path: str | Path,
    requirements_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three CSVs and apply NaN-fill rules so downstream Pydantic
    validation never receives bare ``float('nan')`` values."""
    claims_df = pd.read_csv(claims_path, dtype=str).fillna("")
    history_df = pd.read_csv(history_path, dtype=str).fillna(_HISTORY_FILL)
    requirements_df = pd.read_csv(requirements_path, dtype=str).fillna("")
    return claims_df, history_df, requirements_df


def _build_history_lookup(history_df: pd.DataFrame) -> dict[str, UserHistory]:
    """Build a dict keyed by user_id for O(1) history look-ups.

    Uses ``to_dict(orient="records")`` instead of ``iterrows`` to avoid the
    per-row Series construction overhead. Pydantic coerces the string digits
    produced by ``dtype=str`` loading into the correct ``int`` fields.
    """
    return {
        str(row["user_id"]): UserHistory(**row)
        for row in history_df.to_dict(orient="records")
        if row.get("user_id")
    }


def _build_requirements_list(requirements_df: pd.DataFrame) -> list[EvidenceRequirement]:
    """Parse all evidence-requirement rows into typed objects."""
    return [
        EvidenceRequirement(**row)
        for row in requirements_df.to_dict(orient="records")
        if row.get("requirement_id")
    ]


# ---------------------------------------------------------------------------
# Context joiner — the main generator consumed by the inference layer
# ---------------------------------------------------------------------------

ClaimContext = dict  # {"claim": ClaimInput, "history": UserHistory, "requirements": list[EvidenceRequirement]}


def load_and_join(
    claims_path: str | Path,
    history_path: str | Path,
    requirements_path: str | Path,
) -> Generator[ClaimContext, None, None]:
    """Yield one context dict per claim row, joining history and requirements.

    Each yielded dict has the shape::

        {
            "claim":        ClaimInput,
            "history":      UserHistory,          # default if user not found
            "requirements": list[EvidenceRequirement],  # filtered to this claim_object + "all"
        }

    Args:
        claims_path:       Path to claims.csv (or sample_claims.csv).
        history_path:      Path to user_history.csv.
        requirements_path: Path to evidence_requirements.csv.
    """
    claims_df, history_df, requirements_df = _load_dataframes(
        claims_path, history_path, requirements_path
    )

    history_lookup: dict[str, UserHistory] = _build_history_lookup(history_df)
    all_requirements: list[EvidenceRequirement] = _build_requirements_list(requirements_df)

    for _, row in claims_df.iterrows():
        raw_row = {k: ("" if v is None else str(v)) for k, v in row.to_dict().items()}

        # Per-row parsing is guarded so a single malformed row (e.g. an
        # unrecognised claim_object value or a missing column) yields an error
        # marker instead of crashing the whole run mid-dataset.
        try:
            claim = ClaimInput(
                user_id=row["user_id"],
                image_paths=row["image_paths"],
                user_claim=row["user_claim"],
                claim_object=ClaimObject(row["claim_object"]),
            )
        except Exception as exc:  # noqa: BLE001
            yield {"claim": None, "raw": raw_row, "error": str(exc)}
            continue

        history = history_lookup.get(
            claim.user_id,
            UserHistory(user_id=claim.user_id),  # sensible default for unknown users
        )

        # Normalise both sides to plain str so the comparison is safe whether
        # claim_object is a ClaimObject enum or a bare string on either model.
        claim_obj_str = getattr(claim.claim_object, "value", claim.claim_object)
        relevant_requirements = [
            req for req in all_requirements
            if getattr(req.claim_object, "value", req.claim_object) in (claim_obj_str, "all")
        ]

        yield {
            "claim": claim,
            "history": history,
            "requirements": relevant_requirements,
            "raw": raw_row,
        }


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

#: Base directory that image paths in the CSV are relative to.
#: CSV paths look like "images/sample/case_001/img_1.jpg"; prepend this to get
#: the real filesystem path.
DATASET_DIR: Path = Path(__file__).parent.parent / "dataset"

#: Gemini model used when the Google provider is active.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

#: Claude model used when the Anthropic provider is active.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

# ---------------------------------------------------------------------------
# Provider selection  —  prefer Anthropic when ANTHROPIC_API_KEY is set,
# fall back to Google.  Override with VISION_PROVIDER=google|anthropic.
# ---------------------------------------------------------------------------
_has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
VISION_PROVIDER: str = os.getenv(
    "VISION_PROVIDER",
    "anthropic" if _has_anthropic_key else "google",
)
VISION_MODEL = ANTHROPIC_MODEL if VISION_PROVIDER == "anthropic" else GEMINI_MODEL

# ---------------------------------------------------------------------------
# Retry helper for quota / transient errors
# ---------------------------------------------------------------------------

_RETRY_DELAYS = (12, 25, 50)  # seconds to wait before each retry (minute-level 429s)
_QUOTA_RE = re.compile(r"retryDelay.*?(\d+)s", re.DOTALL)
_DAILY_QUOTA_RE = re.compile(r"PerDay", re.IGNORECASE)


def _call_with_retry(client: genai.Client, model: str, contents: list, config) -> Any:
    """
    Call ``client.models.generate_content`` with automatic retry on 429s.

    Two kinds of quota errors:
    - **Minute-level rate limits** (``retryDelay`` ≤ 120 s): retried up to 3
      times after the server-suggested delay (or the ``_RETRY_DELAYS`` fallback).
    - **Daily quota exhausted** (``quotaId`` contains "PerDay"): no retry is
      possible within the same calendar day; the exception is re-raised
      immediately with a human-readable explanation so the caller can log it
      and move on rather than hanging for minutes.

    Non-quota exceptions are always re-raised immediately.
    """
    last_exc: Exception | None = None
    for attempt, fallback_delay in enumerate((*_RETRY_DELAYS, None)):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            err_str = str(exc)
            is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if not is_quota:
                raise  # non-quota error — propagate immediately

            # Daily quota cannot be recovered by retrying — fail fast
            if _DAILY_QUOTA_RE.search(err_str):
                raise RuntimeError(
                    f"[DAILY QUOTA EXHAUSTED] {model} free-tier daily limit reached. "
                    "Reset happens at midnight Pacific time. "
                    "To continue today: add billing in Google AI Studio or switch to "
                    "a model with a higher free-tier allowance (e.g. gemini-1.5-flash)."
                ) from exc

            last_exc = exc
            # Minute-level rate limit — wait and retry
            match = _QUOTA_RE.search(err_str)
            suggested = int(match.group(1)) if match else 0
            delay = max(suggested + 2, fallback_delay or 0)
            if fallback_delay is None:
                break  # exhausted retries
            print(
                f"\n  [RATE LIMIT 429] attempt {attempt + 1}/3 — sleeping {delay}s …",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# API clients — built once and reused so HTTP connection pools stay warm.
# Creating a fresh client per claim wastes time and prevents keep-alive.
# ---------------------------------------------------------------------------
_anthropic_client: Optional[Any] = None
_gemini_client: Optional[Any] = None


def _get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = _anthropic_sdk.Anthropic()  # reads ANTHROPIC_API_KEY
    return _anthropic_client


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()  # reads GEMINI_API_KEY
    return _gemini_client


# Backoff schedule for retryable Anthropic errors (rate limit, overload, network).
_ANTHROPIC_RETRY_DELAYS = (5, 15, 30)


def _anthropic_messages_with_retry(**kwargs: Any) -> Any:
    """
    Call ``messages.create`` with retry on transient Anthropic failures.

    Retries on rate limits (429), server overload (529), 5xx, and network
    errors using a fixed backoff schedule. Client errors (400/401/403) are
    NOT retryable and propagate immediately to the caller's ``except`` block.
    """
    client = _get_anthropic_client()
    last_exc: Exception | None = None
    for delay in (*_ANTHROPIC_RETRY_DELAYS, None):
        try:
            return client.messages.create(**kwargs)
        except (
            _anthropic_sdk.RateLimitError,
            _anthropic_sdk.InternalServerError,
            _anthropic_sdk.OverloadedError,
            _anthropic_sdk.APIConnectionError,  # includes APITimeoutError
        ) as exc:
            last_exc = exc
            if delay is None:
                break  # retries exhausted
            print(
                f"\n  [ANTHROPIC RETRY] {type(exc).__name__} — sleeping {delay}s …",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Provider-agnostic vision dispatcher
# ---------------------------------------------------------------------------

#: Long edge (px) images are downscaled to before upload. Claude downscales
#: anything larger server-side anyway, so sending smaller images cuts upload
#: time and token cost with no accuracy loss.
_MAX_IMAGE_EDGE = 1568


def _pil_to_base64_jpeg(img: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG string for the Anthropic API."""
    if max(img.size) > _MAX_IMAGE_EDGE:
        img = img.copy()
        img.thumbnail((_MAX_IMAGE_EDGE, _MAX_IMAGE_EDGE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _call_vision_model(
    prompt: str,
    images: list[Image.Image],
    schema_cls: type[_VisionResponseBase],
) -> _VisionResponseBase:
    """
    Route a vision call to either Anthropic (Claude) or Google (Gemini).

    Returns a fully validated Pydantic instance of ``schema_cls``.
    Raises on failure — the caller's ``except`` block handles fallback.
    """
    if VISION_PROVIDER == "anthropic":
        return _call_anthropic(prompt, images, schema_cls)
    return _call_gemini(prompt, images, schema_cls)


def _call_anthropic(
    prompt: str,
    images: list[Image.Image],
    schema_cls: type[_VisionResponseBase],
) -> _VisionResponseBase:
    """
    Call Claude via tool-use to obtain structured JSON output.

    Anthropic's tool-use is the equivalent of Gemini's ``response_schema``:
    the model is forced to call the named tool with the matching JSON schema,
    so the output is always a valid Pydantic instance with no extra parsing.
    """
    # Build content: images first, then the prompt text
    content: list[dict] = []
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _pil_to_base64_jpeg(img),
            },
        })
    content.append({"type": "text", "text": prompt})

    # Build the tool schema from the Pydantic model and inject enum constraints
    # for the fields that have a fixed allowed-values list.  The plain Pydantic
    # schema types these as "string" with no restriction, which lets Claude invent
    # values like "glass_shatter".  Adding an explicit JSON-schema "enum" array
    # forces Claude to pick from the allowed set at the API level — no prompt
    # instruction can achieve the same hard guarantee.
    tool_schema = schema_cls.model_json_schema()
    tool_schema.pop("title", None)
    props = tool_schema.setdefault("properties", {})
    props["issue_type"]   = {"type": "string", "enum": [e.value for e in IssueType]}
    props["claim_status"] = {"type": "string", "enum": [e.value for e in ClaimStatus]}
    props["severity"]     = {"type": "string", "enum": [e.value for e in Severity]}
    props["raw_flags"]    = {
        "type": "array",
        "items": {"type": "string", "enum": [e.value for e in RiskFlag]},
    }
    tool_name = schema_cls.__name__

    response = _anthropic_messages_with_retry(
        model=VISION_MODEL,
        max_tokens=2048,
        temperature=0,  # deterministic/reproducible output (per problem contract)
        tools=[{
            "name": tool_name,
            "description": "Structured damage assessment output for the claims pipeline.",
            "input_schema": tool_schema,
        }],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": content}],
    )

    # Extract and validate the tool-use block
    for block in response.content:
        if block.type == "tool_use":
            return schema_cls(**block.input)

    raise ValueError(f"Claude did not return a tool_use block; stop_reason={response.stop_reason}")


def _call_gemini(
    prompt: str,
    images: list[Image.Image],
    schema_cls: type[_VisionResponseBase],
) -> _VisionResponseBase:
    """Call Gemini with structured-output enforcement and quota-aware retry."""
    client = _get_gemini_client()
    content_parts: list[Any] = [prompt, *images]
    response = _call_with_retry(
        client,
        VISION_MODEL,
        content_parts,
        genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_cls,
            temperature=0,  # deterministic/reproducible output (per problem contract)
        ),
    )
    parsed = response.parsed
    if parsed is None:
        raise ValueError("Gemini response.parsed was None — model returned non-conforming JSON")
    return parsed


#: Shared prompt additions appended to every vision-node prompt.
#: Kept in one place so accuracy tuning is applied uniformly across
#: the car, laptop, and package nodes without risk of drift.
_PROMPT_ADDITIONS = """\
Analyze the image and categorize the damage. Every field in the JSON schema is
required — do NOT omit any key and do NOT invent values outside the allowed lists.

--- CLAIM STATUS RULES (choose exactly one) ---
"supported"              — image clearly shows the exact damage type described
"contradicted"           — object is visible but the claimed damage is absent OR
                           an entirely different damage type is visible instead
"not_enough_information" — image quality too poor / wrong object shown / cannot tell

--- ISSUE TYPE — disambiguation rules (read ALL before choosing) ---

GLASS / SCREEN DAMAGE (windshield, laptop screen, phone screen):
  • ANY crack, chip, spider-web pattern, or shattered glass that is still within
    the frame → "crack". This includes badly shattered windshields and screens.
  • DO NOT use "glass_shatter" for windshields, laptop screens, or phone screens.
    Those are ALWAYS "crack". "glass_shatter" is essentially never correct here.

BODY PANEL vs. STRUCTURAL:
  • Surface deformation/indentation, part still attached and in place → "dent"
  • Linear surface mark, paint scraped, no deformation → "scratch"
  • NEVER upgrade a dent or a scratch to "broken_part". A dent stays "dent" and a
    scratch stays "scratch" no matter how large, unless a component has physically
    snapped off / detached / collapsed.
  • "broken_part" = a component physically detached, snapped, hanging, missing, or
    no longer functioning. This INCLUDES a damaged or knocked-off side_mirror,
    headlight, or taillight, a broken hinge, or a bumper torn away.
  → A bumper dent is "dent". A scratched panel is "scratch". A damaged side
    mirror or light assembly is "broken_part".

LIQUID / MOISTURE:
  • Residue, staining, or discolouration on ELECTRONICS (keyboard, laptop body,
    screen housing) — liquid has soaked in or dried → "stain"
  • Visible wetness or water marks on PACKAGING (cardboard, box, outer wrap)
    → "water_damage"

MISSING CONTENT:
  • A specific part or component is clearly absent in the image → "missing_part"
  • Contents may be missing but cannot be confirmed from the image → "unknown"

NONE vs. UNKNOWN:
  • "none"    — claim is CONTRADICTED: the relevant area IS visible but shows zero
                damage (e.g., trackpad looks intact, seal is unbroken)
  • "unknown" — image is physically unreadable: completely black, extreme blur,
                completely wrong object. Never use for mere label uncertainty.

CONTRADICTED CLAIMS:
  • Different/lesser damage than claimed is visible → issue_type = the type
    ACTUALLY seen, even if minor (e.g., user claims severe damage but image shows
    only a small scratch → issue_type = "scratch", claim_status = "contradicted").
  • Use issue_type = "none" ONLY when truly zero damage of any kind is visible on
    a clearly-shown part (e.g., an intact trackpad, an unbroken seal).
  • Do NOT invent damage that is not visible. If the claimed part looks intact,
    the issue is "none", not a guessed damage type.

CANNOT VERIFY THE CLAIM (use issue_type = "unknown", claim_status = "not_enough_information"):
  • The specific part the claim is about is NOT actually shown in the image set
    (e.g., claim is about the headlight but only the side of the car is visible).
  • The claim is about MISSING items/contents but the image does not clearly show
    the opened package interior / the contents area, so absence cannot be confirmed.
  • The object shown is not clearly the claimed object, or the image is too poor to
    judge the claimed condition.
  In these cases do NOT guess a concrete damage type and do NOT mark the claim
  supported — you simply do not have the evidence to decide.

--- SEVERITY GUIDE ---
  • none    → no damage (e.g. contradicted claim, intact part)
  • low     → minor cosmetic: light scratch, small scuff, tiny dent
  • medium  → clearly visible/functional: crack, moderate dent, stain, torn seal
  • high    → severe/structural: shattering, broken-off part, major collision,
              crushed package, extensive water damage
  • unknown → cannot assess (unusable image)
When claim_status is "not_enough_information", severity is usually "unknown".
When claim_status is "contradicted" with no damage, severity is "none".

--- EVIDENCE_STANDARD_MET GUIDE ---
  • true  → the claimed object/part is visible clearly enough to actually judge
            the claim (even if the verdict turns out to be contradicted)
  • false → image too poor, wrong object, or claimed part not shown — cannot judge

--- FEW-SHOT EXAMPLES (valid JSON only) ---
Windshield cracked in spider-web pattern, matches claim:
{"issue_type":"crack","object_part":"windshield","valid_image":true,"evidence_standard_met":true,"severity":"medium","claim_status":"supported","supporting_image_ids":["img_1"],"raw_flags":[],"vision_justification":"Spider-web crack pattern visible on windshield in img_1."}

Laptop screen cracked / shattered but glass still in frame:
{"issue_type":"crack","object_part":"screen","valid_image":true,"evidence_standard_met":true,"severity":"high","claim_status":"supported","supporting_image_ids":["img_1"],"raw_flags":[],"vision_justification":"Screen shows heavy cracking; glass remains in frame — classified as crack."}

Rear bumper with minor indentation — dent, NOT broken_part:
{"issue_type":"dent","object_part":"rear_bumper","valid_image":true,"evidence_standard_met":true,"severity":"low","claim_status":"supported","supporting_image_ids":["img_1"],"raw_flags":[],"vision_justification":"Small indentation on rear bumper; part is intact and in place."}

Side mirror hanging off car — structural failure → broken_part:
{"issue_type":"broken_part","object_part":"side_mirror","valid_image":true,"evidence_standard_met":true,"severity":"high","claim_status":"supported","supporting_image_ids":["img_1"],"raw_flags":[],"vision_justification":"Mirror housing detached and hanging; structural failure."}

Laptop keyboard with dried liquid residue → stain (not water_damage):
{"issue_type":"stain","object_part":"keyboard","valid_image":true,"evidence_standard_met":true,"severity":"medium","claim_status":"supported","supporting_image_ids":["img_1"],"raw_flags":[],"vision_justification":"Dried liquid staining on keyboard keys in img_1."}

User claims dent but image shows only a surface scratch — contradicted, show visible type:
{"issue_type":"scratch","object_part":"front_bumper","valid_image":true,"evidence_standard_met":true,"severity":"low","claim_status":"contradicted","supporting_image_ids":[],"raw_flags":["claim_mismatch"],"vision_justification":"Image shows a surface scratch, not the dent described."}

Trackpad claim, image shows area clearly but no damage — contradicted, none:
{"issue_type":"none","object_part":"trackpad","valid_image":true,"evidence_standard_met":true,"severity":"none","claim_status":"contradicted","supporting_image_ids":[],"raw_flags":["damage_not_visible"],"vision_justification":"Trackpad area fully visible; no physical damage present."}

Package image completely blurry — unreadable:
{"issue_type":"unknown","object_part":"unknown","valid_image":false,"evidence_standard_met":false,"severity":"unknown","claim_status":"not_enough_information","supporting_image_ids":[],"raw_flags":["blurry_image"],"vision_justification":"Image is too blurry to assess any damage."}
"""


# ---------------------------------------------------------------------------
# Structured-output schema for vision nodes
# ---------------------------------------------------------------------------


class _VisionResponseBase(BaseModel):
    """
    Shared Pydantic schema for all three vision nodes.

    Field names mirror ``VisualFindings`` exactly so parsed objects can be
    converted to a findings dict with a single attribute sweep.
    All subclasses inherit this structure; separate subclasses exist so each
    node passes its own type to ``response_schema``, giving Gemini an
    unambiguous object name in the JSON spec.
    """

    issue_type: str
    object_part: str
    valid_image: bool
    supporting_image_ids: list[str]
    raw_flags: list[str]
    vision_justification: str
    # VLM expresses the claim verdict directly; used by posterior_risk_node.
    # Allowed values: "supported", "contradicted", "not_enough_information"
    claim_status: str = "not_enough_information"
    # Whether the image set meets the evidence standard for this object/part,
    # judged against the injected evidence-requirement text. Drives the final
    # evidence_standard_met column and gates claim_status.
    evidence_standard_met: bool = False
    # Damage severity assessed from the image. Allowed values:
    # "none", "low", "medium", "high", "unknown".
    severity: str = "unknown"


class CarVisionResponse(_VisionResponseBase):
    """Structured-output schema for car damage assessment."""


class LaptopVisionResponse(_VisionResponseBase):
    """Structured-output schema for laptop damage assessment."""


class PackageVisionResponse(_VisionResponseBase):
    """Structured-output schema for package damage assessment."""


_E = type[enum.Enum]  # short alias for the type hint below


def _safe_enum(value: str, enum_cls: _E, fallback: enum.Enum) -> str:
    """
    Normalise a raw VLM string into a valid enum value string.

    Steps:
    1. Strip surrounding whitespace and lower-case the input.
    2. Try an exact match against the enum's values.
    3. Try a case-insensitive substring match (catches common VLM paraphrases
       such as ``"Rear Bumper"`` → ``"rear_bumper"``).
    4. Return ``fallback.value`` if nothing matches, so a hallucinated or
       malformed response never propagates an out-of-spec string downstream.

    Args:
        value:     Raw string returned by the VLM.
        enum_cls:  The enum class to validate against (e.g. ``IssueType``).
        fallback:  The enum member to use when no match is found.
    """
    normalised = value.strip().lower().replace(" ", "_").replace("-", "_")
    if not normalised:
        return fallback.value
    # exact match
    try:
        return enum_cls(normalised).value
    except ValueError:
        pass
    # substring / partial match — take the first member whose value appears
    # in the normalised string or whose normalised string appears in the value.
    # Guard against the empty-string edge case: only test non-empty substrings.
    for member in enum_cls:
        mv = member.value
        if mv and (mv in normalised or normalised in mv):
            return member.value
    return fallback.value


def _safe_flags(raw: list[str]) -> list[str]:
    """
    Filter a list of raw VLM flag strings to only valid ``RiskFlag`` values.

    Any hallucinated or mis-cased flag is silently dropped; the resulting list
    is deduplicated while preserving order.
    """
    valid: list[str] = []
    seen: set[str] = set()
    for item in raw:
        normalised = item.strip().lower().replace(" ", "_").replace("-", "_")
        try:
            flag_val = RiskFlag(normalised).value
        except ValueError:
            continue  # drop unrecognised flags
        if flag_val not in seen:
            seen.add(flag_val)
            valid.append(flag_val)
    return valid


def _findings_from_parsed(parsed: _VisionResponseBase, preflight_flags: list[str]) -> VisualFindings:
    """Convert any parsed vision response into a ``VisualFindings`` dict.

    All string fields from the VLM are run through the safe-enum normaliser
    before being stored, so hallucinated or malformed values are caught here
    rather than causing a crash or a bad CSV row later.
    Preflight flags (e.g. missing images) are merged and deduplicated.
    """
    issue_type = _safe_enum(parsed.issue_type, IssueType, IssueType.unknown)
    # The labelling convention folds all in-frame glass/screen damage into
    # "crack"; the model persistently over-emits "glass_shatter" for ordinary
    # cracks (and flips between the two run-to-run), so normalise it to "crack".
    if issue_type == IssueType.glass_shatter.value:
        issue_type = IssueType.crack.value

    return VisualFindings(
        issue_type=issue_type,
        object_part=parsed.object_part.strip().lower().replace(" ", "_").replace("-", "_") or "unknown",
        claim_status=_safe_enum(parsed.claim_status, ClaimStatus, ClaimStatus.not_enough_information),
        evidence_standard_met=parsed.evidence_standard_met,
        severity=_safe_enum(parsed.severity, Severity, Severity.unknown),
        valid_image=parsed.valid_image,
        supporting_image_ids=parsed.supporting_image_ids,
        raw_flags=list(dict.fromkeys(preflight_flags + _safe_flags(parsed.raw_flags))),
        vision_justification=parsed.vision_justification,
    )


# ---------------------------------------------------------------------------
# LangGraph pipeline
# ---------------------------------------------------------------------------

# -- State -------------------------------------------------------------------

class VisualFindings(TypedDict, total=False):
    """Structured output produced by one of the three vision nodes."""
    issue_type: str           # IssueType value
    object_part: str          # object-specific part value
    claim_status: str         # ClaimStatus value determined by VLM
    evidence_standard_met: bool  # VLM judgement against the evidence requirements
    severity: str             # Severity value assessed from the image
    valid_image: bool
    supporting_image_ids: list[str]
    raw_flags: list[str]      # RiskFlag values detected during visual pass
    vision_justification: str # image-grounded reasoning from the vision node


def empty_findings(
    *,
    object_part: str = "unknown",
    valid_image: bool = False,
    justification: str = "",
) -> VisualFindings:
    """
    Return a fully-populated ``VisualFindings`` dict with safe defaults.

    Every key is always present so downstream nodes can use plain dict access
    (``findings["valid_image"]``) without risking a ``KeyError``, and so that
    partial overwrites never leave stale values from a previous run.

    ``valid_image`` defaults to ``False`` so that any unset or failed findings
    automatically trigger the fast-fail path in ``route_post_evaluation``.
    Each vision node explicitly sets it to ``True`` only when the API succeeds
    and returns usable evidence.

    Args:
        object_part:  Default object part string; callers pass the per-object
                      ``unknown`` value (e.g. ``CarObjectPart.unknown.value``).
        valid_image:  Validity flag; defaults to ``False``.
        justification: Seed text for ``vision_justification``.
    """
    return VisualFindings(
        issue_type=IssueType.unknown.value,
        object_part=object_part,
        claim_status=ClaimStatus.not_enough_information.value,
        evidence_standard_met=False,
        severity=Severity.unknown.value,
        valid_image=valid_image,
        supporting_image_ids=[],
        raw_flags=[],
        vision_justification=justification,
    )


class AgentState(TypedDict, total=False):
    """
    Shared mutable state threaded through every node in the graph.

    Fields are populated progressively:
      - ``context``  is set by the caller before invoking the graph.
      - ``findings`` is written by whichever vision node runs.
      - ``output``   is written by the risk_and_compliance node.
    """
    # ---- set by the caller (input) ----
    context: ClaimContext  # {"claim": ClaimInput, "history": UserHistory, "requirements": [...]}

    # ---- written by vision nodes ----
    findings: VisualFindings

    # ---- written by risk_and_compliance node (final output) ----
    output: ClaimOutput


# -- Router ------------------------------------------------------------------

def route_by_object(state: AgentState) -> Literal["evaluate_car", "evaluate_laptop", "evaluate_package"]:
    """Conditional edge: inspect claim_object and direct to the matching vision node."""
    obj = state["context"]["claim"].claim_object.value
    if obj == ClaimObject.car.value:
        return "evaluate_car"
    if obj == ClaimObject.laptop.value:
        return "evaluate_laptop"
    return "evaluate_package"


# -- Vision nodes -------------------------------------------------------------

def _evaluate_claim_images(
    state: AgentState,
    domain: str,
    part_enum: type[enum.Enum],
    schema_cls: type[_VisionResponseBase],
) -> dict[str, Any]:
    """
    Shared implementation for all three vision nodes.

    Loads images, builds the prompt, calls the active vision provider
    (Anthropic or Google), and returns ``{"findings": VisualFindings}``.
    Failures degrade gracefully to an ``unknown`` placeholder so the graph
    never crashes mid-run.

    Args:
        domain:     Human-readable object type for the prompt ("car", "laptop", "package").
        part_enum:  The object-specific ``ObjectPart`` enum supplying allowed values.
        schema_cls: Pydantic response class passed to the VLM for structured output.
    """
    claim: ClaimInput = state["context"]["claim"]
    requirements: list[EvidenceRequirement] = state["context"].get("requirements", [])

    # ---- load images --------------------------------------------------------
    images: list[Image.Image] = []
    preflight_flags: list[str] = []
    for rel_path in claim.image_path_list:
        full_path = DATASET_DIR / rel_path
        try:
            images.append(Image.open(full_path).convert("RGB"))
        except FileNotFoundError:
            # Path declared in CSV but file absent — missing evidence.
            preflight_flags.append(RiskFlag.cropped_or_obstructed.value)
        except Exception:  # noqa: BLE001
            # Corrupt, truncated, or unsupported image. Treat as unusable
            # evidence rather than letting the error drop the entire claim
            # (which would leave a missing row in output.csv).
            preflight_flags.append(RiskFlag.cropped_or_obstructed.value)

    if not images:
        ef = empty_findings(
            object_part=next(m for m in part_enum if "unknown" in m.value).value,  # type: ignore[union-attr]
            valid_image=False,
            justification="No images could be loaded; claim cannot be evaluated.",
        )
        ef["raw_flags"] = list(
            dict.fromkeys(preflight_flags + [RiskFlag.manual_review_required.value])
        )
        return {"findings": ef}

    # ---- build prompt -------------------------------------------------------
    valid_parts  = [f"  - {v.value}" for v in part_enum]
    valid_issues = [f"  - {v.value}" for v in IssueType]
    valid_flags  = [f"  - {v.value}" for v in RiskFlag]
    image_ids    = claim.image_id_list

    # Inject the evidence standards this claim must be judged against so the
    # model decides evidence_standard_met against the documented requirements
    # rather than an implicit notion of "good enough".
    if requirements:
        evidence_standard_text = "\n".join(
            f"  - {req.applies_to}: {req.minimum_image_evidence}" for req in requirements
        )
    else:
        evidence_standard_text = "  - The claimed object and part must be clearly visible."

    prompt = (
        f"You are an expert claims adjuster specialising in {domain} damage assessment.\n\n"
        f"User claim text:\n\"{claim.user_claim}\"\n\n"
        f"Submitted image IDs (in order): {image_ids}\n\n"
        "EVIDENCE STANDARDS for this claim (judge evidence_standard_met against these):\n"
        + evidence_standard_text + "\n\n"
        "Review each provided image carefully and answer the following:\n"
        "1. Is the image set usable for automated review? (valid_image: true/false)\n"
        "2. Does the image set MEET the evidence standards listed above, i.e. is it\n"
        "   sufficient to actually evaluate this claim? (evidence_standard_met: true/false)\n"
        "3. What is the visible issue type? Choose the single closest value from:\n"
        + "\n".join(valid_issues) + "\n"
        f"4. Which {domain} part is affected? Choose the single closest value from:\n"
        + "\n".join(valid_parts) + "\n"
        "5. How severe is the damage? (severity) Choose one of:\n"
        "  - none    : no damage present\n"
        "  - low     : minor / cosmetic (light scratch, small scuff, single small dent)\n"
        "  - medium  : clearly visible functional or moderate damage (crack, dent, stain)\n"
        "  - high    : severe / structural (shattering, broken-off part, major collision,\n"
        "              crushed package, extensive water damage)\n"
        "  - unknown : cannot be assessed from the image\n"
        "6. List the image IDs (e.g. img_1) that directly support the decision "
        "(supporting_image_ids). Use an empty list if none qualify.\n"
        "7. List any applicable risk flags from:\n"
        + "\n".join(valid_flags) + "\n"
        "   Use an empty list if none apply.\n"
        "8. Write a concise, image-grounded justification (vision_justification). "
        "Reference specific image IDs where helpful.\n"
        + _PROMPT_ADDITIONS
    )

    # ---- call vision model (provider-agnostic) ------------------------------
    try:
        parsed = _call_vision_model(prompt, images, schema_cls)
        findings = _findings_from_parsed(parsed, preflight_flags)
    except Exception as exc:
        unknown_part = next(m for m in part_enum if "unknown" in m.value).value  # type: ignore[union-attr]
        findings = empty_findings(
            object_part=unknown_part,
            valid_image=False,
            justification=f"Vision analysis failed: {exc}",
        )
        findings["raw_flags"] = list(
            dict.fromkeys(preflight_flags + [RiskFlag.manual_review_required.value])
        )

    return {"findings": findings}


def evaluate_car_node(state: AgentState) -> dict[str, Any]:
    """Route car damage claims through the shared vision evaluator."""
    return _evaluate_claim_images(state, "car", CarObjectPart, CarVisionResponse)


def evaluate_laptop_node(state: AgentState) -> dict[str, Any]:
    """Route laptop damage claims through the shared vision evaluator."""
    return _evaluate_claim_images(state, "laptop", LaptopObjectPart, LaptopVisionResponse)


def evaluate_package_node(state: AgentState) -> dict[str, Any]:
    """Route package damage claims through the shared vision evaluator."""
    return _evaluate_claim_images(state, "package", PackageObjectPart, PackageVisionResponse)


# -- Post-evaluation router ---------------------------------------------------

def route_post_evaluation(state: AgentState) -> Literal["posterior_risk_node", "fast_fail_node"]:
    """
    Conditional edge executed after every vision node.

    Inspects findings.valid_image to decide whether the evidence is usable:
      - True  (or placeholder)  → full posterior risk & compliance analysis
      - False                   → fast-fail path, skip expensive downstream work
    """
    # evaluate_laptop_node now populates valid_image from real VLM output.
    # Car and package nodes still use placeholder True; update the comment
    # as each node is implemented.
    if not state.get("findings", {}).get("valid_image", True):
        return "fast_fail_node"
    return "posterior_risk_node"


# -- Fast-fail node -----------------------------------------------------------

def fast_fail_node(state: AgentState) -> dict[str, Any]:
    """
    Short-circuit path for claims where visual evidence is immediately unusable
    (e.g. completely blurry, wrong object, zero images).

    Produces a minimal ClaimOutput so output.csv still has a valid row.
    """
    ctx: ClaimContext = state["context"]
    claim: ClaimInput = ctx["claim"]
    findings: VisualFindings = state.get("findings") or empty_findings(valid_image=False)

    collected_flags = list(findings["raw_flags"])
    if not collected_flags:
        collected_flags = [RiskFlag.damage_not_visible.value]
    risk_flags_str = ";".join(dict.fromkeys(collected_flags))  # dedup, preserve order

    output = ClaimOutput(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=False,
        evidence_standard_met_reason="Visual evidence is unusable; claim cannot be evaluated.",
        risk_flags=risk_flags_str,
        issue_type=IssueType.unknown,
        object_part="unknown",
        claim_status=ClaimStatus.not_enough_information,
        claim_status_justification=(
            findings.get("vision_justification")
            or "Image evidence was disqualified before full analysis."
        ),
        supporting_image_ids="none",
        valid_image=False,
        severity=Severity.unknown,
    )

    return {"output": output}


# -- Posterior risk & compliance node -----------------------------------------

def posterior_risk_node(state: AgentState) -> dict[str, Any]:
    """
    Runs after vision analysis is confirmed usable.

    Cross-references visual findings with user history and evidence requirements
    to produce the final ClaimOutput.

    Decision logic:
    - evidence_standard_met  : VLM judgement against the evidence requirements,
                               gated on a usable image
    - risk_flags             : union of vision flags + history-derived flags
    - claim_status           : VLM verdict, forced to NEI when evidence not met
    - severity               : VLM-assessed severity, escalated by risky history
    """
    ctx: ClaimContext = state["context"]
    claim: ClaimInput = ctx["claim"]
    history: UserHistory = ctx["history"]
    findings: VisualFindings = state.get("findings") or empty_findings()

    # ---- evidence standard ----
    # Trust the VLM's requirement-grounded judgement, but never accept it when
    # the image itself was unusable. Fall back to "supporting image present" if
    # the VLM did not supply an evidence verdict.
    supporting_ids = findings["supporting_image_ids"]
    valid_image = findings["valid_image"]
    vlm_evidence_met = findings.get("evidence_standard_met", bool(supporting_ids))
    evidence_met = valid_image and vlm_evidence_met
    evidence_reason = (
        "Image set meets the evidence standard for the claimed object and part."
        if evidence_met
        else "Image set does not meet the evidence standard to evaluate this claim."
    )

    # ---- risk flags ----
    collected_flags: list[str] = list(findings["raw_flags"])
    if history.rejected_claim > 1 or history.manual_review_claim > 2:
        collected_flags.append(RiskFlag.user_history_risk.value)
    if history.manual_review_claim > 0 or history.rejected_claim > 0:
        collected_flags.append(RiskFlag.manual_review_required.value)
    # deduplicate while preserving order
    seen: set[str] = set()
    deduped_flags: list[str] = []
    for f in collected_flags:
        if f not in seen:
            seen.add(f)
            deduped_flags.append(f)
    risk_flags_str = ";".join(deduped_flags) if deduped_flags else RiskFlag.none.value

    # ---- claim status — use VLM verdict when evidence is present ----
    # The VLM observed the images directly and is best placed to decide whether
    # the damage is supported, contradicted, or indeterminate.  We only override
    # with not_enough_information when the evidence standard itself was not met
    # (no valid image / no supporting image IDs), because in that case no verdict
    # can be trusted regardless of what the model said.
    vision_just = findings["vision_justification"]
    if not evidence_met:
        claim_status = ClaimStatus.not_enough_information
    else:
        vlm_status_raw = findings.get("claim_status", ClaimStatus.not_enough_information.value)
        claim_status = _safe_enum(vlm_status_raw, ClaimStatus, ClaimStatus.not_enough_information)

    # ---- severity: derive deterministically from the final issue_type ----
    # Ground-truth severity is, in practice, a near-deterministic function of
    # the issue type (cosmetic marks are low, real damage is medium, "high" is
    # reserved for catastrophic structural loss). Deriving it from issue_type is
    # both more accurate and more stable than free-form VLM grading, and it
    # automatically tracks any improvement in issue_type classification.
    final_issue = findings["issue_type"]
    base_severity = _SEVERITY_BY_ISSUE.get(final_issue, Severity.unknown.value)

    output = ClaimOutput(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=evidence_met,
        evidence_standard_met_reason=evidence_reason,
        risk_flags=risk_flags_str,
        issue_type=IssueType(findings["issue_type"]),
        object_part=findings["object_part"],
        claim_status=claim_status,
        claim_status_justification=vision_just or "Pending full vision analysis.",
        supporting_image_ids=";".join(supporting_ids) if supporting_ids else "none",
        valid_image=valid_image,
        severity=base_severity,
    )

    return {"output": output}


# -- Graph compilation --------------------------------------------------------

#: Vision-node names used in both routing and edge registration.
_VISION_NODES = ("evaluate_car", "evaluate_laptop", "evaluate_package")


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph for claim verification.

    Topology::

        START
          │  (conditional — route_by_object)
          ├──▶ evaluate_car_node
          ├──▶ evaluate_laptop_node
          └──▶ evaluate_package_node
                    │  (conditional — route_post_evaluation)
                    ├──▶ posterior_risk_node ──▶ END
                    └──▶ fast_fail_node      ──▶ END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("evaluate_car", evaluate_car_node)
    graph.add_node("evaluate_laptop", evaluate_laptop_node)
    graph.add_node("evaluate_package", evaluate_package_node)
    graph.add_node("fast_fail_node", fast_fail_node)
    graph.add_node("posterior_risk_node", posterior_risk_node)

    # START → object router → matching vision node
    graph.add_conditional_edges(
        START,
        route_by_object,
        {
            "evaluate_car": "evaluate_car",
            "evaluate_laptop": "evaluate_laptop",
            "evaluate_package": "evaluate_package",
        },
    )

    # Each vision node → post-evaluation router → posterior_risk_node | fast_fail_node
    _post_eval_map = {
        "posterior_risk_node": "posterior_risk_node",
        "fast_fail_node": "fast_fail_node",
    }
    for vision_node in _VISION_NODES:
        graph.add_conditional_edges(vision_node, route_post_evaluation, _post_eval_map)

    # Both terminal nodes → END
    graph.add_edge("posterior_risk_node", END)
    graph.add_edge("fast_fail_node", END)

    return graph.compile()


#: Module-level compiled graph — import and invoke directly.
claim_graph = build_graph()


def _fallback_csv_row(
    *,
    user_id: str,
    image_paths: str,
    user_claim: str,
    claim_object: str,
    reason: str,
) -> dict[str, str]:
    """
    Build a fully-formed output row for a claim that could not be processed.

    Guarantees output.csv always contains exactly one row per input claim,
    even when row parsing or graph invocation fails — never a missing row.
    """
    return {
        "user_id": user_id,
        "image_paths": image_paths,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": reason,
        "risk_flags": RiskFlag.manual_review_required.value,
        "issue_type": IssueType.unknown.value,
        "object_part": "unknown",
        "claim_status": ClaimStatus.not_enough_information.value,
        "claim_status_justification": reason,
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": Severity.unknown.value,
    }


if __name__ == "__main__":
    import argparse
    import csv
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from tqdm import tqdm

    REPO_ROOT = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Run the evidence-review pipeline.")
    parser.add_argument(
        "--claims",
        default=str(REPO_ROOT / "dataset" / "sample_claims.csv"),
        help="Input claims CSV. Use dataset/claims.csv for the final test run.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "output.csv"),
        help="Destination CSV for predictions.",
    )
    args = parser.parse_args()

    claims_path       = Path(args.claims)
    history_path      = REPO_ROOT / "dataset" / "user_history.csv"
    requirements_path = REPO_ROOT / "dataset" / "evidence_requirements.csv"
    output_path       = Path(args.output)

    print(f"Provider: {VISION_PROVIDER}  |  Model: {VISION_MODEL}")
    print(f"Claims:   {claims_path}")
    print(f"Output:   {output_path}")

    # Pre-count data rows so tqdm can show an accurate total.
    with open(claims_path, encoding="utf-8") as _f:
        total_claims = sum(1 for _ in _f) - 1  # subtract header line

    rows_written = 0
    start_time = time.time()
    # Separate counters per failure category for the end-of-run summary.
    err_parse   = 0  # malformed input row (bad claim_object, missing column)
    err_vlm     = 0  # API / quota / network failures bubbling out of a node
    err_routing = 0  # LangGraph routing or graph-level exceptions
    err_other   = 0  # anything else (serialisation, unexpected crash)

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        contexts = load_and_join(claims_path, history_path, requirements_path)

        for context in tqdm(contexts, total=total_claims, desc="Processing claims",
                            unit="claim", dynamic_ncols=True):
            # --- Malformed input row: write a fallback row, keep going --------
            if context.get("claim") is None:
                raw = context.get("raw", {})
                err_parse += 1
                writer.writerow(_fallback_csv_row(
                    user_id=raw.get("user_id", ""),
                    image_paths=raw.get("image_paths", ""),
                    user_claim=raw.get("user_claim", ""),
                    claim_object=raw.get("claim_object", ""),
                    reason=f"Input row could not be parsed: {context.get('error')}",
                ))
                fout.flush()
                rows_written += 1
                tqdm.write(
                    f"[PARSE ERROR] {raw.get('user_id', '?')}: {context.get('error')}",
                    file=sys.stderr,
                )
                continue

            claim = context["claim"]
            try:
                result = claim_graph.invoke({"context": context})
                output: ClaimOutput = result["output"]
                writer.writerow(output.to_csv_row())
                fout.flush()   # persist each row immediately so partial runs are readable
                rows_written += 1

            # --- Vision / VLM failures (API error, quota, network) ------------
            except (ConnectionError, TimeoutError, OSError) as exc:
                err_vlm += 1
                writer.writerow(_fallback_csv_row(
                    user_id=claim.user_id, image_paths=claim.image_paths,
                    user_claim=claim.user_claim, claim_object=claim.claim_object.value,
                    reason=f"Vision/API failure: {exc}",
                ))
                fout.flush()
                rows_written += 1
                tqdm.write(
                    f"[VLM ERROR] {claim.user_id} ({claim.claim_object.value}): {exc}",
                    file=sys.stderr,
                )

            # --- Graph / routing failures (LangGraph internal errors) ----------
            except (KeyError, ValueError, TypeError) as exc:
                err_routing += 1
                writer.writerow(_fallback_csv_row(
                    user_id=claim.user_id, image_paths=claim.image_paths,
                    user_claim=claim.user_claim, claim_object=claim.claim_object.value,
                    reason=f"Graph/routing error: {type(exc).__name__}: {exc}",
                ))
                fout.flush()
                rows_written += 1
                tqdm.write(
                    f"[GRAPH ERROR] {claim.user_id} ({claim.claim_object.value}): "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            # --- Catch-all for anything unexpected ----------------------------
            except Exception as exc:  # noqa: BLE001
                err_other += 1
                writer.writerow(_fallback_csv_row(
                    user_id=claim.user_id, image_paths=claim.image_paths,
                    user_claim=claim.user_claim, claim_object=claim.claim_object.value,
                    reason=f"Unexpected error: {type(exc).__name__}: {exc}",
                ))
                fout.flush()
                rows_written += 1
                tqdm.write(
                    f"[UNKNOWN ERROR] {claim.user_id} ({claim.claim_object.value}): "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    elapsed_seconds = round(time.time() - start_time, 2)
    total_failed = err_parse + err_vlm + err_routing + err_other

    # Run-stats sidecar so the evaluator can pick up the real model, runtime,
    # and row count automatically (no need to pass --runtime-seconds/--model).
    stats_path = output_path.with_name(output_path.stem + ".run_stats.json")
    run_stats = {
        "provider":        VISION_PROVIDER,
        "model":           VISION_MODEL,
        "claims_path":     str(claims_path),
        "output_path":     str(output_path),
        "total_claims":    total_claims,
        "rows_written":    rows_written,
        "runtime_seconds": elapsed_seconds,
        "failures": {
            "total":   total_failed,
            "parse":   err_parse,
            "vlm":     err_vlm,
            "routing": err_routing,
            "other":   err_other,
        },
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(stats_path, "w", encoding="utf-8") as fstats:
            json.dump(run_stats, fstats, indent=2)
    except OSError as exc:
        print(f"[WARN] Could not write run-stats sidecar: {exc}", file=sys.stderr)

    print(f"\nDone. {rows_written}/{total_claims} rows written → {output_path}")
    print(f"Runtime: {elapsed_seconds:.2f}s  |  Run stats → {stats_path}")
    if total_failed:
        print(
            f"Failures: {total_failed} total  "
            f"(parse={err_parse}, VLM={err_vlm}, routing={err_routing}, other={err_other})"
        )
