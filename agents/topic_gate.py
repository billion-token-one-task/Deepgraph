"""Topic gate: three questions before compute, then surprise-driven allocation.

The engine used to pick topics by "how paradigm-breaking does this sound" and
then spend the same compute on every one of them. That burns GPU on questions
whose answer is already in a textbook, and it treats "the experiment ran" as
success.

Three gates replace that:

闸一 (zero cost, before ANY compute) -- three questions per topic:
  1. What do we predict will happen, and how confident are we?
  2. Do both outcomes lead to the same next action?
  3. Has this already been published?
  Failing any one of them means the experiment cannot pay for itself.

闸二 (pilot first) -- a pilot passes when it REFUTES the prediction, not when it
  runs. Passing is measured in bits of surprise, so a 0.6-confidence prediction
  that comes true (0.74 bits) does not clear the bar while its refutation
  (1.32 bits) does.

闸三 (routing) -- surprising + rigorous goes to the public case page;
  unsurprising + rigorous goes to client delivery. The two channels never share
  a claim.

Compute follows surprise: every topic starts on the cheapest lane with a small
slice of its planned budget, and only buys the next lane by producing bits.

Everything up to ``elicit_prediction`` is pure and DB-free so it can be tested
and reused by the agenda selector, the auto-research scheduler and the web API.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from config import (
    TOPIC_GATE_ENABLED,
    TOPIC_GATE_MAX_CONFIDENCE,
    TOPIC_GATE_MIN_SEEDS,
    TOPIC_GATE_SURPRISE_BITS,
)

# Stage ladder. A topic never starts above ``pilot``.
STAGES = ("pilot", "confirm", "full")

STAGE_BUDGET_FRACTION = {
    "pilot": 0.10,
    "confirm": 0.35,
    "full": 1.0,
}

# Cheapest lane a stage is allowed to use, keyed by the lane the plan asked for.
_PILOT_LANE = {"cpu": "cpu", "gpu_small": "gpu_small", "gpu_large": "gpu_small"}
_CONFIRM_LANE = {"cpu": "cpu", "gpu_small": "gpu_small", "gpu_large": "gpu_small"}

_NO_ACTION_MARKERS = {
    "",
    "none",
    "nothing",
    "no action",
    "no-op",
    "n/a",
    "na",
    "tbd",
    "unknown",
    "不做",
    "无",
    "没有",
}

PUBLISHED_YES = {"yes", "true", "published", "exists"}


# ---------- information-theory helpers ----------


def surprisal_bits(probability: float) -> float:
    """Bits of surprise from observing an event we gave ``probability``."""
    p = min(max(float(probability), 1e-6), 1.0)
    return -math.log2(p)


def binary_entropy(probability: float) -> float:
    """Expected bits from a yes/no question we answer correctly with ``p``."""
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ---------- 闸一: three questions, zero compute ----------


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_action(value: Any) -> str:
    text = _text(value).lower()
    for ch in ".。,，;；!！?？ \t\n":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _is_no_action(value: str) -> bool:
    return value in _NO_ACTION_MARKERS


def normalize_prediction(raw: Any) -> dict[str, Any] | None:
    """Coerce a stored / LLM-produced prediction record into the gate's shape."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, Mapping):
        return None
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    return {
        "predicted_outcome": _text(raw.get("predicted_outcome") or raw.get("outcome")),
        "confidence": confidence,
        "action_if_confirmed": _text(raw.get("action_if_confirmed")),
        "action_if_refuted": _text(raw.get("action_if_refuted")),
        "already_published": _text(raw.get("already_published")).lower() or "unsure",
        "already_published_evidence": _text(raw.get("already_published_evidence")),
    }


def screen_topic(
    insight: Mapping[str, Any],
    *,
    prediction: Any = None,
) -> dict[str, Any]:
    """闸一. Returns {passed, blockers, expected_bits, refute_bits, prediction}.

    ``blockers`` entries are {"question": ..., "reason": ...}. An empty list
    means the topic is worth spending a pilot on.
    """
    if prediction is None:
        prediction = load_gate_record(insight).get("prediction")
    pred = normalize_prediction(prediction)
    blockers: list[dict[str, str]] = []

    # Q1 -- is there a written prediction with a confidence?
    if not pred or not pred["predicted_outcome"] or pred["confidence"] is None:
        return {
            "passed": False,
            "blockers": [
                {
                    "question": "prediction",
                    "reason": "no written prediction + confidence before spending compute",
                }
            ],
            "expected_bits": 0.0,
            "refute_bits": 0.0,
            "prediction": pred,
        }
    confidence = min(max(pred["confidence"], 0.0), 1.0)
    if confidence >= TOPIC_GATE_MAX_CONFIDENCE:
        blockers.append(
            {
                "question": "prediction",
                "reason": (
                    f"confidence={confidence:.2f} >= {TOPIC_GATE_MAX_CONFIDENCE:.2f}: "
                    "the answer is already known, running it buys no information"
                ),
            }
        )

    # Q2 -- do both outcomes lead to the same next action?
    act_true = _normalized_action(pred["action_if_confirmed"])
    act_false = _normalized_action(pred["action_if_refuted"])
    if not act_true or not act_false:
        blockers.append(
            {
                "question": "decision_relevance",
                "reason": "next action undeclared for at least one outcome",
            }
        )
    elif act_true == act_false or (_is_no_action(act_true) and _is_no_action(act_false)):
        blockers.append(
            {
                "question": "decision_relevance",
                "reason": "both outcomes lead to the same next action",
            }
        )

    # Q3 -- has it already been published?
    novelty = _text(insight.get("novelty_status")).lower()
    if novelty == "exists":
        blockers.append({"question": "prior_work", "reason": "novelty_status=exists"})
    elif pred["already_published"] in PUBLISHED_YES:
        evidence = pred["already_published_evidence"] or "no citation given"
        blockers.append({"question": "prior_work", "reason": f"already published ({evidence})"})

    return {
        "passed": not blockers,
        "blockers": blockers,
        "expected_bits": round(binary_entropy(confidence), 4),
        "refute_bits": round(surprisal_bits(1.0 - confidence), 4),
        "prediction": pred,
    }


# ---------- 闸二: the pilot passes by refuting, not by running ----------


def pilot_verdict(
    prediction: Any,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """闸二. Returns {verdict, surprise_bits, reasons}.

    ``verdict`` is one of:
      escalate     -- the pilot produced enough bits to buy the next lane
      stop         -- it ran and told us what we already expected
      inconclusive -- it ran but cannot distinguish the outcomes
      invalid      -- it did not run, or the improvement is not attributable

    ``observation`` keys: ran (bool), outcome (confirmed|refuted|inconclusive),
    null_model_control (passed|failed|missing).
    """
    pred = normalize_prediction(prediction) or {}
    confidence = pred.get("confidence")
    confidence = 0.5 if confidence is None else min(max(float(confidence), 0.0), 1.0)
    outcome = _text(observation.get("outcome")).lower()
    control = _text(observation.get("null_model_control")).lower() or "missing"
    reasons: list[str] = []

    if not observation.get("ran", False):
        return {
            "verdict": "invalid",
            "surprise_bits": 0.0,
            "reasons": ["pilot did not run"],
        }

    # A "win" that survives with the model removed is not the model's win.
    if control == "failed":
        return {
            "verdict": "invalid",
            "surprise_bits": 0.0,
            "reasons": ["null-model control did not drop: the effect is not attributable"],
        }

    if outcome == "refuted":
        # We gave the predicted outcome ``confidence``, so what actually
        # happened had probability 1 - confidence.
        bits = surprisal_bits(1.0 - confidence)
    elif outcome == "confirmed":
        bits = surprisal_bits(confidence)
    else:
        return {
            "verdict": "inconclusive",
            "surprise_bits": 0.0,
            "reasons": [f"pilot outcome={outcome or 'unknown'}: prediction not decided"],
        }

    bits = round(bits, 4)
    reasons.append(f"prediction {outcome} at confidence {confidence:.2f} -> {bits:.2f} bits")
    if control == "missing":
        reasons.append("null-model control not run yet")
    if bits >= TOPIC_GATE_SURPRISE_BITS:
        return {"verdict": "escalate", "surprise_bits": bits, "reasons": reasons}
    reasons.append(
        f"{bits:.2f} bits < {TOPIC_GATE_SURPRISE_BITS:.2f}: running is not the same as passing"
    )
    return {"verdict": "stop", "surprise_bits": bits, "reasons": reasons}


# ---------- 闸三: two channels, never mixed ----------


def route_outcome(
    verdict: Mapping[str, Any],
    rigor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """闸三. Surprising + rigorous -> case page; unsurprising + rigorous -> client.

    ``rigor`` keys: seeds (int), null_model_control (passed|failed|missing),
    p_value (float|None), packet_complete (bool).
    """
    rigor = dict(rigor or {})
    bits = float(verdict.get("surprise_bits") or 0.0)
    reasons: list[str] = []
    blockers: list[str] = []
    disclosures: list[str] = []

    if verdict.get("verdict") == "invalid":
        return {
            "channel": "withhold",
            "reasons": list(verdict.get("reasons") or ["pilot invalid"]),
            "blockers": ["result is not attributable to the claimed mechanism"],
            "required_disclosures": [],
        }

    try:
        seeds = int(rigor.get("seeds") or 0)
    except (TypeError, ValueError):
        seeds = 0
    if seeds < TOPIC_GATE_MIN_SEEDS:
        blockers.append(f"seeds={seeds} < {TOPIC_GATE_MIN_SEEDS}")
    p_value = rigor.get("p_value")
    if isinstance(p_value, (int, float)) and p_value > 0.05:
        blockers.append(f"p={p_value:.3g} > 0.05")
    if rigor.get("packet_complete") is False:
        blockers.append("evidence packet incomplete")

    control = _text(rigor.get("null_model_control")).lower() or "missing"
    if control == "failed":
        return {
            "channel": "withhold",
            "reasons": ["null-model control did not drop"],
            "blockers": ["effect is not attributable to the model"],
            "required_disclosures": [],
        }
    if control == "missing":
        disclosures.append("null-model control not run")

    if blockers:
        return {
            "channel": "withhold",
            "reasons": [f"{bits:.2f} bits of surprise, but the evidence is not yet publishable"],
            "blockers": blockers,
            "required_disclosures": disclosures,
        }

    if bits >= TOPIC_GATE_SURPRISE_BITS:
        # The public case page is the one place a claim goes out with our name
        # on it, so the null-model control is mandatory rather than disclosed.
        if control == "missing":
            return {
                "channel": "withhold",
                "reasons": [f"{bits:.2f} bits would justify a case page"],
                "blockers": ["null-model control must run before any public number"],
                "required_disclosures": disclosures,
            }
        reasons.append(f"{bits:.2f} bits: the result contradicted what we predicted")
        return {
            "channel": "case_page",
            "reasons": reasons,
            "blockers": [],
            "required_disclosures": disclosures,
        }

    reasons.append(f"{bits:.2f} bits: rigorous but unsurprising, no public claim")
    return {
        "channel": "client_delivery",
        "reasons": reasons,
        "blockers": [],
        "required_disclosures": disclosures,
    }


# ---------- compute allocation ----------


def next_stage(stage: str) -> str | None:
    try:
        index = STAGES.index(stage)
    except ValueError:
        return STAGES[0]
    return STAGES[index + 1] if index + 1 < len(STAGES) else None


def allocate_compute(
    *,
    stage: str = "pilot",
    resource_class: str = "cpu",
    surprise_bits: float | None = None,
    expected_bits: float = 0.0,
) -> dict[str, Any]:
    """Map a stage + bits earned so far onto a lane and a slice of the budget."""
    stage = stage if stage in STAGES else "pilot"
    planned = (resource_class or "cpu").strip() or "cpu"
    if stage == "pilot":
        lane = _PILOT_LANE.get(planned, "cpu")
        reason = "pilot runs on the cheapest lane that can execute it"
    elif stage == "confirm":
        lane = _CONFIRM_LANE.get(planned, planned)
        reason = f"{surprise_bits:.2f} bits bought the confirm lane" if surprise_bits else "confirm lane"
    else:
        lane = planned
        reason = f"{surprise_bits:.2f} bits bought the full planned budget" if surprise_bits else "full lane"
    return {
        "stage": stage,
        "resource_class": lane,
        "budget_fraction": STAGE_BUDGET_FRACTION[stage],
        "priority": round(float(surprise_bits if surprise_bits is not None else expected_bits), 4),
        "reason": reason,
    }


# ---------- prediction elicitation + persistence ----------


ELICIT_SYSTEM = """You are screening a research topic BEFORE any compute is spent.

Answer three questions about the topic you are given. Be honest rather than
encouraging: a topic whose answer you already know is a topic we should not run.

1. What do you predict the experiment will show, and with what probability?
   Give the probability that your predicted outcome is what the experiment
   actually produces. If it is textbook knowledge, say so with a high number.
2. What would we do next if the prediction is confirmed, and what would we do
   next if it is refuted? If the two answers are the same, say so plainly.
3. Has this already been published? Name the paper if you know one.

Return JSON only:
{"predicted_outcome": "...", "confidence": 0.0-1.0,
 "action_if_confirmed": "...", "action_if_refuted": "...",
 "already_published": "yes|no|unsure", "already_published_evidence": "..."}
"""


def _insight_brief(insight: Mapping[str, Any]) -> str:
    parts = []
    for key in ("title", "problem_statement", "existing_weakness", "formal_structure", "hypothesis"):
        value = _text(insight.get(key))
        if value:
            parts.append(f"{key}: {value[:600]}")
    method = insight.get("proposed_method")
    if isinstance(method, (dict, list)):
        parts.append(f"proposed_method: {json.dumps(method, ensure_ascii=False)[:600]}")
    elif _text(method):
        parts.append(f"proposed_method: {_text(method)[:600]}")
    plan = insight.get("experimental_plan")
    if isinstance(plan, (dict, list)):
        parts.append(f"experimental_plan: {json.dumps(plan, ensure_ascii=False)[:600]}")
    elif _text(plan):
        parts.append(f"experimental_plan: {_text(plan)[:600]}")
    return "\n".join(parts)


def elicit_prediction(insight: Mapping[str, Any]) -> dict[str, Any] | None:
    """One cheap LLM call that writes the prediction down. No GPU, no dataset."""
    from agents.llm_client import call_llm_json

    try:
        raw, _tokens = call_llm_json(ELICIT_SYSTEM, _insight_brief(insight))
    except Exception as exc:  # elicitation failure must not crash the scheduler
        print(f"[TOPIC_GATE] prediction elicitation failed: {exc}", flush=True)
        return None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    return normalize_prediction(raw)


def load_gate_record(insight: Mapping[str, Any]) -> dict[str, Any]:
    raw = insight.get("topic_gate")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def persist_gate_record(insight_id: int, record: Mapping[str, Any]) -> None:
    from db import database as db

    db.execute(
        "UPDATE deep_insights SET topic_gate=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (json.dumps(dict(record), ensure_ascii=False), int(insight_id)),
    )
    db.commit()


def screen_insight(
    insight: Mapping[str, Any],
    *,
    allow_elicit: bool = True,
    persist: bool = True,
) -> dict[str, Any]:
    """闸一 for one insight: reuse the stored prediction, else elicit and store it.

    Returns the ``screen_topic`` result. With the gate disabled it passes
    everything so the scheduler keeps its previous behaviour.
    """
    if not TOPIC_GATE_ENABLED:
        return {
            "passed": True,
            "blockers": [],
            "expected_bits": 0.0,
            "refute_bits": 0.0,
            "prediction": None,
            "gate_disabled": True,
        }

    record = load_gate_record(insight)
    prediction = normalize_prediction(record.get("prediction"))
    elicited = False
    elicitation_failed = False
    if not prediction or prediction["confidence"] is None:
        if not allow_elicit:
            elicitation_failed = True
        else:
            prediction = elicit_prediction(insight)
            elicited = prediction is not None and prediction["confidence"] is not None
            elicitation_failed = not elicited

    result = screen_topic(insight, prediction=prediction)
    # An unreachable LLM is not evidence about the topic. Callers fail open on
    # this flag (the pilot lane caps what a wrongly-admitted topic can spend)
    # and fail closed on every judgement the gate was actually able to make.
    result["elicitation_failed"] = elicitation_failed
    if persist and elicited and insight.get("id"):
        record["prediction"] = result["prediction"]
        record["screen"] = {
            "passed": result["passed"],
            "blockers": result["blockers"],
            "expected_bits": result["expected_bits"],
        }
        try:
            persist_gate_record(int(insight["id"]), record)
        except Exception as exc:  # persistence failure must not block scheduling
            print(f"[TOPIC_GATE] could not persist gate record: {exc}", flush=True)
    return result
