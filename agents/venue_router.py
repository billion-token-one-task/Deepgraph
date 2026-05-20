"""Venue router: pick the best template for a given insight/run state.

Issue #11/#12 (D1 Foundation). Mirrors the structure of ``agenda_selector``:
each candidate venue is scored by rule weights loaded from
``manuscript_venues/venues_v1.yaml``; the highest-scoring non-rejected venue
wins, the rest are persisted under ``rejected_venues_json`` with a reason.

Public API
~~~~~~~~~~
- ``load_venue_config(path=None) -> list[dict]``
- ``score_venue(venue_cfg, state) -> dict``
- ``evaluate_venues(state, venue_cfgs) -> dict``
- ``route_and_persist(selection_id, state) -> dict``
- ``get_routing(selection_id) -> dict | None``

``state`` is a free-form dict the caller assembles from the deep_insight,
experiment_run, evidence_gate, and manuscript_run rows. Recognised keys::

    {
        "title": str,
        "claim_type": "empirical" | "theory" | "position" | ...,
        "tier": int,                  # AAA-quality tier 1/2/3
        "domain": str,                # e.g. "vision", "nlp", "ml"
        "has_real_data": bool,
        "experiment_status": "completed" | "pending" | "failed",
        "page_count_estimate": int,   # rough LaTeX page estimate
        "novelty_status": str,
    }

Each venue in YAML carries::

    template_id: str            # must resolve via manuscript_templates.get_adapter
    schema_version: int         # bump to invalidate cached routes
    triggers:                   # additive score
      claim_types: [str, ...]
      domains: [str, ...]
      keywords: [str, ...]       # case-insensitive title/keyword match
      requires_real_data: bool
    rejects:                    # hard reject
      claim_types: [str, ...]
      page_count_max: int
    max_pages: int              # informational; flows into DB
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from agents.manuscript_templates import get_adapter, list_adapters
from db import database as db


# ---------- scoring weights ----------

W_CLAIM_TYPE_MATCH = 0.30
W_DOMAIN_MATCH = 0.20
W_KEYWORD_MATCH_EACH = 0.10
W_KEYWORD_MATCH_MAX = 0.30
W_REAL_DATA_BONUS = 0.15
W_TIER_BONUS = 0.05  # multiplied by (3 - tier); higher tier => smaller bonus


DEFAULT_RULE_SET = "venues_v1"

# D3 (#14): score-difference threshold below which the tiebreaker is invoked.
# Two venues whose rule-based scores differ by less than this number are
# considered a tie and the LLM (or deterministic fallback) decides.
TIEBREAK_SCORE_DELTA = 0.05


# ---------- loading ----------


@dataclass
class VenueConfig:
    template_id: str
    schema_version: int = 1
    triggers: dict[str, Any] = field(default_factory=dict)
    rejects: dict[str, Any] = field(default_factory=dict)
    max_pages: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VenueConfig":
        return cls(
            template_id=str(payload["template_id"]),
            schema_version=int(payload.get("schema_version", 1)),
            triggers=dict(payload.get("triggers") or {}),
            rejects=dict(payload.get("rejects") or {}),
            max_pages=int(payload["max_pages"]) if payload.get("max_pages") is not None else None,
        )


def load_venue_config(path: str | Path | None = None) -> list[VenueConfig]:
    """Load venue config from YAML/JSON.

    ``path=None`` resolves to the value of ``VENUES_CONFIG_PATH`` (see
    ``config.py``). Returns an ordered list so deterministic tie-breaking
    can fall back to file order.
    """
    if path is None:
        from config import VENUES_CONFIG_PATH
        path = VENUES_CONFIG_PATH
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"venues config not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML venue config") from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text) if text.strip() else {}
    raw_venues = payload.get("venues") or []
    out: list[VenueConfig] = []
    for raw in raw_venues:
        if not isinstance(raw, Mapping):
            continue
        out.append(VenueConfig.from_dict(raw))
    return out


# ---------- scoring ----------


def _norm_text_blob(state: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "problem_statement", "domain", "claim_type"):
        v = state.get(key)
        if v:
            parts.append(str(v))
    return " ".join(parts).lower()


def score_venue(venue: VenueConfig, state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a full scoring breakdown for one venue against the state."""
    blob = _norm_text_blob(state)
    triggers = venue.triggers or {}
    rejects = venue.rejects or {}

    block_reasons: list[str] = []
    claim = str(state.get("claim_type") or "").lower()
    reject_claims = {str(c).lower() for c in (rejects.get("claim_types") or [])}
    if reject_claims and claim and claim in reject_claims:
        block_reasons.append(f"claim_type={claim} in rejects.claim_types")
    reject_page_max = rejects.get("page_count_max")
    if reject_page_max is not None:
        try:
            est = int(state.get("page_count_estimate") or 0)
            if est > int(reject_page_max):
                block_reasons.append(
                    f"page_count_estimate={est} > rejects.page_count_max={int(reject_page_max)}"
                )
        except (TypeError, ValueError):
            pass

    # Positive components
    components: dict[str, float] = {}

    accepted_claims = {str(c).lower() for c in (triggers.get("claim_types") or [])}
    components["claim_type"] = W_CLAIM_TYPE_MATCH if (claim and claim in accepted_claims) else 0.0

    accepted_domains = {str(d).lower() for d in (triggers.get("domains") or [])}
    domain = str(state.get("domain") or "").lower()
    components["domain"] = W_DOMAIN_MATCH if (domain and domain in accepted_domains) else 0.0

    keywords = [str(k).lower() for k in (triggers.get("keywords") or [])]
    kw_hits = [k for k in keywords if k and k in blob]
    components["keywords"] = min(len(kw_hits) * W_KEYWORD_MATCH_EACH, W_KEYWORD_MATCH_MAX)

    requires_real = bool(triggers.get("requires_real_data"))
    has_real = bool(state.get("has_real_data"))
    if requires_real and not has_real:
        block_reasons.append("triggers.requires_real_data set but state.has_real_data is False")
    components["real_data"] = W_REAL_DATA_BONUS if (requires_real and has_real) else 0.0

    tier_bonus = 0.0
    try:
        tier = int(state.get("tier") or 3)
        tier_bonus = max(0.0, W_TIER_BONUS * (3 - tier))
    except (TypeError, ValueError):
        pass
    components["tier"] = tier_bonus

    total = sum(components.values())
    return {
        "template_id": venue.template_id,
        "score": round(total, 4),
        "blocked": bool(block_reasons),
        "block_reasons": block_reasons,
        "components": {k: round(v, 4) for k, v in components.items()},
        "matched_keywords": kw_hits,
    }


def evaluate_venues(
    state: Mapping[str, Any],
    venue_cfgs: Iterable[VenueConfig],
) -> dict[str, Any]:
    """Score all venues; return ``{selected, rejected, all_scored}``."""
    scored: list[dict[str, Any]] = []
    for cfg in venue_cfgs:
        scored.append({"venue": cfg, "breakdown": score_venue(cfg, state)})
    eligible = [s for s in scored if not s["breakdown"]["blocked"]]
    blocked = [s for s in scored if s["breakdown"]["blocked"]]
    # Stable order: score desc, then list order
    eligible.sort(key=lambda s: s["breakdown"]["score"], reverse=True)
    selected = eligible[0] if eligible else None
    rejected: list[dict[str, Any]] = []
    for s in eligible[1:]:
        rejected.append(
            {
                "template_id": s["venue"].template_id,
                "score": s["breakdown"]["score"],
                "reason": "lower_score_than_selected",
            }
        )
    for s in blocked:
        rejected.append(
            {
                "template_id": s["venue"].template_id,
                "score": s["breakdown"]["score"],
                "reason": "; ".join(s["breakdown"]["block_reasons"]) or "blocked",
            }
        )
    return {"selected": selected, "rejected": rejected, "all_scored": scored}


# ---------- persistence ----------


def route_and_persist(
    selection_id: int,
    state: Mapping[str, Any],
    *,
    venue_cfgs: Iterable[VenueConfig] | None = None,
    rule_set: str = DEFAULT_RULE_SET,
) -> dict[str, Any]:
    """Score venues for ``state``, persist into ``manuscript_venue_selections``."""
    if venue_cfgs is None:
        venue_cfgs = load_venue_config()
    venue_cfgs = list(venue_cfgs)
    if not venue_cfgs:
        raise RuntimeError("no venues configured; check VENUES_CONFIG_PATH")
    # Verify every referenced template_id resolves to a registered adapter.
    registered = set(list_adapters())
    for cfg in venue_cfgs:
        if cfg.template_id not in registered:
            raise RuntimeError(
                f"venue refers to unknown template_id {cfg.template_id!r}; "
                f"registered adapters: {sorted(registered)}"
            )

    result = evaluate_venues(state, venue_cfgs)
    selected = result["selected"]
    rejected = result["rejected"]

    if selected is None:
        chosen_template_id = None
        score = None
        rationale = f"all {len(venue_cfgs)} venues blocked by rule_set={rule_set}"
    else:
        chosen_template_id = selected["venue"].template_id
        score = float(selected["breakdown"]["score"])
        adapter = get_adapter(chosen_template_id)
        matched = selected["breakdown"].get("matched_keywords") or []
        rationale = (
            f"venue={chosen_template_id} (label={adapter.venue_label}) "
            f"score={score:.2f}"
            + (f"; keywords={','.join(matched)}" if matched else "")
        )

    new_id = db.insert_returning_id(
        """
        INSERT INTO manuscript_venue_selections
            (selection_id, chosen_template_id, score, rationale,
             rejected_venues_json, scoring_breakdown_json, rule_set, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            int(selection_id),
            chosen_template_id,
            score,
            rationale,
            json.dumps(rejected, ensure_ascii=False),
            json.dumps(
                {"all_scored": [s["breakdown"] for s in result["all_scored"]]},
                ensure_ascii=False,
            ),
            rule_set,
            "selected" if selected else "blocked",
        ),
    )
    db.commit()
    return {
        "routing_id": int(new_id),
        "selection_id": int(selection_id),
        "chosen_template_id": chosen_template_id,
        "score": score,
        "rationale": rationale,
        "rejected_venues": rejected,
        "rule_set": rule_set,
    }


# ---------- tiebreaker (D3 #14) ----------


def needs_tiebreak(scored: list[dict[str, Any]]) -> bool:
    """Return True iff the top two non-blocked venues differ by < threshold."""
    eligible = [s for s in scored if not s["breakdown"]["blocked"]]
    eligible.sort(key=lambda s: s["breakdown"]["score"], reverse=True)
    if len(eligible) < 2:
        return False
    top = eligible[0]["breakdown"]["score"]
    runner_up = eligible[1]["breakdown"]["score"]
    return (top - runner_up) < TIEBREAK_SCORE_DELTA


def tiebreak_with_llm(
    state: Mapping[str, Any],
    scored: list[dict[str, Any]],
    *,
    llm_caller=None,
) -> dict[str, Any]:
    """Resolve a near-tie between the top venues by calling an LLM.

    Parameters
    ----------
    state
        The same dict passed to ``evaluate_venues``.
    scored
        The ``all_scored`` list returned by ``evaluate_venues``.
    llm_caller
        Optional ``(prompt: str) -> str`` callable. The string return value
        MUST be the chosen ``template_id``. If ``None``, a deterministic
        fallback picks the higher-tier-first / file-order-first venue so
        tests stay reproducible without network access.

    Returns
    -------
    dict
        ``{"chosen_template_id": str, "rationale": str, "candidates": [...],
        "used_llm": bool}``. Caller should record this alongside the
        scoring breakdown.
    """
    eligible = [s for s in scored if not s["breakdown"]["blocked"]]
    eligible.sort(key=lambda s: s["breakdown"]["score"], reverse=True)
    candidates = eligible[:2]
    if not candidates:
        return {
            "chosen_template_id": None,
            "rationale": "no eligible venues to tiebreak between",
            "candidates": [],
            "used_llm": False,
        }
    if len(candidates) == 1:
        return {
            "chosen_template_id": candidates[0]["venue"].template_id,
            "rationale": "only one eligible venue; no tiebreak needed",
            "candidates": [c["venue"].template_id for c in candidates],
            "used_llm": False,
        }

    cand_ids = [c["venue"].template_id for c in candidates]
    if llm_caller is None:
        # Deterministic fallback: keep the first candidate (already the
        # higher-scoring one; the runner-up only got here because the gap
        # was sub-threshold). This matches the file-order tie-break used
        # by ``evaluate_venues``.
        return {
            "chosen_template_id": cand_ids[0],
            "rationale": (
                f"near-tie ({candidates[0]['breakdown']['score']:.3f} vs "
                f"{candidates[1]['breakdown']['score']:.3f}); deterministic "
                f"fallback picked file-order leader {cand_ids[0]!r}"
            ),
            "candidates": cand_ids,
            "used_llm": False,
        }

    prompt = (
        "Two venues are tied for a manuscript routing decision. State summary:\n"
        f"  title={state.get('title')!r}\n"
        f"  claim_type={state.get('claim_type')!r}\n"
        f"  domain={state.get('domain')!r}\n"
        f"  has_real_data={state.get('has_real_data')}\n"
        f"  page_count_estimate={state.get('page_count_estimate')}\n"
        f"Candidates (pick exactly one template_id verbatim):\n"
    )
    for c in candidates:
        bd = c["breakdown"]
        prompt += (
            f"  - template_id={c['venue'].template_id!r} "
            f"score={bd['score']:.3f} "
            f"matched_keywords={bd.get('matched_keywords') or []}\n"
        )
    prompt += "Return only the chosen template_id, no other text."

    raw = llm_caller(prompt)
    chosen = (raw or "").strip()
    if chosen not in cand_ids:
        # Defensive: if the LLM hallucinated, fall back to file-order leader
        # so the pipeline doesn't crash on a bad response.
        return {
            "chosen_template_id": cand_ids[0],
            "rationale": (
                f"LLM returned {chosen!r} not in candidates {cand_ids!r}; "
                f"defensive fallback picked {cand_ids[0]!r}"
            ),
            "candidates": cand_ids,
            "used_llm": True,
        }
    return {
        "chosen_template_id": chosen,
        "rationale": (
            f"near-tie ({candidates[0]['breakdown']['score']:.3f} vs "
            f"{candidates[1]['breakdown']['score']:.3f}); LLM picked {chosen!r}"
        ),
        "candidates": cand_ids,
        "used_llm": True,
    }


def get_routing(selection_id: int) -> dict[str, Any] | None:
    """Return the latest routing row for ``selection_id`` as a dict, or None."""
    row = db.fetchone(
        """
        SELECT * FROM manuscript_venue_selections
        WHERE selection_id=?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (int(selection_id),),
    )
    if not row:
        return None

    def _decode(field_name: str, default: Any) -> Any:
        v = row.get(field_name)
        if isinstance(v, str):
            try:
                return json.loads(v) if v else default
            except json.JSONDecodeError:
                return default
        return v if v is not None else default

    return {
        "id": row.get("id"),
        "selection_id": row.get("selection_id"),
        "chosen_template_id": row.get("chosen_template_id"),
        "score": row.get("score"),
        "rationale": row.get("rationale"),
        "rejected_venues": _decode("rejected_venues_json", []),
        "scoring_breakdown": _decode("scoring_breakdown_json", {}),
        "rule_set": row.get("rule_set"),
        "status": row.get("status"),
        "created_at": row.get("created_at"),
    }
