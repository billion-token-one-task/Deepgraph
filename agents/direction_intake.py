"""User research-direction intake: deterministic YAML -> ResearchAgenda mapping.

Input schema (user-facing, see research_agendas/inbox/README.md):

    direction: "..."            # required, natural language
    keywords: [a, b, c]         # optional
    constraints:                # optional, free text
      compute: "..."
      data: "..."
    goal: experiment_plan       # idea_only | experiment_plan | signal | verified_evidence
    contact: "..."              # required, nickname or email
    token_budget: 500000        # optional token cap; default 0 = no cap,
                                # usage is recorded either way

Mapping (rule-based, no LLM):
    direction            -> description (+ auto-generated slug name)
    keywords             -> focus
    constraints.compute  -> prefer.resource_class via keyword rules (original
                            text is preserved in raw_config)
    goal                 -> required_output.goal
    contact              -> submitter

`parse_direction_yaml` also returns an echo dict: a templated summary of what
the system understood, for the submitter to confirm.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

import yaml  # type: ignore

from agents.agenda_loader import parse_agenda
from contracts.agenda import ResearchAgenda
from contracts.base import ContractValidationError, ensure_string_list


VALID_GOALS = ("idea_only", "experiment_plan", "signal", "verified_evidence")
DEFAULT_GOAL = "experiment_plan"

GOAL_LABELS_ZH = {
    "idea_only": "仅研究想法",
    "experiment_plan": "可执行实验计划",
    "signal": "结构化信号报告",
    "verified_evidence": "经验证的实验证据",
}

# Compute-constraint keyword rules, checked in order; first match wins.
# Conservative on purpose: anything that sounds like "small machine" maps to
# the cheaper resource classes used by deep_insights.resource_class.
_COMPUTE_RULES: tuple[tuple[tuple[str, ...], list[str]], ...] = (
    (
        ("cpu", "无gpu", "no gpu", "笔记本", "laptop", "notebook"),
        ["cpu"],
    ),
    (
        (
            "单卡", "单gpu", "single gpu", "single-gpu", "1 gpu", "one gpu", "1gpu",
            "colab", "t4", "消费级", "consumer gpu",
        ),
        ["cpu", "gpu_small"],
    ),
)


class DirectionParseError(ValueError):
    """Raised when a direction submission cannot be mapped to an agenda."""


def map_compute_constraint(text: Any) -> list[str] | None:
    """Best-effort keyword mapping from a free-text compute constraint."""
    blob = str(text or "").strip().lower()
    if not blob:
        return None
    for markers, resource_classes in _COMPUTE_RULES:
        if any(marker in blob for marker in markers):
            return list(resource_classes)
    return None


def _ascii_tokens(text: str, *, min_len: int = 3, max_tokens: int = 8) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", str(text or "")):
        tok = tok.lower()
        if len(tok) < min_len or tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
        if len(tokens) >= max_tokens:
            break
    return tokens


def _slug_name(direction: str, keywords: list[str]) -> str:
    """Deterministic agenda name: slug of keywords/direction + short hash."""
    basis = keywords if keywords else _ascii_tokens(direction, min_len=3, max_tokens=4)
    slug = "-".join(re.sub(r"[^a-z0-9]+", "-", k.lower()).strip("-") for k in basis[:4])
    slug = re.sub(r"-{2,}", "-", slug).strip("-")[:48]
    digest = hashlib.sha1(direction.encode("utf-8")).hexdigest()[:8]
    return f"direction-{slug}-{digest}" if slug else f"direction-{digest}"


def parse_direction_payload(payload: Mapping[str, Any]) -> ResearchAgenda:
    """Map a parsed direction dict to a validated ResearchAgenda."""
    if not isinstance(payload, Mapping):
        raise DirectionParseError("direction submission must be a YAML mapping")

    direction = str(payload.get("direction") or "").strip()
    if not direction:
        raise DirectionParseError("'direction' is required (natural-language research direction)")

    contact = str(payload.get("contact") or "").strip()
    if not contact:
        raise DirectionParseError("'contact' is required (nickname or email)")

    goal = str(payload.get("goal") or DEFAULT_GOAL).strip().lower()
    if goal not in VALID_GOALS:
        raise DirectionParseError(
            f"'goal' must be one of {list(VALID_GOALS)}, got '{goal}'"
        )

    keywords = ensure_string_list(payload.get("keywords") or [])
    focus = keywords or _ascii_tokens(direction)

    constraints = payload.get("constraints")
    if constraints is not None and not isinstance(constraints, Mapping):
        raise DirectionParseError("'constraints' must be a mapping of free-text fields")
    constraints = dict(constraints or {})

    prefer: dict[str, Any] = {}
    resource_classes = map_compute_constraint(constraints.get("compute"))
    if resource_classes:
        prefer["resource_class"] = resource_classes

    if not focus and not prefer:
        raise DirectionParseError(
            "could not derive any scope keywords; add a 'keywords' list "
            "(the direction text has no extractable terms)"
        )

    token_budget = payload.get("token_budget")
    if token_budget is not None:
        try:
            token_budget = int(token_budget)
        except (TypeError, ValueError) as exc:
            raise DirectionParseError("'token_budget' must be an integer") from exc

    agenda_payload: dict[str, Any] = {
        "version": "v1",
        "name": _slug_name(direction, keywords),
        "description": direction,
        "focus": focus,
        "prefer": prefer,
        "required_output": {"goal": goal},
        "submitter": contact,
        # Keep the original submission verbatim for auditability.
        "source": "direction_intake_v1",
        "direction": direction,
        "keywords": keywords,
        "constraints": constraints,
        "goal": goal,
        "contact": contact,
    }
    if token_budget is not None:
        agenda_payload["token_budget"] = token_budget

    try:
        return parse_agenda(agenda_payload)
    except ContractValidationError as exc:
        raise DirectionParseError(f"mapped agenda failed validation: {exc}") from exc


def build_echo(agenda: ResearchAgenda, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Templated confirmation of what the system understood (for the submitter)."""
    constraints = payload.get("constraints")
    constraints = dict(constraints) if isinstance(constraints, Mapping) else {}
    goal = str(agenda.required_output.get("goal") or DEFAULT_GOAL)
    resource_classes = ensure_string_list((agenda.prefer or {}).get("resource_class"))

    parts = [
        f"已登记研究方向：{agenda.description}",
        f"识别到的范围关键词：{'、'.join(agenda.focus) if agenda.focus else '（无，建议补充 keywords）'}",
        f"目标产出：{GOAL_LABELS_ZH.get(goal, goal)}（{goal}）",
    ]
    if constraints.get("compute"):
        mapped = "、".join(resource_classes) if resource_classes else "未识别（原文已保留）"
        parts.append(f"算力约束：{constraints['compute']} → 资源档位 {mapped}")
    if constraints.get("data"):
        parts.append(f"数据约束：{constraints['data']}（原文保留，供研究执行时参考）")
    parts.append(f"联系人：{agenda.submitter}")

    return {
        "type": "direction_intake_echo",
        "name": agenda.name,
        "direction": agenda.description,
        "focus": list(agenda.focus),
        "goal": goal,
        "constraints": constraints,
        "resource_class": resource_classes or None,
        "submitter": agenda.submitter,
        "summary": "；".join(parts),
    }


def parse_direction_yaml(text: str) -> tuple[ResearchAgenda, dict[str, Any]]:
    """Parse a direction YAML document into (ResearchAgenda, echo dict)."""
    if not str(text or "").strip():
        raise DirectionParseError("empty submission")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DirectionParseError(f"invalid YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DirectionParseError("direction submission must be a YAML mapping")
    agenda = parse_direction_payload(payload)
    return agenda, build_echo(agenda, payload)
