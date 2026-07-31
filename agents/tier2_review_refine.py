"""Pre-insert review and iterative refinement for Tier 2 paper ideas."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from agents.evosci_requirements import evosci_binary_path
from agents.llm_client import (
    call_llm_json,
    call_llm_json_with_provider,
    get_provider_models,
)
from config import (
    TIER2_DEBATE_ROUNDS,
    TIER2_EVOSCI_PREINSERT_REVIEW,
    TIER2_EVOSCI_REVIEW_TIMEOUT_SECONDS,
)


EVOSCI_TIER2_REVIEW_PROMPT = """Run this headless EvoScientist novelty review now.

Do not ask the user questions. Do not wait for approval. Do not stop until you create `final_report.md`.

Required actions:
1. Use the research-agent to search the web/literature for the idea title, method name, and the closest baseline phrases in the Idea JSON.
2. If search is unavailable or weak, continue from internal literature knowledge and say so.
3. Write `final_report.md` with exactly these sections:
   - Verdict: NOVEL / PARTIALLY_EXISTS / EXISTS
   - Search Notes
   - Exact Matches
   - Partial Matches
   - Closest Prior Work
   - Field Baselines
   - Experiment Critique
   - Recommended Refinement

Keep the report concise, concrete, and under 900 words. Use `write_file` to create `final_report.md`.

Idea JSON:
{idea_json}
"""


REVIEWER_A_SYSTEM = """You are Reviewer A, a novelty and related-work specialist.

You are given a Tier-2 ML paper idea and an EvoScientist research report. Your job is to attack overlap with prior work and force sharper novelty.

Return JSON:
{
  "reviewer": "A",
  "novelty_risk": 0-10,
  "overlap_findings": ["..."],
  "must_change": ["..."],
  "method_improvements": ["..."],
  "experiment_improvements": ["..."],
  "accept_if": ["..."]
}

Be concrete. Do not praise. Identify what must change before this is a real paper."""


REVIEWER_B_SYSTEM = """You are Reviewer B, a methods and experiments specialist.

You are given a Tier-2 ML paper idea, EvoScientist's research report, and Reviewer A's critique. Your job is to debate A and improve the algorithm and experiment design.

Return JSON:
{
  "reviewer": "B",
  "agreement_with_A": 0-10,
  "method_failure_modes": ["..."],
  "algorithmic_revisions": ["..."],
  "experimental_design_revisions": ["..."],
  "falsification_tests": ["..."],
  "remaining_blockers": ["..."]
}

Focus on whether the method is technically discriminative and experimentally testable."""


INVENTOR_SYSTEM = """You are the Inventor/Refactorer in a Tier-2 idea debate.

Your job is not to review defensively. Your job is to raise the idea's field leverage.
Given the idea, EvoScientist report, and reviewer critiques, ask:
- Can this evaluation or diagnostic idea be converted into a method, optimizer, loss, training objective, inference policy, or reusable algorithm?
- Is there a deeper formal object hidden underneath the current formulation?
- What mechanism would make the contribution useful beyond one benchmark or dataset?
- What should be penalized as "just another benchmark paper"?

Return JSON:
{
  "role": "inventor_refactorer",
  "excitement_score": 0-10,
  "field_leverage": 0-10,
  "benchmark_paper_risk": 0-10,
  "deeper_formal_object": "...",
  "method_conversion": {
    "is_possible": true,
    "converted_method_name": "...",
    "converted_method_sketch": "...",
    "new_objective_or_algorithm": "..."
  },
  "refactor_actions": ["..."],
  "keep_as_evaluation_only_if": ["..."],
  "stronger_experiment": ["..."]
}

Be ambitious but concrete. If the idea should remain an evaluation paper, say exactly what would make it high leverage."""


REFINER_SYSTEM = """You are the Tier-2 Idea Refiner.

Use the EvoScientist report, both reviewer critiques, and the Inventor/Refactorer proposal to revise the paper idea. Keep the idea executable and top-venue plausible. Do not invent large compute requirements unless necessary.

Return JSON:
{
  "title": "...",
  "problem_statement": "...",
  "existing_weakness": "...",
  "proposed_method": {
    "name": "...",
    "type": "...",
    "one_line": "...",
    "definition": "...",
    "mechanism_repair": "...",
    "why_novel": "...",
    "falsification_hook": "..."
  },
  "experimental_plan": {
    "baselines": ["..."],
    "datasets": ["..."],
    "metrics": {},
    "ablations": ["..."],
    "expected_results": {},
    "compute_budget": {},
    "risks": ["..."]
  },
  "related_work_positioning": {},
  "problem_awareness": {
    "central_question": "...",
    "motivation": "...",
    "method_answer": "...",
    "result_claim": "...",
    "falsification_result": "..."
  },
  "excitement_refactor": {
    "excitement_score": 0-10,
    "field_leverage": 0-10,
    "benchmark_paper_risk": 0-10,
    "what_changed_from_diagnostic_to_mechanism": "..."
  },
  "change_log": ["..."]
}

Preserve valid pieces of the current idea. Replace weak, overlapping, non-discriminative, or merely benchmark-like pieces."""


def _json_load(value: Any, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _cleanup_stale_evosci_langgraph() -> None:
    """Stop stale EvoScientist langgraph dev servers that block port 6174."""
    try:
        import signal

        me = os.getpid()
        needle = "/root/EvoScientist/.venv/bin/langgraph dev"
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == me:
                continue
            try:
                cmd = (Path("/proc") / pid / "cmdline").read_bytes().decode(
                    "utf-8", errors="ignore"
                ).replace("\x00", " ")
            except Exception:
                continue
            if needle in cmd:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except Exception:
                    pass
    except Exception:
        pass


def _idea_for_review(insight: dict) -> dict:
    payload = dict(insight)
    for key in ("proposed_method", "experimental_plan", "related_work_positioning", "problem_awareness", "source_node_ids", "signal_mix", "evidence_packet"):
        payload[key] = _json_load(payload.get(key), payload.get(key))
    return payload


def _truncate_for_prompt(value: Any, *, limit: int = 2200) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "..."
    if isinstance(value, list):
        return [_truncate_for_prompt(item, limit=limit) for item in value[:8]]
    if isinstance(value, dict):
        return {key: _truncate_for_prompt(val, limit=limit) for key, val in value.items()}
    return value


def _named_items_for_prompt(value: Any, *, limit: int = 5) -> list[Any]:
    if not isinstance(value, list):
        return []
    items: list[Any] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            compact = {
                key: item.get(key)
                for key in ("name", "source_paper", "model")
                if item.get(key)
            }
            items.append(_truncate_for_prompt(compact, limit=500))
        else:
            items.append(_truncate_for_prompt(item, limit=500))
    return items


def _metrics_for_prompt(value: Any) -> Any:
    if not isinstance(value, dict):
        return _truncate_for_prompt(value, limit=160)
    return {
        "primary": _truncate_for_prompt(value.get("primary"), limit=160),
    }


def _evidence_for_prompt(value: dict) -> dict:
    return {
        "signal_mix": _truncate_for_prompt(value.get("signal_mix"), limit=400),
        "non_numeric_evidence": _truncate_for_prompt((value.get("non_numeric_evidence") or [])[:2], limit=160),
        "structural_evidence": _truncate_for_prompt((value.get("structural_evidence") or [])[:1], limit=220),
    }


def _idea_for_evosci_review(insight: dict) -> dict:
    """Pass EvoScientist a bounded research packet, not the full Tier2 spec."""
    idea = _idea_for_review(insight)
    method = idea.get("proposed_method") if isinstance(idea.get("proposed_method"), dict) else {}
    experiments = idea.get("experimental_plan") if isinstance(idea.get("experimental_plan"), dict) else {}
    related = idea.get("related_work_positioning") if isinstance(idea.get("related_work_positioning"), dict) else {}
    awareness = idea.get("problem_awareness") if isinstance(idea.get("problem_awareness"), dict) else {}
    evidence = idea.get("evidence_packet") if isinstance(idea.get("evidence_packet"), dict) else {}
    main_claim = experiments.get("main_claim") if isinstance(experiments.get("main_claim"), dict) else experiments.get("main_claim")
    if isinstance(main_claim, dict):
        main_claim = main_claim.get("solid") or main_claim.get("exciting") or next(iter(main_claim.values()), None)

    compact = {
        "title": _truncate_for_prompt(idea.get("title"), limit=220),
        "problem_statement": _truncate_for_prompt(idea.get("problem_statement"), limit=260),
        "existing_weakness": _truncate_for_prompt(idea.get("existing_weakness"), limit=220),
        "method": {
            "name": _truncate_for_prompt(method.get("name"), limit=180),
            "type": method.get("type"),
            "one_line": _truncate_for_prompt(method.get("one_line"), limit=260),
            "definition_summary": _truncate_for_prompt(method.get("definition"), limit=260),
            "key_properties": _truncate_for_prompt((method.get("key_properties") or [])[:1], limit=140),
        },
        "experimental_plan": {
            "datasets": _named_items_for_prompt(experiments.get("datasets") or experiments.get("benchmarks")),
            "baselines": _named_items_for_prompt(experiments.get("baselines")),
            "metrics": _metrics_for_prompt(experiments.get("metrics")),
            "main_claim": _truncate_for_prompt(main_claim or experiments.get("expected_results"), limit=180),
            "falsification": _truncate_for_prompt(experiments.get("falsification"), limit=160),
        },
        "related_work_positioning": {
            "abstract_sketch": _truncate_for_prompt(related.get("abstract_sketch"), limit=220),
            "related_work_sections": _truncate_for_prompt((related.get("related_work_sections") or [])[:5], limit=120),
        },
        "evidence_summary": _truncate_for_prompt(idea.get("evidence_summary"), limit=260),
        "signal_mix": idea.get("signal_mix"),
        "source_node_ids": (idea.get("source_node_ids") or [])[:5],
        "evidence_packet": _evidence_for_prompt(evidence),
    }
    return compact


def _parse_evosci_verdict(report: str) -> str:
    lower = (report or "").lower()
    verdict_window = lower.split("verdict", 1)[1][:200] if "verdict" in lower else lower[:500]
    if "partially_exists" in verdict_window or "partially exists" in verdict_window:
        return "partially_exists"
    if "exists" in verdict_window and "does not exist" not in verdict_window:
        return "exists"
    if "novel" in verdict_window:
        return "novel"
    return "unknown"


def run_evosci_tier2_review(insight: dict, *, timeout_seconds: int | None = None) -> dict:
    """Run EvoScientist synchronously for a pre-insert Tier2 novelty review."""
    if not TIER2_EVOSCI_PREINSERT_REVIEW:
        return {"status": "skipped", "reason": "disabled"}

    timeout_seconds = timeout_seconds or TIER2_EVOSCI_REVIEW_TIMEOUT_SECONDS
    evosci_bin = Path(evosci_binary_path())
    if not evosci_bin.exists():
        return {"status": "failed", "error": f"EvoScientist not found at {evosci_bin}"}

    from agents.novelty_verifier import _build_evosci_env

    stamp = int(time.time())
    safe_title = "".join(ch if ch.isalnum() else "_" for ch in str(insight.get("title") or "tier2")[:48])
    workdir = Path.home() / "research" / f"tier2_preinsert_{safe_title}_{stamp}"
    workdir.mkdir(parents=True, exist_ok=True)
    idea_json = json.dumps(_idea_for_evosci_review(insight), ensure_ascii=False, indent=2)
    prompt = EVOSCI_TIER2_REVIEW_PROMPT.format(idea_json=idea_json)
    (workdir / "preinsert_review_prompt.md").write_text(prompt, encoding="utf-8")
    cli_prompt = (
        "Read /preinsert_review_prompt.md, execute its instructions exactly, "
        "use research-agent search as requested, and write /final_report.md before stopping."
    )
    log_path = workdir / "evoscientist.log"

    try:
        env = _build_evosci_env(workdir)
    except Exception as exc:
        return {"status": "failed", "workdir": str(workdir), "error": str(exc)}

    _cleanup_stale_evosci_langgraph()
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        try:
            completed = subprocess.run(
                [
                    str(evosci_bin),
                    "--workdir",
                    str(workdir),
                    "--auto-approve",
                    "--auto-mode",
                    "--no-thinking",
                    "--ui",
                    "cli",
                    "-p",
                    cli_prompt,
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=workdir,
                env=env,
                timeout=timeout_seconds,
                check=False,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(
                    ["pkill", "-f", "/root/EvoScientist/.venv/bin/langgraph dev"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
            return {
                "status": "failed",
                "workdir": str(workdir),
                "error": f"EvoScientist timed out after {timeout_seconds}s",
            }

    report_text = ""
    report_path = None
    for name in ("novelty_report.md", "final_report.md"):
        path = workdir / name
        if path.exists() and path.stat().st_size > 50:
            report_text = path.read_text(encoding="utf-8", errors="replace")
            report_path = str(path)
            break
    if not report_text and log_path.exists():
        report_text = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]

    verdict = _parse_evosci_verdict(report_text)
    return {
        "status": "complete" if report_path else "failed",
        "workdir": str(workdir),
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 2),
        "verdict": verdict,
        "report_preview": report_text[:4000],
        "report_path": report_path,
        "error": None if report_path else "EvoScientist did not write novelty_report.md or final_report.md",
    }


def _review_prompt(insight: dict, evosci_review: dict, history: list[dict], round_idx: int) -> str:
    return json.dumps(
        {
            "round": round_idx,
            "idea": _idea_for_review(insight),
            "evoscientist_review": evosci_review,
            "recent_debate_history": history[-3:],
        },
        ensure_ascii=False,
        indent=2,
    )


def _reviewer_context(review_a: dict, review_b: dict, history: list[dict]) -> dict:
    return {
        "current_reviewer_A": review_a,
        "current_reviewer_B": review_b,
        "recent_rounds": history[-3:],
        "instruction": (
            "Reviewer A and Reviewer B critiques are upstream constraints. "
            "Downstream agents must preserve unresolved blockers, must_change items, "
            "falsification tests, and experiment requirements unless they explicitly "
            "replace them with a stronger formulation."
        ),
    }


def _call_reviewer(system: str, prompt: str, provider_index: int) -> tuple[dict, int, dict]:
    payload, tokens, provider = call_llm_json_with_provider(
        system,
        prompt,
        provider_index=provider_index,
    )
    return payload if isinstance(payload, dict) else {}, tokens, provider


def _apply_refinement(insight: dict, refined: dict) -> dict:
    updated = dict(insight)
    if refined.get("title"):
        updated["title"] = refined["title"]
    for key in ("problem_statement", "existing_weakness"):
        if refined.get(key):
            updated[key] = refined[key]
    for key in ("proposed_method", "experimental_plan", "related_work_positioning", "problem_awareness"):
        value = refined.get(key)
        if isinstance(value, (dict, list)) and value:
            updated[key] = _json_dump(value)
    return updated


def debate_and_refine_tier2_idea(
    insight: dict,
    evosci_review: dict,
    *,
    rounds: int | None = None,
) -> tuple[dict, dict]:
    """Run reviewer debate and return (updated_insight, metadata)."""
    rounds = max(0, int(TIER2_DEBATE_ROUNDS if rounds is None else rounds))
    providers = get_provider_models()
    if rounds and len(providers) < 2:
        raise RuntimeError(
            "Tier-2 evaluator debate requires two independently routed providers; "
            "manual review required"
        )
    if rounds and (
        providers[0].get("name") == providers[1].get("name")
        and providers[0].get("model_family") == providers[1].get("model_family")
    ):
        raise RuntimeError(
            "Tier-2 evaluator routes are not independent; manual review required"
        )
    reviewer_a_provider = 0
    reviewer_b_provider = 1
    history: list[dict] = []
    total_tokens = 0
    updated = dict(insight)

    for round_idx in range(1, rounds + 1):
        print(f"[TIER2_DEBATE] Round {round_idx}/{rounds}: reviewer A", flush=True)
        prompt_a = _review_prompt(updated, evosci_review, history, round_idx)
        review_a, tokens_a, provider_a = _call_reviewer(REVIEWER_A_SYSTEM, prompt_a, reviewer_a_provider)
        total_tokens += tokens_a

        print(f"[TIER2_DEBATE] Round {round_idx}/{rounds}: reviewer B", flush=True)
        prompt_b = json.dumps(
            {
                "round": round_idx,
                "idea": _idea_for_review(updated),
                "evoscientist_review": evosci_review,
                "reviewer_A": review_a,
                "recent_debate_history": history[-3:],
            },
            ensure_ascii=False,
            indent=2,
        )
        review_b, tokens_b, provider_b = _call_reviewer(REVIEWER_B_SYSTEM, prompt_b, reviewer_b_provider)
        total_tokens += tokens_b

        print(f"[TIER2_DEBATE] Round {round_idx}/{rounds}: inventor/refactorer", flush=True)
        inventor_prompt = json.dumps(
            {
                "round": round_idx,
                "idea": _idea_for_review(updated),
                "evoscientist_review": evosci_review,
                "reviewer_context": _reviewer_context(review_a, review_b, history),
                "recent_debate_history": history[-3:],
            },
            ensure_ascii=False,
            indent=2,
        )
        inventor, tokens_i, provider_i = _call_reviewer(INVENTOR_SYSTEM, inventor_prompt, reviewer_a_provider)
        total_tokens += tokens_i

        print(f"[TIER2_DEBATE] Round {round_idx}/{rounds}: refiner", flush=True)
        refine_prompt = json.dumps(
            {
                "round": round_idx,
                "idea": _idea_for_review(updated),
                "evoscientist_review": evosci_review,
                "reviewer_context": _reviewer_context(review_a, review_b, history),
                "inventor_refactorer": inventor,
                "recent_debate_history": history[-3:],
            },
            ensure_ascii=False,
            indent=2,
        )
        refined, tokens_r = call_llm_json(REFINER_SYSTEM, refine_prompt)
        total_tokens += tokens_r
        refined_payload = refined if isinstance(refined, dict) else {}
        updated = _apply_refinement(updated, refined_payload)
        print(
            f"[TIER2_DEBATE] Round {round_idx}/{rounds} done: "
            f"excitement={((refined_payload.get('excitement_refactor') or {}).get('excitement_score'))}",
            flush=True,
        )
        history.append(
            {
                "round": round_idx,
                "reviewer_A": review_a,
                "reviewer_B": review_b,
                "inventor_refactorer": inventor,
                "refiner_change_log": refined_payload.get("change_log", []),
                "excitement_refactor": refined_payload.get("excitement_refactor", {}),
                "providers": {
                    "A": {"name": provider_a.get("name"), "model": provider_a.get("model")},
                    "B": {"name": provider_b.get("name"), "model": provider_b.get("model")},
                    "inventor": {"name": provider_i.get("name"), "model": provider_i.get("model")},
                },
            }
        )

    metadata = {
        "rounds_requested": rounds,
        "rounds_completed": len(history),
        "tokens": total_tokens,
        "providers": providers,
        "same_provider_fallback": len(providers) < 2,
        "history_preview": history[-3:],
    }
    return updated, metadata


def review_and_refine_tier2_idea(insight: dict) -> dict:
    """Run required EvoScientist review plus debate refinement for a Tier2 idea."""
    from agents.idea_taste import score_excitement

    evosci_review = run_evosci_tier2_review(insight)
    if TIER2_EVOSCI_PREINSERT_REVIEW and evosci_review.get("status") != "complete":
        return {
            "accepted": False,
            "reason": "evoscientist_review_failed",
            "evosci_review": evosci_review,
            "insight": insight,
        }
    if evosci_review.get("verdict") == "exists":
        return {
            "accepted": False,
            "reason": "evoscientist_found_existing_overlap",
            "evosci_review": evosci_review,
            "insight": insight,
        }

    refined, debate_meta = debate_and_refine_tier2_idea(insight, evosci_review)
    excitement = score_excitement(refined)
    novelty_payload = _json_load(refined.get("novelty_report"), {})
    novelty_payload["evosci_preinsert_review"] = evosci_review
    novelty_payload["tier2_debate_refinement"] = debate_meta
    novelty_payload["excitement"] = excitement
    refined["novelty_report"] = _json_dump(novelty_payload)
    refined["novelty_status"] = (
        "prechecked_partially_exists"
        if evosci_review.get("verdict") == "partially_exists"
        else "prechecked_novel"
    )
    refined["generation_tokens"] = int(refined.get("generation_tokens") or 0) + int(debate_meta.get("tokens") or 0)
    refined["llm_calls"] = int(refined.get("llm_calls") or 0) + int(debate_meta.get("rounds_completed") or 0) * 4
    return {
        "accepted": True,
        "reason": "reviewed_and_refined",
        "evosci_review": evosci_review,
        "debate_refinement": debate_meta,
        "insight": refined,
    }
