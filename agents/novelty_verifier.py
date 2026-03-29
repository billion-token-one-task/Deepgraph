"""Novelty Verifier: use EvoScientist to check if a deep insight already exists in literature.

Modes:
- Verification (5-15 min): quick check for prior work
- Full Research (1-3 hrs): deep dive with experiment planning

For each candidate:
1. Construct search queries from the insight's core claim
2. Launch EvoScientist in verification mode (shorter)
3. Parse results → update candidate status
"""
import json
import os
import subprocess
import time
from pathlib import Path

from db import database as db


def _build_verification_prompt(insight: dict) -> str:
    """Build a focused search prompt for novelty verification."""
    tier = insight.get("tier", 0)
    title = insight.get("title", "")

    if tier == 1:
        formal = insight.get("formal_structure", "")
        field_a = json.loads(insight["field_a"]) if insight.get("field_a") else {}
        field_b = json.loads(insight["field_b"]) if insight.get("field_b") else {}
        transform = insight.get("transformation", "")

        return f"""You are a novelty checker. Search the academic literature (arXiv, Semantic Scholar, Google Scholar) to determine if the following paradigm-level insight already exists.

## Insight to Verify
**Title**: {title}

**Claim**: A structural connection exists between:
- Field A: {field_a.get('node_id', '?')} — {field_a.get('phenomenon', '')}
- Field B: {field_b.get('node_id', '?')} — {field_b.get('phenomenon', '')}

**Formal Structure**: {formal[:500]}

**Transformation**: {transform[:300]}

## Search Strategy
1. Search for papers that EXPLICITLY connect these two fields
2. Search for the mathematical structure described above
3. Search for surveys or position papers noting this connection
4. Check recent workshops (2024-2026) on cross-disciplinary ML

## Output Requirements
Write a file called `novelty_report.md` with:
1. **Verdict**: NOVEL / PARTIALLY_EXISTS / EXISTS
2. **Exact Matches**: Papers that make the SAME claim (if any)
3. **Partial Matches**: Papers that note PART of this connection
4. **Closest Prior Work**: The single most relevant prior paper
5. **What's New**: If partially exists, what specific aspect is still novel
6. **Recommended Refinement**: If partially exists, how to sharpen the novelty

Be thorough but fast. Search at least 5 different queries."""

    else:  # tier 2
        method = json.loads(insight["proposed_method"]) if insight.get("proposed_method") else {}
        problem = insight.get("problem_statement", "")
        weakness = insight.get("existing_weakness", "")

        return f"""You are a novelty checker. Search the academic literature to determine if the following proposed method already exists.

## Method to Verify
**Title**: {title}

**Problem**: {problem[:300]}

**Current Weakness**: {weakness[:300]}

**Proposed Method**:
- Name: {method.get('name', '?')}
- Type: {method.get('type', '?')}
- Summary: {method.get('one_line', '')}
- Definition: {method.get('definition', '')[:400]}

## Search Strategy
1. Search for the method NAME or similar names
2. Search for the TECHNIQUE TYPE applied to this problem
3. Search for papers addressing the SAME failure mode
4. Check if the mathematical formulation matches any existing work
5. Look for concurrent/recent (2025-2026) submissions

## Output Requirements
Write a file called `novelty_report.md` with:
1. **Verdict**: NOVEL / PARTIALLY_EXISTS / EXISTS
2. **Exact Matches**: Papers proposing the SAME method (if any)
3. **Partial Matches**: Papers using SIMILAR techniques for this problem
4. **Closest Prior Work**: The single most relevant prior paper
5. **What's New**: If partially exists, what specific aspect is novel
6. **Technical Differentiation**: How to position against closest work

Be thorough but fast. Search at least 5 different queries."""


def launch_verification(insight_id: int, timeout_minutes: int = None) -> dict:
    """Launch EvoScientist in verification mode for a deep insight.

    Returns immediately with job info. EvoScientist runs in background.
    """
    if timeout_minutes is None:
        from config import EVOSCI_VERIFY_TIMEOUT
        timeout_minutes = EVOSCI_VERIFY_TIMEOUT // 60

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id = ?", (insight_id,))
    if not insight:
        return {"error": f"Deep insight {insight_id} not found"}

    prompt = _build_verification_prompt(dict(insight))

    safe_title = insight["title"][:40].replace(" ", "_").replace("/", "-").replace("'", "")
    workdir = str(Path.home() / "research" / f"verify_di_{insight_id}_{safe_title}")
    os.makedirs(workdir, exist_ok=True)

    # Write the prompt for reference
    prompt_path = Path(workdir) / "verification_prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    evosci_bin = str(Path.home() / "EvoScientist" / ".venv" / "bin" / "EvoSci")
    if not Path(evosci_bin).exists():
        return {"error": f"EvoScientist not found at {evosci_bin}"}

    env = os.environ.copy()
    env["CUSTOM_OPENAI_USE_RESPONSES_API"] = "true"

    log_file = open(Path(workdir) / "evoscientist.log", "w")
    proc = subprocess.Popen(
        [
            evosci_bin,
            "--workdir", workdir,
            "--auto-approve",
            "--ui", "cli",
            "-p", prompt,
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=workdir,
        env=env,
    )

    db.execute(
        "UPDATE deep_insights SET novelty_status='verifying', evoscientist_workdir=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (workdir, insight_id))
    db.commit()

    import threading

    def _watchdog():
        try:
            proc.wait(timeout=timeout_minutes * 60)
        except subprocess.TimeoutExpired:
            print(f"[VERIFY] Timeout ({timeout_minutes}min) for insight {insight_id}, killing EvoScientist", flush=True)
            proc.kill()
        check_verification_result(insight_id)

    threading.Thread(target=_watchdog, daemon=True).start()

    return {
        "status": "started",
        "insight_id": insight_id,
        "workdir": workdir,
        "pid": proc.pid,
        "timeout_minutes": timeout_minutes,
    }


def check_verification_result(insight_id: int) -> dict:
    """Check if verification is complete and parse the result."""
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id = ?", (insight_id,))
    if not insight:
        return {"error": "Not found"}

    workdir = insight.get("evoscientist_workdir")
    if not workdir:
        return {"status": "not_started"}

    workdir = Path(workdir)
    report_path = workdir / "novelty_report.md"
    final_path = workdir / "final_report.md"

    result = {"status": "running", "workdir": str(workdir)}

    # Check log for completion signals
    log_path = workdir / "evoscientist.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        result["log_lines"] = len(log_text.split("\n"))
        result["log_tail"] = log_text[-500:]

    # Check for output files
    report_text = None
    for rpath in [report_path, final_path]:
        if rpath.exists() and rpath.stat().st_size > 100:
            report_text = rpath.read_text(encoding="utf-8", errors="replace")
            break

    if not report_text:
        return result

    # Parse verdict
    result["status"] = "complete"
    result["report"] = report_text[:5000]

    verdict = "unchecked"
    report_lower = report_text.lower()
    if "verdict" in report_lower:
        verdict_parts = report_lower.split("verdict")
        after_verdict = verdict_parts[1][:100] if len(verdict_parts) > 1 else ""
        if "novel" in report_lower and "partially" not in after_verdict:
            verdict = "novel"
        elif "partially_exists" in report_lower or "partially exists" in report_lower:
            verdict = "partially_exists"
        elif "exists" in report_lower:
            verdict = "exists"

    result["verdict"] = verdict

    # Update DB
    novelty_report = json.dumps({
        "verdict": verdict,
        "report_preview": report_text[:2000],
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    status = "verified" if verdict == "novel" else ("refined" if verdict == "partially_exists" else "exists")

    db.execute(
        """UPDATE deep_insights
           SET novelty_status=?, novelty_report=?, status=?, updated_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (verdict, novelty_report, status, insight_id))
    db.commit()

    result["updated_status"] = status
    return result


def launch_full_research(insight_id: int) -> dict:
    """Launch EvoScientist for full research (longer session) on a verified insight.

    This is the existing research_bridge flow but for deep_insights.
    """
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id = ?", (insight_id,))
    if not insight:
        return {"error": f"Deep insight {insight_id} not found"}

    tier = insight["tier"]
    title = insight["title"]

    if tier == 1:
        field_a = json.loads(insight["field_a"]) if insight.get("field_a") else {}
        field_b = json.loads(insight["field_b"]) if insight.get("field_b") else {}
        prompt = f"""Research the following paradigm-level ML insight in depth:

**Title**: {title}
**Formal Structure**: {insight.get('formal_structure', '')}
**Field A**: {field_a.get('node_id', '?')} — {field_a.get('phenomenon', '')}
**Field B**: {field_b.get('node_id', '?')} — {field_b.get('phenomenon', '')}
**Transformation**: {insight.get('transformation', '')}

1. Search for all related work (2020-2026)
2. Validate the mathematical claim with evidence
3. Design experiments to test the predictions
4. Draft a paper outline with title, abstract, contributions
5. Estimate the impact: which papers/methods would be affected

Write your findings to final_report.md."""

    else:
        method = json.loads(insight["proposed_method"]) if insight.get("proposed_method") else {}
        prompt = f"""Research the following proposed ML method in depth:

**Title**: {title}
**Problem**: {insight.get('problem_statement', '')}
**Weakness Addressed**: {insight.get('existing_weakness', '')}
**Method**: {method.get('name', '?')} — {method.get('one_line', '')}
**Definition**: {method.get('definition', '')[:500]}

1. Search for all related methods and baselines (2022-2026)
2. Refine the method based on what you find
3. Design detailed experiments (specific models, datasets, metrics)
4. Identify the most relevant comparison points
5. Draft a complete paper outline

Write your findings to final_report.md."""

    safe_title = title[:40].replace(" ", "_").replace("/", "-").replace("'", "")
    workdir = str(Path.home() / "research" / f"deep_research_di_{insight_id}_{safe_title}")
    os.makedirs(workdir, exist_ok=True)

    evosci_bin = str(Path.home() / "EvoScientist" / ".venv" / "bin" / "EvoSci")
    env = os.environ.copy()
    env["CUSTOM_OPENAI_USE_RESPONSES_API"] = "true"

    log_file = open(Path(workdir) / "evoscientist.log", "w")
    proc = subprocess.Popen(
        [
            evosci_bin,
            "--workdir", workdir,
            "--auto-approve",
            "--ui", "cli",
            "-p", prompt,
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=workdir,
        env=env,
    )

    db.execute(
        "UPDATE deep_insights SET evoscientist_workdir=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (workdir, insight_id))
    db.commit()

    return {
        "status": "started",
        "insight_id": insight_id,
        "workdir": workdir,
        "pid": proc.pid,
    }
