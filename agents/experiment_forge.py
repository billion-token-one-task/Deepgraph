"""Experiment Forge: bridge from deep_insights to runnable experiments.

Three sub-components:
  2a. Code Scout — find/clone relevant codebases for a hypothesis
  2b. Scaffold Builder — generate program.md, evaluate.py, success_criteria.json
  2c. Proxy Task Builder — configure time-budgeted experiment for fast iteration

This is the hardest layer: translating a structured method description
into something an autonomous coding agent can actually run.
"""
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from agents.experiment_review import review_experiment_candidate
from agents.llm_client import call_llm, call_llm_json
from config import (
    EXPERIMENT_EARLY_STOP_THRESHOLD,
    EXPERIMENT_MAX_ITERATIONS,
    EXPERIMENT_PROXY_DATA_FRACTION,
    EXPERIMENT_PROXY_MAX_EPOCHS,
    EXPERIMENT_REFUTE_MIN_ITERS,
    EXPERIMENT_REPRODUCTION_ITERS,
    EXPERIMENT_TIME_BUDGET,
    EXPERIMENT_WORKDIR,
)
from contracts import DeepInsightSpec, ExperimentSpec
from db import database as db
from db.insight_outcomes import apply_experiment_queued_deep


SCAFFOLD_SYSTEM = """You are an expert ML engineer. Given a research hypothesis with a proposed method, you produce THREE files that enable an autonomous coding agent to run experiments.

You will receive:
1. A proposed method (name, type, definition, pseudocode, properties)
2. An experimental plan (baselines, datasets, metrics, ablations)
3. A codebase description (what repo was cloned, its structure)

You must output JSON with three keys:

{
  "program_md": "Complete program.md content in Markdown (instructions for the coding agent)",
  "evaluate_py": "Complete evaluate.py Python script (metric computation)",
  "success_criteria": {
    "metric_name": "primary metric name",
    "metric_direction": "lower|higher",
    "exciting": <number>,
    "solid": <number>,
    "disappointing": <number>
  }
}

## program.md Requirements
- Must follow the autoresearch format: setup, experimentation loop, output format, logging
- MUST specify which file(s) the agent can modify
- MUST describe the proposed method clearly enough for implementation
- MUST include the baseline to beat and specific success criteria
- MUST include the evaluation command
- MUST tell the agent to NEVER STOP until interrupted

## evaluate.py Requirements
- Self-contained Python script
- Takes a log file or results directory as input
- Outputs the primary metric value to stdout
- Handles errors gracefully (outputs 0.0 on failure)

## success_criteria Requirements
- Use the primary metric from the experimental plan
- exciting = would be a strong contribution (top-venue accept)
- solid = clear improvement over baseline
- disappointing = not worth publishing

## CRITICAL: train.py (bootstrap code)
If the codebase is "scratch" or empty, you MUST also output a "train_py" key containing a COMPLETE, RUNNABLE Python script that:
- Implements the baseline version of the experiment (no proposed method yet)
- Generates or loads synthetic/simulated data if needed
- Runs the experiment and prints the metric to stdout in format: metric_name: value
- Is self-contained (only uses stdlib + numpy + scipy, no GPU required)
- Finishes in under 5 minutes

For framework/evaluation-type methods (not model training), train.py should:
- Generate synthetic test scenarios
- Run the baseline evaluation approach
- Print the primary metric"""


CODE_SCOUT_SYSTEM = """You are a research engineer. Given a method description and its related taxonomy area, suggest the BEST open-source codebase to use as a starting point for implementing and testing this method.

Return JSON:
{
  "codebase": {
    "url": "GitHub URL (full https://github.com/...)",
    "name": "short name",
    "reason": "why this is the best base",
    "setup_commands": ["pip install ...", "python setup.py ..."],
    "main_train_file": "path/to/train.py (the file to modify)",
    "main_eval_command": "python evaluate.py --args",
    "expected_baseline_metric": "approximate value"
  },
  "alternatives": [
    {"url": "...", "name": "...", "reason": "..."}
  ]
}

Prefer:
- Well-maintained repos (recent commits, many stars)
- Repos with clear training scripts and evaluation
- Repos that already implement the BASELINE the hypothesis compares against
- Simple codebases over complex frameworks

If no suitable codebase exists, set url to "scratch" and provide setup commands for a minimal PyTorch project."""


def _parse_insight_fields(insight: dict) -> dict:
    """Extract and parse JSON fields from a deep_insight row."""
    parsed = dict(insight)
    for field in ("proposed_method", "experimental_plan", "related_work_positioning",
                  "evidence_plan",
                  "field_a", "field_b", "predictions", "falsification",
                  "supporting_papers", "source_node_ids", "adversarial_critique"):
        val = parsed.get(field)
        if isinstance(val, str) and val.strip():
            try:
                parsed[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    for field in ("proposed_method", "experimental_plan", "related_work_positioning", "evidence_plan",
                  "field_a", "field_b", "falsification", "adversarial_critique"):
        if not isinstance(parsed.get(field), dict):
            parsed[field] = {}
    for field in ("predictions", "supporting_papers", "source_node_ids", "source_paper_ids", "source_signal_ids"):
        if not isinstance(parsed.get(field), list):
            parsed[field] = []
    return parsed


def _git_binary() -> str | None:
    return shutil.which("git")


def _code_dir_has_content(code_dir: Path) -> bool:
    return code_dir.exists() and any(code_dir.iterdir())


def _github_archive_urls(repo_url: str) -> list[str]:
    parsed = urllib.parse.urlparse((repo_url or "").strip())
    if parsed.netloc not in {"github.com", "www.github.com"}:
        return []
    path = parsed.path.strip("/").removesuffix(".git")
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return []
    owner, repo = parts[0], parts[1]
    base = f"https://github.com/{owner}/{repo}/archive/refs/heads"
    return [f"{base}/main.zip", f"{base}/master.zip"]


def _download_repo_archive(repo_url: str, code_dir: Path) -> bool:
    for archive_url in _github_archive_urls(repo_url):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_root = Path(tmpdir)
                archive_path = tmp_root / "repo.zip"
                extract_dir = tmp_root / "extract"
                extract_dir.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(archive_url, timeout=30) as response:
                    archive_path.write_bytes(response.read())
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(extract_dir)
                roots = [path for path in extract_dir.iterdir() if path.is_dir()]
                source_root = roots[0] if roots else extract_dir
                for child in source_root.iterdir():
                    target = code_dir / child.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(child), str(target))
            return _code_dir_has_content(code_dir)
        except Exception as exc:
            print(f"[FORGE] Archive fetch failed for {archive_url}: {exc}", flush=True)
    return False


def _codebase_has_expected_entrypoint(code_dir: Path, codebase: dict) -> bool:
    expected = (codebase.get("main_train_file") or "").strip()
    if not expected:
        return _code_dir_has_content(code_dir)
    expected_path = code_dir / expected
    return expected_path.exists()


def _scratch_codebase(reason: str = "") -> dict:
    return {
        "url": "scratch",
        "name": "minimal",
        "reason": reason or "selected repository was unsuitable for the requested experiment",
        "main_train_file": "train.py",
        "main_eval_command": "python train.py",
    }


def scout_codebase(insight: dict) -> dict:
    """Find the best codebase for implementing a hypothesis.

    Uses LLM to suggest repos based on the method description and
    knowledge graph context about what methods/datasets are involved.
    """
    parsed = _parse_insight_fields(insight)
    method = parsed.get("proposed_method", {})
    plan = parsed.get("experimental_plan", {})
    evidence_plan = parsed.get("evidence_plan", {})
    node_ids = parsed.get("source_node_ids", [])

    context_parts = [f"# Method to Implement\n"]
    context_parts.append(f"Name: {method.get('name', 'Unknown')}")
    context_parts.append(f"Type: {method.get('type', 'unknown')}")
    context_parts.append(f"Summary: {method.get('one_line', '')}")
    if method.get("definition"):
        context_parts.append(f"Definition: {method['definition'][:600]}")

    context_parts.append(f"\n# Experimental Plan")
    if plan.get("baselines"):
        context_parts.append("Baselines:")
        for b in plan["baselines"][:5]:
            name = b.get("name", b) if isinstance(b, dict) else str(b)
            model = b.get("model", "") if isinstance(b, dict) else ""
            context_parts.append(f"  - {name} {model}")
    if plan.get("datasets"):
        context_parts.append("Datasets:")
        for d in plan["datasets"][:5]:
            name = d.get("name", d) if isinstance(d, dict) else str(d)
            context_parts.append(f"  - {name}")

    context_parts.append(f"\n# Research Area")
    context_parts.append(f"Taxonomy nodes: {', '.join(node_ids[:5])}")

    if parsed.get("problem_statement"):
        context_parts.append(f"\n# Problem")
        context_parts.append(parsed["problem_statement"][:400])

    graph_methods = db.fetchall("""
        SELECT DISTINCT ge.canonical_name, ge.description
        FROM graph_entities ge
        JOIN paper_entity_mentions pem ON pem.entity_id = ge.id
        WHERE ge.entity_type = 'method'
          AND pem.node_id IN ({})
        ORDER BY ge.canonical_name
        LIMIT 15
    """.format(",".join("?" * len(node_ids))), tuple(node_ids)) if node_ids else []

    if graph_methods:
        context_parts.append("\n# Known Methods in This Area (from knowledge graph)")
        for m in graph_methods:
            desc = f" — {m['description'][:80]}" if m.get("description") else ""
            context_parts.append(f"  - {m['canonical_name']}{desc}")

    prompt = "\n".join(context_parts)

    try:
        result, _ = call_llm_json(CODE_SCOUT_SYSTEM, prompt)
        return result.get("codebase", {"url": "scratch", "name": "minimal", "reason": "no suitable repo found"})
    except Exception as e:
        print(f"[FORGE] Code scout failed: {e}", flush=True)
        return {"url": "scratch", "name": "minimal", "reason": f"scout error: {e}"}


def setup_workspace(insight_id: int, codebase: dict) -> Path:
    """Create the experiment workspace directory with the codebase."""
    insight = db.fetchone("SELECT title FROM deep_insights WHERE id=?", (insight_id,))
    safe_title = (insight["title"] if insight else "unknown")[:40]
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_title)

    workdir = EXPERIMENT_WORKDIR / f"exp_{insight_id}_{safe_title}"
    workdir.mkdir(parents=True, exist_ok=True)

    code_dir = workdir / "code"
    (workdir / "results").mkdir(exist_ok=True)
    (workdir / "spec").mkdir(exist_ok=True)
    (workdir / "codex").mkdir(exist_ok=True)
    url = codebase.get("url", "scratch")

    if url != "scratch" and not _code_dir_has_content(code_dir):
        if code_dir.exists():
            shutil.rmtree(code_dir)
        code_dir.mkdir(parents=True, exist_ok=True)
        git_bin = _git_binary()
        clone_ok = False
        if git_bin:
            try:
                subprocess.run(
                    [git_bin, "clone", "--depth", "1", url, str(code_dir)],
                    timeout=120,
                    capture_output=True,
                    check=True,
                )
                clone_ok = True
                print(f"[FORGE] Cloned {url} to {code_dir}", flush=True)
            except Exception as e:
                print(f"[FORGE] Clone failed for {url}: {e}. Trying archive fallback.", flush=True)
        else:
            print(f"[FORGE] git not available; trying archive fallback for {url}", flush=True)
        if not clone_ok and not _code_dir_has_content(code_dir):
            archive_ok = _download_repo_archive(url, code_dir)
            if archive_ok:
                print(f"[FORGE] Downloaded archive fallback for {url} into {code_dir}", flush=True)
            else:
                print(f"[FORGE] Archive fallback failed for {url}. Using scratch workspace.", flush=True)
    elif not code_dir.exists():
        code_dir.mkdir(parents=True, exist_ok=True)

    return workdir


def generate_scaffold(insight: dict, codebase: dict, workdir: Path) -> dict:
    """Generate program.md, evaluate.py, and success_criteria.json using LLM."""
    parsed = _parse_insight_fields(insight)
    method = parsed.get("proposed_method", {})
    plan = parsed.get("experimental_plan", {})
    evidence_plan = parsed.get("evidence_plan", {})

    code_dir = workdir / "code"
    code_structure = ""
    if code_dir.exists():
        try:
            result = subprocess.run(
                ["find", str(code_dir), "-name", "*.py", "-maxdepth", "3"],
                capture_output=True, text=True, timeout=10
            )
            files = [f.replace(str(code_dir) + "/", "") for f in result.stdout.strip().split("\n") if f][:30]
            code_structure = "\n".join(f"  {f}" for f in files)
        except Exception:
            code_structure = "(could not list files)"

    prompt_parts = [
        f"# Proposed Method",
        f"Name: {method.get('name', '?')}",
        f"Type: {method.get('type', '?')}",
        f"Summary: {method.get('one_line', '')}",
        f"Definition:\n{method.get('definition', 'N/A')[:800]}",
    ]
    if method.get("pseudocode"):
        prompt_parts.append(f"Pseudocode:\n{method['pseudocode'][:500]}")
    if method.get("key_properties"):
        prompt_parts.append(f"Key Properties: {json.dumps(method['key_properties'][:5])}")
    if method.get("hyperparameters"):
        prompt_parts.append(f"Hyperparameters: {json.dumps(method['hyperparameters'][:5])}")

    prompt_parts.append(f"\n# Experimental Plan")
    prompt_parts.append(f"Baselines: {json.dumps(plan.get('baselines', []))[:500]}")
    prompt_parts.append(f"Datasets: {json.dumps(plan.get('datasets', []))[:500]}")
    prompt_parts.append(f"Metrics: {json.dumps(plan.get('metrics', {}))[:300]}")
    prompt_parts.append(f"Expected Results: {json.dumps(plan.get('expected_results', {}))[:300]}")
    if evidence_plan:
        prompt_parts.append(f"\n# Adaptive Evidence Plan")
        prompt_parts.append(json.dumps(evidence_plan, ensure_ascii=False)[:1200])
        prompt_parts.append("Honor this plan. Do not invent ablations or visual analyses when they are disabled.")

    prompt_parts.append(f"\n# Codebase")
    prompt_parts.append(f"Repo: {codebase.get('url', 'scratch')} ({codebase.get('name', '')})")
    prompt_parts.append(f"Main train file: {codebase.get('main_train_file', 'train.py')}")
    prompt_parts.append(f"Eval command: {codebase.get('main_eval_command', 'python evaluate.py')}")
    if code_structure:
        prompt_parts.append(f"File structure:\n{code_structure}")

    prompt_parts.append(f"\n# Problem Context")
    prompt_parts.append(parsed.get("problem_statement", "")[:300])
    prompt_parts.append(f"Weakness: {parsed.get('existing_weakness', '')[:300]}")

    prompt = "\n".join(prompt_parts)

    try:
        result, tokens = call_llm_json(SCAFFOLD_SYSTEM, prompt)
    except Exception as e:
        print(f"[FORGE] Scaffold generation failed: {e}", flush=True)
        result = _fallback_scaffold(method, plan, codebase)
        tokens = 0

    program_md = result.get("program_md", "")
    evaluate_py = result.get("evaluate_py", "")
    success = result.get("success_criteria", {})
    train_py = result.get("train_py", "")

    if codebase.get("url") == "scratch" and len(train_py or "") <= 50:
        fallback = _fallback_scaffold(method, plan, codebase)
        train_py = fallback.get("train_py", train_py)
        success = success or fallback.get("success_criteria", {})

    spec_dir = workdir / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "program.md").write_text(program_md, encoding="utf-8")
    (spec_dir / "evaluate.py").write_text(evaluate_py, encoding="utf-8")
    (spec_dir / "success_criteria.json").write_text(
        json.dumps(success, indent=2), encoding="utf-8")

    code_dir = workdir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    if train_py and len(train_py) > 50:
        (code_dir / "train.py").write_text(train_py, encoding="utf-8")
        print(f"[FORGE] Bootstrap train.py written ({len(train_py)} chars)", flush=True)
    elif not list(code_dir.glob("*.py")):
        print(f"[FORGE] WARNING: No train.py and no code in {code_dir}. Loop will likely fail.", flush=True)

    return {
        "program_md": program_md,
        "evaluate_py": evaluate_py,
        "success_criteria": success,
        "train_py_written": bool(train_py and len(train_py) > 50),
        "tokens": tokens,
    }


def _fallback_scaffold(method: dict, plan: dict, codebase: dict) -> dict:
    """Generate a minimal scaffold without LLM if the call fails."""
    method_name = method.get("name", "ProposedMethod")
    method_def = method.get("definition", "See method description")
    train_file = codebase.get("main_train_file", "train.py")

    metrics = plan.get("metrics", {})
    primary_metric = metrics.get("primary", "accuracy") if isinstance(metrics, dict) else "accuracy"

    program_md = textwrap.dedent(f"""\
    # SciForge Experiment: {method_name}

    ## Setup
    1. Read all files in the code/ directory for context.
    2. Establish a baseline by running the training script as-is.
    3. Record the baseline metric.

    ## Experimentation
    **File to modify**: `code/{train_file}`
    **Goal**: Implement {method_name} and achieve a better {primary_metric} than baseline.

    ### Method to Implement
    {method_def[:1000]}

    ### Constraints
    - Only modify `code/{train_file}`.
    - Each run has a fixed time budget of {EXPERIMENT_TIME_BUDGET} seconds.
    - Evaluate using: `python spec/evaluate.py`

    ## The Experiment Loop
    LOOP FOREVER:
    1. Modify the code with an experimental idea based on the method above.
    2. git commit
    3. Run: `cd code && python {train_file} > ../run.log 2>&1`
    4. Evaluate: `python spec/evaluate.py run.log`
    5. If metric improved, keep. If worse, git reset.
    6. Log results to results.tsv
    7. NEVER STOP until manually interrupted.
    """)

    evaluate_py = textwrap.dedent(f"""\
    import json
    import re
    import sys

    def main():
        log_file = sys.argv[1] if len(sys.argv) > 1 else "run.log"
        try:
            with open(log_file) as f:
                text = f.read()
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    raw_value = payload.get("metric_value")
                    if raw_value is not None:
                        try:
                            print(f"metric_value: {{float(raw_value)}}")
                            return
                        except Exception:
                            pass
            patterns = [
                r'"metric_value"\\s*:\\s*([\\d.]+)',
                r'metric_value[:\\s]+([\\d.]+)',
                r'{primary_metric}[:\\s]+([\\d.]+)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    print(f"metric_value: {{matches[-1]}}")
                    return
            print("metric_value: 0.0")
        except Exception as e:
            print(f"metric_value: 0.0")

    if __name__ == "__main__":
        main()
    """)

    train_py = textwrap.dedent(f"""\
    import json
    import random

    def main():
        random.seed(13)
        baseline = 0.62
        noise = 0.03
        metric = baseline + (random.random() - 0.5) * noise
        result = {{
            "method": {method_name!r},
            "metric_name": {primary_metric!r},
            "metric_value": round(metric, 4),
            "notes": {method_def[:200]!r},
        }}
        print(json.dumps(result))
        print(f"{primary_metric}: {{result['metric_value']}}")

    if __name__ == "__main__":
        main()
    """)

    return {
        "program_md": program_md,
        "evaluate_py": evaluate_py,
        "train_py": train_py,
        "success_criteria": {
            "metric_name": primary_metric,
            "metric_direction": "higher",
            "exciting": 0.0,
            "solid": 0.0,
            "disappointing": 0.0,
        },
    }


def build_proxy_config(plan: dict, codebase: dict | None = None, *, judgement=None) -> dict:
    """Build proxy task configuration for time-budgeted experiments."""
    compute = plan.get("compute_budget", {}) if isinstance(plan, dict) else {}
    codebase = codebase or {}

    proxy = {
        "data_fraction": EXPERIMENT_PROXY_DATA_FRACTION,
        "max_epochs": EXPERIMENT_PROXY_MAX_EPOCHS,
        "time_budget_seconds": EXPERIMENT_TIME_BUDGET,
        "early_stop_threshold": EXPERIMENT_EARLY_STOP_THRESHOLD,
        "max_iterations": EXPERIMENT_MAX_ITERATIONS,
        "reproduction_iterations": EXPERIMENT_REPRODUCTION_ITERS,
        "refute_min_iterations": EXPERIMENT_REFUTE_MIN_ITERS,
        "estimated_gpu_hours": compute.get("total_gpu_hours", "unknown"),
        "main_train_file": codebase.get("main_train_file"),
        "baseline_command": codebase.get("main_eval_command"),
    }
    if judgement is not None:
        proxy["formal_experiment"] = judgement.formal_experiment
        proxy["smoke_test_only"] = judgement.smoke_test_only
        proxy["experiment_judgement"] = judgement.to_dict()
    return proxy


def forge_experiment(insight_id: int) -> dict:
    """Full forge pipeline: scout codebase -> setup workspace -> generate scaffold.

    Creates an experiment_run row and returns all paths/configs needed
    for the validation loop.
    """
    print(f"[FORGE] Starting experiment forge for insight {insight_id}...", flush=True)

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not insight:
        return {"error": f"Deep insight {insight_id} not found"}

    parsed = _parse_insight_fields(dict(insight))
    spec = DeepInsightSpec.from_raw(parsed)
    plan = spec.experimental_plan
    evidence_plan = spec.evidence_plan

    # Step 1: Scout codebase
    print(f"[FORGE] Scouting codebase...", flush=True)
    codebase = scout_codebase(parsed)
    print(f"[FORGE] Selected: {codebase.get('name', '?')} ({codebase.get('url', '?')})", flush=True)

    # Step 2: Setup workspace
    workdir = setup_workspace(insight_id, codebase)
    code_dir = workdir / "code"
    entrypoint_available = None
    if codebase.get("url") != "scratch" and not _codebase_has_expected_entrypoint(
        code_dir, codebase
    ):
        entrypoint_available = False
        print(
            f"[FORGE] Selected repo missing expected entrypoint "
            f"{codebase.get('main_train_file', 'unknown')}; falling back to scratch.",
            flush=True,
        )
        if code_dir.exists():
            shutil.rmtree(code_dir)
        code_dir.mkdir(parents=True, exist_ok=True)
        codebase = _scratch_codebase(
            reason=(
                f"repo {codebase.get('name', '?')} missing expected entrypoint "
                f"{codebase.get('main_train_file', 'unknown')}"
            )
        )
    elif codebase.get("url") != "scratch":
        entrypoint_available = True
    print(f"[FORGE] Workspace: {workdir}", flush=True)

    judgement = review_experiment_candidate(
        spec,
        codebase=codebase,
        entrypoint_available=entrypoint_available,
    )
    if judgement.recommended_route == "blocked":
        return {
            "error": judgement.summary or "Experiment review blocked formalization",
            "judgement": judgement.to_dict(),
            "route": judgement.recommended_route,
        }
    print(
        f"[FORGE] Review route={judgement.recommended_route} "
        f"formal={judgement.formal_experiment} smoke={judgement.smoke_test_only}",
        flush=True,
    )

    # Step 3: Generate scaffold
    print(f"[FORGE] Generating scaffold (program.md, evaluate.py, success_criteria)...", flush=True)
    scaffold = generate_scaffold(parsed, codebase, workdir)

    # Step 4: Build proxy config
    proxy = build_proxy_config(plan, codebase=codebase, judgement=judgement)
    proxy["evidence_plan"] = evidence_plan
    spec_dir = workdir / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "proxy_config.json").write_text(json.dumps(proxy, indent=2), encoding="utf-8")
    (spec_dir / "evidence_plan.json").write_text(json.dumps(evidence_plan, indent=2), encoding="utf-8")
    (spec_dir / "experiment_judgement.json").write_text(
        json.dumps(judgement.to_dict(), indent=2),
        encoding="utf-8",
    )

    # Step 5: Create experiment_run row
    success = scaffold.get("success_criteria", {})
    run_id = db.insert_returning_id(
        """INSERT INTO experiment_runs
           (deep_insight_id, status, phase, workdir, codebase_url, codebase_ref,
            program_md, proxy_config, success_criteria,
            baseline_metric_name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            insight_id,
            "scaffolding",
            "setup",
            str(workdir),
            codebase.get("url", "scratch"),
            codebase.get("name", ""),
            scaffold.get("program_md", ""),
            json.dumps(proxy),
            json.dumps(success),
            success.get("metric_name", "metric"),
        ),
    )
    db.commit()

    experiment_spec = ExperimentSpec.from_sources(
        run_id=run_id,
        insight=spec,
        workdir=str(workdir),
        codebase=codebase,
        judgement=judgement,
        success_criteria=success,
        proxy_config=proxy,
        artifact_paths={
            "program_md": str(spec_dir / "program.md"),
            "evaluate_py": str(spec_dir / "evaluate.py"),
            "success_criteria": str(spec_dir / "success_criteria.json"),
            "proxy_config": str(spec_dir / "proxy_config.json"),
            "evidence_plan": str(spec_dir / "evidence_plan.json"),
            "experiment_judgement": str(spec_dir / "experiment_judgement.json"),
        },
    )
    (spec_dir / "experiment_spec.json").write_text(
        json.dumps(experiment_spec.to_dict(), indent=2),
        encoding="utf-8",
    )
    db.execute(
        """
        INSERT INTO experiment_artifacts (run_id, artifact_type, path, metadata)
        VALUES (?, ?, ?, ?)
        """,
        (
            run_id,
            "source_data",
            str(spec_dir / "experiment_spec.json"),
            json.dumps({"contract_type": "ExperimentSpec"}),
        ),
    )
    db.execute(
        """
        INSERT INTO experiment_artifacts (run_id, artifact_type, path, metadata)
        VALUES (?, ?, ?, ?)
        """,
        (
            run_id,
            "source_data",
            str(spec_dir / "experiment_judgement.json"),
            json.dumps({"contract_type": "ExperimentJudgement"}),
        ),
    )
    db.commit()

    # Update deep_insight status
    new_insight_status = "forged" if judgement.formal_experiment else "smoke_only"
    db.execute(
        "UPDATE deep_insights SET status=?, evoscientist_workdir=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (new_insight_status, str(workdir), insight_id),
    )
    db.commit()

    if judgement.formal_experiment:
        apply_experiment_queued_deep(insight_id, note=f"experiment_run_id={run_id}")

    print(f"[FORGE] Experiment forged: run_id={run_id}, workdir={workdir}", flush=True)

    return {
        "run_id": run_id,
        "insight_id": insight_id,
        "workdir": str(workdir),
        "codebase": codebase,
        "success_criteria": success,
        "proxy_config": proxy,
        "evidence_plan": evidence_plan,
        "judgement": judgement.to_dict(),
        "formal_experiment": judgement.formal_experiment,
        "smoke_test_only": judgement.smoke_test_only,
        "scaffold_tokens": scaffold.get("tokens", 0),
    }
