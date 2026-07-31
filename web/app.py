"""Flask web application for DeepGraph dashboard."""
import json
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from flask import Flask, Response, abort, jsonify, render_template, request, send_file, url_for
from agents.workspace_layout import get_idea_workspace, list_paper_assets, plan_file_path, resolve_paper_asset
import config as cfg
from config import APP_NAME, APP_SUBTITLE, PROFILE, ROOT_NODE_ID
from db import database as db
from db import evidence_graph as graph
from db import opportunity_engine as opp
from db import taxonomy as tax
from orchestrator.pipeline import get_events, log_event, get_stats_dict

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")
from web.meta_harness_routes import blueprint as meta_harness_blueprint

app.register_blueprint(meta_harness_blueprint)

_pipeline_running = False
_pipeline_lock = threading.Lock()

_ACTIVE_EXPERIMENT_STATUSES = {"pending", "scaffolding", "reproducing", "testing", "running_gpu", "running_cpu"}
_PROTECTED_GENERATED_PAPER_IDS = {
    int(part)
    for part in re.split(r"[,\s]+", os.getenv("DEEPGRAPH_PROTECTED_GENERATED_PAPER_IDS", "3,4,8").strip())
    if part.isdigit()
}


def _json_load(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _manual_api_removed_response():
    return jsonify(
        {
            "error": "Manual web actions have been removed. This deployment is fixed-flow and read-only from the UI.",
            "mode": "fixed_flow_read_only",
            "replacement": "/api/meta-harness/v1",
        }
    ), 410


def _required_agenda_query_id() -> int | None:
    try:
        agenda_id = int(request.args.get("agenda_id", ""))
    except (TypeError, ValueError):
        return None
    return agenda_id if agenda_id > 0 else None


def _api_failure(scope: str, exc: Exception, status: int = 500):
    try:
        db.rollback()
    except Exception:
        pass
    message = str(exc)
    log_event("error", {"step": scope, "error": message})
    print(f"[API] {scope} failed: {message}\n{traceback.format_exc()}", flush=True)
    return jsonify({"status": "error", "scope": scope, "error": message}), status


def _process_rss_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return round(int(out) / 1024, 1) if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _total_memory_mb() -> float | None:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return round(int(out) / (1024 * 1024), 1)
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024 * 1024), 1)
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return None


def _disk_snapshot(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        return {
            "total_gb": round(usage.total / (1024 ** 3), 1),
            "used_gb": round(usage.used / (1024 ** 3), 1),
            "free_gb": round(usage.free / (1024 ** 3), 1),
        }
    except OSError:
        return {}

def _cpu_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"count": os.cpu_count()}
    try:
        one, five, fifteen = os.getloadavg()
        snapshot.update({
            "load_1m": round(one, 2),
            "load_5m": round(five, 2),
            "load_15m": round(fifteen, 2),
            "load_pct_1m": round((one / max(os.cpu_count() or 1, 1)) * 100, 1),
        })
    except (OSError, AttributeError):
        pass
    return snapshot


def _gpu_snapshot() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "gpus": [], "processes": []}
    query = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        gpus = []
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 8:
                continue
            idx, name, mem_total, mem_used, util, temp, power, power_limit = parts[:8]
            def _float(value: str) -> float | None:
                try:
                    return round(float(value), 1)
                except ValueError:
                    return None
            total = _float(mem_total) or 0
            used = _float(mem_used) or 0
            gpus.append({
                "index": int(idx),
                "name": name,
                "memory_total_mb": total,
                "memory_used_mb": used,
                "memory_used_pct": round((used / total) * 100, 1) if total else None,
                "utilization_pct": _float(util),
                "temperature_c": _float(temp),
                "power_w": _float(power),
                "power_limit_w": _float(power_limit),
            })
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"available": False, "gpus": [], "processes": []}

    processes = []
    try:
        pout = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        for line in pout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_uuid, pid, process_name, used_memory = parts[:4]
            try:
                used_memory_mb = round(float(used_memory), 1)
            except ValueError:
                used_memory_mb = None
            processes.append({
                "gpu_uuid": gpu_uuid,
                "pid": pid,
                "process_name": process_name,
                "used_memory_mb": used_memory_mb,
            })
    except (OSError, subprocess.SubprocessError):
        pass

    return {"available": True, "gpus": gpus, "processes": processes}



@app.before_request
def block_manual_experiment_post_apis():
    if (
        request.method == "POST"
        and request.path.startswith("/api/")
        and not request.path.startswith("/api/meta-harness/v1/")
    ):
        return _manual_api_removed_response()


def _pick_canonical_run(runs: list[dict], canonical_run_id: int | None = None) -> dict | None:
    if not runs:
        return None
    if canonical_run_id:
        for run in runs:
            if int(run.get("id") or 0) == int(canonical_run_id):
                return run
    for run in runs:
        if (run.get("status") or "") in _ACTIVE_EXPERIMENT_STATUSES:
            return run
    return runs[0]


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _deep_insight_is_displayable(row: dict) -> bool:
    """Hide incomplete mechanism-first placeholders from the dashboard."""
    title = " ".join(str(row.get("title") or "").strip().lower().split())
    generic_title = title in {"mechanism-first insight", "mechanism first insight", "deep insight", "paper idea"}
    core_fields = (
        "formal_structure",
        "transformation",
        "problem_statement",
        "existing_weakness",
        "proposed_method",
        "experimental_plan",
        "evidence_summary",
    )
    has_core = any(str(row.get(field) or "").strip() for field in core_fields)
    if generic_title and not has_core:
        return False
    if generic_title and int(row.get("tier") or 0) == 1:
        field_a = _json_load(row.get("field_a"), {})
        field_b = _json_load(row.get("field_b"), {})
        if not row.get("formal_structure") and not row.get("transformation") and not field_a and not field_b:
            return False
    return True


def _deep_insight_is_archived(row: dict) -> bool:
    """Hide soft-cleaned / archived ideas from live dashboards by default."""
    status = str(row.get("status") or "").strip().lower()
    novelty = str(row.get("novelty_status") or "").strip().lower()
    outcome = str(row.get("outcome") or "").strip().lower()
    submission = str(row.get("submission_status") or "").strip().lower()
    return (
        outcome in {"cleaned", "archived"}
        or novelty in {"cleaned_similar_duplicate"}
        or submission in {"stale"}
        or status in {"exists"}
    )


def _include_archived_requested() -> bool:
    return str(request.args.get("include_archived") or "").strip().lower() in {"1", "true", "yes"}


def _plan_snapshot(insight: dict) -> dict[str, Any]:
    insight_id = int(insight["id"])
    return {
        "experiment_spec": _read_json_file(plan_file_path(insight_id, "experiment_spec.json", insight=insight)),
        "proxy_config": _read_json_file(plan_file_path(insight_id, "proxy_config.json", insight=insight)),
        "evidence_plan": _read_json_file(plan_file_path(insight_id, "evidence_plan.json", insight=insight)),
        "manuscript_input_state": _read_json_file(plan_file_path(insight_id, "manuscript_input_state.json", insight=insight)),
        "manuscript_blockers": _read_json_file(plan_file_path(insight_id, "manuscript_blockers.json", insight=insight)),
        "latest_status": _read_json_file(plan_file_path(insight_id, "latest_status.json", insight=insight)),
    }


def _paper_preview_urls(
    insight_id: int,
    assets: list[dict],
    *,
    agenda_id: int,
) -> dict[str, str | None]:
    asset_paths = {str(asset.get("path") or "") for asset in assets}
    pdf_path = next((path for path in sorted(asset_paths) if path.endswith("/main.pdf") or path == "current/main.pdf"), None)
    tex_path = next((path for path in sorted(asset_paths) if path.endswith("/main.tex") or path == "current/main.tex"), None)
    return {
        "index": url_for(
            "paper_preview_index",
            insight_id=insight_id,
            agenda_id=agenda_id,
        ),
        "pdf": url_for(
            "paper_preview_pdf",
            insight_id=insight_id,
            agenda_id=agenda_id,
        ) if pdf_path else None,
        "tex": url_for(
            "paper_preview_tex",
            insight_id=insight_id,
            agenda_id=agenda_id,
        ) if tex_path else None,
    }


def _workspace_payload(insight: dict) -> dict[str, Any]:
    layout = get_idea_workspace(int(insight["id"]), insight=insight, create=True, sync_db=True)
    assets = list_paper_assets(int(insight["id"]), insight=insight)
    preview_urls = _paper_preview_urls(
        int(insight["id"]),
        assets,
        agenda_id=int(insight["agenda_id"]),
    )
    return {
        "workspace_root": str(layout["workspace_root"]),
        "experiment_root": str(layout["experiment_root"]),
        "plan_root": str(layout["plan_root"]),
        "paper_root": str(layout["paper_root"]),
        "canonical_run_id": insight.get("canonical_run_id"),
        "plan_snapshot": _plan_snapshot(insight),
        "paper_assets": assets,
        "paper_preview_urls": preview_urls,
    }


def _artifact_counts_for_runs(run_ids: list[int]) -> dict[int, dict[str, int]]:
    if not run_ids:
        return {}
    placeholders = ",".join("?" for _ in run_ids)
    rows = db.fetchall(
        f"""
        SELECT run_id, artifact_type, COUNT(*) AS c
        FROM experiment_artifacts
        WHERE run_id IN ({placeholders})
        GROUP BY run_id, artifact_type
        """,
        tuple(run_ids),
    )
    grouped: dict[int, dict[str, int]] = {}
    for row in rows:
        run_id = int(row["run_id"])
        grouped.setdefault(run_id, {})[str(row["artifact_type"])] = int(row["c"])
    return grouped


def _claim_counts_for_runs(run_ids: list[int]) -> dict[int, int]:
    if not run_ids:
        return {}
    placeholders = ",".join("?" for _ in run_ids)
    rows = db.fetchall(
        f"""
        SELECT run_id, COUNT(*) AS c
        FROM experimental_claims
        WHERE run_id IN ({placeholders})
        GROUP BY run_id
        """,
        tuple(run_ids),
    )
    return {int(row["run_id"]): int(row["c"]) for row in rows}


def _summarize_run(run: dict, artifact_counts: dict[str, int] | None = None, claim_count: int = 0) -> dict:
    artifact_counts = artifact_counts or {}
    return {
        **dict(run),
        "artifact_counts": artifact_counts,
        "artifact_total": sum(artifact_counts.values()),
        "has_plot_artifacts": artifact_counts.get("plot", 0) > 0,
        "has_bundle": bool(run.get("submission_bundle_id")),
        "claim_count": claim_count,
    }


def _planned_tracks(insight: dict, runs: list[dict]) -> list[dict]:
    evidence_plan = _json_load(insight.get("evidence_plan"), {})
    experimental_plan = _json_load(insight.get("experimental_plan"), {})
    ablations = experimental_plan.get("ablations") or []
    has_plot = any((run.get("artifact_counts") or {}).get("plot", 0) > 0 for run in runs)
    has_bundle = any(run.get("has_bundle") for run in runs) or (insight.get("submission_status") == "bundle_ready")
    main_state = "not_started"
    canonical = _pick_canonical_run(runs, insight.get("canonical_run_id"))
    if canonical:
        main_state = canonical.get("status") or "unknown"
    return [
        {"key": "main", "label": "主实验", "enabled": True, "state": main_state},
        {
            "key": "ablation",
            "label": "消融",
            "enabled": bool((evidence_plan.get("ablation") or {}).get("enabled") or ablations),
            "state": f"{len(ablations)} planned" if ablations else ("enabled" if (evidence_plan.get("ablation") or {}).get("enabled") else "not_planned"),
        },
        {
            "key": "visualization",
            "label": "可视化",
            "enabled": bool((evidence_plan.get("visualization") or {}).get("enabled") or has_plot),
            "state": "artifacts_ready" if has_plot else ("planned" if (evidence_plan.get("visualization") or {}).get("enabled") else "not_planned"),
        },
        {
            "key": "bundle",
            "label": "论文包",
            "enabled": True,
            "state": "bundle_ready" if has_bundle else (insight.get("submission_status") or "not_started"),
        },
    ]


def _planned_tracks(insight: dict, runs: list[dict]) -> list[dict]:
    evidence_plan = _json_load(insight.get("evidence_plan"), {})
    experimental_plan = _json_load(insight.get("experimental_plan"), {})
    ablations = experimental_plan.get("ablations") or []
    has_plot = any((run.get("artifact_counts") or {}).get("plot", 0) > 0 for run in runs)
    has_bundle = any(run.get("has_bundle") for run in runs) or (insight.get("submission_status") == "bundle_ready")
    canonical = _pick_canonical_run(runs, insight.get("canonical_run_id"))
    return [
        {
            "key": "main",
            "label": "Main experiment",
            "enabled": True,
            "state": (canonical or {}).get("status") or "not_started",
        },
        {
            "key": "ablation",
            "label": "Ablation",
            "enabled": bool((evidence_plan.get("ablation") or {}).get("enabled") or ablations),
            "state": f"{len(ablations)} planned" if ablations else ("enabled" if (evidence_plan.get("ablation") or {}).get("enabled") else "not_planned"),
        },
        {
            "key": "visualization",
            "label": "Visualization",
            "enabled": bool((evidence_plan.get("visualization") or {}).get("enabled") or has_plot),
            "state": "artifacts_ready" if has_plot else ("planned" if (evidence_plan.get("visualization") or {}).get("enabled") else "not_planned"),
        },
        {
            "key": "bundle",
            "label": "Paper bundle",
            "enabled": True,
            "state": "bundle_ready" if has_bundle else (insight.get("submission_status") or "not_started"),
        },
    ]


def _build_experiment_group_payload(*, insight: dict, auto_job: dict | None, runs: list[dict]) -> dict:
    runs = sorted(
        runs,
        key=lambda row: (str(row.get("created_at") or ""), int(row.get("id") or 0)),
        reverse=True,
    )
    latest_run = runs[0] if runs else None
    canonical_run = _pick_canonical_run(runs, insight.get("canonical_run_id"))
    run_status_counts: dict[str, int] = {}
    for run in runs:
        key = str(run.get("status") or "unknown")
        run_status_counts[key] = run_status_counts.get(key, 0) + 1
    updated_at = (
        (auto_job or {}).get("updated_at")
        or (canonical_run or {}).get("created_at")
        or (latest_run or {}).get("created_at")
        or insight.get("updated_at")
        or insight.get("created_at")
    )
    workspace = _workspace_payload(insight)
    return {
        "insight": insight,
        "auto_job": auto_job,
        "latest_run": latest_run,
        "canonical_run": canonical_run,
        "run_count": len(runs),
        "run_status_counts": run_status_counts,
        "planned_tracks": _planned_tracks(insight, runs),
        "updated_at": updated_at,
        "runs": runs,
        **workspace,
    }


def _service_counts(table: str, status_column: str = "status") -> dict[str, int]:
    rows = db.fetchall(
        f"""
        SELECT {status_column} AS status, COUNT(*) AS c
        FROM {table}
        GROUP BY {status_column}
        """
    )
    return {str(row["status"] or "unknown"): int(row["c"] or 0) for row in rows}


def _safe_service_payload(name: str, fn):
    try:
        return fn()
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        log_event("error", {"step": f"{name}_status", "error": str(exc)})
        return {"status": "error", "error": str(exc)}


def _recent_evoscientist_sessions(limit: int = 5) -> list[dict]:
    rows = db.fetchall(
        """
        SELECT arj.deep_insight_id, arj.status, arj.stage, arj.research_workdir,
               arj.last_note, arj.last_error, arj.updated_at, di.title
        FROM auto_research_jobs arj
        JOIN deep_insights di ON di.id = arj.deep_insight_id
        WHERE arj.research_workdir IS NOT NULL
           OR arj.status IN ('verifying', 'researching')
           OR arj.stage LIKE '%evosci%'
           OR arj.stage LIKE '%research%'
        ORDER BY arj.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    sessions = []
    for row in rows:
        item = dict(row)
        workdir = item.get("research_workdir")
        if workdir:
            wd = Path(str(workdir))
            session: dict[str, Any] = {"workdir": str(wd), "exists": wd.exists()}
            if wd.exists():
                final_report = wd / "final_report.md"
                proposal = wd / "research_proposal.md"
                log_path = wd / "evoscientist.log"
                session["final_report_ready"] = final_report.is_file() and final_report.stat().st_size > 100
                session["proposal_ready"] = proposal.is_file()
                if log_path.is_file():
                    try:
                        mtime = log_path.stat().st_mtime
                        session["log_age_seconds"] = round(time.time() - mtime, 2)
                        with log_path.open("rb") as fh:
                            fh.seek(0, 2)
                            size = fh.tell()
                            fh.seek(max(0, size - 2000))
                            session["log_tail"] = fh.read().decode("utf-8", errors="replace")
                    except OSError as exc:
                        session["log_error"] = str(exc)
            item["session"] = session
        sessions.append(item)
    return sessions


def _current_work_snapshot() -> dict[str, list[dict]]:
    recent_run_window = (
        "er.created_at > NOW() - INTERVAL '12 hours'"
        if db.use_postgres()
        else "er.created_at > datetime('now', '-12 hours')"
    )
    recent_pipeline = []
    for event in reversed(get_events(0)[-80:]):
        data = event.get("data") or {}
        event_type = event.get("type") or ""
        step = data.get("step") or event_type
        if event_type not in {
            "paper_worker",
            "step",
            "pipeline_start",
            "pipeline_done",
            "recovery",
            "abstraction_done",
            "bridge_done",
            "error",
        } and not step:
            continue
        title = step.replace("_", " ").strip().title() if step else event_type
        detail_parts = []
        for key in ("paper_id", "node_id", "max_papers", "batch_size", "papers_processed", "error"):
            value = data.get(key)
            if value not in (None, ""):
                detail_parts.append(f"{key}={value}")
        recent_pipeline.append(
            {
                "title": title,
                "status": event_type,
                "stage": step,
                "last_note": ", ".join(detail_parts),
                "updated_at": event.get("timestamp"),
            }
        )
        if len(recent_pipeline) >= 6:
            break

    papers = db.fetchall(
        """
        SELECT id, title, status, processing_stage, stage_last_error, updated_at
        FROM papers
        WHERE status IN ('processing', 'extracted')
        ORDER BY updated_at DESC
        LIMIT 8
        """
    )
    plans = db.fetchall(
        """
        SELECT arj.deep_insight_id, arj.status, arj.stage, arj.last_note, arj.last_error,
               arj.updated_at, di.title
        FROM auto_research_jobs arj
        JOIN deep_insights di ON di.id = arj.deep_insight_id
        WHERE arj.status IN ('queued', 'eligible', 'review_pending', 'smoke_only', 'harness_required')
           OR arj.stage LIKE '%review%'
           OR arj.stage LIKE '%forge%'
           OR arj.stage LIKE '%formal%'
        ORDER BY arj.updated_at DESC
        LIMIT 8
        """
    )
    experiments = db.fetchall(
        f"""
        SELECT er.id, er.deep_insight_id, er.status, er.phase, er.iterations_total,
               er.iterations_kept, er.hypothesis_verdict, er.effect_pct,
               er.created_at, di.title
        FROM experiment_runs er
        JOIN deep_insights di ON di.id = er.deep_insight_id
        WHERE er.status IN ('pending', 'scaffolding', 'reproducing', 'testing', 'running_gpu', 'running_cpu')
          AND (er.status IN ('running_gpu', 'running_cpu', 'testing') OR {recent_run_window})
          AND (
                EXISTS (
                    SELECT 1 FROM gpu_jobs gj
                    WHERE gj.experiment_run_id=er.id
                      AND gj.status IN ('queued', 'running')
                )
                OR NOT EXISTS (
                    SELECT 1 FROM gpu_jobs gj
                    WHERE gj.experiment_run_id=er.id
                )
              )
          AND NOT EXISTS (
                SELECT 1 FROM experiment_runs newer
                WHERE newer.deep_insight_id=er.deep_insight_id
                  AND newer.created_at > er.created_at
                  AND newer.status IN ('completed', 'failed', 'bundle_ready')
              )
        ORDER BY er.created_at DESC
        LIMIT 8
        """
    )
    manuscripts = db.fetchall(
        """
        SELECT mr.id, mr.experiment_run_id, mr.deep_insight_id, mr.status, mr.workdir,
               mr.updated_at, di.title
        FROM manuscript_runs mr
        LEFT JOIN deep_insights di ON di.id = mr.deep_insight_id
        ORDER BY mr.updated_at DESC
        LIMIT 8
        """
    )
    return {
        "pipeline": recent_pipeline,
        "papers": papers,
        "experiment_plans": plans,
        "experiments": experiments,
        "manuscripts": manuscripts,
    }


def _automation_snapshot() -> dict:
    from orchestrator import auto_research, paper_worker

    paper_worker_status = _safe_service_payload("paper_worker", paper_worker.get_status)
    auto_research_status = _safe_service_payload("auto_research", auto_research.get_status)

    def gpu_status():
        workers = db.fetchall(
            """
            SELECT id, hostname, gpu_index, gpu_model, total_mem_gb, status, heartbeat_at
            FROM gpu_workers
            ORDER BY gpu_index, id
            """
        )
        counts = _service_counts("gpu_jobs")
        return {
            "workers": workers,
            "total_jobs": sum(counts.values()),
            "queued_jobs": counts.get("queued", 0),
            "running_jobs": counts.get("running", 0),
            "completed_jobs": counts.get("completed", 0),
            "failed_jobs": counts.get("failed", 0),
        }

    def evoscientist_status():
        from agents.evosci_requirements import evosci_binary_path, evosci_installed

        sessions = _recent_evoscientist_sessions(limit=5)
        active = [
            item for item in sessions
            if (item.get("session") or {}).get("active") or item.get("status") in {"verifying", "researching"}
        ]
        return {
            "available": evosci_installed(),
            "binary_path": str(evosci_binary_path()),
            "active_count": len(active),
            "recent_sessions": sessions,
        }

    def paperorchestra_status():
        manuscripts = db.fetchall(
            """
            SELECT mr.*, di.title AS insight_title
            FROM manuscript_runs mr
            LEFT JOIN deep_insights di ON di.id = mr.deep_insight_id
            ORDER BY mr.updated_at DESC
            LIMIT 5
            """
        )
        counts = _service_counts("manuscript_runs")
        return {
            "available": importlib.util.find_spec("agents.paperorchestra") is not None,
            "backend": "paper_orchestra",
            "latex_available": bool(shutil.which("latexmk") or shutil.which("pdflatex")),
            "counts": counts,
            "active_count": int(counts.get("drafting", 0)),
            "recent_runs": manuscripts,
        }

    return {
        "paper_worker": paper_worker_status,
        "auto_research": auto_research_status,
        "gpu_scheduler": _safe_service_payload("gpu_scheduler", gpu_status),
        "evoscientist": _safe_service_payload("evoscientist", evoscientist_status),
        "paperorchestra": _safe_service_payload("paperorchestra", paperorchestra_status),
        "current_work": _safe_service_payload("current_work", _current_work_snapshot),
    }


def _load_experiment_groups(
    *,
    agenda_id: int,
    include_archived: bool = False,
) -> list[dict]:
    insights = db.fetchall(
        """
        SELECT di.*, arj.id AS auto_job_id, arj.status AS auto_status, arj.stage AS auto_stage,
               arj.cpu_eligible, arj.cpu_reason, arj.assigned_worker, arj.artifact_bundle_id,
               arj.experiment_run_id AS auto_experiment_run_id, arj.research_workdir,
               arj.last_note, arj.last_error, arj.updated_at AS auto_updated_at,
               arj.created_at AS auto_created_at
        FROM deep_insights di
        LEFT JOIN auto_research_jobs arj
          ON arj.deep_insight_id = di.id AND arj.agenda_id = di.agenda_id
        WHERE di.agenda_id=?
          AND (
               arj.id IS NOT NULL
               OR EXISTS (
                   SELECT 1 FROM experiment_runs er
                   WHERE er.deep_insight_id = di.id
                     AND er.agenda_id = di.agenda_id
               )
          )
        ORDER BY COALESCE(arj.updated_at, di.updated_at, di.created_at) DESC
        """,
        (agenda_id,),
    )
    if not insights:
        return []
    if not include_archived:
        insights = [
            row for row in insights
            if _deep_insight_is_displayable(row) and not _deep_insight_is_archived(row)
        ]
    if not insights:
        return []

    insight_ids = [int(row["id"]) for row in insights]
    placeholders = ",".join("?" for _ in insight_ids)
    run_rows = db.fetchall(
        f"""
        SELECT er.*, di.title AS insight_title, di.tier AS insight_tier
        FROM experiment_runs er
        JOIN deep_insights di
          ON di.id = er.deep_insight_id AND di.agenda_id = er.agenda_id
        WHERE er.agenda_id=?
          AND er.deep_insight_id IN ({placeholders})
        ORDER BY er.created_at DESC, er.id DESC
        """,
        (agenda_id, *insight_ids),
    )
    run_ids = [int(row["id"]) for row in run_rows]
    artifact_counts = _artifact_counts_for_runs(run_ids)
    claim_counts = _claim_counts_for_runs(run_ids)

    grouped_runs: dict[int, list[dict]] = {}
    for run in run_rows:
        deep_insight_id = int(run["deep_insight_id"])
        grouped_runs.setdefault(deep_insight_id, []).append(
            _summarize_run(
                run,
                artifact_counts=artifact_counts.get(int(run["id"]), {}),
                claim_count=claim_counts.get(int(run["id"]), 0),
            )
        )

    groups = []
    for row in insights:
        insight = dict(row)
        insight_id = int(insight["id"])
        auto_job = None
        if insight.get("auto_job_id") is not None:
            auto_job = {
                "id": insight.get("auto_job_id"),
                "status": insight.get("auto_status"),
                "stage": insight.get("auto_stage"),
                "cpu_eligible": insight.get("cpu_eligible"),
                "cpu_reason": insight.get("cpu_reason"),
                "assigned_worker": insight.get("assigned_worker"),
                "artifact_bundle_id": insight.get("artifact_bundle_id"),
                "experiment_run_id": insight.get("auto_experiment_run_id"),
                "research_workdir": insight.get("research_workdir"),
                "last_note": insight.get("last_note"),
                "last_error": insight.get("last_error"),
                "updated_at": insight.get("auto_updated_at"),
                "created_at": insight.get("auto_created_at"),
            }
        groups.append(
            _build_experiment_group_payload(
                insight=insight,
                auto_job=auto_job,
                runs=grouped_runs.get(insight_id, []),
            )
        )
    groups.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return groups


@app.route("/")
def index():
    return render_template(
        "index.html",
        app_name=APP_NAME,
        subtitle=APP_SUBTITLE,
        root_node_id=ROOT_NODE_ID,
        profile=PROFILE,
    )


@app.route("/api/meta")
def api_meta():
    backend = db.describe_backend()
    return jsonify({
        "app_name": APP_NAME,
        "subtitle": APP_SUBTITLE,
        "root_node_id": ROOT_NODE_ID,
        "profile": PROFILE,
        "database": backend,
    })


# ── Stats ──────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats_dict())


@app.route("/api/providers")
def api_providers():
    """Get LLM provider stats (round-robin load balancing)."""
    from agents.llm_client import get_provider_stats
    return jsonify(get_provider_stats())


@app.route("/api/runtime-config", methods=["GET", "POST"])
def api_runtime_config():
    """Expose model/provider and runtime configuration for the dashboard."""
    if request.method == "POST":
        return _manual_api_removed_response()

    try:
        db_info = db.describe_backend()
        workspace_disk = _disk_snapshot(cfg.WORKSPACE_DIR)
        experiment_disk = _disk_snapshot(cfg.EXPERIMENT_WORKDIR)
        cpu_info = _cpu_snapshot()
        gpu_info = _gpu_snapshot()
        return jsonify({
            "llm": {
                "primary": {
                    "model": cfg.LLM_MODEL,
                    "base_url": cfg.LLM_BASE_URL,
                    "protocol": cfg.LLM_PROTOCOL,
                    "rpm": cfg.LLM_RPM,
                    "api_key_configured": bool(cfg.LLM_API_KEY),
                },
                "secondary": {
                    "enabled": cfg.LLM_SECONDARY_ENABLED,
                    "model": cfg.LLM_SECONDARY_MODEL,
                    "base_url": cfg.LLM_SECONDARY_BASE_URL,
                    "protocol": cfg.LLM_SECONDARY_PROTOCOL,
                    "rpm": cfg.LLM_SECONDARY_RPM,
                    "api_key_configured": bool(cfg.LLM_SECONDARY_API_KEY),
                },
                "limits": {
                    "max_input_tokens": cfg.LLM_MAX_INPUT_TOKENS,
                    "max_output_tokens": cfg.LLM_MAX_OUTPUT_TOKENS,
                    "request_timeout_seconds": cfg.LLM_REQUEST_TIMEOUT_SECONDS,
                    "connect_timeout_seconds": cfg.LLM_CONNECT_TIMEOUT_SECONDS,
                },
                "extra_providers_configured": bool(str(cfg.LLM_EXTRA_PROVIDERS_JSON or "").strip()),
            },
            "runtime": {
                "app_name": cfg.APP_NAME,
                "profile": cfg.PROFILE,
                "root_node_id": cfg.ROOT_NODE_ID,
                "pid": os.getpid(),
                "python": cfg.RUNTIME_PYTHON,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "cpu": cpu_info,
                "process_rss_mb": _process_rss_mb(),
                "total_memory_mb": _total_memory_mb(),
                "database": db_info,
                "workspace_dir": str(cfg.WORKSPACE_DIR),
                "workspace_disk": workspace_disk,
            },
            "experiment": {
                "auto_pipeline_enabled": cfg.AUTO_PIPELINE_ENABLED,
                "auto_research_enabled": cfg.AUTO_RESEARCH_ENABLED,
                "pipeline_concurrency": cfg.PIPELINE_CONCURRENCY,
                "require_real_benchmark": cfg.EXPERIMENT_REQUIRE_REAL_BENCHMARK,
                "allow_synthetic_fallback": cfg.EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK,
                "real_llm_model": cfg.EXPERIMENT_REAL_LLM_MODEL,
                "benchmark_dataset": cfg.EXPERIMENT_REAL_BENCHMARK_DATASET,
                "benchmark_dataset_config": cfg.EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG,
                "real_benchmark_max_examples": cfg.EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
                "real_benchmark_seeds": cfg.EXPERIMENT_REAL_BENCHMARK_SEEDS,
                "full_benchmark_min_examples": cfg.EXPERIMENT_FULL_BENCHMARK_MIN_EXAMPLES,
                "full_benchmark_min_datasets": cfg.EXPERIMENT_FULL_BENCHMARK_MIN_DATASETS,
                "full_benchmark_min_models": cfg.EXPERIMENT_FULL_BENCHMARK_MIN_MODELS,
                "full_benchmark_min_baselines": cfg.EXPERIMENT_FULL_BENCHMARK_MIN_BASELINES,
                "full_benchmark_require_significance": cfg.EXPERIMENT_FULL_BENCHMARK_REQUIRE_SIGNIFICANCE,
                "full_benchmark_require_strongest_win": cfg.EXPERIMENT_FULL_BENCHMARK_REQUIRE_STRONGEST_WIN,
                "gpu_mode": cfg.GPU_MODE,
                "gpu_worker_slots": cfg.GPU_WORKER_SLOTS,
                "gpu_visible_devices": cfg.GPU_VISIBLE_DEVICES,
                "gpu_default_model": cfg.GPU_DEFAULT_MODEL,
                "gpu_default_vram_gb": cfg.GPU_DEFAULT_VRAM_GB,
                "gpu_snapshot": gpu_info,
                "experiment_workdir": str(cfg.EXPERIMENT_WORKDIR),
                "experiment_disk": experiment_disk,
            },
            "editable_keys": [],
        })
    except Exception as exc:
        return _api_failure("runtime_config", exc)


def _office_clip(value: Any, limit: int = 110) -> str:
    text = "" if value is None else " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 1)] + "..."


def _office_leaf(path: str) -> str:
    leaf = str(path or "").split(".")[-1]
    leaf = leaf.replace("_", " ").replace("-", " ").strip()
    if leaf.startswith("run "):
        leaf = leaf[4:]
    return leaf.title() or "Agent"


def _office_item(title, status="working", detail="", kind="task") -> dict:
    return {
        "title": _office_clip(title, 92),
        "status": _office_clip(status, 32) or "working",
        "detail": _office_clip(detail, 130),
        "kind": kind,
    }


def _office_department_state(items: list[dict], service_running: bool = False) -> str:
    statuses = " ".join(str(item.get("status") or "") for item in items).lower()
    if any(token in statuses for token in ("blocked", "error", "failed", "stale")):
        return "blocked"
    if service_running or items:
        return "working"
    return "idle"


@app.route("/api/agent_office")
def api_agent_office():
    """Lightweight agent registry plus current work for the overview office."""
    try:
        from agents.agent_registry import iter_agent_boundaries
        from orchestrator import auto_research, paper_worker

        work = _current_work_snapshot()
        paper_worker_status = _safe_service_payload("paper_worker", paper_worker.get_status)
        auto_research_status = _safe_service_payload("auto_research", auto_research.get_status)
        gpu_counts = _service_counts("gpu_jobs")
        manuscript_counts = _service_counts("manuscript_runs")

        pipeline_events = work.get("pipeline") or []
        stage_markers = {
            "paper_extraction": ("ingest", "prefetch", "download", "extract", "paper_"),
            "graph_construction": ("graph", "taxonomy", "node_", "opportunity", "signal", "abstraction", "bridge"),
            "idea_generation": ("reasoning", "contradiction", "insight", "novelty", "idea", "research"),
        }

        items_by_key = {key: [] for key in (
            "paper_extraction", "graph_construction", "idea_generation",
            "experiment_planning", "experiment_execution", "manuscript_generation", "orchestration"
        )}

        for paper in (work.get("papers") or [])[:5]:
            detail = "{} | {}".format(paper.get("id") or "", paper.get("processing_stage") or paper.get("status") or "")
            items_by_key["paper_extraction"].append(_office_item(paper.get("title") or paper.get("id"), paper.get("status") or "processing", detail, "paper"))

        for event in pipeline_events[:8]:
            stage = str(event.get("stage") or event.get("title") or "pipeline")
            status = str(event.get("status") or "step")
            detail = event.get("last_note") or event.get("updated_at") or ""
            lowered = stage.lower()
            placed = False
            for key, markers in stage_markers.items():
                if any(marker in lowered for marker in markers):
                    items_by_key[key].append(_office_item(event.get("title") or stage, status, detail, "pipeline"))
                    placed = True
                    break
            if not placed:
                items_by_key["orchestration"].append(_office_item(event.get("title") or stage, status, detail, "pipeline"))

        for plan in (work.get("experiment_plans") or [])[:5]:
            detail = plan.get("last_note") or plan.get("last_error") or plan.get("updated_at") or ""
            items_by_key["experiment_planning"].append(_office_item(plan.get("title"), plan.get("stage") or plan.get("status"), detail, "plan"))

        for run in (work.get("experiments") or [])[:5]:
            detail = "run {} | {}".format(run.get("id") or "", run.get("phase") or run.get("status") or "")
            items_by_key["experiment_execution"].append(_office_item(run.get("title") or run.get("deep_insight_id"), run.get("status") or "running", detail, "experiment"))

        running_gpu = int(gpu_counts.get("running", 0) or 0)
        queued_gpu = int(gpu_counts.get("queued", 0) or 0)
        if running_gpu or queued_gpu:
            items_by_key["experiment_execution"].append(_office_item("GPU scheduler", "running", "{} running, {} queued".format(running_gpu, queued_gpu), "gpu"))

        for manuscript in (work.get("manuscripts") or [])[:5]:
            detail = "run {} | {}".format(manuscript.get("id") or "", manuscript.get("updated_at") or "")
            items_by_key["manuscript_generation"].append(_office_item(manuscript.get("title") or manuscript.get("deep_insight_id"), manuscript.get("status") or "manuscript", detail, "manuscript"))

        if manuscript_counts:
            blocked = int(manuscript_counts.get("manuscript_blocked", 0) or 0)
            drafting = int(manuscript_counts.get("drafting", 0) or 0)
            if blocked or drafting:
                items_by_key["manuscript_generation"].append(_office_item("PaperOrchestra", "blocked" if blocked else "drafting", "{} drafting, {} blocked".format(drafting, blocked), "service"))

        if paper_worker_status.get("running"):
            items_by_key["orchestration"].append(_office_item("Paper worker", "running", "batch {} every {}s".format(paper_worker_status.get("batch_size") or "?", paper_worker_status.get("interval_seconds") or "?"), "service"))
        if auto_research_status.get("running"):
            items_by_key["orchestration"].append(_office_item("Auto research", "running", "review {} | blocked {} | completed {}".format(auto_research_status.get("review_pending") or 0, auto_research_status.get("blocked") or 0, auto_research_status.get("completed") or 0), "service"))

        service_running = {
            "paper_extraction": bool(paper_worker_status.get("running")) or bool(items_by_key["paper_extraction"]),
            "idea_generation": bool(auto_research_status.get("running")) and bool((auto_research_status.get("researching") or 0) or (auto_research_status.get("verifying") or 0)),
            "experiment_planning": bool(auto_research_status.get("review_pending") or 0),
            "experiment_execution": bool(running_gpu or queued_gpu),
            "manuscript_generation": bool(manuscript_counts.get("drafting", 0) or manuscript_counts.get("manuscript_blocked", 0)),
            "orchestration": bool(paper_worker_status.get("running") or auto_research_status.get("running")),
        }

        accents = ["blue", "green", "gold", "purple", "red", "cyan", "slate"]
        departments = []
        total_sub_agents = 0
        for index, boundary in enumerate(iter_agent_boundaries()):
            sub_agents = []
            for module in boundary.modules:
                sub_agents.append({"name": _office_leaf(module), "path": module, "kind": "module"})
            for script in boundary.scripts:
                sub_agents.append({"name": _office_leaf(script), "path": script, "kind": "script"})
            total_sub_agents += len(sub_agents)
            items = items_by_key.get(boundary.key, [])[:8]
            departments.append({
                "key": boundary.key,
                "title": boundary.title.replace(" Agent", ""),
                "responsibility": boundary.responsibility,
                "accent": accents[index % len(accents)],
                "status": _office_department_state(items, service_running.get(boundary.key, False)),
                "sub_agents": sub_agents,
                "items": items,
                "item_count": len(items_by_key.get(boundary.key, [])),
            })

        return jsonify({
            "departments": departments,
            "summary": {
                "departments": len(departments),
                "sub_agents": total_sub_agents,
                "working": sum(1 for dep in departments if dep["status"] == "working"),
                "blocked": sum(1 for dep in departments if dep["status"] == "blocked"),
            },
            "updated_at": time.time(),
        })
    except Exception as exc:
        return _api_failure("agent_office", exc)


@app.route("/api/processing")
def api_processing():
    """Get papers currently being processed + recently completed (last 15s)."""
    rows = db.fetchall(
        f"""SELECT id, title, status FROM papers
           WHERE status IN ('processing', 'extracted')
              OR (status IN ('reasoned', 'error') AND {db.sql_updated_after_seconds(15)})
           ORDER BY CASE status WHEN 'processing' THEN 0 WHEN 'extracted' THEN 1 ELSE 2 END, updated_at DESC
           LIMIT 30"""
    )
    processing_count = db.fetchone("SELECT COUNT(*) as c FROM papers WHERE status='processing'")["c"]
    paper_worker_status = {}
    try:
        from orchestrator import paper_worker
        paper_worker_status = paper_worker.get_status()
    except Exception as exc:
        paper_worker_status = {"running": False, "error": str(exc)}
    with _pipeline_lock:
        is_running = _pipeline_running or bool(paper_worker_status.get("running")) or processing_count > 0
    return jsonify({"papers": rows, "pipeline_running": is_running, "paper_worker": paper_worker_status})


@app.route("/api/automation")
def api_automation():
    """Read-only status for background automation workers."""
    try:
        return jsonify(_automation_snapshot())
    except Exception as exc:
        return _api_failure("automation", exc)


# ── Taxonomy Navigation ───────────────────────────────────────────

@app.route("/api/taxonomy")
def api_taxonomy():
    """Return the full taxonomy tree as a flat list."""
    return jsonify(tax.get_taxonomy_flat())


@app.route("/api/taxonomy/<node_id>")
def api_taxonomy_node(node_id):
    """Get a node, its children (with counts), breadcrumb path, papers, and matrix."""
    node = tax.get_node(node_id)
    if not node:
        return jsonify({"error": "Node not found"}), 404

    children = tax.get_children(node_id)
    breadcrumb = tax.get_breadcrumb(node_id)
    papers = tax.get_node_papers(node_id, limit=50)
    paper_clusters = tax.get_node_paper_clusters(node_id)
    is_leaf = len(children) == 0
    # Only load heavy data for leaf nodes
    intersections = tax.get_subfield_intersection_matrix(node_id) if not is_leaf else {}
    matrix = tax.get_method_dataset_matrix(node_id) if is_leaf else {"methods": [], "datasets": [], "metrics": [], "cells": {}}
    gaps = tax.get_node_gaps(node_id) if is_leaf else []
    # Only return cached data - never block on LLM generation during page load
    opportunities = opp.get_node_opportunities(node_id)
    summary = tax.get_node_summary(node_id)
    graph_summary = graph.get_node_graph_summary(node_id)

    return jsonify({
        "node": dict(node),
        "children": children,
        "breadcrumb": breadcrumb,
        "is_leaf": is_leaf,
        "papers": papers,
        "paper_clusters": paper_clusters,
        "intersections": intersections,
        "matrix": matrix,
        "gaps": gaps,
        "opportunities": opportunities,
        "summary": summary,
        "graph_summary": graph_summary,
    })


@app.route("/api/taxonomy/<node_id>/children")
def api_taxonomy_children(node_id):
    """Get just the children of a node with counts."""
    children = tax.get_children(node_id)
    return jsonify(children)


@app.route("/api/taxonomy/<node_id>/matrix")
def api_taxonomy_matrix(node_id):
    """Get the method x dataset matrix for a node."""
    matrix = tax.get_method_dataset_matrix(node_id)
    return jsonify(matrix)


@app.route("/api/taxonomy/<node_id>/intersections")
def api_taxonomy_intersections(node_id):
    """Get the subfield intersection matrix for a node."""
    return jsonify(tax.get_subfield_intersection_matrix(node_id))


@app.route("/api/taxonomy/<node_id>/papers")
def api_taxonomy_papers(node_id):
    """Get papers for a node."""
    limit = request.args.get("limit", 50, type=int)
    papers = tax.get_node_papers(node_id, limit=limit)
    return jsonify(papers)


@app.route("/api/taxonomy/<node_id>/paper_clusters")
def api_taxonomy_paper_clusters(node_id):
    """Get paper clusters for a node."""
    return jsonify(tax.get_node_paper_clusters(node_id))


@app.route("/api/taxonomy/<node_id>/gaps")
def api_taxonomy_gaps(node_id):
    """Get matrix gaps for a node."""
    gaps = tax.get_node_gaps(node_id)
    return jsonify(gaps)


@app.route("/api/taxonomy/<node_id>/opportunities")
def api_taxonomy_opportunities(node_id):
    """Get richer deterministic opportunity themes for a node."""
    return jsonify(opp.get_node_opportunities(node_id))


@app.route("/api/insights")
def api_insights():
    """Get deep research insights from the insight agent."""
    limit = request.args.get("limit", 50, type=int)
    node_id = request.args.get("node_id", "")
    insight_type = request.args.get("type", "")

    sql = "SELECT * FROM insights WHERE 1=1"
    params = []
    if node_id:
        sql += " AND (node_id=? OR node_id LIKE ? || '.%')"
        params.extend([node_id, node_id])
    if insight_type:
        sql += " AND insight_type=?"
        params.append(insight_type)
    sql += " ORDER BY (novelty_score + feasibility_score) DESC, created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    return jsonify(rows)


@app.route("/api/patterns")
def api_patterns():
    """Get cross-domain abstract patterns."""
    limit = request.args.get("limit", 50, type=int)
    node_id = request.args.get("node_id", "")
    level = request.args.get("level", "")

    sql = "SELECT * FROM patterns WHERE 1=1"
    params = []
    if node_id:
        sql += " AND node_id=?"
        params.append(node_id)
    if level:
        sql += " AND abstraction_level=?"
        params.append(level)
    sql += " ORDER BY domain_count DESC, created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    return jsonify(rows)


@app.route("/api/bridges")
def api_bridges():
    """Get cross-domain bridge insights."""
    limit = request.args.get("limit", 20, type=int)
    rows = db.fetchall(
        "SELECT * FROM insights WHERE insight_type='cross_domain_bridge' "
        "ORDER BY (novelty_score + feasibility_score) DESC LIMIT ?",
        (limit,)
    )
    return jsonify(rows)


@app.route("/api/taxonomy/<node_id>/graph")
def api_taxonomy_graph(node_id):
    """Get the entity-relation graph summary for a node."""
    return jsonify(graph.ensure_node_graph_summary(node_id) or {})


@app.route("/api/papers/<paper_id>/graph")
def api_paper_graph(paper_id):
    """Get entity-relation evidence for one paper."""
    return jsonify(graph.get_paper_graph(paper_id))


@app.route("/api/graph/merge_candidates")
def api_graph_merge_candidates():
    """List entity merge candidates."""
    status = request.args.get("status", "pending")
    entity_type = request.args.get("entity_type", "") or None
    limit = request.args.get("limit", 100, type=int)
    return jsonify(graph.list_merge_candidates_with_context(status=status, limit=limit, entity_type=entity_type))


@app.route("/api/graph/merge_candidates/<int:candidate_id>")
def api_graph_merge_candidate(candidate_id: int):
    """Get one merge candidate with supporting context."""
    row = graph.get_merge_candidate_context(candidate_id)
    if not row:
        return jsonify({"error": "Candidate not found"}), 404
    return jsonify(row)


@app.route("/api/graph/merge_candidates/refresh", methods=["POST"])
def api_graph_merge_candidates_refresh():
    """Refresh heuristic merge candidates."""
    return _manual_api_removed_response()


@app.route("/api/graph/merge_candidates/<int:candidate_id>/decision", methods=["POST"])
def api_graph_merge_candidate_decision(candidate_id: int):
    """Accept or reject a merge candidate."""
    return _manual_api_removed_response()


# ── Search ─────────────────────────────────────────────────────────

@app.route("/api/search")
def api_search():
    """Search across papers, methods, gaps, and taxonomy nodes."""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2:
        return jsonify({"papers": [], "methods": [], "gaps": [], "nodes": [], "opportunities": []})

    search_term = f"%{q}%"

    papers = db.fetchall(
        """SELECT p.id, p.title, p.status, p.published_date,
                  pi.plain_summary, pi.work_type
           FROM papers p
           LEFT JOIN paper_insights pi ON p.id = pi.paper_id
           WHERE p.title LIKE ? OR p.abstract LIKE ? OR pi.plain_summary LIKE ?
           ORDER BY p.published_date DESC
           LIMIT 15""",
        (search_term, search_term, search_term),
    )

    methods = db.fetchall(
        """SELECT DISTINCT method_name as name, COUNT(*) as result_count,
                  COUNT(DISTINCT paper_id) as paper_count
           FROM results
           WHERE method_name LIKE ?
           GROUP BY method_name
           ORDER BY paper_count DESC
           LIMIT 10""",
        (search_term,),
    )

    gaps = db.fetchall(
        """SELECT mg.*, tn.name as node_name
           FROM matrix_gaps mg
           JOIN taxonomy_nodes tn ON mg.node_id = tn.id
           WHERE mg.gap_description LIKE ? OR mg.method_name LIKE ?
              OR mg.dataset_name LIKE ? OR mg.research_proposal LIKE ?
           ORDER BY mg.value_score DESC
           LIMIT 10""",
        (search_term, search_term, search_term, search_term),
    )

    nodes = db.fetchall(
        """SELECT t.*,
                  (SELECT COUNT(DISTINCT pt.paper_id)
                   FROM paper_taxonomy pt
                   WHERE pt.node_id = t.id OR pt.node_id LIKE t.id || '.%') AS paper_count
           FROM taxonomy_nodes t
           WHERE t.name LIKE ? OR t.description LIKE ? OR t.id LIKE ?
           ORDER BY paper_count DESC
           LIMIT 10""",
        (search_term, search_term, search_term),
    )

    opportunities = db.fetchall(
        """SELECT no.*, tn.name as node_name
           FROM node_opportunities no
           JOIN taxonomy_nodes tn ON no.node_id = tn.id
           WHERE no.title LIKE ? OR no.description LIKE ?
           ORDER BY no.value_score DESC
           LIMIT 10""",
        (search_term, search_term),
    )

    return jsonify({
        "papers": papers,
        "methods": methods,
        "gaps": gaps,
        "nodes": nodes,
        "opportunities": opportunities,
    })


# ── Recently Discovered ───────────────────────────────────────────

@app.route("/api/recent_discoveries")
def api_recent_discoveries():
    """Get recently discovered gaps, contradictions, opportunities, and taxonomy expansions."""
    limit = request.args.get("limit", 10, type=int)

    recent_gaps = db.fetchall(
        """SELECT mg.*, tn.name as node_name
           FROM matrix_gaps mg
           JOIN taxonomy_nodes tn ON mg.node_id = tn.id
           ORDER BY mg.created_at DESC LIMIT ?""",
        (limit,),
    )

    recent_contradictions = db.fetchall(
        """SELECT c.*, ca.claim_text as claim_a_text, ca.paper_id as paper_a,
                  cb.claim_text as claim_b_text, cb.paper_id as paper_b
           FROM contradictions c
           LEFT JOIN claims ca ON c.claim_a_id = ca.id
           LEFT JOIN claims cb ON c.claim_b_id = cb.id
           ORDER BY c.created_at DESC LIMIT ?""",
        (limit,),
    )

    recent_opportunities = db.fetchall(
        """SELECT no.*, tn.name as node_name
           FROM node_opportunities no
           JOIN taxonomy_nodes tn ON no.node_id = tn.id
           ORDER BY no.created_at DESC LIMIT ?""",
        (limit,),
    )

    recent_papers = db.fetchall(
        """SELECT p.id, p.title, p.published_date, p.status,
                  pi.plain_summary, pi.work_type
           FROM papers p
           LEFT JOIN paper_insights pi ON p.id = pi.paper_id
           WHERE p.status IN ('extracted', 'reasoned')
           ORDER BY p.updated_at DESC LIMIT ?""",
        (limit,),
    )

    return jsonify({
        "gaps": recent_gaps,
        "contradictions": recent_contradictions,
        "opportunities": recent_opportunities,
        "papers": recent_papers,
    })


# ── Taxonomy Expansion Trigger ────────────────────────────────────

@app.route("/api/taxonomy/expand", methods=["POST"])
def api_taxonomy_expand():
    """Manually trigger taxonomy expansion."""
    return _manual_api_removed_response()


# ── Legacy Endpoints (kept for compatibility) ─────────────────────

@app.route("/api/papers")
def api_papers():
    limit = request.args.get("limit", 50, type=int)
    status = request.args.get("status", "")
    if status:
        papers = db.fetchall(
            """SELECT id, title, authors, abstract, categories, published_date, pdf_url,
                      status, token_cost, processing_stage, created_at, updated_at
               FROM papers WHERE status=? ORDER BY updated_at DESC LIMIT ?""",
            (status, limit))
    else:
        papers = db.fetchall(
            """SELECT id, title, authors, abstract, categories, published_date, pdf_url,
                      status, token_cost, processing_stage, created_at, updated_at
               FROM papers ORDER BY updated_at DESC LIMIT ?""",
            (limit,))
    return jsonify(papers)


@app.route("/api/claims")
def api_claims():
    limit = request.args.get("limit", 100, type=int)
    paper_id = request.args.get("paper_id", "")
    if paper_id:
        claims = db.fetchall("SELECT * FROM claims WHERE paper_id=?", (paper_id,))
    else:
        claims = db.fetchall("SELECT * FROM claims ORDER BY id DESC LIMIT ?", (limit,))
    return jsonify(claims)


@app.route("/api/results")
def api_results():
    """Get structured results, optionally filtered."""
    limit = request.args.get("limit", 100, type=int)
    paper_id = request.args.get("paper_id", "")
    node_id = request.args.get("node_id", "")
    method = request.args.get("method", "")

    sql = "SELECT DISTINCT r.*, p.title as paper_title FROM results r JOIN papers p ON r.paper_id = p.id"
    params = []
    if node_id:
        sql += " JOIN result_taxonomy rt ON rt.result_id = r.id"
    sql += " WHERE 1=1"
    if paper_id:
        sql += " AND r.paper_id=?"
        params.append(paper_id)
    if node_id:
        sql += " AND (rt.node_id=? OR rt.node_id LIKE ? || '.%')"
        params.extend([node_id, node_id])
    if method:
        sql += " AND r.method_name=?"
        params.append(method)
    sql += " ORDER BY r.id DESC LIMIT ?"
    params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    return jsonify(rows)


@app.route("/api/contradictions")
def api_contradictions():
    limit = request.args.get("limit", 50, type=int)
    rows = db.fetchall("""
        SELECT c.*, ca.claim_text as claim_a_text, ca.paper_id as paper_a,
               cb.claim_text as claim_b_text, cb.paper_id as paper_b
        FROM contradictions c
        LEFT JOIN claims ca ON c.claim_a_id = ca.id
        LEFT JOIN claims cb ON c.claim_b_id = cb.id
        ORDER BY c.id DESC LIMIT ?
    """, (limit,))
    return jsonify(rows)


@app.route("/api/matrix_gaps")
def api_matrix_gaps():
    """Get all matrix gaps, optionally filtered by node."""
    limit = request.args.get("limit", 50, type=int)
    node_id = request.args.get("node_id", "")
    if node_id:
        rows = db.fetchall(
            """SELECT mg.*, tn.name as node_name
               FROM matrix_gaps mg
               JOIN taxonomy_nodes tn ON mg.node_id = tn.id
               WHERE mg.node_id=? OR mg.node_id LIKE ? || '.%'
               ORDER BY mg.value_score DESC LIMIT ?""",
            (node_id, node_id, limit))
    else:
        rows = db.fetchall(
            """SELECT mg.*, tn.name as node_name
               FROM matrix_gaps mg
               JOIN taxonomy_nodes tn ON mg.node_id = tn.id
               ORDER BY mg.value_score DESC LIMIT ?""",
            (limit,))
    return jsonify(rows)


# ── Events (SSE) ──────────────────────────────────────────────────

@app.route("/api/events")
def api_events():
    """SSE endpoint for real-time updates."""
    def generate():
        # Start from near the end - only send last 20 events on connect
        all_events = get_events(0)
        last_seq = max(0, all_events[-1]["seq"] - 20) if all_events else 0
        while True:
            events = get_events(last_seq)
            for e in events:
                yield f"data: {json.dumps(e, ensure_ascii=False, default=str)}\n\n"
                last_seq = e["seq"] + 1
            time.sleep(2)  # slower polling = less browser load

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Pipeline Control ──────────────────────────────────────────────

@app.route("/api/start", methods=["POST"])
def api_start():
    """Start the pipeline."""
    return _manual_api_removed_response()


@app.route("/api/backfill_graph", methods=["POST"])
def api_backfill_graph():
    """Backfill graph evidence from existing structured records."""
    return _manual_api_removed_response()


# ── EvoScientist Bridge ──────────────────────────────────────────────

@app.route("/api/research/launch", methods=["POST"])
def api_research_launch():
    """Launch EvoScientist research from a DeepGraph insight."""
    return _manual_api_removed_response()


@app.route("/api/research/status")
def api_research_status():
    """Check status of an EvoScientist research session."""
    return _manual_api_removed_response()


@app.route("/api/research/proposal/<int:insight_id>")
def api_research_proposal(insight_id):
    """Preview the research proposal that would be sent to EvoScientist."""
    return _manual_api_removed_response()


@app.route("/api/insights/rank", methods=["POST"])
def api_rank_insights():
    """Block the unscoped legacy LLM ranker.

    Ranking/admission for v1 is performed by agenda-scoped decision packets
    and the portfolio API; the legacy endpoint has no candidate identity or
    ResourceGrant boundary.
    """
    return jsonify(
        {
            "error": "legacy_unscoped_ranker_disabled",
            "replacement": "/api/meta-harness/portfolio/decide",
        }
    ), 410


# ── Deep Insights (Tier 1 / Tier 2) ─────────────────────────────────

@app.route("/api/deep_insights")
def api_deep_insights():
    """List deep insights with optional tier/status filter."""
    tier = request.args.get("tier", "", type=str)
    status = request.args.get("status", "")
    limit = request.args.get("limit", 50, type=int)

    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        return jsonify({"error": "positive agenda_id query parameter required"}), 400
    sql = "SELECT * FROM deep_insights WHERE agenda_id=?"
    params = [agenda_id]
    if tier:
        sql += " AND tier=?"
        try:
            params.append(int(tier))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid tier value"}), 400
    if status:
        sql += " AND status=?"
        params.append(status)
    fetch_limit = min(max(limit * 4, limit), 200)
    sql += " ORDER BY CASE WHEN adversarial_score IS NOT NULL THEN adversarial_score ELSE 0 END DESC, created_at DESC LIMIT ?"
    params.append(fetch_limit)

    try:
        rows = db.fetchall(sql, tuple(params))
        if request.args.get("include_placeholders") not in {"1", "true", "yes"}:
            rows = [row for row in rows if _deep_insight_is_displayable(row)]
        if not _include_archived_requested():
            rows = [row for row in rows if not _deep_insight_is_archived(row)]
        return jsonify(rows[:limit])
    except Exception as exc:
        return _api_failure("deep_insights", exc)


@app.route("/api/deep_insights/<int:insight_id>")
def api_deep_insight_detail(insight_id):
    """Get full detail for one deep insight."""
    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        return jsonify({"error": "positive agenda_id query parameter required"}), 400
    row = db.fetchone(
        "SELECT * FROM deep_insights WHERE id=? AND agenda_id=?",
        (insight_id, agenda_id),
    )
    if not row:
        return jsonify({"error": "Not found"}), 404
    return jsonify(dict(row))


@app.route("/api/deep_insights/generate", methods=["POST"])
def api_generate_deep_insights():
    """Trigger discovery pipeline (harvest + Tier 1 + Tier 2).

    JSON body (optional):
      tier: "1" | "2" | "both" (default both)
      bulk: true — use DISCOVERY_BULK_* wider signals + expand all Tier2 problems
    """
    return _manual_api_removed_response()


@app.route("/api/deep_insights/<int:insight_id>/verify", methods=["POST"])
def api_verify_deep_insight(insight_id):
    """Launch novelty verification via EvoScientist."""
    return _manual_api_removed_response()


@app.route("/api/deep_insights/<int:insight_id>/verify_status")
def api_verify_status(insight_id):
    """Check verification status."""
    return _manual_api_removed_response()


@app.route("/api/deep_insights/<int:insight_id>/research", methods=["POST"])
def api_deep_insight_research(insight_id):
    """Launch full EvoScientist research session."""
    return _manual_api_removed_response()


@app.route("/api/deep_insights/signals")
def api_deep_insight_signals():
    """Get current signal harvester data."""
    try:
        overlaps = db.fetchall(
            "SELECT * FROM node_entity_overlap ORDER BY overlap_score DESC LIMIT 20")
        pattern_ms = db.fetchall(
            "SELECT * FROM pattern_matches ORDER BY similarity_score DESC LIMIT 15")
        clusters = db.fetchall(
            "SELECT * FROM contradiction_clusters ORDER BY cluster_size DESC")
        plateaus = db.fetchall(
            "SELECT * FROM performance_plateaus ORDER BY method_count DESC LIMIT 15")
        protocol = db.fetchall(
            "SELECT * FROM protocol_artifacts ORDER BY support_count DESC LIMIT 15")
        negative = db.fetchall(
            "SELECT * FROM negative_space_gaps ORDER BY support_count DESC LIMIT 15")
        bridges = db.fetchall(
            "SELECT * FROM hidden_variable_bridges ORDER BY score DESC LIMIT 15")
        claim_gaps = db.fetchall(
            "SELECT * FROM claim_method_gaps ORDER BY support_count DESC LIMIT 15")
        return jsonify({
            "entity_overlaps": overlaps,
            "pattern_matches": pattern_ms,
            "contradiction_clusters": clusters,
            "performance_plateaus": plateaus,
            "protocol_artifacts": protocol,
            "negative_space_gaps": negative,
            "hidden_variable_bridges": bridges,
            "claim_method_gaps": claim_gaps,
        })
    except Exception as exc:
        return _api_failure("deep_insight_signals", exc)


@app.route("/api/discovery/candidates")
def api_discovery_candidates():
    from agents.discovery_supervisor import collect_candidate_pool

    limit = request.args.get("limit", 50, type=int)
    return jsonify(collect_candidate_pool(limit=limit))


@app.route("/api/discovery/rankings")
def api_discovery_rankings():
    from agents.discovery_supervisor import rank_candidates

    limit = request.args.get("limit", 20, type=int)
    return jsonify(rank_candidates(limit=limit))


# ── SciForge: Experiment Validation ──────────────────────────────────

@app.route("/api/experiments")
def api_experiments():
    """List experiment runs with optional status/insight filter."""
    status = request.args.get("status", "")
    insight_id = request.args.get("insight_id", "", type=str)
    limit = request.args.get("limit", 50, type=int)

    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        return jsonify({"error": "positive agenda_id query parameter required"}), 400
    sql = """SELECT er.*, di.title as insight_title, di.tier as insight_tier
             FROM experiment_runs er
             JOIN deep_insights di
               ON er.deep_insight_id = di.id AND er.agenda_id = di.agenda_id
             WHERE er.agenda_id=?"""
    params = [agenda_id]
    if status:
        sql += " AND er.status=?"
        params.append(status)
    if insight_id:
        sql += " AND er.deep_insight_id=?"
        try:
            params.append(int(insight_id))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid insight_id value"}), 400
    sql += " ORDER BY er.created_at DESC LIMIT ?"
    params.append(limit)

    try:
        rows = db.fetchall(sql, tuple(params))
        return jsonify(rows)
    except Exception as exc:
        return _api_failure("experiments", exc)


@app.route("/api/experiment_groups")
def api_experiment_groups():
    """List idea-centric experiment groups for the dashboard."""
    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        return jsonify({"error": "positive agenda_id query parameter required"}), 400
    status = request.args.get("status", "")
    limit = request.args.get("limit", 50, type=int)
    groups = _load_experiment_groups(
        agenda_id=agenda_id,
        include_archived=_include_archived_requested(),
    )
    if status:
        groups = [
            group
            for group in groups
            if ((group.get("canonical_run") or {}).get("status") == status)
        ]
    return jsonify(groups[:limit])


@app.route("/api/experiment_groups/<int:insight_id>")
def api_experiment_group_detail(insight_id):
    """Get one idea-centric experiment group with run history."""
    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        return jsonify({"error": "positive agenda_id query parameter required"}), 400
    groups = _load_experiment_groups(agenda_id=agenda_id, include_archived=True)
    for group in groups:
        if int(group["insight"]["id"]) == insight_id:
            return jsonify(group)
    return jsonify({"error": "Not found"}), 404


def _load_paper_preview_context(insight_id: int) -> tuple[dict, dict]:
    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        abort(400)
    insight = db.fetchone(
        "SELECT * FROM deep_insights WHERE id=? AND agenda_id=?",
        (insight_id, agenda_id),
    )
    if not insight:
        abort(404)
    payload = _workspace_payload(dict(insight))
    return dict(insight), payload


def _pick_main_asset(assets: list[dict], suffix: str) -> str | None:
    for asset in assets:
        path = str(asset.get("path") or "")
        if path.endswith(f"/main{suffix}") or path == f"current/main{suffix}":
            return path
    return None


def _read_paper_asset_text(insight_id: int, asset: str | None, insight: dict) -> str:
    if not asset:
        return ""
    try:
        path = resolve_paper_asset(insight_id, asset, insight=insight)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    return ""


def _strip_latex_inline(text: str) -> str:
    text = re.sub(r"\\(?:textbf|emph|textit|method)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = text.replace("{", "").replace("}", "")
    return " ".join(text.split())


def _extract_latex_abstract(tex: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    return _strip_latex_inline(match.group(1)) if match else ""


def _extract_latex_title(tex: str) -> str:
    match = re.search(r"\\title\{(.*?)\}", tex, re.S)
    return _strip_latex_inline(match.group(1)) if match else ""


def _paper_generation_complete(paper: dict[str, Any]) -> bool:
    status = str(paper.get("status") or paper.get("manuscript_status") or "").strip().lower()
    complete_statuses = {"bundle_ready", "completed", "paper_ready", "ready", "submitted", "accepted"}
    return bool(
        paper.get("bundle_count")
        or paper.get("pdf_url")
        or paper.get("tex_url")
        or paper.get("main_pdf")
        or paper.get("main_tex")
        or status in complete_statuses
    )


@app.route("/api/generated_papers")
def api_generated_papers():
    """List DeepGraph-generated manuscripts, not imported arXiv papers."""
    try:
        agenda_id = _required_agenda_query_id()
        if agenda_id is None:
            return jsonify({"error": "positive agenda_id query parameter required"}), 400
        limit = request.args.get("limit", 100, type=int)
        include_archived = _include_archived_requested()
        rows = db.fetchall(
            """
            SELECT *
            FROM deep_insights
            WHERE agenda_id=?
            ORDER BY
              CASE WHEN submission_status='bundle_ready' THEN 0 ELSE 1 END,
              updated_at DESC
            LIMIT ?
            """,
            (agenda_id, limit),
        )
        papers: list[dict[str, Any]] = []
        for raw in rows:
            insight = dict(raw)
            if not _deep_insight_is_displayable(insight):
                continue
            insight_id = int(insight["id"])
            if (
                not include_archived
                and insight_id not in _PROTECTED_GENERATED_PAPER_IDS
                and _deep_insight_is_archived(insight)
            ):
                continue
            assets = list_paper_assets(insight_id, insight=insight)
            main_pdf = _pick_main_asset(assets, ".pdf")
            main_tex = _pick_main_asset(assets, ".tex")
            has_paper_asset = bool(main_pdf or main_tex or assets)
            if not has_paper_asset and (insight.get("submission_status") or "not_started") == "not_started":
                continue

            preview_urls = _paper_preview_urls(
                insight_id,
                assets,
                agenda_id=agenda_id,
            )
            tex_source = _read_paper_asset_text(insight_id, main_tex, insight)
            manuscript_rows = db.fetchall(
                """
                SELECT mr.id, mr.status, mr.workdir, mr.created_at, mr.updated_at,
                       sb.id AS bundle_id, sb.bundle_format, sb.status AS bundle_status,
                       sb.bundle_path, sb.created_at AS bundle_created_at
                FROM manuscript_runs mr
                LEFT JOIN submission_bundles sb
                  ON sb.manuscript_run_id=mr.id AND sb.agenda_id=mr.agenda_id
                WHERE mr.deep_insight_id=? AND mr.agenda_id=?
                ORDER BY COALESCE(sb.created_at, mr.updated_at) DESC
                """,
                (insight_id, agenda_id),
            )
            bundle_rows = [
                {
                    "bundle_id": r.get("bundle_id"),
                    "bundle_format": r.get("bundle_format"),
                    "bundle_status": r.get("bundle_status"),
                    "bundle_path": r.get("bundle_path"),
                    "bundle_created_at": r.get("bundle_created_at"),
                    "manuscript_run_id": r.get("id"),
                    "manuscript_status": r.get("status"),
                    "workdir": r.get("workdir"),
                }
                for r in manuscript_rows
                if r.get("bundle_id")
            ]
            latest_run = dict(manuscript_rows[0]) if manuscript_rows else {}
            status = insight.get("submission_status") or latest_run.get("status") or "draft"
            title = _extract_latex_title(tex_source) or insight.get("title") or f"Idea {insight_id}"
            abstract = _extract_latex_abstract(tex_source) or insight.get("problem_statement") or insight.get("evidence_summary") or ""
            paper = {
                "id": f"idea_{insight_id}",
                "insight_id": insight_id,
                "title": title,
                "status": status,
                "tier": insight.get("tier"),
                "updated_at": insight.get("updated_at"),
                "created_at": insight.get("created_at"),
                "paper_root": insight.get("paper_root"),
                "preview_url": preview_urls.get("index"),
                "pdf_url": preview_urls.get("pdf"),
                "tex_url": preview_urls.get("tex"),
                "main_pdf": main_pdf,
                "main_tex": main_tex,
                "asset_count": len(assets),
                "assets": assets[:80],
                "bundle_count": len(bundle_rows),
                "bundles": bundle_rows[:8],
                "manuscript_status": latest_run.get("status"),
                "abstract": abstract,
                "problem_statement": insight.get("problem_statement"),
                "proposed_method": _json_load(insight.get("proposed_method"), {}),
                "evidence_summary": insight.get("evidence_summary"),
                "source_node_ids": _json_load(insight.get("source_node_ids"), []),
                "canonical_run_id": insight.get("canonical_run_id"),
            }
            paper["paper_complete"] = _paper_generation_complete(paper)
            papers.append(paper)
        papers.sort(key=lambda paper: str(paper.get("updated_at") or paper.get("created_at") or ""), reverse=True)
        papers.sort(key=lambda paper: not bool(paper.get("paper_complete")))
        return jsonify(papers)
    except Exception as exc:
        return _api_failure("generated_papers", exc)


@app.route("/papers/<int:insight_id>")
def paper_preview_index(insight_id):
    insight, payload = _load_paper_preview_context(insight_id)
    return render_template(
        "paper_preview.html",
        insight=insight,
        paper_assets=payload["paper_assets"],
        preview_urls=payload["paper_preview_urls"],
        plan_snapshot=payload["plan_snapshot"],
    )


@app.route("/papers/<int:insight_id>/view/<path:asset>")
def paper_preview_asset(insight_id, asset):
    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        abort(400)
    insight = db.fetchone(
        "SELECT * FROM deep_insights WHERE id=? AND agenda_id=?",
        (insight_id, agenda_id),
    )
    if not insight:
        abort(404)
    try:
        resolved = resolve_paper_asset(insight_id, asset, insight=dict(insight))
    except ValueError:
        abort(404)
    if not resolved.exists() or not resolved.is_file():
        abort(404)
    return send_file(resolved, as_attachment=False)


@app.route("/papers/<int:insight_id>/pdf")
def paper_preview_pdf(insight_id):
    insight, payload = _load_paper_preview_context(insight_id)
    pdf_asset = _pick_main_asset(payload["paper_assets"], ".pdf")
    if not pdf_asset:
        return jsonify({"error": "Compiled PDF not found for this idea"}), 404
    resolved = resolve_paper_asset(insight_id, pdf_asset, insight=insight)
    return send_file(resolved, as_attachment=False)


@app.route("/papers/<int:insight_id>/tex")
def paper_preview_tex(insight_id):
    insight, payload = _load_paper_preview_context(insight_id)
    tex_asset = _pick_main_asset(payload["paper_assets"], ".tex")
    if not tex_asset:
        return jsonify({"error": "main.tex not found for this idea"}), 404
    resolved = resolve_paper_asset(insight_id, tex_asset, insight=insight)
    return send_file(resolved, as_attachment=False)


@app.route("/api/experiments/<int:run_id>")
def api_experiment_detail(run_id):
    """Get full detail for one experiment run including iterations."""
    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        return jsonify({"error": "positive agenda_id query parameter required"}), 400
    run = db.fetchone(
        """SELECT er.*, di.title as insight_title, di.tier as insight_tier
           FROM experiment_runs er
           JOIN deep_insights di
             ON er.deep_insight_id = di.id AND er.agenda_id = di.agenda_id
           WHERE er.id=? AND er.agenda_id=?""", (run_id, agenda_id))
    if not run:
        return jsonify({"error": "Not found"}), 404

    iterations = db.fetchall(
        """SELECT * FROM experiment_iterations
           WHERE run_id=? AND agenda_id=?
           ORDER BY iteration_number""", (run_id, agenda_id))

    claims = db.fetchall(
        "SELECT * FROM experimental_claims WHERE run_id=? AND agenda_id=?",
        (run_id, agenda_id),
    )

    return jsonify({
        "run": dict(run),
        "iterations": iterations,
        "claims": claims,
    })


@app.route("/api/experiments/forge", methods=["POST"])
def api_forge_experiment():
    """Forge an experiment from a deep insight (scaffold + codebase)."""
    return _manual_api_removed_response()


@app.route("/api/experiments/<int:run_id>/run", methods=["POST"])
def api_run_experiment(run_id):
    """Launch the validation loop for a forged experiment."""
    return _manual_api_removed_response()


@app.route("/api/experiments/run_full", methods=["POST"])
def api_run_full_experiment():
    """Full pipeline: forge + validation loop + knowledge loop for a deep insight."""
    return _manual_api_removed_response()


@app.route("/api/meta_report")
def api_meta_report():
    """Get the meta-learning report on hypothesis quality."""
    from agents.meta_learner import get_full_meta_report
    try:
        return jsonify(get_full_meta_report(request.args.get("agenda_id")))
    except Exception as exc:
        return _api_failure("meta_report", exc)


# ── Auto Research ────────────────────────────────────────────────────

@app.route("/api/auto_research/status")
def api_auto_research_status():
    return _manual_api_removed_response()


@app.route("/api/auto_research/jobs")
def api_auto_research_jobs():
    return _manual_api_removed_response()


@app.route("/api/auto_research/start", methods=["POST"])
def api_auto_research_start():
    return _manual_api_removed_response()


@app.route("/api/auto_research/stop", methods=["POST"])
def api_auto_research_stop():
    return _manual_api_removed_response()


@app.route("/api/gpu/status")
def api_gpu_status():
    return _manual_api_removed_response()


@app.route("/api/gpu/jobs")
def api_gpu_jobs():
    return _manual_api_removed_response()


@app.route("/api/manuscripts")
def api_manuscripts():
    from agents.manuscript_pipeline import list_manuscripts

    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        return jsonify({"error": "positive agenda_id query parameter required"}), 400
    limit = request.args.get("limit", 50, type=int)
    return jsonify(list_manuscripts(agenda_id=agenda_id, limit=limit))


@app.route("/api/manuscripts/<int:run_id>/bundle", methods=["POST"])
def api_manuscript_bundle(run_id):
    return _manual_api_removed_response()


@app.route("/api/submission_bundles/<int:bundle_id>")
def api_submission_bundle(bundle_id):
    agenda_id = _required_agenda_query_id()
    if agenda_id is None:
        return jsonify({"error": "positive agenda_id query parameter required"}), 400
    row = db.fetchone(
        "SELECT * FROM submission_bundles WHERE id=? AND agenda_id=?",
        (bundle_id, agenda_id),
    )
    if not row:
        return jsonify({"error": "Not found"}), 404
    manuscript_run = db.fetchone(
        "SELECT * FROM manuscript_runs WHERE id=? AND agenda_id=?",
        (row["manuscript_run_id"], agenda_id),
    )
    assets = db.fetchall(
        """SELECT * FROM manuscript_assets
           WHERE manuscript_run_id=? AND agenda_id=? ORDER BY id""",
        (row["manuscript_run_id"], agenda_id),
    )
    return jsonify({"bundle": row, "manuscript_run": manuscript_run, "assets": assets})
