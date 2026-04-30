"""Generate manuscript bundles using PaperOrchestra as the only backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import SUBMISSION_BUNDLE_FORMATS
from contracts import ContractValidationError, ManuscriptInputState
from db import database as db


def _json_load(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _json_list(value) -> list:
    loaded = _json_load(value, [])
    return loaded if isinstance(loaded, list) else []


def _json_dict(value) -> dict[str, Any]:
    loaded = _json_load(value, {})
    return loaded if isinstance(loaded, dict) else {}


def _dedupe(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _best_iteration(iterations: list[dict]) -> dict | None:
    kept = [it for it in iterations if it.get("status") == "keep" and it.get("metric_value") is not None]
    if not kept:
        return None
    return max(kept, key=lambda item: item.get("metric_value") or 0)


def _load_result_packet(run: dict, claims: list[dict]) -> dict[str, Any]:
    workdir_raw = str(run.get("workdir") or "").strip()
    if workdir_raw:
        workdir = Path(workdir_raw)
        packet_path = workdir / "results" / "experiment_result_packet.json"
        if packet_path.exists():
            return _json_dict(packet_path.read_text(encoding="utf-8"))
    for claim in claims:
        packet = _json_dict(_json_dict(claim.get("supporting_data")).get("result_packet"))
        if packet:
            return packet
    proxy = _json_dict(run.get("proxy_config"))
    return {
        "run_id": run.get("id"),
        "deep_insight_id": run.get("deep_insight_id"),
        "formal_experiment": bool(proxy.get("formal_experiment")),
        "smoke_test_only": bool(proxy.get("smoke_test_only")),
        "metric_name": run.get("baseline_metric_name") or "metric",
        "verdict": run.get("hypothesis_verdict") or "inconclusive",
        "baseline": run.get("baseline_metric_value"),
        "best": run.get("best_metric_value"),
        "effect_pct": run.get("effect_pct"),
    }


def _claim_source_papers(claim: dict, fallback_ids: list[str]) -> list[str]:
    supporting_data = _json_dict(claim.get("supporting_data"))
    candidates: list[str] = []
    for raw in (
        claim.get("source_paper_ids"),
        supporting_data.get("source_paper_ids"),
        supporting_data.get("supporting_papers"),
        supporting_data.get("paper_ids"),
        fallback_ids,
    ):
        for item in _json_list(raw) if not isinstance(raw, list) else raw:
            if item:
                candidates.append(str(item))
    return _dedupe(candidates)


def _claim_source_nodes(claim: dict, fallback_ids: list[str]) -> list[str]:
    supporting_data = _json_dict(claim.get("supporting_data"))
    candidates: list[str] = []
    for raw in (
        claim.get("source_node_ids"),
        supporting_data.get("source_node_ids"),
        fallback_ids,
    ):
        for item in _json_list(raw) if not isinstance(raw, list) else raw:
            if item:
                candidates.append(str(item))
    return _dedupe(candidates)


def _build_claim_records(claims: list[dict], *, fallback_papers: list[str], fallback_nodes: list[str], evidence_summary: str) -> list[dict]:
    out: list[dict] = []
    for claim in claims:
        supporting_data = _json_dict(claim.get("supporting_data"))
        out.append(
            {
                "id": claim.get("id"),
                "claim_text": claim.get("claim_text") or "",
                "claim_type": claim.get("claim_type") or "experimental",
                "verdict": claim.get("verdict") or "inconclusive",
                "effect_size": claim.get("effect_size"),
                "confidence": claim.get("confidence"),
                "supporting_data": supporting_data,
                "source_paper_ids": _claim_source_papers(claim, fallback_papers),
                "source_node_ids": _claim_source_nodes(claim, fallback_nodes),
                "evidence_summary": supporting_data.get("evidence_summary") or evidence_summary,
            }
        )
    return out


def build_manuscript_input_state(run: dict, insight: dict, iterations: list[dict], claims: list[dict]) -> ManuscriptInputState:
    method = _json_dict(insight.get("proposed_method"))
    plan = _json_dict(insight.get("experimental_plan"))
    evidence_plan = _json_dict(insight.get("evidence_plan"))
    related_raw = _json_load(insight.get("related_work_positioning"), {})
    related = related_raw if isinstance(related_raw, dict) else {}
    supporting_papers = [str(x) for x in _json_list(insight.get("supporting_papers")) if x]
    source_paper_ids = [str(x) for x in _json_list(insight.get("source_paper_ids")) if x]
    source_node_ids = [str(x) for x in _json_list(insight.get("source_node_ids")) if x]
    citation_seed_paper_ids = _dedupe(supporting_papers + source_paper_ids)
    evidence_packet = _json_dict(insight.get("evidence_packet"))
    evidence_summary = insight.get("evidence_summary") or insight.get("related_work_positioning") or ""
    best_iter = _best_iteration(iterations)
    result_packet = _load_result_packet(run, claims)

    contributions = [
        method.get("one_line") or "Mechanism-first insight generation and automated experiment loop.",
        f"Validated with baseline {run.get('baseline_metric_value')} and best metric {run.get('best_metric_value')}.",
        f"Generated as a {insight.get('mechanism_type') or 'mechanism-first'} DeepGraph insight.",
    ]

    claim_records = _build_claim_records(
        claims,
        fallback_papers=citation_seed_paper_ids,
        fallback_nodes=source_node_ids,
        evidence_summary=str(evidence_summary),
    )

    return ManuscriptInputState(
        run_id=run.get("id"),
        deep_insight_id=run.get("deep_insight_id"),
        formal_experiment=bool(result_packet.get("formal_experiment")),
        smoke_test_only=bool(result_packet.get("smoke_test_only")),
        title=insight.get("title") or f"DeepGraph Run {run['id']}",
        problem_statement=insight.get("problem_statement") or insight.get("existing_weakness") or "",
        existing_weakness=insight.get("existing_weakness") or "",
        method_name=method.get("name") or insight.get("title") or "DeepGraph Method",
        method_summary=method.get("one_line") or method.get("definition") or "",
        method_payload=method,
        mechanism_type=insight.get("mechanism_type") or "mechanism_first",
        resource_class=run.get("resource_class") or insight.get("resource_class") or "cpu",
        baseline_metric_name=run.get("baseline_metric_name") or result_packet.get("metric_name") or "metric",
        baseline_metric_value=result_packet.get("baseline", run.get("baseline_metric_value")),
        best_metric_value=result_packet.get("best", run.get("best_metric_value")),
        effect_pct=result_packet.get("effect_pct", run.get("effect_pct")),
        verdict=result_packet.get("verdict", run.get("hypothesis_verdict") or "inconclusive"),
        claims=claim_records,
        iterations=iterations,
        best_iteration=best_iter or _json_dict(result_packet.get("best_iteration")),
        datasets=plan.get("datasets", []),
        baselines=plan.get("baselines", []),
        paper_outline=related,
        contributions=contributions,
        supporting_papers=supporting_papers,
        source_paper_ids=source_paper_ids,
        source_node_ids=source_node_ids,
        citation_seed_paper_ids=citation_seed_paper_ids,
        evidence_summary=str(evidence_summary),
        evidence_packet=evidence_packet,
        evidence_plan=evidence_plan,
        experimental_plan=plan,
        submission_keywords=related.get("submission_keywords", [insight.get("mechanism_type"), insight.get("resource_class")]),
        result_packet=result_packet,
    )


def _build_canonical_state(run: dict, insight: dict, iterations: list[dict], claims: list[dict]) -> dict:
    return build_manuscript_input_state(run, insight, iterations, claims).to_dict()


def _bundle_manifest(bundle_root: Path) -> dict:
    manifest = {"files": []}
    for path in sorted(bundle_root.rglob("*")):
        if path.is_file():
            manifest["files"].append({"path": str(path.relative_to(bundle_root)), "size": path.stat().st_size})
    return manifest


def _store_assets(manuscript_run_id: int, bundle_root: Path, bundle_format: str) -> int:
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        asset_type = path.suffix.lstrip(".") or "text"
        if path.name == "cover_letter.md":
            asset_type = "cover_letter"
        elif path.name == "references.bib":
            asset_type = "bib"
        elif path.suffix == ".tex":
            asset_type = "tex"
        elif path.suffix in {".svg", ".pdf", ".png"}:
            asset_type = "figure"
        db.execute(
            """
            INSERT INTO manuscript_assets (manuscript_run_id, asset_type, label, path)
            VALUES (?, ?, ?, ?)
            """,
            (manuscript_run_id, asset_type, f"{bundle_format}:{path.name}", str(path)),
        )
    bid = db.insert_returning_id(
        """
        INSERT INTO submission_bundles (manuscript_run_id, bundle_format, status, bundle_path, manifest_path)
        VALUES (?, ?, 'ready', ?, ?)
        RETURNING id
        """,
        (manuscript_run_id, bundle_format, str(bundle_root), str(bundle_root / "artifact_manifest.json")),
    )
    db.commit()
    return bid


def generate_submission_bundle(run_id: int, bundle_formats: list[str] | None = None) -> dict:
    from agents.paper_orchestra_pipeline import generate_bundle_paper_orchestra

    return generate_bundle_paper_orchestra(run_id, bundle_formats=bundle_formats or list(SUBMISSION_BUNDLE_FORMATS))


def list_manuscripts(limit: int = 50) -> list[dict]:
    db.init_db()
    return db.fetchall(
        """
        SELECT mr.*, di.title AS insight_title, er.hypothesis_verdict
        FROM manuscript_runs mr
        LEFT JOIN deep_insights di ON di.id = mr.deep_insight_id
        LEFT JOIN experiment_runs er ON er.id = mr.experiment_run_id
        ORDER BY mr.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    )
