#!/usr/bin/env python3
"""Prepare a local CGGR top-venue adaptive-reasoning baseline shard template."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import experiment_forge  # noqa: E402


TOP_VENUE_METHODS = [
    "CAR-Style Certainty Adaptive Routing",
    "Self-Route-Style Mode Routing",
    "Rational-Metareasoning VOC Routing",
]

NON_DEPLOYABLE_DIAGNOSTICS = {
    "Oracle Routing Upper Bound",
    "Oracle-Router Upper Bound",
}

CANONICAL_ABLATIONS = [
    "no_counterfactual_delta",
    "no_lcb",
    "no_self_divergence_penalty",
    "no_qstruct_term",
]

OLD_LEARNED_ROUTER_CLAIM = (
    "CGGR learns a per-input stopping and routing policy by directly estimating the "
    "counterfactual utility gain of additional reasoning relative to immediate answering, "
    "and only allocates extra budget when a lower confidence bound on gain is positive."
)

FIXED_PROXY_CLAIM = (
    "CGGR is evaluated as a fixed proxy-gated executable instantiation of a "
    "counterfactual-gain selective-deliberation formulation. This shard does not "
    "establish that a learned gain estimator or learned routing policy has been trained."
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _as_name_rows(names: list[str]) -> list[dict[str, str]]:
    return [{"name": str(name)} for name in names if str(name).strip()]


def _without_duplicates(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _deployable_only(values: list[str]) -> list[str]:
    return [
        value
        for value in _without_duplicates(values)
        if value not in NON_DEPLOYABLE_DIAGNOSTICS
    ]


def _replace_stale_claim_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_stale_claim_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_stale_claim_text(item) for item in value]
    if isinstance(value, str):
        return value.replace(OLD_LEARNED_ROUTER_CLAIM, FIXED_PROXY_CLAIM)
    return value


def _append_methods(target: Any, field: str) -> None:
    if not isinstance(target, dict):
        return
    values = target.get(field)
    if isinstance(values, list):
        target[field] = _deployable_only(
            [
                str(row.get("name") or row.get("method") or row)
                if isinstance(row, dict)
                else str(row)
                for row in values
            ]
            + TOP_VENUE_METHODS
        )


def _align_claim_scope(plan: dict[str, Any]) -> dict[str, Any]:
    plan = _replace_stale_claim_text(plan)
    publication_contract = plan.setdefault("publication_evidence_contract", {})
    if not isinstance(publication_contract, dict):
        publication_contract = {}
        plan["publication_evidence_contract"] = publication_contract

    publication_contract.update(
        {
            "claim_to_validate": FIXED_PROXY_CLAIM,
            "active_executable_instantiation": "fixed_proxy_gated",
            "trained_estimator_claim_allowed": False,
            "learned_router_claim_allowed": False,
            "oracle_router_in_active_contract": False,
            "non_deployable_diagnostics_blocked": sorted(NON_DEPLOYABLE_DIAGNOSTICS),
            "broad_top_venue_or_sota_superiority_claim_allowed": False,
            "required_top_venue_baselines": TOP_VENUE_METHODS,
            "strict_top_venue_audit_command": "scripts/audit_paper_benchmark_artifacts.py <merged_workdir> --require-full --require-top-venue-baselines",
        }
    )
    _append_methods(publication_contract, "required_baselines")

    paper_intent = plan.setdefault("paper_intent", {})
    if isinstance(paper_intent, dict):
        paper_intent["central_claim"] = FIXED_PROXY_CLAIM
    nested_intent = publication_contract.setdefault("paper_intent", {})
    if isinstance(nested_intent, dict):
        nested_intent["central_claim"] = FIXED_PROXY_CLAIM

    for manifest in (
        plan.get("benchmark_manifest"),
        publication_contract.get("benchmark_manifest"),
    ):
        if not isinstance(manifest, dict):
            continue
        full_stage = manifest.get("full_benchmark_stage")
        if isinstance(full_stage, dict):
            _append_methods(full_stage, "baselines")
            _append_methods(full_stage, "methods")

    plan["active_method_claim"] = FIXED_PROXY_CLAIM
    plan["trained_estimator_claim_allowed"] = False
    plan["learned_router_claim_allowed"] = False
    plan["broad_top_venue_or_sota_superiority_claim_allowed"] = False
    return plan


def _contract_plan(source_run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec_dir = source_run / "spec"
    experiment_spec = _load_json(spec_dir / "experiment_spec.json")
    benchmark_manifest = _load_json(spec_dir / "benchmark_manifest.json")
    source_plan = experiment_spec.get("experimental_plan") if isinstance(experiment_spec.get("experimental_plan"), dict) else {}
    full_stage = benchmark_manifest.get("full_benchmark_stage") if isinstance(benchmark_manifest.get("full_benchmark_stage"), dict) else {}

    dataset_names = [str(name) for name in full_stage.get("datasets") or [] if str(name).strip()]
    if not dataset_names:
        dataset_names = [
            str(row.get("name") or row.get("dataset"))
            for row in source_plan.get("datasets") or []
            if isinstance(row, dict) and str(row.get("name") or row.get("dataset") or "").strip()
        ]
    if not dataset_names:
        raise ValueError("source run does not declare benchmark datasets")

    model_names = [str(name) for name in full_stage.get("models") or [] if str(name).strip()]
    if not model_names:
        model_names = [
            str(row.get("name") or row.get("hf_model") or row.get("model"))
            for row in source_plan.get("model_targets") or []
            if isinstance(row, dict) and str(row.get("name") or row.get("hf_model") or row.get("model") or "").strip()
        ]
    if not model_names:
        model_names = ["Qwen/Qwen2.5-14B-Instruct"]

    source_baselines = [
        str(row.get("name") or row.get("method") or row)
        for row in source_plan.get("baselines") or full_stage.get("baselines") or []
        if (isinstance(row, dict) and str(row.get("name") or row.get("method") or "").strip())
        or (not isinstance(row, dict) and str(row).strip())
    ]
    baselines = _deployable_only(source_baselines + TOP_VENUE_METHODS)

    plan = dict(source_plan)
    plan.update(
        {
            "real_benchmark_required": True,
            "benchmark_targets": [
                experiment_forge._normalize_benchmark_target({"name": name})
                for name in dataset_names
            ],
            "datasets": _as_name_rows(dataset_names),
            "model_targets": [{"name": model_names[0], "hf_model": model_names[0]}],
            "baselines": _as_name_rows(baselines),
            "ablations": _as_name_rows(CANONICAL_ABLATIONS),
            "minimum_seeds": 5,
            "max_eval_examples": 128,
            "metrics": {"primary": "cost_adjusted_accuracy"},
            "top_venue_baseline_shard": True,
        }
    )
    plan = _align_claim_scope(plan)
    return experiment_spec, benchmark_manifest, plan


def prepare(source_run: Path, out_workdir: Path, *, force: bool = False) -> dict[str, Any]:
    source_run = source_run.resolve()
    out_workdir = out_workdir.resolve()
    if out_workdir.exists():
        if not force:
            raise FileExistsError(f"output workdir already exists: {out_workdir}")
        shutil.rmtree(out_workdir)
    (out_workdir / "code").mkdir(parents=True)
    (out_workdir / "spec").mkdir(parents=True)

    experiment_spec, benchmark_manifest, plan = _contract_plan(source_run)
    method = experiment_spec.get("proposed_method") if isinstance(experiment_spec.get("proposed_method"), dict) else {}
    method_name = str(method.get("name") or "Counterfactual Gain Gated Reasoning (CGGR)")
    metric_name = "cost_adjusted_accuracy"

    train_py = experiment_forge._real_llm_benchmark_train_py(
        method_name=method_name,
        metric_name=metric_name,
        plan=plan,
    )
    (out_workdir / "code" / "train.py").write_text(train_py, encoding="utf-8")
    (out_workdir / "code" / "requirements.txt").write_text(
        experiment_forge._real_llm_requirements_txt(),
        encoding="utf-8",
    )

    for name in (
        "experiment_spec.json",
        "benchmark_manifest.json",
        "success_criteria.json",
        "program.md",
        "evaluate.py",
    ):
        _copy_if_exists(source_run / "spec" / name, out_workdir / "spec" / name)

    shard_config = {
        "schema_version": "cggr_top_venue_baseline_shard_v1",
        "source_run": str(source_run),
        "source_benchmark_manifest": str(source_run / "spec" / "benchmark_manifest.json"),
        "method_subset": TOP_VENUE_METHODS,
        "env": {
            "DEEPGRAPH_BENCHMARK_FULL_RUN": "1",
            "DEEPGRAPH_BENCHMARK_INCLUDE_TOP_VENUE_BASELINES": "1",
            "DEEPGRAPH_BENCHMARK_METHODS": ",".join(TOP_VENUE_METHODS),
            "DEEPGRAPH_BENCHMARK_MAX_EXAMPLES": "128",
            "DEEPGRAPH_BENCHMARK_SEEDS": "5",
        },
        "merge_policy": {
            "merge_with_existing_shards": [
                "run_45",
                "run_46",
                "this_top_venue_baseline_shard",
            ],
            "required_audit": "scripts/audit_paper_benchmark_artifacts.py <merged_workdir> --require-full --require-top-venue-baselines",
        },
        "notes": [
            "This shard is not part of the active run45/run46 contract.",
            "Use only after registering a new run id and preserving the raw prediction rows.",
            "Do not use this shard to retrofit claims into run47; merge into a new artifact if SOTA/general-superiority claims are needed.",
        ],
    }
    _write_json(out_workdir / "spec" / "shard_config.json", shard_config)
    _write_json(out_workdir / "spec" / "top_venue_baseline_plan.json", plan)
    _write_json(
        out_workdir / "spec" / "source_benchmark_manifest_snapshot.json",
        benchmark_manifest,
    )

    readme = "\n".join(
        [
            "# CGGR Top-Venue Baseline Shard Template",
            "",
            "This template is prepared from an existing CGGR full-benchmark contract.",
            "It is not a completed experiment artifact.",
            "",
            "Required environment:",
            "",
            "```powershell",
            "$env:DEEPGRAPH_BENCHMARK_FULL_RUN='1'",
            "$env:DEEPGRAPH_BENCHMARK_INCLUDE_TOP_VENUE_BASELINES='1'",
            "$env:DEEPGRAPH_BENCHMARK_METHODS='" + ",".join(TOP_VENUE_METHODS) + "'",
            "$env:DEEPGRAPH_BENCHMARK_MAX_EXAMPLES='128'",
            "$env:DEEPGRAPH_BENCHMARK_SEEDS='5'",
            "python code\\train.py",
            "```",
            "",
            "A merged artifact that uses this shard for broad top-venue claims must pass:",
            "",
            "```powershell",
            "python scripts\\audit_paper_benchmark_artifacts.py <merged_workdir> --require-full --require-top-venue-baselines",
            "```",
            "",
        ]
    )
    (out_workdir / "README.md").write_text(readme, encoding="utf-8")

    return {
        "ok": True,
        "source_run": str(source_run),
        "out_workdir": str(out_workdir),
        "methods": TOP_VENUE_METHODS,
        "written": [
            str(out_workdir / "code" / "train.py"),
            str(out_workdir / "code" / "requirements.txt"),
            str(out_workdir / "spec" / "shard_config.json"),
            str(out_workdir / "spec" / "top_venue_baseline_plan.json"),
            str(out_workdir / "README.md"),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Existing CGGR run workdir whose benchmark contract should be reused.",
    )
    parser.add_argument("--out-workdir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare(args.source_run, args.out_workdir, force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
