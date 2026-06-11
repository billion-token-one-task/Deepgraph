#!/usr/bin/env python3
"""Run one idea8 benchmark shard in an isolated workdir.

This avoids the old one-shot full-matrix timeout. Each shard is scoped to one
model, one dataset, one seed, and a small method set, then writes a local
benchmark_summary.json plus raw_predictions.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:120] or "shard"


def _copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _patch_compact_crpp_direct(train_path: Path) -> bool:
    """Inject a compact non-routed CRPP prompt into the copied shard code."""
    text = train_path.read_text(encoding="utf-8")
    if "compact_choice_direct" in text:
        return False
    direct_anchor = '    if kind == "direct":\n        return "Answer the question. Give only the final answer.\\nQuestion: " + question + "\\nAnswer:"\n'
    compact_branch = (
        direct_anchor
        + '    if kind == "compact_choice_direct":\n'
        + '        return (\n'
        + '            "Return only the single best option letter, such as A, B, C, D, E, F, G, or H. "\n'
        + '            "Do not include the option text, reasoning, punctuation, or any extra words."\n'
        + '            "\\nQuestion: " + question + "\\nAnswer:"\n'
        + '        )\n'
    )
    if direct_anchor not in text:
        raise RuntimeError("could not locate direct prompt anchor for compact CRPP patch")
    text = text.replace(direct_anchor, compact_branch, 1)
    route_anchor = '        strategy = "voc_metareasoning" if deliberate else "direct"\n'
    route_replacement = (
        '        compact_direct = _env_flag("DEEPGRAPH_CRPP_COMPACT_DIRECT")\n'
        '        strategy = "voc_metareasoning" if deliberate else ("compact_choice_direct" if compact_direct else "direct")\n'
    )
    if route_anchor not in text:
        raise RuntimeError("could not locate VOC routing anchor for compact CRPP patch")
    text = text.replace(route_anchor, route_replacement, 1)
    train_path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=Path("/root/deepgraph_ideas/idea_8/experiments/main/runs/run_13"))
    parser.add_argument("--out-root", type=Path, default=Path("/root/deepgraph_ideas/idea_8/experiments/main/shards"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--methods", required=True, help="Comma-separated exact method names")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods:
        raise SystemExit("--methods must not be empty")
    source_code = args.source_run / "code"
    source_spec = args.source_run / "spec"
    if not source_code.exists():
        raise SystemExit(f"missing source code dir: {source_code}")

    shard_name = args.name or "__".join([
        _safe_slug(args.model),
        _safe_slug(args.dataset),
        f"seed{args.seed}",
        _safe_slug("_".join(methods)),
    ])
    workdir = args.out_root / shard_name
    code_dir = workdir / "code"
    spec_dir = workdir / "spec"
    results_dir = workdir / "results"
    workdir.mkdir(parents=True, exist_ok=True)
    _copytree_clean(source_code, code_dir)
    if source_spec.exists():
        _copytree_clean(source_spec, spec_dir)
    compact_crpp_patch = os.getenv("DEEPGRAPH_CRPP_COMPACT_DIRECT", "").strip().lower() in {"1", "true", "yes", "on"}
    compact_crpp_patch_applied = False
    if compact_crpp_patch:
        compact_crpp_patch_applied = _patch_compact_crpp_direct(code_dir / "train.py")
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    shard_config = {
        "schema_version": "idea8_benchmark_shard_v1",
        "source_run": str(args.source_run),
        "workdir": str(workdir),
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "methods": methods,
        "gpu": args.gpu,
        "timeout": args.timeout,
        "max_examples": args.max_examples,
        "policy_env": {
            key: os.getenv(key)
            for key in (
                "DEEPGRAPH_CRPP_COMPACT_DIRECT",
                "DEEPGRAPH_VOC_THRESHOLD",
                "DEEPGRAPH_VOC_REASONING_COST",
                "DEEPGRAPH_VOC_DELIBERATE_MAX_NEW_TOKENS",
            )
            if os.getenv(key) is not None
        },
        "compact_crpp_patch_applied": compact_crpp_patch_applied,
        "created_at": time.time(),
    }
    (spec_dir / "shard_config.json").parent.mkdir(parents=True, exist_ok=True)
    (spec_dir / "shard_config.json").write_text(json.dumps(shard_config, indent=2, ensure_ascii=False), encoding="utf-8")

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "DEEPGRAPH_BENCHMARK_MODEL": args.model,
        "DEEPGRAPH_BENCHMARK_MAX_EXAMPLES": str(args.max_examples),
        "DEEPGRAPH_BENCHMARK_SEEDS": "5",
        "DEEPGRAPH_BENCHMARK_SEED_OFFSET": str(args.seed),
        "DEEPGRAPH_BENCHMARK_SEED_COUNT": "1",
        "DEEPGRAPH_BENCHMARK_TARGET_NAMES": args.dataset,
        "DEEPGRAPH_BENCHMARK_METHODS": ",".join(methods),
        "DEEPGRAPH_BENCHMARK_FULL_RUN": "1",
        "DEEPGRAPH_BENCHMARK_INCLUDE_TOP_VENUE_BASELINES": "1",
        "HF_HUB_DISABLE_XET": env.get("HF_HUB_DISABLE_XET", "1"),
    })
    cmd = [sys.executable, "train.py"]
    log_path = workdir / "run.log"
    started = time.time()
    status = "ok"
    error = ""
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=str(code_dir), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        try:
            proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            status = "timeout"
            error = "timeout"
    duration = time.time() - started
    if status == "ok" and proc.returncode != 0:
        status = "failed"
        error = f"returncode={proc.returncode}"
    summary = _load_json(results_dir / "benchmark_summary.json", {})
    raw_path = results_dir / "raw_predictions.jsonl"
    raw_lines = 0
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as handle:
            raw_lines = sum(1 for _ in handle)
    done = {
        **shard_config,
        "status": status,
        "error": error,
        "duration_seconds": duration,
        "returncode": proc.returncode,
        "raw_predictions_lines": raw_lines,
        "final_results_present": bool(summary),
        "log_path": str(log_path),
        "results_dir": str(results_dir),
        "completed_at": time.time(),
    }
    (workdir / "shard_status.json").write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(done, indent=2, ensure_ascii=False))
    return 0 if status == "ok" and summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
