#!/usr/bin/env python3
"""Watch idea8 long baseline shards, launch the remaining queue, then merge."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


REPO = Path("/root/hk/Deepgraph")
SHARDS = Path("/root/deepgraph_ideas/idea_8/experiments/main/shards")
OUT_DIR = Path("/root/deepgraph_ideas/idea_8/experiments/main/merged/idea8_qwen_v7_full")
MODEL = "/root/hf_models/Qwen/Qwen2-7B-Instruct"
METHODS = (
    "Always-Reason Chain-of-Thought,"
    "Self-Consistency Reasoning,"
    "Least-to-Most Prompting,"
    "Disagreement Routing,"
    "Random Budget-Matched Routing,"
    "CAR-Style Certainty Adaptive Routing,"
    "Self-Route-Style Mode Routing,"
    "Rational-Metareasoning VOC Routing"
)
TIMEOUT = "57600"
MAX_EXAMPLES = "1000"


def _job(dataset: str, seed: int, gpu: int) -> dict:
    slug = dataset.lower()
    return {
        "dataset": dataset,
        "seed": seed,
        "gpu": str(gpu),
        "name": f"qwen_{slug}_seed{seed}_missing_baselines_v2_gpu{gpu}",
        "screen": f"idea8_{slug}_s{seed}_base_v2_g{gpu}",
    }


def _status_path(job: dict) -> Path:
    return SHARDS / job["name"] / "shard_status.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_status(payload: dict) -> None:
    path = OUT_DIR / "idea8_completion_supervisor_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _screen_names() -> set[str]:
    proc = subprocess.run(["screen", "-ls"], cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    names = set()
    for line in proc.stdout.splitlines():
        item = line.strip().split()[0] if line.strip().split() else ""
        if "." not in item:
            continue
        names.add(item.split(".", 1)[1])
    return names


def _launch(job: dict) -> None:
    cmd = [
        "python3",
        "-u",
        "plugins/examples/cggr/scripts/launch_idea8_shard_screen.py",
        "--model",
        MODEL,
        "--dataset",
        job["dataset"],
        "--seed",
        str(job["seed"]),
        "--methods",
        METHODS,
        "--gpu",
        job["gpu"],
        "--timeout",
        TIMEOUT,
        "--max-examples",
        MAX_EXAMPLES,
        "--name",
        job["name"],
        "--screen-name",
        job["screen"],
    ]
    subprocess.run(cmd, cwd=str(REPO), check=True)


def _merge() -> None:
    subprocess.run(
        [
            "python3",
            "plugins/examples/cggr/scripts/merge_idea8_shards.py",
            "--out-dir",
            str(OUT_DIR),
            "--require-full-matrix",
        ],
        cwd=str(REPO),
        check=False,
    )


def main() -> int:
    running = {gpu: _job("QASC", gpu, gpu) for gpu in range(4)}
    pending = [_job("QASC", 4, 0)] + [_job("OpenBookQA", seed, 0) for seed in range(5)]
    completed: list[dict] = []
    failed: list[dict] = []
    started_at = time.time()

    while running or pending:
        screens = _screen_names()
        for gpu, job in list(running.items()):
            status = _read_json(_status_path(job))
            done = status.get("status") in {"ok", "failed", "timeout"}
            screen_gone = job["screen"] not in screens
            if not done and not screen_gone:
                continue
            if status.get("status") == "ok":
                completed.append(job)
            else:
                failed.append({**job, "status": status.get("status") or "screen_gone_without_status"})
            del running[gpu]

            if pending:
                next_job = pending.pop(0)
                next_job["gpu"] = str(gpu)
                next_job["name"] = f"qwen_{next_job['dataset'].lower()}_seed{next_job['seed']}_missing_baselines_v2_gpu{gpu}"
                next_job["screen"] = f"idea8_{next_job['dataset'].lower()}_s{next_job['seed']}_base_v2_g{gpu}"
                _launch(next_job)
                running[gpu] = next_job

        _write_status(
            {
                "status": "running" if running or pending else "complete",
                "running": list(running.values()),
                "pending": pending,
                "completed_count": len(completed),
                "failed": failed,
                "elapsed_seconds": time.time() - started_at,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        time.sleep(120)

    _merge()
    _write_status(
        {
            "status": "complete_with_failures" if failed else "complete",
            "completed_count": len(completed),
            "failed": failed,
            "elapsed_seconds": time.time() - started_at,
            "merged_out_dir": str(OUT_DIR),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
