"""Tests for deep benchmark artifact materialization."""

from __future__ import annotations

import json
from pathlib import Path

from agents.benchmark_artifacts import materialize_deep_benchmark_artifacts


def test_materialize_run2_artifacts(tmp_path: Path) -> None:
    src = Path("/root/deepgraph_ideas/idea_2/experiments/main/runs/run_2/results/raw_predictions.jsonl")
    if not src.is_file():
        return
    (tmp_path / "raw_predictions.jsonl").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    report = materialize_deep_benchmark_artifacts(
        tmp_path,
        publication_contract={"required_ablations": ["disable_routing", "no_verifier"]},
        metric_name="primary_score",
        min_lines=50,
    )
    assert report.get("ok")
    assert (tmp_path / "ablation_table.json").is_file()
    assert (tmp_path / "per_dataset_results.json").is_file()
    assert (tmp_path / "seed_variance_table.json").is_file()
    ablation = json.loads((tmp_path / "ablation_table.json").read_text(encoding="utf-8"))
    assert isinstance(ablation, dict)
    assert any(row.get("executed") is False for row in ablation.values() if isinstance(row, dict))
