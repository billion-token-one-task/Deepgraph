"""Mocked bubblewrap invocation; no candidate command or suite is executed."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from meta_harness.evaluator_runner import (
    EvaluatorSuiteSpec,
    IsolatedEvaluatorRunner,
)
from meta_harness.harness_evolution import HarnessCandidate, HarnessPolicy


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
            digest.update(b"\n")
    return digest.hexdigest()


class EvaluatorIsolationTests(unittest.TestCase):
    def test_runner_mounts_candidate_read_only_and_clears_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidates = root / "candidates"
            candidate_dir = candidates / "candidate-1"
            evaluators = root / "evaluators"
            evaluator_dir = evaluators / "v1"
            holdouts = root / "holdouts"
            suite_dir = holdouts / "held-out-v1"
            artifacts = root / "artifacts"
            output = artifacts / "candidate-1-held-out"
            production = root / "production"
            for path in (
                candidate_dir,
                evaluator_dir,
                suite_dir,
                artifacts,
                production,
            ):
                path.mkdir(parents=True)
            (candidate_dir / "candidate.py").write_text("VALUE = 1\n")
            entrypoint = evaluator_dir / "run-evaluator"
            entrypoint.write_text("#!/bin/sh\nexit 0\n")
            entrypoint.chmod(0o755)
            (suite_dir / "cases.json").write_text('{"cases":[]}\n')
            candidate = HarnessCandidate(
                agenda_id=1,
                candidate_ref="meta-harness/candidate-1",
                base_commit="a" * 40,
                worktree_path=str(candidate_dir),
                database_namespace="meta_harness_candidate_candidate_1",
                artifact_namespace="meta_harness_candidate_candidate_1/artifacts",
            )
            policy = HarnessPolicy(candidate_root=str(candidates))
            seen: dict = {}

            def fake_run(command, **kwargs):
                seen["command"] = list(command)
                seen["kwargs"] = dict(kwargs)
                bind_positions = [
                    index
                    for index, value in enumerate(command)
                    if value == "--bind"
                ]
                host_output = Path(command[bind_positions[-1] + 1])
                (host_output / "result.json").write_text(
                    json.dumps({"status": "passed"})
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            runner = IsolatedEvaluatorRunner(
                policy=policy,
                production_path=str(production),
                production_database_namespace="production",
                evaluator_root=str(evaluators),
                holdout_root=str(holdouts),
                artifact_root=str(artifacts),
                runner=fake_run,
            )
            spec = EvaluatorSuiteSpec(
                suite="held_out",
                evaluator_root=str(evaluator_dir),
                evaluator_entrypoint="run-evaluator",
                evaluator_hash=_tree_hash(evaluator_dir),
                suite_root=str(suite_dir),
                suite_hash=_tree_hash(suite_dir),
                output_dir=str(output),
                timeout_seconds=30,
            )
            with mock.patch(
                "meta_harness.evaluator_runner.shutil.which",
                # The subprocess is mocked, but the production guard still
                # requires the resolved isolation path to be a real file.
                return_value=str(entrypoint),
            ):
                result = runner.run(candidate=candidate, spec=spec)

            self.assertEqual(result.status, "passed")
            command = seen["command"]
            self.assertIn("--unshare-all", command)
            self.assertIn("--clearenv", command)
            candidate_index = command.index(str(candidate_dir))
            self.assertEqual(command[candidate_index - 1], "--ro-bind")
            self.assertNotIn("DEEPGRAPH_DATABASE_URL", seen["kwargs"]["env"])


if __name__ == "__main__":
    unittest.main()
