import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import watch_and_merge_cggr_shards


class WatchAndMergeCggrShardsTests(unittest.TestCase):
    def test_compile_latex_skips_when_no_tex_path(self):
        result = watch_and_merge_cggr_shards._compile_latex(None)
        self.assertTrue(result["ok"])
        self.assertTrue(result["skipped"])

    def test_compile_latex_blocks_missing_tex_file(self):
        result = watch_and_merge_cggr_shards._compile_latex(Path("missing-main.tex"))
        self.assertFalse(result["ok"])
        self.assertIn("does not exist", result["error"])

    def test_compile_latex_runs_latexmk_in_tex_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tex = root / "main.tex"
            tex.write_text(r"\documentclass{article}\begin{document}ok\end{document}", encoding="utf-8")
            completed = subprocess.CompletedProcess(
                args=["latexmk"],
                returncode=0,
                stdout="compiled",
                stderr="",
            )
            with mock.patch.object(watch_and_merge_cggr_shards.subprocess, "run", return_value=completed) as run:
                result = watch_and_merge_cggr_shards._compile_latex(tex)

            self.assertTrue(result["ok"], result)
            run.assert_called_once()
            _args, kwargs = run.call_args
            self.assertEqual(kwargs["cwd"], str(root))
            self.assertIn("latexmk", _args[0][0])
            self.assertIn("main.tex", _args[0])

    def test_claim_values_summary_reports_claim_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "cggr_utility": 0.42,
                "claim_support_decision": "downgraded",
                "top_venue_general_superiority_decision": "blocked_missing_strict_top_venue_baseline_audit",
                "paired_permutation_p": 0.125,
            }
            (root / "claim_values.json").write_text(json.dumps(payload), encoding="utf-8")

            result = watch_and_merge_cggr_shards._claim_values_summary(root)

        self.assertEqual(result["cggr_utility"], 0.42)
        self.assertEqual(result["claim_support_decision"], "downgraded")
        self.assertEqual(
            result["top_venue_general_superiority_decision"],
            "blocked_missing_strict_top_venue_baseline_audit",
        )

    def test_watch_passes_strict_top_venue_flag_to_audit_and_materializer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("shard1", "shard2", "merged", "materialized"):
                (root / name).mkdir()

            def fake_run(run_id):
                if run_id == 1:
                    return {"id": 1, "status": "completed", "workdir": str(root / "shard1")}
                if run_id == 2:
                    return {"id": 2, "status": "completed", "workdir": str(root / "shard2")}
                if run_id == 3:
                    return {"id": 3, "status": "pending", "workdir": str(root / "merged")}
                return {}

            with (
                mock.patch.object(watch_and_merge_cggr_shards, "_run", side_effect=fake_run),
                mock.patch.object(
                    watch_and_merge_cggr_shards,
                    "_jobs_for",
                    return_value=[
                        {"experiment_run_id": 1, "status": "completed"},
                        {"experiment_run_id": 2, "status": "completed"},
                    ],
                ),
                mock.patch.object(watch_and_merge_cggr_shards, "merge", return_value={"ok": True}) as merge,
                mock.patch.object(watch_and_merge_cggr_shards, "audit", return_value={"ok": True}) as audit,
                mock.patch.object(
                    watch_and_merge_cggr_shards,
                    "materialize",
                    return_value={"ok": True, "written": []},
                ) as materialize,
                mock.patch.object(watch_and_merge_cggr_shards, "_compile_latex", return_value={"ok": True}),
                mock.patch.object(watch_and_merge_cggr_shards, "_mark_merged_completed"),
            ):
                result = watch_and_merge_cggr_shards.watch(
                    shard_run_ids=[1, 2],
                    merged_run_id=3,
                    materialize_out_dir=root / "materialized",
                    poll_seconds=30,
                    require_top_venue_baselines=True,
                )

            self.assertTrue(result["ok"], result)
            merge.assert_called_once()
            audit.assert_called_once_with(
                root / "merged",
                require_full=True,
                require_top_venue_baselines=True,
            )
            materialize.assert_called_once_with(
                root / "merged",
                root / "materialized",
                require_top_venue_baselines=True,
            )


if __name__ == "__main__":
    unittest.main()
