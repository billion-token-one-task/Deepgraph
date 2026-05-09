import subprocess
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
