import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_cggr_shard_contract import audit


class CggrShardContractAuditTests(unittest.TestCase):
    def _write_run(self, root: Path, *, shard: bool, methods: list[str], model_id: str = "Qwen/Qwen2.5-14B-Instruct") -> None:
        (root / "results").mkdir(parents=True)
        (root / "spec").mkdir(parents=True)
        config = {
            "model_id": model_id,
            "targets": [
                {"name": "MuSiQue-Ans"},
                {"name": "StrategyQA"},
                {"name": "2WikiMultihopQA"},
                {"name": "Stress Test Split: Simple-vs-Hard Counterfactual Partition"},
            ],
            "seeds": 5,
            "seed_values": [0, 1, 2, 3, 4],
            "max_examples_per_dataset_seed": 128,
            "cost_lambda": 0.03,
            "decoding": {"default": "greedy"},
            "reasoning_budget": {"direct": 48, "cggr": "64-224 max_new_tokens by difficulty"},
            "methods": methods,
            "sharded_run": shard,
            "shard_axes": {"method": shard, "target": False, "seed": False},
        }
        (root / "results" / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    def test_matching_method_shard_passes_contract_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = root / "canonical"
            shard = root / "shard"
            shard_methods = ["CGGR", "CGGR/no_counterfactual_delta"]
            self._write_run(canonical, shard=False, methods=["Vanilla Direct Answering", *shard_methods])
            self._write_run(shard, shard=True, methods=shard_methods)
            (shard / "spec" / "shard_config.json").write_text(
                json.dumps({"method_subset": shard_methods}),
                encoding="utf-8",
            )

            result = audit(canonical, shard)

        self.assertTrue(result["ok"], result)

    def test_model_mismatch_blocks_contract_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = root / "canonical"
            shard = root / "shard"
            shard_methods = ["CGGR", "CGGR/no_counterfactual_delta"]
            self._write_run(canonical, shard=False, methods=["Vanilla Direct Answering", *shard_methods])
            self._write_run(shard, shard=True, methods=shard_methods, model_id="other-model")
            (shard / "spec" / "shard_config.json").write_text(
                json.dumps({"method_subset": shard_methods}),
                encoding="utf-8",
            )

            result = audit(canonical, shard)

        self.assertFalse(result["ok"])
        self.assertIn("contract field mismatch: model_id", result["blockers"])


if __name__ == "__main__":
    unittest.main()
