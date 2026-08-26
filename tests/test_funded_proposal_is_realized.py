"""Authority that has been signed must be spent, not reissued.

A funded proposal parks at (deferred, proposal_generation_granted) and waits for
`execute_bounded_proposal`. That consumer is complete, but its only caller was
scripts/run_bounded_proposal.py -- a manual tool -- so an ordinary advance pass
signed the authority and then walked past it to generate more candidates. Four
hours later the grant expired, the concurrency slot reopened, and the next pass
signed another. Agendas 16, 17 and 18 entered that loop the moment they were
first funded on 2026-08-26 and would never have left it.
"""

import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "scripts" / "auto_advance.py"


class FundedProposalWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.text)

    def _function(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def test_the_ordinary_pass_has_a_realizer(self):
        self.assertIsNotNone(
            self._function("realize_funded_proposals"),
            "no ordinary-pass consumer for (deferred, proposal_generation_granted)")

    def test_it_calls_the_real_consumer_not_a_reimplementation(self):
        node = self._function("realize_funded_proposals")
        called = {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        self.assertIn("execute_bounded_proposal", called)
        self.assertIn("authorize_bounded_proposal", called)

    def test_it_selects_only_live_proposal_authority(self):
        node = self._function("realize_funded_proposals")
        sql = " ".join(
            child.value for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str))
        for clause in ("status = 'deferred'", "stage = 'proposal_generation_granted'",
                       "rg.stage = 'proposal'", "rg.status = 'active'"):
            self.assertIn(clause, sql, f"selector is missing {clause!r}")

    def test_the_soonest_expiry_is_realized_first(self):
        node = self._function("realize_funded_proposals")
        sql = " ".join(
            child.value for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str))
        self.assertIn("ORDER BY rg.expires_at ASC", sql,
                      "authority closest to expiry must be spent first")

    def test_one_bad_proposal_does_not_end_the_pass(self):
        node = self._function("realize_funded_proposals")
        handlers = [h for h in ast.walk(node) if isinstance(h, ast.ExceptHandler)]
        self.assertGreaterEqual(len(handlers), 2,
                                "a refusal and an unexpected error must both be survivable")
        self.assertTrue(
            any(isinstance(stmt, ast.Continue) for h in handlers for stmt in ast.walk(h)),
            "a failed proposal must continue to the next, not abort the sweep")

    def test_realization_runs_before_discovery_in_the_pass(self):
        realize = self.text.index("realize_funded_proposals(discovery_agenda_id")
        discover = self.text.index("from orchestrator.discovery_scheduler import run_tier2_discovery",
                                   realize)
        self.assertLess(realize, discover,
                        "committed money must be spent before more candidates are bought")

    def test_every_branch_is_journalled(self):
        node = self._function("realize_funded_proposals")
        logged = {
            child.args[0].value
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute) and child.func.attr == "log"
            and child.args and isinstance(child.args[0], ast.Constant)
        }
        self.assertEqual(
            logged,
            {"bounded_proposal_already_realized", "bounded_proposal_refused",
             "bounded_proposal_failed", "bounded_proposal_realized"},
            "a silent branch here is how the original gap stayed invisible")


if __name__ == "__main__":
    unittest.main()
