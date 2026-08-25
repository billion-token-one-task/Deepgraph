"""No role may depend on a single network path, and independence must survive
whichever route a role actually used.

On 2026-08-25 the proposer had exactly one route. Seven calls succeeded, then
three failed inside two seconds with in=0 out=0 -- the request never left the
host -- and `fail_closed_or_manual` declared all_explicit_routes_failed. Two
paper ingestion jobs died, corpus backfill made no further progress and tripped
its three-hour breaker, and the same route completed sixteen calls twelve
minutes later. A second route at the same vendor would not have helped: a
transport blip follows the network path, not the model.

The evaluator then needs a second route for a reason that only exists because
of the first fix. Independence is judged against the route the proposer
actually used, so a proposer that falls back to deepseek leaves a
deepseek-only evaluator with nothing independent to offer, and the ladder
stalls one rung above the blip.
"""

import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - interpreter without tomllib
    tomllib = None

CONFIG = Path(__file__).resolve().parent.parent / "deepgraph.toml"


def _load():
    if tomllib is None:
        raise unittest.SkipTest("tomllib is unavailable on this interpreter")
    with CONFIG.open("rb") as handle:
        return tomllib.load(handle)


class RoleRouteRedundancyTest(unittest.TestCase):
    def setUp(self):
        self.config = _load()
        self.routes = self.config["llm_routes"]
        self.providers = {p["name"]: p for p in self.config["llm"]["providers"]}

    def _families(self, role):
        out = []
        for entry in self.routes.get(role, []):
            family = str(entry.get("model_family_ref") or "")
            if family.startswith("env:"):
                # Env-bound routes are resolved at runtime; the literal ones are
                # what this test can assert about statically.
                out.append(("env", family))
            else:
                out.append(("literal", family))
        return out

    def test_every_role_has_more_than_one_route(self):
        for role in ("proposer", "evaluator", "reviewer"):
            with self.subTest(role=role):
                self.assertGreaterEqual(
                    len(self.routes.get(role, [])), 2,
                    f"{role} has a single route and therefore a single network path")

    def test_the_proposer_fallback_is_a_different_vendor(self):
        literals = [e for e in self.routes["proposer"]
                    if not str(e.get("provider_ref", "")).startswith("env:")]
        self.assertTrue(literals, "the proposer fallback must name a provider directly")
        for entry in literals:
            provider = self.providers.get(entry["provider_ref"])
            self.assertIsNotNone(provider, f"unknown provider {entry['provider_ref']}")
            self.assertTrue(provider.get("enabled"), "a disabled provider is not a fallback")

    def test_an_independent_evaluator_exists_for_every_proposer_family(self):
        proposer_families = {"gemini-flash"}  # env-bound primary, per .env
        for entry in self.routes["proposer"]:
            family = str(entry.get("model_family_ref") or "")
            if not family.startswith("env:"):
                proposer_families.add(family)
        evaluator_families = set()
        for entry in self.routes["evaluator"]:
            family = str(entry.get("model_family_ref") or "")
            evaluator_families.add("deepseek" if family.startswith("env:") else family)
        for proposer_family in proposer_families:
            with self.subTest(proposer=proposer_family):
                self.assertTrue(
                    evaluator_families - {proposer_family},
                    f"no evaluator family is independent of a {proposer_family} proposer")

    def test_independence_is_still_required(self):
        self.assertTrue(self.routes.get("require_independent_evaluator"))

    def test_failure_policy_is_still_fail_closed(self):
        self.assertEqual(self.routes.get("failure_policy"), "fail_closed_or_manual")

    def test_every_referenced_provider_is_declared_and_enabled(self):
        for role in ("proposer", "evaluator", "reviewer"):
            for entry in self.routes.get(role, []):
                ref = str(entry.get("provider_ref") or "")
                if ref.startswith("env:"):
                    continue
                with self.subTest(role=role, provider=ref):
                    self.assertIn(ref, self.providers)
                    self.assertTrue(self.providers[ref].get("enabled"))


if __name__ == "__main__":
    unittest.main()
