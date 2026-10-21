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

    def test_the_judging_roles_have_more_than_one_route(self):
        """A verdict must not be blocked by one vendor being unreachable.

        The proposer is deliberately excluded, see the test below.
        """
        for role in ("evaluator", "reviewer"):
            with self.subTest(role=role):
                self.assertGreaterEqual(
                    len(self.routes.get(role, [])), 2,
                    f"{role} has a single route and therefore a single network path")

    def test_the_proposer_is_deliberately_single_route(self):
        """Redundancy here bought availability and paid for it in silence.

        The proposer's second route was added on 2026-08-25 so a transport blip
        could not stall corpus ingestion, and it did that job. It also did one
        nobody asked for: whenever the primary gateway rate-limited, ingestion
        fell through to the other vendor and carried on at full speed. A week of
        that put 1,300 calls and 8.2M output tokens on the fallback vendor, and
        no human ever decided to spend it.

        The roles differ in what a stall costs. A blocked verdict is a blocked
        scientific result and stays redundant. Corpus ingestion is a background
        job with no deadline, so the right answer to a rate limit is to wait --
        which is what the provider rpm cap and the widened transient retry
        window do instead.
        """
        self.assertEqual(
            len(self.routes.get("proposer", [])), 1,
            "a second proposer route turns a rate limit into a bill at another vendor")
        primary = self.routes["proposer"][0]
        self.assertTrue(
            str(primary.get("provider_ref", "")).startswith("env:"),
            "the single proposer route must stay switchable without a code change")

    def test_the_primary_gateway_is_rate_capped(self):
        """With no fallback, the throttle has to be ours rather than theirs."""
        capped = [p for p in self.config["llm"]["providers"]
                  if p.get("name") == "sora2_gemini"]
        self.assertTrue(capped, "the primary proposer gateway is not declared")
        self.assertGreater(
            int(capped[0].get("rpm") or 0), 0,
            "an uncapped gateway with no fallback fails closed instead of waiting")
        self.assertGreaterEqual(
            int(self.config["llm"].get("transient_retries") or 0), 4,
            "the retry window must outlast a rate limit now that nothing catches it")

    def test_an_independent_evaluator_exists_for_every_proposer_family(self):
        proposer_families = {"gemini-flash"}  # env-bound primary, per .env
        for entry in self.routes["proposer"]:
            family = str(entry.get("model_family_ref") or "")
            if not family.startswith("env:"):
                proposer_families.add(family)
        evaluator_families = set()
        for entry in self.routes["evaluator"]:
            family = str(entry.get("model_family_ref") or "")
            # The env-bound evaluator is claude since 2026-08-30; deepseek is
            # no longer any role's primary.
            evaluator_families.add("claude" if family.startswith("env:") else family)
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
