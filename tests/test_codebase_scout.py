import unittest
from unittest import mock

from agents import codebase_scout


class CodebaseScoutTests(unittest.TestCase):
    def test_dedupe_candidates_prefers_paper_linked_repos(self):
        rows = codebase_scout._dedupe_candidates(
            [
                {"full_name": "org/popular", "url": "https://github.com/org/popular", "stars": 9999, "source_kind": "repository_search"},
                {"full_name": "org/paper", "url": "https://github.com/org/paper", "stars": 1, "source_kind": "baseline_paper"},
            ]
        )
        self.assertEqual(rows[0]["full_name"], "org/paper")

    def test_collect_paper_linked_repositories_from_supporting_paper_text(self):
        parsed = {
            "supporting_papers": ["2606.05158"],
            "source_paper_ids": [],
        }
        plan = {"baselines": [{"name": "Multi-Agent Debate (MAD)"}]}
        paper = {
            "id": "2606.05158",
            "title": "Streaming Communication in Multi-Agent Reasoning",
            "abstract": "Code at https://github.com/example/stream-comm",
            "full_text": "",
            "appendix_text": "",
            "pdf_url": "",
        }

        with (
            mock.patch.object(codebase_scout.db, "fetchone", return_value=paper),
            mock.patch.object(codebase_scout.db, "fetchall", return_value=[]),
        ):
            rows = codebase_scout.collect_paper_linked_repositories(parsed, plan)

        self.assertEqual(rows[0]["url"], "https://github.com/example/stream-comm")
        self.assertEqual(rows[0]["source_kind"], "supporting_paper")

    def test_match_baseline_method_papers_links_named_baselines(self):
        with mock.patch.object(
            codebase_scout.db,
            "fetchall",
            return_value=[
                {"name": "Multi-Agent Debate", "first_paper_id": "2305.19118"},
            ],
        ):
            matches = codebase_scout._match_baseline_method_papers(["Multi-Agent Debate (MAD) Under Matched Budget"])

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["paper_id"], "2305.19118")

    def test_search_github_repositories_parses_api_payload(self):
        payload = {
            "items": [
                {
                    "full_name": "EleutherAI/lm-evaluation-harness",
                    "html_url": "https://github.com/EleutherAI/lm-evaluation-harness",
                    "description": "LLM eval harness",
                    "stargazers_count": 5000,
                    "updated_at": "2026-01-01T00:00:00Z",
                    "topics": ["llm"],
                }
            ]
        }

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return payload

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, headers=None, params=None):
                return FakeResponse()

        with mock.patch.object(codebase_scout.httpx, "Client", FakeClient):
            rows = codebase_scout.search_github_repositories("lm evaluation harness")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["full_name"], "EleutherAI/lm-evaluation-harness")
        self.assertEqual(rows[0]["stars"], 5000)

    def test_fetch_arxiv_metadata_extracts_github_from_summary(self):
        atom = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Encouraging Divergent Thinking</title>
    <summary>Code: https://github.com/Skytliang/Multi-Agents-Debate</summary>
    <author><name>Shun Zhang</name></author>
  </entry>
</feed>"""

        class FakeResponse:
            text = atom

            def raise_for_status(self):
                return None

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, params=None):
                return FakeResponse()

        with mock.patch.object(codebase_scout.httpx, "Client", FakeClient):
            meta = codebase_scout._fetch_arxiv_metadata("2305.19118")

        self.assertIsNotNone(meta)
        self.assertIn("github.com/Skytliang/Multi-Agents-Debate", meta["full_text"])

    def test_search_github_for_paper_when_text_has_no_url(self):
        paper = {
            "id": "2305.19118",
            "title": "Encouraging Divergent Thinking in LLMs",
            "abstract": "We study multi-agent debate.",
            "authors": ["Shun Zhang"],
        }
        search_hit = {
            "full_name": "Skytliang/Multi-Agents-Debate",
            "url": "https://github.com/Skytliang/Multi-Agents-Debate",
            "description": "official code",
            "stars": 120,
            "updated_at": "",
            "topics": [],
            "source_query": "2305.19118",
            "source_kind": "repository_search",
        }

        with (
            mock.patch.object(codebase_scout, "_fetch_arxiv_metadata", return_value=None),
            mock.patch.object(codebase_scout, "search_github_repositories", return_value=[search_hit]),
        ):
            rows = codebase_scout._search_github_for_paper(
                paper,
                baseline_name="Multi-Agent Debate",
                method_name="Multi-Agent Debate",
                source_kind="baseline_paper",
            )

        self.assertEqual(rows[0]["url"], "https://github.com/Skytliang/Multi-Agents-Debate")
        self.assertEqual(rows[0]["source_kind"], "paper_github_search")

    def test_scout_auto_picks_verified_paper_repo(self):
        insight = {
            "id": 100,
            "tier": 2,
            "title": "MAD experiment",
            "resource_class": "cpu",
            "supporting_papers": ["2305.19118"],
            "proposed_method": {"name": "Router", "type": "algorithm", "one_line": "x"},
            "experimental_plan": {
                "baselines": [{"name": "Multi-Agent Debate"}],
                "datasets": [{"name": "GSM8K"}],
            },
            "source_node_ids": [],
        }
        paper_repo = {
            "full_name": "Skytliang/Multi-Agents-Debate",
            "url": "https://github.com/Skytliang/Multi-Agents-Debate",
            "stars": 120,
            "source_kind": "baseline_paper",
        }

        with (
            mock.patch.object(codebase_scout, "call_llm_json") as llm_mock,
            mock.patch.object(codebase_scout, "search_paper_repositories_complete", return_value=[paper_repo]),
            mock.patch.object(
                codebase_scout,
                "_verify_codebase_download",
                return_value={
                    "ok": True,
                    "codebase": {
                        "url": paper_repo["url"],
                        "name": "Multi-Agents-Debate",
                        "main_train_file": "train.py",
                        "main_eval_command": "python train.py",
                    },
                },
            ),
            mock.patch("agents.experiment_forge._parse_insight_fields", side_effect=lambda insight: dict(insight)),
            mock.patch("agents.experiment_forge._ensure_real_benchmark_plan", side_effect=lambda parsed, method, plan, resource_class: dict(plan)),
        ):
            picked = codebase_scout.scout_codebase_agentic(insight)

        self.assertEqual(picked["url"], paper_repo["url"])
        llm_mock.assert_not_called()

    def test_scout_codebase_agentic_verifies_llm_pick(self):
        insight = {
            "id": 99,
            "tier": 2,
            "title": "Routing experiment",
            "resource_class": "cpu",
            "proposed_method": {
                "name": "Router",
                "type": "algorithm",
                "one_line": "Route between agents",
                "definition": "f(x)",
            },
            "experimental_plan": {
                "baselines": [{"name": "Direct"}, {"name": "Always-CoT"}],
                "datasets": [{"name": "GSM8K"}],
                "metrics": {"primary": "accuracy"},
            },
            "source_node_ids": [],
        }

        def fake_llm(system_prompt, user_prompt, temperature=0.0):
            if "search_queries" in system_prompt or "GitHub search queries" in system_prompt:
                return (
                    {
                        "search_queries": ["multi-agent llm evaluation"],
                        "code_search_queries": ["gsm8k evaluate.py"],
                    },
                    10,
                )
            return (
                {
                    "action": "pick",
                    "codebase": {
                        "url": "https://github.com/example/harness",
                        "name": "harness",
                        "main_train_file": "train.py",
                        "main_eval_command": "python train.py",
                        "reason": "matches baselines",
                    },
                },
                20,
            )

        with (
            mock.patch.object(codebase_scout, "call_llm_json", side_effect=fake_llm),
            mock.patch.object(codebase_scout, "search_paper_repositories_complete", return_value=[]),
            mock.patch.object(codebase_scout, "_try_auto_pick_verified_paper_repo", return_value=None),
            mock.patch.object(codebase_scout, "_execute_search_plan", return_value=[{"full_name": "example/harness", "url": "https://github.com/example/harness", "stars": 1, "source_kind": "repository_search"}]),
            mock.patch.object(codebase_scout, "enrich_repository", side_effect=lambda row: {**row, "entrypoint_hints": ["train.py"]}),
            mock.patch.object(
                codebase_scout,
                "_verify_codebase_download",
                return_value={
                    "ok": True,
                    "codebase": {
                        "url": "https://github.com/example/harness",
                        "name": "harness",
                        "main_train_file": "train.py",
                        "main_eval_command": "python train.py",
                    },
                },
            ),
            mock.patch("agents.experiment_forge._parse_insight_fields", side_effect=lambda insight: dict(insight)),
            mock.patch("agents.experiment_forge._ensure_real_benchmark_plan", side_effect=lambda parsed, method, plan, resource_class: dict(plan)),
            mock.patch("agents.experiment_forge._normalize_codebase_metadata", side_effect=lambda codebase: dict(codebase)),
        ):
            picked = codebase_scout.scout_codebase_agentic(insight)

        self.assertEqual(picked["url"], "https://github.com/example/harness")
        self.assertEqual(picked["main_train_file"], "train.py")

    def test_experiment_forge_uses_agentic_scout_first(self):
        from agents import experiment_forge

        with (
            mock.patch("agents.codebase_scout.scout_codebase_agentic", return_value={"url": "https://github.com/example/repo", "name": "repo", "main_train_file": "train.py"}),
            mock.patch.object(experiment_forge, "_scout_codebase_single_shot") as fallback,
        ):
            result = experiment_forge.scout_codebase({"id": 1})

        self.assertEqual(result["url"], "https://github.com/example/repo")
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
