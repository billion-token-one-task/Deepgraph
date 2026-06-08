import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from agents import workspace_layout
from db import database
from web import app as web_app


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()

    def test_api_events_returns_short_polling_json_and_serializes_datetimes(self):
        with mock.patch.object(
            web_app,
            "get_events",
            return_value=[{"seq": 6, "created_at": datetime(2026, 4, 21, 12, 0, 0)}],
        ) as get_events:
            response = self.client.get("/api/events?since=5")
            payload = response.get_json()

        get_events.assert_called_once_with(5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        self.assertEqual(payload["events"][0]["created_at"], "2026-04-21 12:00:00")
        self.assertEqual(payload["events"][0]["seq"], 6)
        self.assertEqual(payload["next_seq"], 7)

    def test_read_requests_roll_back_open_transaction_on_teardown(self):
        with mock.patch.object(web_app.db, "rollback") as rollback:
            response = self.client.get("/api/meta")

        self.assertEqual(response.status_code, 200)
        rollback.assert_called()

    def test_api_meta_includes_database_backend_summary(self):
        response = self.client.get("/api/meta")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("database", payload)
        self.assertIn("backend", payload["database"])
        self.assertIn("target", payload["database"])

    def test_manual_post_api_returns_gone_in_fixed_flow_mode(self):
        response = self.client.post("/api/experiments/run_full", json={"insight_id": 1})
        payload = response.get_json()

        self.assertEqual(response.status_code, 410)
        self.assertEqual(payload["mode"], "fixed_flow_read_only")
        self.assertIn("removed", payload["error"].lower())


class DashboardRefreshTests(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()

    def test_dashboard_and_stats_routes_return_200(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_db_path = database.DB_PATH
            old_database_url = database.DATABASE_URL
            for attr in ("pg_conn", "sqlite_conn", "conn"):
                if hasattr(database._local, attr):
                    try:
                        getattr(database._local, attr).close()
                    except Exception:
                        pass
                    setattr(database._local, attr, None)
            try:
                database.DATABASE_URL = ""
                database.DB_PATH = Path(tmpdir) / "stats.db"
                database.init_db()

                self.assertEqual(self.client.get("/").status_code, 200)
                self.assertEqual(self.client.get("/api/stats").status_code, 200)
            finally:
                for attr in ("pg_conn", "sqlite_conn", "conn"):
                    if hasattr(database._local, attr):
                        try:
                            getattr(database._local, attr).close()
                        except Exception:
                            pass
                        setattr(database._local, attr, None)
                database.DATABASE_URL = old_database_url
                database.DB_PATH = old_db_path

    def test_dashboard_main_tabs_are_phase1_six_with_advanced_collapsed(self):
        html = _read("web/templates/index.html")
        main_tabs = re.findall(r'<button class="nav-item(?: active)?" data-tab="([^"]+)"', html)
        self.assertEqual(
            main_tabs,
            ["overview", "explore", "evidence", "generated-papers", "insights", "papers"],
        )
        self.assertRegex(html, r"<details[^>]+class=\"[^\"]*advanced-nav[^\"]*\"[^>]*>")
        self.assertNotRegex(html, r"<details[^>]+class=\"[^\"]*advanced-nav[^\"]*\"[^>]*open")

    def test_dashboard_i18n_keys_match_and_static_labels_are_marked(self):
        i18n = _read("web/static/js/i18n.js")
        en_match = re.search(r"en:\s*\{(?P<body>.*?)\n\s*\},\n\s*zh:", i18n, re.S)
        zh_match = re.search(r"zh:\s*\{(?P<body>.*?)\n\s*\},?\n\s*\};", i18n, re.S)
        self.assertIsNotNone(en_match)
        self.assertIsNotNone(zh_match)
        key_pattern = r'^\s*"([^"]+)":'
        en_keys = set(re.findall(key_pattern, en_match.group("body"), re.M))
        zh_keys = set(re.findall(key_pattern, zh_match.group("body"), re.M))
        self.assertEqual(en_keys, zh_keys)
        self.assertIn("navigator.language", i18n)
        self.assertIn("startsWith('zh')", i18n)
        self.assertIn("localStorage", i18n)

        html = _read("web/templates/index.html")
        self.assertGreaterEqual(html.count("data-i18n="), 40)
        for key in ("nav.overview", "nav.explore", "nav.evidence", "nav.generated", "nav.insights", "nav.papers"):
            self.assertIn(f'data-i18n="{key}"', html)
        for key in (
            "empty.experimentGroups",
            "search.researchAreas",
            "discoveries.tier1",
            "discoveries.tier2",
            "insights.type.crossDomainBridge",
            "agenda.noSelection",
            "manuscript.routePreview",
        ):
            self.assertIn(f'"{key}"', i18n)

    def test_dashboard_stat_cards_have_i18n_tooltips(self):
        expected_tips = {
            "sourcePapers": (
                "Papers extracted/parsed (excludes fetched-but-unprocessed)",
                "已抽取/解析完成的文献数(不含仅入库未处理的)",
            ),
            "results": (
                "Benchmark result records produced by experiments",
                "基准实验产出的结果记录条数",
            ),
            "researchAreas": (
                "Nodes in the research-area taxonomy tree",
                "研究领域分类树的节点数",
            ),
            "contradictions": (
                "Pairs of conflicting claims found across papers",
                "从文献中发现的互相矛盾的论断对数",
            ),
            "researchInsights": (
                "Research insights (insights table; distinct from Discoveries)",
                "研究洞见条数(insights 表;与\"深度发现\"是不同的表)",
            ),
            "tokens": (
                "Total LLM tokens consumed processing papers",
                "处理文献累计消耗的 LLM token 总量",
            ),
            "experimentRuns": (
                "Total experiment runs",
                "实验运行的总次数",
            ),
            "discoveries": (
                "Discoveries that can grow into papers (deep_insights table; distinct from Research Insights)",
                "可发展成论文的深度发现条数(deep_insights 表;与\"研究洞见\"是不同的表)",
            ),
            "submissionBundles": (
                "Completed paper bundles ready for submission",
                "打包完成、可投稿的完整论文数",
            ),
        }
        html = _read("web/templates/index.html")
        i18n = _read("web/static/js/i18n.js")

        self.assertEqual(html.count('class="stat-card'), 9)
        self.assertEqual(html.count("data-i18n-title=\"overview."), 9)
        for key, (en_tip, zh_tip) in expected_tips.items():
            i18n_key = f"overview.{key}.tip"
            self.assertIn(f'data-i18n-title="{i18n_key}"', html)
            self.assertIn(f'"{i18n_key}": {json.dumps(en_tip, ensure_ascii=False)}', i18n)
            self.assertIn(f'"{i18n_key}": {json.dumps(zh_tip, ensure_ascii=False)}', i18n)

    def test_dashboard_events_use_short_polling_not_eventsource(self):
        app_js = _read("web/static/js/app.js")
        self.assertNotIn("EventSource", app_js)
        self.assertIn("function fetchEvents()", app_js)
        self.assertIn("/api/events?since=", app_js)
        self.assertIn("setInterval(fetchEvents, 2000)", app_js)

    def test_dashboard_legacy_visible_labels_are_gone(self):
        frontend = "\n".join(
            _read(path)
            for path in (
                "web/templates/index.html",
                "web/static/js/app.js",
                "web/static/js/agenda.js",
                "web/static/js/manuscript_routing.js",
            )
        )
        forbidden_labels = [
            "Paper DB",
            "Pipeline Papers",
            "Paper Ideas",
            "Method x Dataset Matrix",
            "Method × Dataset Matrix",
            "Taxonomy Map",
            "Opportunity Map",
            "Deep Insight",
            "Deep Insights",
            "IDEA #",
            "RUN #",
            "Main run",
            "Research Paper Generation",
            "Generated Papers",
            "Complete Papers",
            "Taxonomy Nodes",
            "experiment ideas",
            "PARADIGM",
            "DISCOVERY",
        ]
        for label in forbidden_labels:
            self.assertNotIn(label, frontend)
        self.assertIsNone(re.search(r"\bforge\b", frontend, re.I))

    def test_dashboard_target_files_have_no_known_i18n_residuals(self):
        frontend = "\n".join(
            _read(path)
            for path in (
                "web/templates/index.html",
                "web/static/js/app.js",
                "web/static/js/agenda.js",
                "web/static/js/manuscript_routing.js",
            )
        )
        frontend = re.sub(r"<!--.*?-->", "", frontend, flags=re.S)
        frontend = re.sub(r"//.*", "", frontend)
        forbidden_labels = [
            "Methods & Datasets",
            "What People Are Working On",
            "Where The Gaps Are",
            "Recurring Themes",
            "Paper Clusters",
            "Core Entities",
            "Key Links",
            "NOVEL",
            "PARTIAL",
            "EXISTS",
            "UNCHECKED",
            "Universal",
            "Cross-domain",
            "Method:",
            "Baselines:",
            "Datasets:",
            "Compute:",
            "Strongest Challenge:",
            "Fixed automatic pipeline",
            "RUNNING",
            "STOPPED",
            "Auto Research status unavailable.",
            "Paste YAML first.",
            "VENUES",
            "ROUTE PREVIEW",
            "LINT PREVIEW",
            "ERROR (",
        ]
        for label in forbidden_labels:
            self.assertNotIn(label, frontend)
        self.assertIsNone(re.search(r"\bNo [A-Z][^\"'<>]* yet\b", frontend))

    def test_dashboard_init_is_progressive_and_uses_idle_prefetch(self):
        app_js = _read("web/static/js/app.js")
        initial_block = re.search(
            r"// Initial data loads(?P<body>.*?)// Stats refresh",
            app_js,
            re.S,
        )
        self.assertIsNotNone(initial_block)
        initial_loads = set(re.findall(r"\b(load[A-Za-z0-9_]+)\(", initial_block.group("body")))
        self.assertFalse(
            initial_loads
            & {
                "loadTaxonomyDropdown",
                "loadPapers",
                "loadPaperProgressTab",
                "loadGeneratedPapersTab",
                "loadDiscoveriesTab",
                "loadExperimentsTab",
                "loadInsightsTab",
                "loadProviders",
                "loadOverviewGraph",
            }
        )
        self.assertRegex(app_js, r"requestIdleCallback|setTimeout")
        self.assertIn("prefetchInactiveTabs", app_js)
        self.assertIn("overviewGraphLoaded", app_js)

    def test_dashboard_dead_code_is_removed(self):
        app_py = _read("web/app.py")
        app_js = _read("web/static/js/app.js")
        self.assertEqual(app_py.count("def _planned_tracks("), 1)
        self.assertNotIn('label": "主实验"', app_py)
        self.assertNotIn("function renderExperimentGroups(groups)", app_js)


class ExperimentGroupApiTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = ""
        database.DB_PATH = self.db_path
        database.init_db()
        self.workspace_root = Path(self.tmpdir.name) / "ideas"
        self.workspace_patch = mock.patch.object(workspace_layout, "IDEA_WORKSPACE_DIR", self.workspace_root)
        self.workspace_patch.start()
        self.client = web_app.app.test_client()

        database.execute(
            """
            INSERT INTO deep_insights
            (id, tier, title, submission_status, evidence_plan, experimental_plan)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                2,
                "Idea One",
                "not_started",
                json.dumps({"ablation": {"enabled": True}, "visualization": {"enabled": True}}),
                json.dumps({"ablations": [{"name": "drop_gate"}]}),
            ),
        )
        database.execute(
            """
            INSERT INTO auto_research_jobs
            (deep_insight_id, status, stage, last_note)
            VALUES (?, ?, ?, ?)
            """,
            (1, "running_gpu", "gpu_scheduler", "Main run still progressing"),
        )
        database.execute(
            """
            INSERT INTO experiment_runs
            (id, deep_insight_id, status, hypothesis_verdict, effect_pct, iterations_total, iterations_kept, workdir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (10, 1, "completed", "confirmed", 12.5, 8, 3, str(self.workspace_root / "legacy_run_10")),
        )
        database.execute(
            """
            INSERT INTO experiment_runs
            (id, deep_insight_id, status, iterations_total, iterations_kept, workdir)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (11, 1, "testing", 2, 0, str(self.workspace_root / "legacy_run_11")),
        )
        database.execute(
            "INSERT INTO experiment_artifacts (run_id, artifact_type, path) VALUES (?, ?, ?)",
            (11, "plot", "/tmp/plot.svg"),
        )
        database.execute(
            "INSERT INTO experimental_claims (run_id, deep_insight_id, claim_text, verdict) VALUES (?, ?, ?, ?)",
            (10, 1, "Improves metric", "confirmed"),
        )
        plan_root = self.workspace_root / "idea_1" / "plan"
        paper_root = self.workspace_root / "idea_1" / "paper" / "current"
        plan_root.mkdir(parents=True, exist_ok=True)
        paper_root.mkdir(parents=True, exist_ok=True)
        (plan_root / "latest_status.json").write_text(json.dumps({"stage": "testing", "status": "testing"}), encoding="utf-8")
        (plan_root / "experiment_spec.json").write_text(json.dumps({"run_id": 11, "note": "spec"}), encoding="utf-8")
        (paper_root / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
        database.commit()

    def tearDown(self):
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = self.old_database_url
        database.DB_PATH = self.old_db_path
        self.workspace_patch.stop()
        self.tmpdir.cleanup()

    def test_api_experiment_groups_returns_idea_centric_cards(self):
        response = self.client.get("/api/experiment_groups")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload), 1)
        group = payload[0]
        self.assertEqual(group["insight"]["title"], "Idea One")
        self.assertEqual(group["run_count"], 2)
        self.assertEqual(group["canonical_run"]["id"], 11)
        self.assertEqual(group["latest_run"]["id"], 11)
        self.assertEqual(group["auto_job"]["stage"], "gpu_scheduler")
        self.assertTrue(any(track["key"] == "ablation" and track["enabled"] for track in group["planned_tracks"]))
        self.assertTrue(group["workspace_root"].endswith("idea_1"))
        self.assertIn("latest_status", group["plan_snapshot"])
        self.assertTrue(group["paper_preview_urls"]["index"].endswith("/papers/1"))

    def test_api_experiment_group_detail_includes_run_history_and_artifacts(self):
        response = self.client.get("/api/experiment_groups/1")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["insight"]["id"], 1)
        self.assertEqual(len(payload["runs"]), 2)
        active_run = payload["runs"][0]
        self.assertEqual(active_run["id"], 11)
        self.assertTrue(active_run["has_plot_artifacts"])
        historical_run = next(run for run in payload["runs"] if run["id"] == 10)
        self.assertEqual(historical_run["claim_count"], 1)

    def test_paper_preview_routes_serve_current_tex(self):
        index_response = self.client.get("/papers/1")
        tex_response = self.client.get("/papers/1/tex")

        self.assertEqual(index_response.status_code, 200)
        self.assertIn("Idea 1", index_response.get_data(as_text=True))
        self.assertEqual(tex_response.status_code, 200)
        self.assertIn("\\documentclass", tex_response.get_data(as_text=True))
        tex_response.close()


if __name__ == "__main__":
    unittest.main()
