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
        # The unreachable "opportunities" rendering path (no DOM target exists;
        # loadOpportunities was never called) is removed, not just orphaned.
        for dead in (
            "function loadOpportunities",
            "function renderOpportunities",
            "const insightTypeColors",
            "function insightTypeLabel",
            "allOpportunities",
            "oppsLoaded",
        ):
            self.assertNotIn(dead, app_js, f"dead opportunities symbol still present: {dead}")

    def test_evidence_matrix_is_a_heatmap(self):
        """Acceptance B — the benchmark matrix shades filled cells by value.

        The old matrix gave every filled cell one flat class (`.cell-filled`,
        a single background). A heatmap must compute a per-cell background from
        the cell's value via a colour scale. We pin to source: a value→colour
        builder exists, the cell markup carries an inline `background:` derived
        from it, and the scale endpoints live in :root as CSS variables. The
        "more than one distinct fill colour" wall-clock proof lives in the
        Playwright gate (dashboard_e2e.mjs) and the perf microbench.
        """
        app_js = _read("web/static/js/app.js")
        css = _read("web/static/css/style.css")
        self.assertIn("function buildMatrixHeat(", app_js)
        body = re.search(
            r"function renderMatrix\(container, matrix\)\s*\{(?P<body>.*?)\n\}",
            app_js,
            re.S,
        )
        self.assertIsNotNone(body, "renderMatrix not found")
        fn = body.group("body")
        # A filled cell's background is computed from the heat scale, inline.
        self.assertRegex(fn, r"heat\(", "renderMatrix must colour cells via the heat scale")
        self.assertRegex(fn, r"background:\$\{", "filled cells must carry an inline heat background")
        # Switching the metric must recolour, not just relabel.
        update = re.search(
            r"function updateMatrixMetric\(selectEl\)\s*\{(?P<body>.*?)\n\}",
            app_js,
            re.S,
        )
        self.assertIsNotNone(update, "updateMatrixMetric not found")
        self.assertRegex(update.group("body"), r"buildMatrixHeat\(|heat\(",
                         "updateMatrixMetric must recompute the heat colours")
        # Scale endpoints are themeable CSS variables.
        self.assertIn("--heat-lo:", css)
        self.assertIn("--heat-hi:", css)

    def test_style_scales_exist_and_are_applied(self):
        """Acceptance B — :root carries spacing / type / shadow scales and the
        scattered hard-coded values are pulled into them (not left inline)."""
        css = _read("web/static/css/style.css")
        # Scales are defined.
        for token in ("--space-8:", "--space-12:", "--space-16:",
                      "--text-sm:", "--text-base:", "--text-lg:",
                      "--shadow-sm:", "--shadow-lg:", "--shadow-focus:"):
            self.assertIn(token, css, f"missing scale token {token}")
        # Scales are actually applied (not just declared).
        self.assertGreaterEqual(css.count("var(--space-"), 50)
        self.assertGreaterEqual(css.count("var(--text-"), 20)
        self.assertGreaterEqual(css.count("var(--shadow-"), 4)
        # The hard-coded elevation shadows are collected into tokens: no raw
        # `box-shadow: 0 <n>px ...` literals remain outside the :root scale.
        root = re.search(r":root\s*\{.*?\n\}", css, re.S)
        css_outside_root = css.replace(root.group(0), "") if root else css
        self.assertIsNone(
            re.search(r"box-shadow:\s*0\s+\d+px", css_outside_root),
            "a hard-coded box-shadow literal still lives outside the shadow scale",
        )

    def test_heavy_list_renders_are_chunked_not_one_shot(self):
        """Acceptance A — lists that grow with data are built AND inserted in
        chunks, so no single synchronous task scales with the list length.

        The old helper chunked only the DOM insertion; the card HTML (parsing
        several JSON fields per row on ~0.6–0.7 MB payloads) was still built in
        one synchronous `items.map(...)`. The new renderListChunked takes
        (container, items, renderItem) and builds each chunk lazily. We pin to
        source; the wall-clock proof is the perf microbench + the E2E.
        """
        app_js = _read("web/static/js/app.js")
        self.assertIn("function renderListChunked(container, items, renderItem", app_js)
        # No O(n^2) `innerHTML +=` anywhere in the frontend bundle.
        self.assertIsNone(re.search(r"\.innerHTML\s*\+=", app_js),
                          "`innerHTML +=` (O(n^2)) must not appear in app.js")
        # The heavy tabs render through the chunked builder, not one-shot
        # `list.innerHTML = X.map(...).join('')`.
        for fn_name in ("loadDiscoveriesTab", "loadInsightsTab",
                        "renderExperiments", "renderAutoResearchJobs"):
            self.assertIn(fn_name, app_js)
        self.assertGreaterEqual(app_js.count("renderListChunked(list"), 6)

    def test_taxonomy_dropdown_build_is_not_quadratic(self):
        """Regression guard for the main-thread freeze (Acceptance A).

        The taxonomy ``<select>`` holds ~3300 nodes. Building it with
        ``sel.innerHTML += '<option>...'`` inside a loop re-serializes and
        re-parses the entire <select> on every iteration — O(n²) — which
        froze the main thread ~4s after load (when prefetchInactiveTabs ran
        loadTaxonomyDropdown). The fix must batch the option strings and
        assign innerHTML (or append a DocumentFragment) exactly once.

        We assert against the *source* so the O(n) property is pinned to the
        real file; the wall-clock proof lives in the node perf microbench
        (tests/perf/taxonomy_dropdown_perf.mjs) and the Playwright E2E.
        """
        app_js = _read("web/static/js/app.js")
        body = re.search(
            r"async function loadTaxonomyDropdown\(\)\s*\{(?P<body>.*?)\n\}",
            app_js,
            re.S,
        )
        self.assertIsNotNone(body, "loadTaxonomyDropdown not found")
        fn = body.group("body")
        # No `someEl.innerHTML += ...` accumulation anywhere in the function:
        # that is the quadratic pattern we are forbidding.
        self.assertIsNone(
            re.search(r"\.innerHTML\s*\+=", fn),
            "loadTaxonomyDropdown must not use `innerHTML +=` (O(n^2))",
        )
        # And it must build the options in a batch (join an array) before a
        # single assignment / fragment append.
        self.assertRegex(
            fn,
            r"\.join\(|createDocumentFragment|insertAdjacentHTML",
            "loadTaxonomyDropdown must build options in one batch",
        )


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


class StatsCacheTests(unittest.TestCase):
    """Issue #34 · Feature 1 — /api/stats served from an in-process TTL cache."""

    def setUp(self):
        self.client = web_app.app.test_client()

    def test_api_stats_served_from_cache_not_recomputed_each_request(self):
        sentinel = {
            "papers_processed": 1,
            "results_total": 2,
            "insights_total": 3,
            "deep_insights_total": 4,
            "submission_bundles_total": 5,
        }
        with mock.patch.object(
            web_app, "get_stats_dict", return_value=sentinel
        ) as heavy:
            web_app._stats_cache.invalidate()
            web_app._stats_cache.prewarm()
            for _ in range(8):
                response = self.client.get("/api/stats")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json(), sentinel)

        # The heavy COUNT(*) computation must run at most once across the
        # warm-up plus eight requests — proving the cache is hit, not recomputed.
        self.assertLessEqual(heavy.call_count, 1)

    def test_api_stats_returns_correct_fields(self):
        sentinel = {
            "papers_processed": 11,
            "results_total": 22,
            "insights_total": 33,
            "deep_insights_total": 44,
            "submission_bundles_total": 55,
        }
        with mock.patch.object(web_app, "get_stats_dict", return_value=sentinel):
            web_app._stats_cache.invalidate()
            web_app._stats_cache.prewarm()
            payload = self.client.get("/api/stats").get_json()

        # Caching must not change the contract: every field and value the
        # underlying computation produced is served verbatim.
        for key in (
            "papers_processed",
            "results_total",
            "insights_total",
            "deep_insights_total",
            "submission_bundles_total",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload, sentinel)

    def test_import_web_app_does_not_block_on_heavy_stats(self):
        import importlib
        from orchestrator import pipeline

        with mock.patch.object(pipeline, "get_stats_dict") as heavy:
            importlib.reload(web_app)
            # Importing / building the app must not synchronously run the heavy
            # stats query (tests import web.app; deploy imports it at startup).
            heavy.assert_not_called()
        # Restore a clean module bound to the real computation.
        importlib.reload(web_app)


class EventsTailTests(unittest.TestCase):
    """Issue #34 · Feature 2 — /api/events?since=0 returns only the tail."""

    def setUp(self):
        self.client = web_app.app.test_client()

    def test_api_events_since_zero_returns_tail_only(self):
        full = [{"seq": i, "type": "x"} for i in range(120)]
        with mock.patch.object(web_app, "get_events", return_value=list(full)):
            payload = self.client.get("/api/events?since=0").get_json()

        self.assertLessEqual(len(payload["events"]), 50)
        # next_seq must still point past the newest event so subsequent
        # ?since=next_seq polling keeps advancing correctly.
        self.assertEqual(payload["next_seq"], 120)
        self.assertEqual(payload["events"][-1]["seq"], 119)

    def test_api_events_incremental_since_unchanged(self):
        incremental = [{"seq": i, "type": "x"} for i in range(50, 120)]
        with mock.patch.object(
            web_app, "get_events", return_value=list(incremental)
        ) as get_events:
            payload = self.client.get("/api/events?since=50").get_json()

        get_events.assert_called_once_with(50)
        # Incremental polling is untouched: every event after the cursor is
        # returned, no truncation.
        self.assertEqual(len(payload["events"]), 70)
        self.assertEqual(payload["events"][0]["seq"], 50)
        self.assertEqual(payload["next_seq"], 120)


class FirstPaintE2ETests(unittest.TestCase):
    """Issue #34 · end-to-end — the first-paint endpoints are 200 and bounded."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
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
        database.DB_PATH = Path(self.tmpdir.name) / "firstpaint.db"
        database.init_db()
        self.client = web_app.app.test_client()

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
        self.tmpdir.cleanup()

    def test_first_paint_endpoints_are_fast_and_bounded(self):
        # Seed more than the tail size so truncation is actually exercised.
        for i in range(80):
            web_app.log_event("seed", {"i": i})

        stats_payload = {
            "papers_processed": 0,
            "results_total": 0,
            "insights_total": 0,
            "deep_insights_total": 0,
            "submission_bundles_total": 0,
        }
        first_paint_paths = [
            "/api/stats",
            "/api/events?since=0",
            "/api/recent_discoveries",
            "/api/insights?limit=6",
            "/api/deep_insights?limit=4",
        ]

        with mock.patch.object(
            web_app, "get_stats_dict", return_value=stats_payload
        ) as heavy:
            web_app._stats_cache.invalidate()
            web_app._stats_cache.prewarm()  # deploy/startup prewarm
            responses = {p: self.client.get(p) for p in first_paint_paths}

            # 2) stats is served from the warm cache: the heavy query is not
            #    recomputed per request (mock count, no wall-clock dependency).
            self.assertLessEqual(heavy.call_count, 1)

        # 1) every first-paint endpoint returns 200.
        for path, response in responses.items():
            self.assertEqual(response.status_code, 200, f"{path} -> {response.status_code}")

        # 3) since=0 returns only the tail (<= 50 events).
        events_payload = responses["/api/events?since=0"].get_json()
        self.assertLessEqual(len(events_payload["events"]), 50)

        # 4) stats keeps its contract fields (cache does not drop/rename them).
        stats_resp = responses["/api/stats"].get_json()
        for key in (
            "papers_processed",
            "results_total",
            "insights_total",
            "deep_insights_total",
            "submission_bundles_total",
        ):
            self.assertIn(key, stats_resp)


if __name__ == "__main__":
    unittest.main()
