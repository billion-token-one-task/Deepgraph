import tempfile
import unittest
import json
from pathlib import Path

from db import database
from orchestrator import manuscript_watchdog


class ManuscriptWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_path = Path(self.tmpdir.name)
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        if hasattr(database._local, "conn"):
            try:
                database._local.conn.close()
            except Exception:
                pass
        database._local.conn = None
        if hasattr(database._local, "pg_conn"):
            try:
                database._local.pg_conn.close()
            except Exception:
                pass
            database._local.pg_conn = None
        database.DB_PATH = self.tmpdir_path / "test.db"
        database.DATABASE_URL = ""
        database.init_db()
        database.execute("INSERT INTO deep_insights (id, tier, title, submission_status) VALUES (1, 2, 'Paper', 'bundle_ready')")
        database.execute(
            "INSERT INTO experiment_runs (id, deep_insight_id, status) VALUES (1, 1, 'bundle_ready')"
        )
        database.execute(
            """
            INSERT INTO manuscript_runs (id, experiment_run_id, deep_insight_id, status, workdir)
            VALUES (1, 1, 1, 'bundle_ready', ?)
            """,
            (str(self.tmpdir_path / "paper"),),
        )
        database.execute(
            """
            INSERT INTO auto_research_jobs (deep_insight_id, status, stage, experiment_run_id)
            VALUES (1, 'bundle_ready', 'writing_submission', 1)
            """
        )
        database.commit()

    def tearDown(self):
        if hasattr(database._local, "conn"):
            try:
                database._local.conn.close()
            except Exception:
                pass
        database._local.conn = None
        if hasattr(database._local, "pg_conn"):
            try:
                database._local.pg_conn.close()
            except Exception:
                pass
            database._local.pg_conn = None
        database.DATABASE_URL = self.old_database_url
        database.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    def test_audit_marks_article_placeholder_bundle_stale(self):
        bundle = self.tmpdir_path / "bundle"
        bundle.mkdir()
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\title{Thin Paper}\n\\author{Your Name}\n\\begin{document}\n"
            "\\maketitle\n\\section{Abstract}Short.\\end{document}\n",
            encoding="utf-8",
        )
        database.execute(
            """
            INSERT INTO submission_bundles (id, manuscript_run_id, bundle_format, status, bundle_path)
            VALUES (1, 1, 'conference', 'ready', ?)
            """,
            (str(bundle),),
        )
        database.commit()

        result = manuscript_watchdog.audit_ready_submission_bundles(limit=10, mark_stale=True)

        self.assertEqual(result["stale_marked"], 1)
        submission = database.fetchone("SELECT status FROM submission_bundles WHERE id=1")
        manuscript = database.fetchone("SELECT status FROM manuscript_runs WHERE id=1")
        run = database.fetchone("SELECT status, error_message FROM experiment_runs WHERE id=1")
        insight = database.fetchone("SELECT submission_status FROM deep_insights WHERE id=1")
        auto_job = database.fetchone("SELECT status, stage, last_error FROM auto_research_jobs WHERE deep_insight_id=1")
        self.assertEqual(submission["status"], "stale")
        self.assertEqual(manuscript["status"], "stale")
        self.assertEqual(run["status"], "completed")
        self.assertIn("ICLR 2026", run["error_message"])
        self.assertEqual(insight["submission_status"], "stale")
        self.assertEqual(auto_job["status"], "completed")
        self.assertEqual(auto_job["stage"], "manuscript_stale")
        self.assertIn("ICLR 2026", auto_job["last_error"])

    def test_iclr_contract_bundle_passes(self):
        bundle = self.tmpdir_path / "good_bundle"
        bundle.mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES | manuscript_watchdog.CONTRACT_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "pass")

    def test_blocked_marker_blocks_bundle_audit(self):
        bundle = self.tmpdir_path / "blocked_bundle"
        bundle.mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES | manuscript_watchdog.CONTRACT_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )
        (bundle / "MANUSCRIPT_BLOCKED.json").write_text('{"status":"manuscript_blocked"}', encoding="utf-8")

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "block")
        self.assertTrue(any("manuscript_blocked" in item["issue"] for item in report["issues"]))

    def test_top_venue_overclaim_is_blocked_when_claim_values_blocks_it(self):
        bundle = self.tmpdir_path / "overclaim_bundle"
        bundle.mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES | manuscript_watchdog.CONTRACT_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        (bundle / "claim_values.json").write_text(
            json.dumps({"top_venue_general_superiority_decision": "blocked_missing_strict_top_venue_baseline_audit"}),
            encoding="utf-8",
        )
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            "CGGR is superior to current adaptive reasoning methods and achieves state-of-the-art performance. "
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "block")
        self.assertTrue(any("top-venue/SOTA" in item["issue"] for item in report["issues"]))

    def test_guarded_top_venue_scope_sentence_is_allowed(self):
        bundle = self.tmpdir_path / "guarded_bundle"
        bundle.mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES | manuscript_watchdog.CONTRACT_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        (bundle / "claim_values.json").write_text(
            json.dumps({"top_venue_general_superiority_decision": "blocked_missing_strict_top_venue_baseline_audit"}),
            encoding="utf-8",
        )
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            "The locked package is not by itself a state-of-the-art comparison; broad superiority requires a stricter audit. "
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "pass")

    def test_ordinal_first_sentence_is_not_first_method_overclaim(self):
        bundle = self.tmpdir_path / "ordinal_first_bundle"
        bundle.mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES | manuscript_watchdog.CONTRACT_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        (bundle / "claim_values.json").write_text(
            json.dumps({"top_venue_general_superiority_decision": "blocked_missing_strict_top_venue_baseline_audit"}),
            encoding="utf-8",
        )
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        prior_art = (
            "We study adaptive routing for selective reasoning. "
            "The manuscript acknowledges CAR-style certainty adaptive routing \\cite{lu2025car}, "
            "Self-Route \\cite{he2025selfroute}, Rational Metareasoning \\cite{desabbata2025rational}, "
            "Route-to-Reason \\cite{pan2025rtr}, and RouteLLM \\cite{ong2024routellm}. "
        )
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            "First, it formalizes selective deliberation as an answer-now versus deliberate-more decision. "
            + prior_art
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "pass")

    def test_adaptive_reasoning_manuscript_requires_nearby_prior_art(self):
        bundle = self.tmpdir_path / "prior_art_gap_bundle"
        bundle.mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES | manuscript_watchdog.CONTRACT_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            "We study adaptive reasoning and selective deliberation for question answering. "
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "block")
        self.assertTrue(any("prior-art acknowledgement" in item["issue"] for item in report["issues"]))

    def test_adaptive_routing_prior_art_acknowledgement_passes(self):
        bundle = self.tmpdir_path / "prior_art_ack_bundle"
        bundle.mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES | manuscript_watchdog.CONTRACT_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        prior_art = (
            "We study adaptive routing for selective reasoning. "
            "The manuscript acknowledges CAR-style certainty adaptive routing \\cite{lu2025car}, "
            "Self-Route \\cite{he2025selfroute}, Rational Metareasoning \\cite{desabbata2025rational}, "
            "Route-to-Reason \\cite{pan2025rtr}, and RouteLLM \\cite{ong2024routellm}. "
        )
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            + prior_art
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "pass")

    def test_top_venue_decision_can_be_loaded_from_audited_results(self):
        bundle = self.tmpdir_path / "nested_claim_values_bundle"
        bundle.mkdir()
        (bundle / "audited_results").mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES | manuscript_watchdog.CONTRACT_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        (bundle / "audited_results" / "claim_values.json").write_text(
            json.dumps({"top_venue_general_superiority_decision": "eligible_under_strict_top_venue_audit"}),
            encoding="utf-8",
        )
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        prior_art = (
            "We study adaptive routing and report state-of-the-art performance under the strict audit. "
            "The manuscript acknowledges CAR-style certainty adaptive routing \\cite{lu2025car}, "
            "Self-Route \\cite{he2025selfroute}, Rational Metareasoning \\cite{desabbata2025rational}, "
            "Route-to-Reason \\cite{pan2025rtr}, and RouteLLM \\cite{ong2024routellm}. "
        )
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            + prior_art
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "pass")

    def test_contract_files_can_be_loaded_from_audited_results(self):
        bundle = self.tmpdir_path / "nested_contract_bundle"
        bundle.mkdir()
        audited = bundle / "audited_results"
        audited.mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        for name in manuscript_watchdog.CONTRACT_FILES:
            (audited / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        (audited / "claim_values.json").write_text(
            json.dumps({"top_venue_general_superiority_decision": "blocked_missing_strict_top_venue_baseline_audit"}),
            encoding="utf-8",
        )
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        prior_art = (
            "We study adaptive routing for selective reasoning. "
            "The manuscript acknowledges CAR-style certainty adaptive routing \\cite{lu2025car}, "
            "Self-Route \\cite{he2025selfroute}, Rational Metareasoning \\cite{desabbata2025rational}, "
            "Route-to-Reason \\cite{pan2025rtr}, and RouteLLM \\cite{ong2024routellm}. "
        )
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            + prior_art
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "pass")

    def test_evidence_pending_result_placeholders_block_submission(self):
        bundle = self.tmpdir_path / "placeholder_result_bundle"
        bundle.mkdir()
        for name in manuscript_watchdog.ICLR_REQUIRED_FILES | manuscript_watchdog.CONTRACT_FILES:
            (bundle / name).write_text("{}", encoding="utf-8")
        (bundle / "main.pdf").write_bytes(b"%PDF-1.4\n")
        body = " ".join(["evidence-backed claim with citation \\cite{a}."] * 400)
        (bundle / "main.tex").write_text(
            "\\documentclass{article}\n\\usepackage{iclr2026_conference}\n\\begin{document}\n"
            "No main-result number is reported until the full benchmark completes. "
            "\\begin{tabular}{cccc} Direct & -- & -- & -- \\\\ \\end{tabular} "
            + body
            + "\\includegraphics{figures/a.png}\\includegraphics{figures/b.png}\\end{document}\n",
            encoding="utf-8",
        )

        report = manuscript_watchdog.audit_bundle_path(bundle, bundle_format="conference")

        self.assertEqual(report["status"], "block")
        self.assertTrue(any("blank-result placeholders" in item["issue"] for item in report["issues"]))

    def test_reconcile_downgrades_existing_stale_manuscript_auto_job(self):
        database.execute("UPDATE manuscript_runs SET status='stale' WHERE id=1")
        database.commit()

        reconciled = manuscript_watchdog.reconcile_stale_manuscript_jobs(limit=10)

        auto_job = database.fetchone("SELECT status, stage, last_error FROM auto_research_jobs WHERE deep_insight_id=1")
        self.assertEqual(reconciled, 1)
        self.assertEqual(auto_job["status"], "completed")
        self.assertEqual(auto_job["stage"], "manuscript_stale")
        self.assertIn("stale", auto_job["last_error"])


if __name__ == "__main__":
    unittest.main()
