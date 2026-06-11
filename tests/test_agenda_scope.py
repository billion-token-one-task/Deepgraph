"""Acceptance tests for agenda-scoped candidate circling (multi-agenda isolation).

Two agendas in different domains (medical imaging vs cryptography) must not
leak into each other's candidate pool: scoped queries return only records from
the agenda's own keyword domain or tagged with its agenda_id.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")


MEDICAL_AGENDA = {
    "version": "v1",
    "name": "medical_imaging_v1",
    "description": "few-shot medical image segmentation",
    "focus": ["medical imaging", "segmentation"],
    "prefer": {"keywords": ["diffusion model"]},
}

CRYPTO_AGENDA = {
    "version": "v1",
    "name": "cryptography_v1",
    "description": "post-quantum cryptography",
    "focus": ["cryptography", "lattice"],
    "prefer": {"keywords": ["post-quantum"]},
}

MEDICAL_TERMS = ("medical", "segmentation", "diffusion", "imaging")
CRYPTO_TERMS = ("cryptography", "lattice", "post-quantum", "encryption")


class AgendaScopeTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        from db import database as db

        self._original_db_path = db.DB_PATH
        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()
        self.db = db

    def tearDown(self):
        from db import database as db

        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = self._original_db_path
        self._tmpdir.cleanup()
        os.environ.pop("DEEPGRAPH_DB_PATH", None)

    def _seed_insight(self, insight_id, title, problem, agenda_id=None):
        self.db.execute(
            """
            INSERT INTO deep_insights
                (id, tier, status, title, problem_statement, adversarial_score,
                 novelty_status, resource_class, experimentability, agenda_id)
            VALUES (?, 2, 'candidate', ?, ?, 7.0, 'novel', 'cpu', 'easy', ?)
            """,
            (insight_id, title, problem, agenda_id),
        )
        self.db.commit()


class InsightPoolIsolationTests(AgendaScopeTestBase):
    def setUp(self):
        super().setUp()
        from agents.agenda_loader import parse_agenda, save_agenda

        self.medical = parse_agenda(MEDICAL_AGENDA)
        save_agenda(self.medical)
        self.crypto = parse_agenda(CRYPTO_AGENDA)
        save_agenda(self.crypto)

        # Untagged corpus: two domains
        self._seed_insight(1, "Diffusion model for medical imaging segmentation",
                           "Few-shot generalization across hospitals")
        self._seed_insight(2, "Cross-center medical imaging benchmark",
                           "Segmentation models break across centers")
        self._seed_insight(3, "Lattice-based post-quantum cryptography scheme",
                           "Encryption against quantum adversaries")
        self._seed_insight(4, "Side-channel analysis of lattice cryptography",
                           "Lattice implementations leak timing information")
        # Tagged outputs: one per agenda
        self._seed_insight(10, "Agenda-produced medical insight",
                           "Generated under the medical agenda",
                           agenda_id=self.medical.agenda_id)
        self._seed_insight(11, "Agenda-produced crypto insight",
                           "Generated under the crypto agenda",
                           agenda_id=self.crypto.agenda_id)

    def _titles(self, rows):
        return [str(r["title"]).lower() for r in rows]

    def test_pools_do_not_cross_domains(self):
        from agents.agenda_selector import _fetch_insight_pool

        medical_pool = self._titles(_fetch_insight_pool(agenda=self.medical))
        crypto_pool = self._titles(_fetch_insight_pool(agenda=self.crypto))

        self.assertTrue(medical_pool)
        self.assertTrue(crypto_pool)
        # No crypto-domain term in the medical pool, and vice versa
        for title in medical_pool:
            for term in CRYPTO_TERMS:
                self.assertNotIn(term, title)
        for title in crypto_pool:
            for term in MEDICAL_TERMS:
                self.assertNotIn(term, title)

    def test_tagged_insights_stay_with_their_agenda(self):
        from agents.agenda_selector import _fetch_insight_pool

        medical_ids = {r["id"] for r in _fetch_insight_pool(agenda=self.medical)}
        crypto_ids = {r["id"] for r in _fetch_insight_pool(agenda=self.crypto)}

        self.assertIn(10, medical_ids)
        self.assertNotIn(11, medical_ids)
        self.assertIn(11, crypto_ids)
        self.assertNotIn(10, crypto_ids)

    def test_unscoped_pool_unchanged(self):
        from agents.agenda_selector import _fetch_insight_pool

        ids = {r["id"] for r in _fetch_insight_pool()}
        self.assertEqual(ids, {1, 2, 3, 4, 10, 11})

    def test_scoped_selection_picks_within_domain(self):
        from agents.agenda_selector import select_and_persist

        sel_med = select_and_persist(self.medical, scope_to_agenda=True)
        sel_cry = select_and_persist(self.crypto, scope_to_agenda=True)
        self.assertIn(sel_med.selected_insight_id, {1, 2, 10})
        self.assertIn(sel_cry.selected_insight_id, {3, 4, 11})


class TaxonomyCircleTests(AgendaScopeTestBase):
    def _seed_node(self, node_id, name, description=""):
        self.db.execute(
            "INSERT INTO taxonomy_nodes (id, name, description, parent_id, depth) "
            "VALUES (?, ?, ?, NULL, 1)",
            (node_id, name, description),
        )
        self.db.commit()

    def test_node_circle_matches_keywords_only(self):
        from agents.signal_harvester import agenda_taxonomy_node_ids

        self._seed_node("ml.medimg", "Medical Imaging", "segmentation and diagnosis")
        self._seed_node("ml.crypto", "Cryptography", "lattice-based encryption")
        self._seed_node("ml.nlp", "Natural Language Processing", "")

        medical_nodes = agenda_taxonomy_node_ids(["medical imaging", "segmentation"])
        crypto_nodes = agenda_taxonomy_node_ids(["cryptography", "lattice"])

        self.assertEqual(medical_nodes, ["ml.medimg"])
        self.assertEqual(crypto_nodes, ["ml.crypto"])
        self.assertEqual(agenda_taxonomy_node_ids([]), [])

    def test_wildcard_keyword_does_not_match_everything(self):
        from agents.signal_harvester import agenda_taxonomy_node_ids

        self._seed_node("ml.medimg", "Medical Imaging", "segmentation")
        self._seed_node("ml.crypto", "Cryptography", "lattice")
        self._seed_node("ml.quant", "Quantization", "keeps 99% accuracy at 4-bit")

        # LIKE wildcards in user keywords must match literally, not widen
        # scope: '%' only hits the node whose text contains a literal '%'.
        self.assertEqual(agenda_taxonomy_node_ids(["%"]), ["ml.quant"])
        self.assertEqual(agenda_taxonomy_node_ids(["_"]), [])
        self.assertEqual(agenda_taxonomy_node_ids(["99% accuracy"]), ["ml.quant"])


class LikeWildcardEscapeTests(AgendaScopeTestBase):
    """User-supplied scope keywords go into SQL LIKE patterns; wildcard
    characters must not widen the candidate pool beyond the literal term."""

    def test_escape_like_escapes_wildcards(self):
        from db.sql_dialect import escape_like

        self.assertEqual(escape_like(r"50%_\x"), r"50\%\_\\x")
        self.assertEqual(escape_like("plain term"), "plain term")

    def test_percent_keyword_does_not_widen_insight_pool(self):
        from agents.agenda_loader import parse_agenda
        from agents.agenda_selector import _fetch_insight_pool

        self._seed_insight(1, "Diffusion model for medical imaging",
                           "Few-shot generalization")
        self._seed_insight(2, "Lattice-based cryptography scheme",
                           "Post-quantum encryption")

        wild = parse_agenda({
            "version": "v1",
            "name": "wildcard_probe",
            "focus": ["%"],
        })
        # Pre-escape this pattern ('%%%') matched the whole table
        self.assertEqual(_fetch_insight_pool(agenda=wild), [])

        underscore = parse_agenda({
            "version": "v1",
            "name": "underscore_probe",
            "focus": ["________"],
        })
        self.assertEqual(_fetch_insight_pool(agenda=underscore), [])

    def test_literal_percent_keyword_matches_only_literal_text(self):
        from agents.agenda_loader import parse_agenda
        from agents.agenda_selector import _fetch_insight_pool

        self._seed_insight(1, "Quantization keeps 99% accuracy at 4-bit",
                           "Compression without quality loss")
        self._seed_insight(2, "Reaching 99 points of accuracy with ensembling",
                           "Ensembles for tabular data")

        agenda = parse_agenda({
            "version": "v1",
            "name": "literal_percent",
            "focus": ["99% accuracy"],
        })
        ids = {r["id"] for r in _fetch_insight_pool(agenda=agenda)}
        self.assertEqual(ids, {1})


if __name__ == "__main__":
    unittest.main()
