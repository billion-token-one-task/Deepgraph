import unittest

from agents.paperorchestra.semantic_scholar import paper_to_bibtex_entry


class SemanticScholarBibtexTests(unittest.TestCase):
    def test_bibtex_entry_escapes_tex_special_chars(self):
        entry = paper_to_bibtex_entry(
            {
                "title": "Fast & Faithful Function_Vectors #1",
                "authors": [{"name": "Alice A&B"}],
                "year": 2026,
                "venue": "Venue_One",
                "paperId": "abc123",
            },
            "test_key",
        )

        self.assertIn(r"Fast \& Faithful Function\_Vectors \#1", entry)
        self.assertIn(r"Alice A\&B", entry)
        self.assertIn(r"Venue\_One", entry)


if __name__ == "__main__":
    unittest.main()
