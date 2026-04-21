import unittest

from db.sql_dialect import to_postgres


class SqlDialectTests(unittest.TestCase):
    def test_to_postgres_escapes_like_wildcards(self):
        sql = "SELECT id FROM taxonomy_nodes WHERE id = ? OR id LIKE ? || '.%'"

        adapted = to_postgres(sql)

        self.assertEqual(
            adapted,
            "SELECT id FROM taxonomy_nodes WHERE id = %s OR id LIKE %s || '.%%'",
        )

    def test_to_postgres_rewrites_group_concat(self):
        sql = "SELECT GROUP_CONCAT(DISTINCT r.method_name) AS methods, GROUP_CONCAT(paper_id) AS paper_ids FROM results r"

        adapted = to_postgres(sql)

        self.assertIn("STRING_AGG(DISTINCT CAST(r.method_name AS TEXT), ',')", adapted)
        self.assertIn("STRING_AGG(CAST(paper_id AS TEXT), ',')", adapted)


if __name__ == "__main__":
    unittest.main()
