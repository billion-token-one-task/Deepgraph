"""The startup schema applier must not lose statements to their own comments.

schema_postgres.sql is split on ";", which leaves every statement carrying the
comment lines that preceded it. A "drop chunks that start with --" filter
therefore threw away the statement too. 19 of 151 statements were lost that
way, including CREATE TABLE papers, deep_insights, experiment_runs,
auto_research_jobs and manuscript_runs, and the additive ALTER TABLE repairs.
Production did not notice because its tables predate the function, so nothing
failed until a database had to be created from scratch.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest import mock

from db import database
from db.database import _schema_statements

SCHEMA = Path(__file__).resolve().parents[1] / "db" / "schema_postgres.sql"


class SchemaStatementSplittingTests(unittest.TestCase):
    def test_no_statement_is_dropped_by_a_leading_comment(self):
        sql = SCHEMA.read_text(encoding="utf-8")
        statements = _schema_statements(sql)

        # Every chunk that holds any executable line must survive.
        executable_chunks = [
            chunk
            for chunk in sql.split(";")
            if any(
                line.strip() and not line.lstrip().startswith("--")
                for line in chunk.splitlines()
            )
        ]
        self.assertEqual(len(statements), len(executable_chunks))

    def test_every_declared_table_reaches_the_applier(self):
        sql = SCHEMA.read_text(encoding="utf-8")
        declared = set(
            re.findall(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", sql, flags=re.IGNORECASE)
        )
        self.assertIn("papers", declared)
        applied = set()
        for statement in _schema_statements(sql):
            match = re.match(
                r"CREATE TABLE IF NOT EXISTS\s+(\w+)", statement, flags=re.IGNORECASE
            )
            if match:
                applied.add(match.group(1))
        self.assertEqual(declared, applied)

    def test_statements_never_begin_with_a_comment_line(self):
        for statement in _schema_statements(SCHEMA.read_text(encoding="utf-8")):
            self.assertFalse(statement.lstrip().startswith("--"), statement[:80])


class _FailingCursor:
    def __init__(self, error):
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt):
        if not stmt.startswith(("SET ", "SAVEPOINT", "RELEASE", "ROLLBACK")):
            raise self.error


class SchemaApplierErrorTests(unittest.TestCase):
    def test_an_unexpected_statement_error_surfaces_as_itself(self):
        # The re-raise sits after the except block, where a bare `raise` has no
        # active exception and turns into "No active exception to reraise".
        error = ValueError("syntax error at or near FOO")
        conn = mock.Mock()
        conn.cursor.return_value = _FailingCursor(error)
        with mock.patch.object(database, "get_conn", return_value=conn):
            with self.assertRaises(ValueError) as caught:
                database._apply_postgres_schema_file()
        self.assertIs(caught.exception, error)
        conn.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
