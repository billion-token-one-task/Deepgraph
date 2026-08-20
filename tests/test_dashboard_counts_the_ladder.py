"""The headline card must count what its own label promises.

The dashboard's "Decided Findings" card says, in its tooltip: "conclusions
that passed the full evidence ladder (sanity, benchmark, audit) and received
a recorded verdict... The only card that counts scientific output."

It counted rows in scientific_decision_records -- a status table any code
path can write, and one that still held 34 rows stamped before the audit
executor existed. On 2026-08-20 it displayed 47 findings when 13 runs had
actually been adjudicated: a 3.6x overstatement on the one number the page
presents as scientific output.

The ladder already records exactly what the label describes, in
evidence_state_transitions with actor='evidence_audit_v1'. Nothing new had to
be built; the card was reading the wrong table.

The legacy backlog is still exposed as its own field rather than dropped,
because a number that quietly disappears is how the next person rediscovers
the same gap the hard way.
"""

import inspect
import unittest

from web import app as web_app


class DashboardCountsTheLadderTests(unittest.TestCase):
    def _source(self):
        return inspect.getsource(web_app._compute_stats_snapshot)

    def test_headline_reads_the_audit_transitions(self):
        source = self._source()
        head = source.split('"agenda_tokens_total"')[0]
        self.assertIn("evidence_state_transitions", head)
        self.assertIn("actor='evidence_audit_v1'", head)

    def test_headline_no_longer_counts_the_status_table_alone(self):
        source = self._source()
        headline = source.split('"scientific_decisions_total"')[1].split("),")[0]
        self.assertNotIn("COUNT(*) AS c FROM scientific_decision_records", headline)

    def test_the_legacy_backlog_is_still_reported(self):
        # surfaced, not silently dropped
        self.assertIn('"legacy_decision_rows"', self._source())

    def test_verdict_breakdown_is_restricted_to_audited_runs(self):
        source = self._source()
        breakdown = source.split('decisions_{verdict}')[1]
        self.assertIn("actor='evidence_audit_v1'", breakdown)

    def test_the_two_counts_are_distinct_questions(self):
        # the headline counts distinct audited runs; the backlog counts rows
        # that never reached the ladder. They must not share a query.
        source = self._source()
        self.assertIn("COUNT(DISTINCT experiment_run_id)", source)


if __name__ == "__main__":
    unittest.main()
