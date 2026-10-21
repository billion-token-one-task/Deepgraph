"""A paper must not be checked for contradictions against itself."""
from unittest import mock

import agents.reasoning_agent as ra


def test_related_claim_lookup_excludes_the_paper_under_test():
    seen = []

    def fake_fetchall(sql, params):
        seen.append((sql, params))
        return []

    claims = [{"claim_type": "performance", "method_name": "CARE",
               "claim_text": "CARE reaches 0.827", "_id": 5,
               "grounding_status": "verified"}]
    with mock.patch.object(ra.db, "fetchall", side_effect=fake_fetchall):
        contras, tokens = ra.detect_contradictions_batch(claims, paper_id="2604.18234")

    assert contras == [] and tokens == 0
    sql, params = seen[0]
    assert "paper_id<>?" in sql.replace(" ", "")
    assert params[-1] == "2604.18234"
