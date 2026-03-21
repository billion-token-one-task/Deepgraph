"""Insight Ranker: Uses LLM to evaluate and rank insights by paradigm-breaking potential.

Feeds all insights to LLM, gets back rankings with rationale,
stores results in the insights table.
"""

import json
from agents.llm_client import call_llm_json
from db import database as db


RANKING_PROMPT = """You are a visionary research strategist. You will receive a list of research insights discovered from {count} recent ML papers.

Your job: rate each insight on a 1-10 "paradigm-breaking" scale:
- 1-3: Incremental improvement within existing framework
- 4-6: Meaningful contribution but still within current paradigms
- 7-8: Challenges a fundamental assumption, could shift how a subfield works
- 9-10: "Relativity-level" — reveals that a widely-held paradigm is structurally wrong, with evidence

For EACH insight, return:
- id: the insight ID
- paradigm_score: 1-10 float
- rationale: 1-2 sentences explaining WHY this score (be harsh — most things are NOT paradigm-breaking)

Be VERY selective with 8+. A true paradigm break means the current approach is not just suboptimal but fundamentally wrong. Most "novel" ideas are still incremental.

Return JSON array: [{{"id": N, "paradigm_score": X.X, "rationale": "..."}}]
"""


def rank_insights_batch(insight_ids: list[int] = None) -> dict:
    """Rank insights by paradigm-breaking potential. Returns stats."""
    if insight_ids:
        placeholders = ",".join("?" * len(insight_ids))
        insights = db.fetchall(f"SELECT id, node_id, insight_type, title, hypothesis, impact, novelty_score, feasibility_score FROM insights WHERE id IN ({placeholders})", tuple(insight_ids))
    else:
        insights = db.fetchall("SELECT id, node_id, insight_type, title, hypothesis, impact, novelty_score, feasibility_score FROM insights ORDER BY created_at DESC")

    if not insights:
        return {"ranked": 0}

    # Build compact insight list for LLM
    insight_text = ""
    for ins in insights:
        insight_text += f"#{ins['id']} [{ins['insight_type']}] N:{ins['novelty_score']} F:{ins['feasibility_score']} ({ins['node_id']})\n"
        insight_text += f"Title: {ins['title']}\n"
        insight_text += f"Hypothesis: {ins['hypothesis'][:300]}\n"
        insight_text += f"Impact: {(ins['impact'] or '')[:200]}\n\n"

    system = RANKING_PROMPT.format(count=len(insights))

    # If too many insights, batch them
    batch_size = 40
    all_rankings = []

    for i in range(0, len(insights), batch_size):
        batch = insights[i:i+batch_size]
        batch_text = ""
        for ins in batch:
            batch_text += f"#{ins['id']} [{ins['insight_type']}] N:{ins['novelty_score']} F:{ins['feasibility_score']} ({ins['node_id']})\n"
            batch_text += f"Title: {ins['title']}\n"
            batch_text += f"Hypothesis: {ins['hypothesis'][:300]}\n"
            batch_text += f"Impact: {(ins['impact'] or '')[:200]}\n\n"

        result, tokens = call_llm_json(system, batch_text)

        if isinstance(result, list):
            all_rankings.extend(result)
        elif isinstance(result, dict) and "rankings" in result:
            all_rankings.extend(result["rankings"])

    # Store rankings
    ranked = 0
    for r in all_rankings:
        if not isinstance(r, dict) or "id" not in r:
            continue
        db.execute(
            "UPDATE insights SET paradigm_score=?, rank_rationale=? WHERE id=?",
            (r.get("paradigm_score", 0), r.get("rationale", ""), r["id"])
        )
        ranked += 1
    db.commit()

    # Compute rank order
    db.execute("""
        UPDATE insights SET rank = (
            SELECT COUNT(*) + 1 FROM insights i2
            WHERE i2.paradigm_score > insights.paradigm_score
        )
        WHERE paradigm_score > 0
    """)
    db.commit()

    return {"ranked": ranked, "tokens": sum(1 for _ in all_rankings)}


if __name__ == "__main__":
    db.init_db()
    stats = rank_insights_batch()
    print(f"Ranked {stats['ranked']} insights")

    # Show top results
    top = db.fetchall("SELECT id, title, paradigm_score, rank_rationale FROM insights WHERE paradigm_score > 0 ORDER BY paradigm_score DESC LIMIT 10")
    for r in top:
        print(f"\n#{r['id']} [P:{r['paradigm_score']}] {r['title']}")
        print(f"  {r['rank_rationale']}")
