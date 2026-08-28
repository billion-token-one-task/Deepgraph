"""Read-only exhibit page for the adjudication path: /judge.

What this page is for: showing, from the ledger and nothing else, that a
claim which reaches "supported" here had to walk a fixed ladder, and that the
same ladder returns "refuted" and "inconclusive" for claims that do not earn
better. It is an exhibit, not a dashboard -- every number on it is read live
from the tables the meta-harness writes during a run, so a reader can ask for
the row behind any figure.

Three things this module deliberately does NOT do, because the page's whole
value is that it can be checked:

  * It never recomputes a verdict. Verdicts come from
    scientific_decision_records, which is the only table a verdict is written
    to, and each one carries the verdict_hash the auditor signed it with.
  * It never widens a query to make an exhibit look better. The headline
    counts are the whole-ledger counts, cherry-picked exhibits included, so a
    reader sees the base rate a supported result was drawn from.
  * It has no mutating endpoint and no operator affordance. Mutations belong
    to the operator-authenticated /api/meta-harness/v1 blueprint.

Provenance honesty is enforced here rather than left to the template: every
exhibit carries the candidate's model_version, and OPERATOR_FROZEN_NOTE is
rendered wherever that value says a human wrote the method down. Nothing on
this page may read as an autonomous discovery when it was a reproduction.
"""
from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, render_template

from db import database as db
from web.provenance_routes import _scrub_text

blueprint = Blueprint("judge_demo", __name__)

# The candidate ideas in agenda 14 were transcribed from published papers by an
# operator, not invented by the system: model_version records which. Any
# exhibit whose candidate carries this value is a reproduction under audit, and
# the page says so next to the verdict rather than in a footnote.
OPERATOR_FROZEN_MODEL_VERSION = "operator_frozen_no_llm"

# The rungs an experiment run must climb, in order. Sourced from
# contracts.meta_harness.EVIDENCE_STATES; spelled out here with the plain
# language a reader outside the project needs, and with the gate that has to
# pass before each one is written.
LADDER = [
    ("planned", "预注册",
     "候选的方法、指标、基准切片和成功阈值在跑之前写死并存档; 之后改不了"),
    ("sanity_passed", "小样本试跑",
     "先用小切片证明这套代码能跑出非空预测, 再申请全量预算"),
    ("full_benchmark_complete", "全量基准",
     "候选和基线在同一份 revision-pinned 数据、同一份预算下各跑一遍"),
    ("evidence_audited", "证据审计",
     "留出集必须自证是留出集; 置换检验出 p 值; 跨厂商的独立评审读原始预测"),
    ("scientifically_decided", "判决",
     "supported / refuted / inconclusive 之一, 连同 verdict_hash 写进不可变账本"),
    ("manuscript_allowed", "允许成稿",
     "只有走完上面全部台阶的判决才允许写成论文"),
]

# Exhibit 3 has no row of its own in the ledger: it is a gate that fires before
# a measurement is allowed to exist, so what it produced was the ABSENCE of a
# number. The record is the commit that installed it and the incident that
# forced it, both quoted verbatim rather than paraphrased.
HOLDOUT_GATE = {
    "commit": "65e2699",
    "date": "2026-08-18",
    "title": "Holdout provenance gate: a holdout must prove it is a holdout",
    "incident": (
        "第一次留出集飞行 (colab 请求 14) 跑出来的数字和被审计的那一趟逐位相同。"
        "原因是那次 run 的 vendored runner 快照早于 example-offset 支持, 环境变量被"
        "静默忽略, test[0:200] 被测了两遍 -- 同一批样本冒充留出集。"
    ),
    "rule": (
        "审计现在拒收两类留出集: dataset_manifest 里没记录 example_offset 的, "
        "以及原始输入 input_sha256 集合与被审计那趟不相交不成立的。"
        "manifest 字段只是一个声明, 哈希集合才是证据, 两个都查。"
    ),
    "source": "meta_harness/evidence_audit.py :: holdout_provenance_problem",
}

# What the system does today versus what it does not. The second column exists
# because the first one is easy to over-read: agenda 10 spent 3.63M tokens and
# roughly 50 GPU-hours on self-improving the harness and produced zero
# supported results, and V1 cannot structurally do it. Saying so here is
# cheaper than being caught not saying it.
#
# Counts are {placeholders} filled from the live ledger at render time. A
# hardcoded "166 判决" was right for about an hour and then quietly became a
# false number on the one page whose entire claim is that its numbers check out.
CAPABILITY_LEDGER = [
    ("预注册 -> 判决 全流程无人介入", True,
     "调度、授权、执行、审计、判决全部由服务自己完成; "
     "展品里每一级台阶都带着写入它的 actor 和时间戳"),
    ("拒绝证据不足的主张", True,
     "全库 {total} 条判决里 supported 只有 {supported} 条; "
     "展品 B 是系统对一个候选说不, 展品 C 是拒绝下结论"),
    ("拒收被污染的留出集", True,
     "展品 D: 同一批样本冒充留出集被拦下, 该趟测量作废重跑"),
    ("跨厂商独立复核", True,
     "评审模型与被测模型来自不同厂商, evaluator_ref 和 evaluator_hash 记在账本里"),
    ("不可变账本", True,
     "判决、原始产物、评审器、留出集各自的 sha256 都随判决一起落盘"),
    ("系统自主选题并做出可复现的发现", False,
     "路线图。本页所有候选都是人手从论文里抽出来冻结的 (operator-frozen); "
     "非 operator-frozen 的 {llm_total} 条判决里 supported 为 {llm_supported}"),
    ("通用程序 runner", False,
     "路线图。目前只覆盖两种任务协议, 换一类实验就要人接线"),
    ("harness 自进化 / RSI", False,
     "路线图, 且 V1 结构上做不了。agenda 10 为此花掉 363 万 token 和约 50 GPU-h, "
     "0 条 supported"),
]



def _rows(sql: str, params: tuple = ()) -> list[dict]:
    try:
        return [dict(row) for row in db.fetchall(sql, params)]
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        return []


def _one(sql: str, params: tuple = ()) -> dict:
    rows = _rows(sql, params)
    return rows[0] if rows else {}


def _loads(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}


def _ts(value: Any) -> str:
    """One clock, one format.

    experiment_runs.created_at comes back naive and the transition rows come
    back tz-aware, so an unformatted ladder mixed "06:09:48" with
    "07:06:22+00:00" and invited the reader to wonder which was local. Both are
    UTC; the page says so once, in the ladder header.
    """
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ", timespec="seconds")[:19]
    return str(value)[:19]


def _exhibit(run_id: int) -> dict | None:
    """Assemble one run's whole ladder from the tables that recorded it.

    Returns None when the run does not exist, so a page built before a demo
    run finishes renders the exhibits it does have instead of a 500.
    """
    run = _one(
        "SELECT id, deep_insight_id, agenda_id, status, phase,"
        " scientific_evidence_state, hypothesis_verdict, baseline_metric_name,"
        " baseline_metric_value, best_metric_value, effect_size, effect_pct,"
        " resource_grant_id, created_at, completed_at"
        " FROM experiment_runs WHERE id=?",
        (run_id,),
    )
    if not run:
        return None

    idea = _one(
        "SELECT id, title, model_version, agenda_id FROM deep_insights WHERE id=?",
        (run["deep_insight_id"],),
    )
    audit = _one(
        "SELECT id, raw_artifacts_hash, claim_ledger_hash, benchmark_contract_hash,"
        " evaluator_ref, evaluator_hash, holdout_ref, holdout_hash, created_at"
        " FROM evidence_audit_records WHERE experiment_run_id=?"
        " ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    decision = _one(
        "SELECT id, verdict, verdict_hash, evidence_decision_json, created_at"
        " FROM scientific_decision_records WHERE experiment_run_id=?"
        " ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    outcome = _one(
        "SELECT id, verdict, effect, baseline, actual_tokens, actual_gpu_hours,"
        " wall_seconds, state_decision, recorded_at"
        " FROM outcome_records WHERE experiment_run_id=?"
        " ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    # The whole grant chain, not just the last one. Waiting for budget is a rung
    # of the ladder the page claims to show, and a run that reached a verdict
    # did so across a separate grant per stage -- pilot, full benchmark, audit --
    # each with its own cap. Showing only run.resource_grant_id showed the audit
    # grant alone and made the wait look like it never happened.
    #
    # Reached through the run's own compute requests rather than by timestamp:
    # run 274's pilot grant was issued 1.3 seconds BEFORE the run row existed,
    # so a "grants created at or after the run" filter silently dropped the
    # first rung of the very chain this row is here to show.
    grants = _rows(
        "SELECT g.id, g.stage, g.token_cap, g.gpu_class, g.max_gpu_hours,"
        " g.status, g.grant_reason, g.created_at FROM resource_grants g"
        " WHERE g.id IN ("
        "   SELECT DISTINCT resource_grant_id FROM colab_work_requests_v1"
        "   WHERE experiment_run_id=? AND resource_grant_id IS NOT NULL"
        " ) ORDER BY g.id",
        (run_id,),
    )

    # Where the work actually ran. resource_grants.gpu_class is what the grant
    # AUTHORISED, and rendering it alone labelled a Colab flight "NVIDIA A10G"
    # because that is the class the grant asked for. The compute account that
    # returned the artifacts is the answer to "where did this run", and it is
    # the one a reader checking the story will ask for.
    accounts = _rows(
        "SELECT DISTINCT stage, account_ref FROM colab_work_requests_v1"
        " WHERE experiment_run_id=? AND status='succeeded' AND account_ref IS NOT NULL"
        " ORDER BY stage",
        (run_id,),
    )

    reached = {
        row["to_state"]: row
        for row in _rows(
            "SELECT to_state, actor, created_at FROM evidence_state_transitions"
            " WHERE experiment_run_id=? ORDER BY id",
            (run_id,),
        )
    }
    # "planned" is the state a run is created in, so no transition writes it;
    # the run's own created_at is when the pre-registration was frozen.
    ladder = []
    for state, label, why in LADDER:
        hit = reached.get(state)
        if state == "planned":
            hit = hit or {"actor": "forge", "created_at": run.get("created_at")}
        ladder.append({
            "state": state,
            "label": label,
            "why": why,
            "reached": bool(hit),
            "actor": (hit or {}).get("actor", ""),
            "at": _ts((hit or {}).get("created_at")),
        })

    decision_input = _loads(decision.get("evidence_decision_json")).get("input", {})
    decision_body = _loads(decision.get("evidence_decision_json")).get("decision", {})

    return {
        "run": run,
        "idea": idea,
        "title": _scrub_text(str(idea.get("title") or "")),
        "operator_frozen": idea.get("model_version") == OPERATOR_FROZEN_MODEL_VERSION,
        "model_version": idea.get("model_version"),
        "grants": grants,
        "accounts": accounts,
        "audit": audit,
        "decision": decision,
        "verdict": (decision.get("verdict") or outcome.get("verdict")
                    or run.get("hypothesis_verdict") or "pending"),
        "p_value": decision_input.get("p_value"),
        "alpha": decision_input.get("alpha"),
        "metric_value": decision_input.get("metric_value"),
        "baseline_value": decision_input.get("baseline_value"),
        "blockers": decision_body.get("blockers") or [],
        "significant": decision_body.get("significant"),
        "outcome": outcome,
        "ladder": ladder,
        "complete": bool(decision),
    }


def _ledger_totals() -> dict:
    counts = {
        row["verdict"]: int(row["c"])
        for row in _rows(
            "SELECT verdict, count(*) c FROM outcome_records GROUP BY 1"
        )
    }
    total = sum(counts.values())
    frozen = _one(
        "SELECT count(*) c FROM scientific_decision_records d"
        " JOIN experiment_runs r ON r.id = d.experiment_run_id"
        " JOIN deep_insights i ON i.id = r.deep_insight_id"
        " WHERE d.verdict='supported' AND i.model_version=?",
        (OPERATOR_FROZEN_MODEL_VERSION,),
    )
    # The roadmap row claims no LLM-authored candidate has ever been supported.
    # It is counted, not asserted: an autonomous supported result appearing
    # here should change the page, not be argued with.
    llm = _one(
        "SELECT count(*) c FROM outcome_records o"
        " JOIN deep_insights i ON i.id = o.idea_id"
        " WHERE i.model_version IS DISTINCT FROM ?",
        (OPERATOR_FROZEN_MODEL_VERSION,),
    )
    llm_supported = _one(
        "SELECT count(*) c FROM outcome_records o"
        " JOIN deep_insights i ON i.id = o.idea_id"
        " WHERE o.verdict='supported' AND i.model_version IS DISTINCT FROM ?",
        (OPERATOR_FROZEN_MODEL_VERSION,),
    )
    return {
        "counts": counts,
        "total": total,
        "supported": counts.get("supported", 0),
        "supported_operator_frozen": int(frozen.get("c") or 0),
        "llm_total": int(llm.get("c") or 0),
        "llm_supported": int(llm_supported.get("c") or 0),
    }


@blueprint.get("/judge")
def judge_demo():
    # Exhibit A is whichever agenda-14 run most recently walked the whole
    # ladder; naming a run id here would freeze the page to one demo.
    newest = _one(
        "SELECT d.experiment_run_id AS id FROM scientific_decision_records d"
        " JOIN experiment_runs r ON r.id = d.experiment_run_id"
        " WHERE r.agenda_id=14 ORDER BY d.id DESC LIMIT 1"
    )
    exhibits = {
        # A: the most recent completed adjudication, whatever it decided. Keyed
        # "latest" and not "supported" on purpose -- the page must not be built
        # around an assumption about a verdict it has not seen yet.
        "latest": _exhibit(int(newest["id"])) if newest.get("id") else None,
        # B: the same ladder saying no. Run 235's candidate produced 0.0 on a
        # 0.32 baseline and the audit refused it at p=0.000999.
        "refuted": _exhibit(235),
        # C: the same ladder declining to conclude anything, at p=0.697.
        "inconclusive": _exhibit(240),
    }
    totals = _ledger_totals()
    capabilities = [
        (name, done, note.format(**totals)) for name, done, note in CAPABILITY_LEDGER
    ]
    return render_template(
        "judge_demo.html",
        exhibits=exhibits,
        gate=HOLDOUT_GATE,
        capabilities=capabilities,
        totals=totals,
    )


def register_judge_demo_routes(app) -> None:
    app.register_blueprint(blueprint)
