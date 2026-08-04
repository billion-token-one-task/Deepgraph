#!/usr/bin/env python3
"""Render an observation window into a single self-contained HTML report.

Reads the JSONL trace written by ``observe_agenda_run.py`` plus the live
database, and lays out what the autonomous loop actually did against what the
agenda asked for. Deliberately separates three questions that have come apart
in this project before:

* did the loop move at all (jobs, stages, grants),
* did it reach compute (GPU jobs, remote execution),
* did it produce evidence (metric values that are not null).

No network, no external assets: the output opens from the local filesystem.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import database as db  # noqa: E402


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in db.fetchall(sql, params)]


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return '<p class="empty">no rows</p>'
    cols = columns or list(rows[0].keys())
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(r.get(c))}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def load_ticks(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    ticks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ticks.append(json.loads(line))
        except ValueError:
            continue
    return ticks


def agenda_section(agenda_id: int, trace: Path) -> str:
    agenda = db.fetchone(
        "SELECT id, name, token_budget, token_spent, token_reserved,"
        " gpu_hours_budget, gpu_hours_spent, required_output_json"
        " FROM research_agendas WHERE id=?",
        (agenda_id,),
    )
    if not agenda:
        return f"<section><h2>agenda {agenda_id}</h2><p class='empty'>not found</p></section>"
    agenda = dict(agenda)
    ticks = load_ticks(trace)
    first, last = (ticks[0] if ticks else {}), (ticks[-1] if ticks else {})

    # The honesty check: artifacts that carry a number versus artifacts that exist.
    artifacts = _rows(
        "SELECT artifact_type, COUNT(*) AS total, COUNT(metric_value) AS with_value"
        " FROM experiment_artifacts WHERE agenda_id=? GROUP BY 1 ORDER BY 2 DESC",
        (agenda_id,),
    )
    total_art = sum(int(r["total"]) for r in artifacts)
    valued = sum(int(r["with_value"]) for r in artifacts)
    yield_pct = round(100.0 * valued / total_art, 1) if total_art else 0.0

    milestones = [
        ("研究问题 research_problems", db.fetchone("SELECT COUNT(*) c FROM research_problems WHERE agenda_id=?", (agenda_id,))["c"]),
        ("候选想法 deep_insights", db.fetchone("SELECT COUNT(*) c FROM deep_insights WHERE agenda_id=?", (agenda_id,))["c"]),
        ("排队作业 auto_research_jobs", db.fetchone("SELECT COUNT(*) c FROM auto_research_jobs WHERE agenda_id=?", (agenda_id,))["c"]),
        ("组合决策 idea_decision_packets", db.fetchone("SELECT COUNT(*) c FROM idea_decision_packets WHERE agenda_id=?", (agenda_id,))["c"]),
        ("资源授权 resource_grants", db.fetchone("SELECT COUNT(*) c FROM resource_grants WHERE agenda_id=?", (agenda_id,))["c"]),
        ("实验运行 experiment_runs", db.fetchone("SELECT COUNT(*) c FROM experiment_runs WHERE agenda_id=?", (agenda_id,))["c"]),
        ("GPU 作业 gpu_jobs", db.fetchone("SELECT COUNT(*) c FROM gpu_jobs WHERE agenda_id=?", (agenda_id,))["c"]),
        ("结算记录 outcome_records", db.fetchone("SELECT COUNT(*) c FROM outcome_records WHERE agenda_id=?", (agenda_id,))["c"]),
    ]
    chain = "".join(
        f'<div class="step {"hit" if int(n) else "miss"}"><span class="n">{int(n)}</span>'
        f'<span class="lbl">{_esc(label)}</span></div>'
        for label, n in milestones
    )
    reached = sum(1 for _, n in milestones if int(n))

    errors = _rows(
        "SELECT id, stage, LEFT(COALESCE(last_error,''),200) AS last_error"
        " FROM auto_research_jobs WHERE agenda_id=? AND last_error IS NOT NULL"
        " ORDER BY updated_at DESC LIMIT 10",
        (agenda_id,),
    )
    return f"""
<section>
  <h2>agenda {agenda_id} &mdash; {_esc(agenda['name'])}</h2>
  <div class="kpis">
    <div class="kpi"><b>{reached}/8</b><span>链条到达阶段</span></div>
    <div class="kpi"><b>{int(agenda['token_spent'] or 0):,}</b><span>token 已花 / {int(agenda['token_budget'] or 0):,}</span></div>
    <div class="kpi"><b>{float(agenda['gpu_hours_spent'] or 0):.2f}</b><span>GPU 卡时已花 / {float(agenda['gpu_hours_budget'] or 0):.0f}</span></div>
    <div class="kpi {'good' if yield_pct >= 50 else 'bad'}"><b>{yield_pct}%</b><span>指标有真数值 ({valued}/{total_art})</span></div>
    <div class="kpi"><b>{len(ticks)}</b><span>采样点</span></div>
  </div>
  <h3>自主链条走到哪一步</h3>
  <div class="chain">{chain}</div>
  <h3>产出物诚实度</h3>
  {_table(artifacts, ['artifact_type','total','with_value'])}
  <h3>作业错误(最近 10 条)</h3>
  {_table(errors, ['id','stage','last_error'])}
  <h3>窗口首尾对照</h3>
  {_table([
      {'时点':'窗口开始','时间':first.get('at',''),'预算':json.dumps(first.get('budget',{}),ensure_ascii=False)},
      {'时点':'窗口结束','时间':last.get('at',''),'预算':json.dumps(last.get('budget',{}),ensure_ascii=False)},
  ], ['时点','时间','预算'])}
</section>"""


def fleet_section() -> str:
    workers = _rows(
        "SELECT id, gpu_model, total_mem_gb, status, heartbeat_at"
        " FROM gpu_workers WHERE id LIKE 'ssh:%' ORDER BY id"
    )
    gpu_jobs = _rows(
        "SELECT COALESCE(agenda_id::text,'(无归属)') AS agenda, status, COUNT(*) AS n,"
        " MAX(created_at) AS last FROM gpu_jobs GROUP BY 1,2 ORDER BY 3 DESC LIMIT 10"
    )
    routes = _rows(
        "SELECT role, provider, model, status, COUNT(*) AS n"
        " FROM llm_route_observations GROUP BY 1,2,3,4 ORDER BY 5 DESC LIMIT 10"
    )
    return f"""
<section>
  <h2>算力与模型路由</h2>
  <h3>GPU 机队</h3>
  {_table(workers, ['id','gpu_model','total_mem_gb','status','heartbeat_at'])}
  <h3>GPU 作业(按归属)</h3>
  {_table(gpu_jobs, ['agenda','status','n','last'])}
  <h3>LLM 路由实际调用</h3>
  {_table(routes, ['role','provider','model','status','n'])}
</section>"""


CSS = """
:root{--bg:#fff;--fg:#16181d;--mut:#5b6270;--line:#e3e6ec;--card:#f7f8fa;--good:#0a7a4a;--bad:#b3261e;--acc:#2c5fd6}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaee;--mut:#9aa2b1;--line:#2a2e37;--card:#1c1f25;--good:#4ade80;--bad:#f87171;--acc:#7aa2f7}}
*{box-sizing:border-box}
body{margin:0;padding:28px 20px 64px;background:var(--bg);color:var(--fg);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;max-width:1100px;margin-inline:auto}
h1{font-size:26px;margin:0 0 4px} h2{font-size:20px;margin:34px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--mut);margin:0 0 8px}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.kpi{flex:1 1 150px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px;line-height:1.2}
.kpi span{display:block;color:var(--mut);font-size:12px;margin-top:3px}
.kpi.good b{color:var(--good)} .kpi.bad b{color:var(--bad)}
.chain{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.step{flex:1 1 120px;border:1px solid var(--line);border-radius:8px;padding:10px;background:var(--card);text-align:center}
.step .n{display:block;font-size:20px;font-weight:700}
.step .lbl{display:block;font-size:11px;color:var(--mut);margin-top:2px}
.step.hit{border-color:var(--good)} .step.hit .n{color:var(--good)}
.step.miss{opacity:.55} .step.miss .n{color:var(--bad)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:420px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--card);font-weight:600;white-space:nowrap}
td{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.empty{color:var(--mut);font-style:italic;margin:6px 0}
footer{margin-top:40px;padding-top:14px;border-top:1px solid var(--line);color:var(--mut);font-size:12px}
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agenda", type=int, action="append", required=True,
                   help="repeatable: --agenda 8 --agenda 9")
    p.add_argument("--trace-dir", required=True,
                   help="directory holding observe_agenda<N>.jsonl")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    trace_dir = Path(args.trace_dir)
    sections = [agenda_section(a, trace_dir / f"observe_agenda{a}.jsonl") for a in args.agenda]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    doc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepGraph 自主运行观察报告</title><style>{CSS}</style></head><body>
<h1>DeepGraph 自主运行观察报告</h1>
<p class="sub">生成于 {now}｜观察方向: {', '.join('agenda ' + str(a) for a in args.agenda)}</p>
{''.join(sections)}
{fleet_section()}
<footer>由 scripts/render_observation_report.py 从实时数据库与采样轨迹生成。
"指标有真数值" 是本报告的核心诚实度指标: 它区分 "流程走完了" 与 "产出了可用证据"。</footer>
</body></html>"""
    out = Path(args.out)
    out.write_text(doc, encoding="utf-8")
    print(f"report written: {out}  ({len(doc):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
