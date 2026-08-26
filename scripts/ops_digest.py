#!/usr/bin/env python3
"""Print the operational status the public front page no longer carries.

The overview page used to double as the ops console: six lifecycle domains,
the reconciliation backlog, orphaned grants, harvest timestamps. That made the
front door of a public site a live readout of how the deployment is run, and
the same figures were reachable over the open internet through the endpoints
behind them.

The figures themselves are not the problem -- on-call needs every one of them
to tell a stalled harvest from a revoked grant. They just belong on the host,
not on the web. This prints them, in-process, reading exactly the sources the
page used to read, so nothing was lost in the move.

Run it on the host:
    python scripts/ops_digest.py
    python scripts/ops_digest.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DOMAINS = (
    ("corpus", "语料 Corpus"),
    ("research_runtime", "研究运行时 Research runtime"),
    ("scoped_ingestion", "限定范围摄取 Scoped ingestion"),
    ("legacy_paper_ingestion", "旧版论文摄取 Legacy ingestion"),
    ("harvest", "文献抓取 Harvest"),
    ("backfill", "回填 Backfill"),
)

# Counters that live in /api/stats rather than in a lifecycle domain.
STAT_LINES = (
    ("papers_total", "论文语料记录 Corpus records"),
    ("papers_processed", "已建成证据图 Graph built"),
    ("papers_pending", "待继续分析 Pending analysis"),
    ("papers_error", "处理错误 Processing errors"),
    ("adjudication_candidates", "待人工对账 Awaiting adjudication"),
)


def _collect() -> dict:
    from web.app import app, _stats_cache

    if _stats_cache.get() is None:
        _stats_cache.prewarm()
    client = app.test_client()
    processing = client.get("/api/processing")
    stats = client.get("/api/stats")
    return {
        "processing": processing.get_json() if processing.status_code == 200 else None,
        "processing_status": processing.status_code,
        "stats": stats.get_json() if stats.status_code == 200 else None,
        "stats_status": stats.status_code,
    }


def _render(data: dict) -> str:
    out: list[str] = []
    processing = data.get("processing") or {}
    stats = data.get("stats") or {}

    out.append("DeepGraph ops digest")
    out.append("=" * 60)
    contract = processing.get("contract_version")
    out.append(f"contract        : {contract or 'UNAVAILABLE'}")
    out.append(f"pipeline_state  : {processing.get('pipeline_state') or '-'}")
    out.append("")

    out.append("-- 计数 counts " + "-" * 45)
    for key, label in STAT_LINES:
        value = stats.get(key)
        out.append(f"  {label:<40} {value if value is not None else '-'}")
    out.append("")

    out.append("-- 六个领域 lifecycle domains " + "-" * 31)
    for key, label in DOMAINS:
        domain = processing.get(key)
        if not isinstance(domain, dict):
            out.append(f"  {label:<40} UNAVAILABLE")
            continue
        state = domain.get("state") or "-"
        out.append(f"  {label:<40} {state}")
        # Every remaining key is a diagnostic someone on call may need, so
        # print them all rather than curating a list that goes stale.
        for field in sorted(domain):
            if field == "state":
                continue
            value = domain[field]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            out.append(f"      {field:<36} {value}")
    out.append("")

    papers = processing.get("papers")
    if isinstance(papers, list):
        out.append(f"-- 正在处理 in flight ({len(papers)}) " + "-" * 30)
        for paper in papers:
            out.append(f"  {paper.get('status','-'):<12} {paper.get('id','-'):<14} {paper.get('title','')}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the raw payloads instead of text")
    args = parser.parse_args()

    data = _collect()
    if data["processing"] is None or data["stats"] is None:
        print(
            f"ops digest unavailable (processing={data['processing_status']}, "
            f"stats={data['stats_status']})",
            file=sys.stderr,
        )
        return 1
    if args.json:
        print(json.dumps({"processing": data["processing"], "stats": data["stats"]},
                         ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_render(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
