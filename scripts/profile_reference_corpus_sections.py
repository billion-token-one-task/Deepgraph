#!/usr/bin/env python3
"""Profile section length and structure from workspace/pdfs reference corpus.

Writes/updates prompts/venue_styles/_EMPIRICAL_STATS.md when run with --write-stats.
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "workspace" / "pdfs"
STATS_MD = REPO / "prompts" / "venue_styles" / "_EMPIRICAL_STATS.md"

SECTION_MARKERS = [
    (r"(?i)\babstract\b", "abstract_start"),
    (r"(?i)(?:\n\s*1\s+introduction|\nintroduction)\b", "intro"),
    (r"(?i)\n\s*(?:2\s+)?related work", "related_work"),
    (r"(?i)\n\s*(?:2\s+)?background", "background"),
    (r"(?i)\n\s*(?:3\s+)?(?:method|approach|model|framework|architecture)\b", "method"),
    (r"(?i)\n\s*(?:4\s+)?(?:experiment|evaluation|empirical|results)\b", "experiments"),
    (r"(?i)\n\s*(?:5\s+)?(?:discussion|analysis)\b", "discussion"),
    (r"(?i)\n\s*(?:6\s+)?conclusion", "conclusion"),
    (r"(?i)\n\s*references\b", "references"),
    (r"(?i)\n\s*bibliography\b", "bibliography"),
]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z0-9\-']*", text))


def _visual_line_count(text: str, chars_per_line: int) -> int:
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    total = 0
    for p in paras:
        words = p.split()
        if not words:
            continue
        line_len = 0
        lines = 1
        for w in words:
            wl = len(w) + (1 if line_len else 0)
            if line_len + wl > chars_per_line:
                lines += 1
                line_len = len(w)
            else:
                line_len += wl
        total += lines
    return total


def _summarize(values: list[float]) -> dict[str, float | int]:
    s = sorted(values)
    n = len(s)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "median": float(statistics.median(s)),
        "mean": float(statistics.mean(s)),
        "p25": float(s[n // 4]),
        "p75": float(s[(3 * n) // 4]),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def _extract_sections(text: str, max_chars: int = 120_000) -> dict[str, str]:
    text = text[:max_chars]
    positions: list[tuple[int, str]] = []
    for pat, name in SECTION_MARKERS:
        for m in re.finditer(pat, text):
            positions.append((m.start(), name))
    positions.sort()
    filtered: list[tuple[int, str]] = []
    for pos, name in positions:
        if filtered and filtered[-1][1] == name:
            continue
        filtered.append((pos, name))
    chunks: dict[str, str] = {}
    for i, (pos, name) in enumerate(filtered):
        end = filtered[i + 1][0] if i + 1 < len(filtered) else len(text)
        chunk = text[pos:end]
        if name == "abstract_start":
            chunk = re.sub(r"(?i)^\s*abstract\s*", "", chunk)
            for pat in (r"(?i)\n\s*1\s+introduction", r"(?i)\nintroduction\b"):
                m = re.search(pat, chunk)
                if m:
                    chunk = chunk[: m.start()]
            chunks["abstract"] = chunk.strip()
        elif name == "intro":
            chunk = re.sub(r"(?i)^\s*(?:1\s+)?introduction\s*", "", chunk)
            chunks["intro"] = chunk.strip()
        elif name in ("related_work", "background"):
            if "related_work" not in chunks:
                chunks["related_work"] = re.sub(
                    r"(?i)^\s*(?:2\s+)?(?:related work|background)\s*", "", chunk
                ).strip()
        elif name == "method" and "method" not in chunks:
            chunks["method"] = re.sub(
                r"(?i)^\s*(?:\d+\s+)?(?:method|approach|model|framework|architecture)\s*",
                "",
                chunk,
            ).strip()
        elif name == "experiments" and "experiments" not in chunks:
            chunks["experiments"] = re.sub(
                r"(?i)^\s*(?:\d+\s+)?(?:experiment|evaluation|empirical|results)s?\s*",
                "",
                chunk,
            ).strip()
        elif name == "conclusion" and "conclusion" not in chunks:
            body = re.sub(r"(?i)^\s*(?:\d+\s+)?conclusion\s*", "", chunk)
            cl = body.lower()
            for pat in (
                r"\n\s*references\b",
                r"\n\s*bibliography\b",
                r"\n\s*acknowledg",
                r"\n\s*appendix\b",
                r"\n\s*ethical",
            ):
                m_end = re.search(pat, cl)
                if m_end:
                    body = body[: m_end.start()]
            chunks["conclusion"] = body.strip()
    return chunks


def profile_corpus(corpus_dir: Path, *, sample_size: int, seed: int) -> dict:
    import random

    pdfs = [p for p in sorted(corpus_dir.glob("*.pdf")) if p.is_file() and p.stat().st_size > 1000]
    random.seed(seed)
    sample = pdfs if len(pdfs) <= sample_size else random.sample(pdfs, sample_size)
    keys = ("abstract", "intro", "related_work", "method", "experiments", "conclusion")
    buckets = {k: {"words": [], "lines85": [], "lines52": []} for k in keys}
    title_words: list[int] = []
    intro_contrib = 0
    intro_total = 0
    rw_multi = 0
    rw_total = 0
    exp_sub_counts: list[int] = []

    for path in sample:
        try:
            doc = fitz.open(str(path))
            text = ""
            for i in range(min(12, doc.page_count)):
                text += doc.load_page(i).get_text("text") + "\n"
            doc.close()
        except Exception:
            continue
        pre = text[:4000]
        lines = [ln.strip() for ln in pre.splitlines() if len(ln.strip()) > 15]
        if lines:
            title_words.append(len(lines[0].split()))
        sec = _extract_sections(text)
        for key in keys:
            body = sec.get(key, "")
            wc = _word_count(body)
            if key == "abstract":
                if wc < 40 or wc > 600:
                    continue
            elif wc < 80:
                continue
            if key == "experiments" and wc > 12_000:
                body = body[:30_000]
            if key == "conclusion" and wc > 400:
                continue
            buckets[key]["words"].append(float(wc))
            buckets[key]["lines85"].append(float(_visual_line_count(body, 85)))
            buckets[key]["lines52"].append(float(_visual_line_count(body, 52)))
        intro_body = sec.get("intro", "")
        if _word_count(intro_body) > 150:
            intro_total += 1
            if re.search(
                r"(?i)(?:contribution|we propose|we introduce|we present|we show|we demonstrate|our main)",
                intro_body,
            ):
                intro_contrib += 1
        rw_body = sec.get("related_work", "")
        if _word_count(rw_body) > 100:
            rw_total += 1
            if len(re.findall(r"(?i)(?:related work|background).{0,200}", rw_body)) >= 1:
                # count numbered subsection headers like "2.1" or capitalized short lines
                heads = len(re.findall(r"\n\s*\d+\.\d+\s+\w", rw_body))
                if heads >= 2 or _word_count(rw_body) > 400:
                    rw_multi += 1
        exp_body = sec.get("experiments", "")
        if _word_count(exp_body) > 150:
            exp_sub_counts.append(len(re.findall(r"\n\s*\d+\.\d+\s+", exp_body)))

    stats: dict = {
        "corpus_dir": str(corpus_dir),
        "pdf_count": len(pdfs),
        "sample_size": len(sample),
        "title_words": _summarize([float(x) for x in title_words]),
        "intro_contribution_rate": (
            float(intro_contrib) / intro_total if intro_total else 0.0
        ),
        "related_work_multi_cluster_rate": (
            float(rw_multi) / rw_total if rw_total else 0.0
        ),
        "sections": {},
    }
    for key in keys:
        stats["sections"][key] = {
            "words": _summarize(buckets[key]["words"]),
            "lines_single_col_85cpl": _summarize(buckets[key]["lines85"]),
            "lines_two_col_52cpl": _summarize(buckets[key]["lines52"]),
        }
    stats["experiments_numbered_subsections"] = _summarize([float(x) for x in exp_sub_counts])
    return stats


def _fmt_row(label: str, block: dict) -> str:
    w = block.get("words") or {}
    l85 = block.get("lines_single_col_85cpl") or {}
    if not w.get("n"):
        return f"| {label} | — | — | — |"
    return (
        f"| {label} | **{w['median']:.0f}** ({w['p25']:.0f}–{w['p75']:.0f}) | "
        f"{l85.get('median', 0):.0f} ({l85.get('p25', 0):.0f}–{l85.get('p75', 0):.0f}) | {w['n']:.0f} |"
    )


def render_stats_markdown(stats: dict) -> str:
    tw = stats.get("title_words") or {}
    lines = [
        "# 参考论文库实测篇幅（自动生成，勿手填）",
        "",
        f"- **语料**：`{stats['corpus_dir']}`，共 **{stats['pdf_count']}** 篇 PDF",
        f"- **抽样**：n={stats['sample_size']}（seed=42）",
        "- **方法**：PyMuPDF 前 12 页；按 Abstract / Introduction / Related Work / Method / Experiments / Conclusion 分段",
        "- **行数**：按排版换行估算（**不是** `.tex` 源码行数）",
        "- **复现**：`python scripts/profile_reference_corpus_sections.py --write-stats`",
        "",
        "## 标题（PDF 首页首行）",
        "",
        f"- 词数 median **{tw.get('median', 0):.0f}**，p25–p75 **{tw.get('p25', 0):.0f}–{tw.get('p75', 0):.0f}**（n={tw.get('n', 0):.0f}）",
        "",
        "## 主文各节词数与单栏行数（median, p25–p75）",
        "",
        "| 章节 | 词数 | 单栏行数 @85 字符/行 | 样本数 |",
        "|------|------|----------------------|--------|",
    ]
    labels = {
        "abstract": "Abstract",
        "intro": "Introduction",
        "related_work": "Related Work",
        "method": "Method",
        "experiments": "Experiments",
        "conclusion": "Conclusion",
    }
    for key, label in labels.items():
        lines.append(_fmt_row(label, stats["sections"].get(key, {})))
    lines += [
        "",
        "## 结构信号（同批样本）",
        "",
        f"- Introduction 含 contribution / we propose 类表述比例：**{100*stats.get('intro_contribution_rate', 0):.0f}%**",
        f"- Related Work 呈多簇/多子主题结构比例：**{100*stats.get('related_work_multi_cluster_rate', 0):.0f}%**",
        "",
        "## 与 DeepGraph 当前草稿差距（典型问题，非语料）",
        "",
        "| 章节 | 参考 median | idea_2 草稿约 |",
        "|------|-------------|---------------|",
        "| Abstract | ~183 词 / ~22 行 | ~105 词 |",
        "| Introduction | ~659 词 / ~101 行 | ~198 词，**无 Contributions itemize** |",
        "| Related Work | ~540 词 / ~93 行 | ~185 词，无主题子节 |",
        "| Method | ~862 词 | ~282 词，子节堆砌无 Overview |",
        "| Experiments | ~473 词 | ~120 词，**无主结果表** |",
        "",
        "## LaTeX 写作注意",
        "",
        "- `\\begin{abstract}` 在源码中常为 **1 段**；目标 **140–250 词**，编译后单栏约 **16–31 行**。",
        "- 禁止按源码 1 行判断摘要长度。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-stats", action="store_true")
    args = ap.parse_args()
    if not args.corpus.exists():
        raise SystemExit(f"corpus missing: {args.corpus}")
    stats = profile_corpus(args.corpus, sample_size=args.sample, seed=args.seed)
    md = render_stats_markdown(stats)
    if args.write_stats:
        STATS_MD.write_text(md, encoding="utf-8")
        print(f"wrote {STATS_MD}")
    print(md)


if __name__ == "__main__":
    main()
