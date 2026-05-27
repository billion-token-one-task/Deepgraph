#!/usr/bin/env python3
"""Generate Kairos overview figure (readable three-column) via Gemini."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperbanana_wrapper import _ensure_paperbanana_env  # noqa: E402
from _gemini_common import PALETTE, run_gemini  # noqa: E402

STYLE_REF = ROOT / "workspace/figures/hidden_stack/fig_overview_gemini.png"
OUT = ROOT / "workspace/figures/kairos/fig_overview_gemini.png"

PROMPT = f"""
Draw a NEW publication-quality overview figure for Kairos (interval-based counterfactual temporal reasoning).

Match attached overview figure style: three readable columns, gray mini-headers, white cards, labeled charts, muted palette, block arrows between columns.
{PALETTE}

TOPIC: Kairos maps event mentions to latent intervals, decodes an event relation graph, and reranks answers with counterfactual temporal supervision.

THREE COLUMNS left-to-right with block arrows (full-width figure):

=== COLUMN 1 — subtle blue tint, header "Event Construction" ===
White card 1: sample query text with highlighted event spans e1 (blue) and e2 (amber)
White card 2: temporal marker chips (before, after, during, then)
White card 3: small table "Reliable relation labels" with rows original relation / perturbed relation

=== COLUMN 2 — subtle cream tint, header "Interval and Relation Graph" ===
White card 1: horizontal timeline with two intervals [s1,e1] blue and [s2,e2] amber, latent coordinates
White card 2: geometry feature box g_ij with endpoint comparisons (precedes / overlaps)
White card 3: event relation graph — two nodes, directed edge with relation label distribution (softmax bars), NO abstract voronoi

=== COLUMN 3 — subtle rose tint, header "Graph-Aware Selection" ===
White card 1: candidate answer pool (3 answer chips from LM)
White card 2: scoring diagram — graph embedding dotted to each candidate, consistency score S(a,G,q)
White card 3: counterfactual branch — original q vs perturbed q', relation edge flips rose, L_cf label as small formula box

Corner legend: blue Event A, amber Event B, rose counterfactual update.

STRICT: NO top figure title, NO Step 1/2/3, NO bullet lists, NO numeric benchmark scores, NO ampersand, NO abstract prototype pools.
Easy to read for ACL reviewers.
""".strip()


def main() -> int:
    _ensure_paperbanana_env()
    if not STYLE_REF.exists():
        print(f"Missing style reference: {STYLE_REF}", file=sys.stderr)
        return 2
    backup = OUT.with_name(OUT.stem + "_prev.png")
    if OUT.exists():
        shutil.copy2(OUT, backup)
        print(f"Backed up to {backup}")
    return run_gemini(style_ref=STYLE_REF, out=OUT, prompt=PROMPT)


if __name__ == "__main__":
    raise SystemExit(main())
