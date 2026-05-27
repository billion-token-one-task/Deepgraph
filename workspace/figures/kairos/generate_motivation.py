#!/usr/bin/env python3
"""Generate Kairos motivation figure (compact single-column) via Gemini."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperbanana_wrapper import _ensure_paperbanana_env  # noqa: E402
from _gemini_common import PALETTE, run_gemini  # noqa: E402

STYLE_REF = ROOT / "workspace/figures/hidden_stack/fig_motivation_gemini_v2.png"
OUT = ROOT / "workspace/figures/kairos/fig_motivation_gemini.png"

PROMPT = f"""
Draw a NEW publication-quality motivation figure for the Kairos paper on counterfactual temporal reasoning.

Match attached figure visual polish: gray section headers, white cards, shadows, readable labels, professional ACL style.
{PALETTE}

TOPIC: Counterfactual event order shifts expose temporal structure inconsistency in prompting based LLMs.

LAYOUT — COMPACT single-column (for ACL intro), top-to-bottom ONLY:
- NO three wide columns left-to-right across the full figure
- Aspect ratio roughly square or slightly tall (max 4:5), NOT an elongated poster
- Down arrows between bands only

BAND 1 — gray header "Shared Event Content"
- One compact white card: two event nodes Event A (blue) and Event B (amber) with labels
- Caption chip: "Same entities, same lexical context"

BAND 2 — gray header "Counterfactual Order Shift"
- One white card, internal left-right split OK:
  LEFT mini-panel "Original query": text line "Event A happens before Event B"
    + simple timeline bar: A block then B block (blue then amber)
  RIGHT mini-panel "Perturbed query": text line "Event A happens after Event B"
    + timeline bar: B block then A block (reversed)
- Small caption: "Non-temporal content preserved"

BAND 3 — gray header "Prediction Behavior"
- One white card, two side-by-side comparison panels:
  LEFT "Prompting (Direct / CoT)":
    - unchanged answer box "Same final answer"
    - relation label "precedes" with red unchanged icon
    - caption "Surface overlap dominates"
  RIGHT "Kairos":
    - small event relation graph with 2 nodes, edge flips from precedes to follows (rose highlight)
    - answer box may show "Updated answer" with green check
    - caption "Relation graph updates"

STRICT: NO top figure title, NO bottom banner, NO Step 1/2/3, NO bullet lists, NO numeric scores, NO ampersand.
Readable and concrete, NOT abstract star constellations or vague clouds.
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
