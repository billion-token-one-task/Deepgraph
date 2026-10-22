"""Official PaperOrchestra agent prompts (verbatim .tex from arXiv:2604.05018 source).

Files live under ``prompts/paper_orchestra/*.tex`` (copied from the paper's TeX submission).
"""

from __future__ import annotations

import os

from config import PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / "prompts" / "paper_orchestra"

# Timeline rule in Song et al.; override via env if needed.
CUTOFF_DATE = os.getenv("DEEPGRAPH_PAPERORCHESTRA_CUTOFF_DATE", "2026-04-01").strip()
