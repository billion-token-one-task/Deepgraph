"""Per-venue manuscript style requirement documents.

Each registered ``template_id`` may have a companion markdown file under
``prompts/venue_styles/{template_id}.md``. These guides are injected into
PaperOrchestra ``conference_guidelines`` so outline / section / refinement
agents follow venue-specific structure, length, and rhetoric norms.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config import PROJECT_ROOT

VENUE_STYLES_DIR = PROJECT_ROOT / "prompts" / "venue_styles"

# Fallback when a venue has no dedicated file yet.
_GENERIC_ML_GUIDE = VENUE_STYLES_DIR / "_generic_ml_conference.md"


def _read_guide_part(name: str) -> str:
    path = VENUE_STYLES_DIR / name
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


@lru_cache(maxsize=32)
def load_venue_style_guide(template_id: str) -> str:
    """Return merged guide: empirical stats + section framework + venue overlay."""
    tid = (template_id or "iclr2026").strip()
    parts: list[str] = []
    for name in (
        "_EMPIRICAL_STATS.md",
        "_SECTION_WRITING_FRAMEWORK.md",
        f"{tid}.md",
    ):
        body = _read_guide_part(name)
        if body:
            parts.append(body)
    if parts:
        return "\n\n---\n\n".join(parts)
    if _GENERIC_ML_GUIDE.is_file():
        return _read_guide_part("_generic_ml_conference.md")
    return (
        f"No style guide for template_id={tid!r}. "
        "See prompts/venue_styles/_SECTION_WRITING_FRAMEWORK.md."
    )


def build_venue_style_guidelines_block(template_id: str) -> str:
    """Wrap the loaded guide for injection into agent prompts."""
    body = load_venue_style_guide(template_id)
    return (
        f"=== Venue style requirements ({template_id}) ===\n"
        "The following rules override generic PaperOrchestra defaults when they conflict.\n\n"
        f"{body}\n"
        "=== End venue style requirements ===\n"
    )
