"""Pluggable manuscript template adapters.

Issue #11/#12 (D1 Foundation):
- ``TemplateAdapter`` is the abstract contract every venue plugin implements.
- ``get_adapter(template_id)`` returns the singleton adapter for a venue id.
- ``list_adapters()`` returns the registered ids (for diagnostics).

The contract intentionally mirrors the four hard-coded ICLR 2026 hooks that
currently live in ``agents.paper_orchestra_pipeline``:

    pipeline call site               | adapter method
    ---------------------------------|--------------------------
    _copy_iclr2026_template_files    | copy_files(bundle_dir)
    _ensure_iclr2026_preamble        | inject_preamble(source)
    normalize_latex_source(force=..) | normalize_source(source)
    assemble_main_tex venue label    | venue_label

Plus two metadata helpers (``bibstyle_name``, ``max_pages``) that the format
linter (D3 #14) needs.

Adding a new venue = drop a module under this package, register via the
``register`` decorator, and reference its id from
``manuscript_venues/venues_v1.yaml``. No core pipeline edits required.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable, Dict, List


class TemplateAdapter(abc.ABC):
    """Abstract base for venue-specific manuscript template plugins.

    Subclasses MUST implement the five abstract methods/properties. They MAY
    override ``venue_label`` to customise the human-readable label that ends
    up on the title page (defaults to ``template_id``).

    Contract invariants enforced by tests in ``tests/test_template_adapter.py``:

    1. ``copy_files`` is idempotent and only copies files that exist on disk;
       missing optional assets must not raise.
    2. ``inject_preamble`` is idempotent: ``f(f(x)) == f(x)`` for any source.
    3. ``normalize_source`` must not crash on empty / whitespace-only input;
       it returns a trailing-newline-terminated string.
    """

    #: Stable string id used by ``venues_v1.yaml`` and ``get_adapter``.
    template_id: str = ""

    @property
    def venue_label(self) -> str:
        """Short label that ends up on the LaTeX title block / cover page."""
        return self.template_id

    @property
    @abc.abstractmethod
    def bibstyle_name(self) -> str:
        """LaTeX bibliography style command argument (e.g. ``"plain"``)."""

    @property
    @abc.abstractmethod
    def max_pages(self) -> int:
        """Soft page-count budget; FormatLinter (D3) uses this for warnings."""

    @abc.abstractmethod
    def copy_files(self, bundle_dir: Path) -> List[str]:
        """Copy venue assets (sty/bst/cls/etc.) into ``bundle_dir``.

        Returns the list of relative filenames that were actually copied so
        the manuscript runner can record provenance.
        """

    @abc.abstractmethod
    def inject_preamble(self, source: str) -> str:
        """Force-inject venue preamble into a LaTeX document body.

        MUST be idempotent: running twice on its own output is a no-op beyond
        the first call.
        """

    @abc.abstractmethod
    def normalize_source(self, source: str) -> str:
        """Apply venue-specific LaTeX cleanups (bibstyle, packages, layout).

        This is the moral equivalent of the legacy
        ``normalize_latex_source(text, force_iclr2026=...)`` call.
        """


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: Dict[str, TemplateAdapter] = {}


def register(template_id: str) -> Callable[[type], type]:
    """Class decorator that registers an adapter under ``template_id``.

    Re-registration is allowed (the latest wins); this keeps test fixtures
    that monkey-patch adapters simple.
    """

    def decorator(cls: type) -> type:
        if not issubclass(cls, TemplateAdapter):
            raise TypeError(f"{cls!r} must subclass TemplateAdapter")
        instance = cls()
        instance.template_id = template_id
        _REGISTRY[template_id] = instance
        return cls

    return decorator


def get_adapter(template_id: str) -> TemplateAdapter:
    """Return the singleton adapter for ``template_id``.

    Triggers module import on first access so users only pay for the venues
    they reference. Raises ``KeyError`` for unknown ids so callers can fall
    back / error explicitly.
    """
    _ensure_builtin_adapters_loaded()
    if template_id not in _REGISTRY:
        raise KeyError(
            f"unknown template_id {template_id!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[template_id]


def list_adapters() -> List[str]:
    """Return ids of all currently-registered adapters (sorted)."""
    _ensure_builtin_adapters_loaded()
    return sorted(_REGISTRY)


_builtins_loaded = False


def _ensure_builtin_adapters_loaded() -> None:
    """Import the ICLR + arXiv plain adapters that ship with the package."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    # imports trigger the @register side-effect
    from agents.manuscript_templates import iclr2026 as _iclr  # noqa: F401
    from agents.manuscript_templates import arxiv_plain as _arx  # noqa: F401
    _builtins_loaded = True


__all__ = [
    "TemplateAdapter",
    "register",
    "get_adapter",
    "list_adapters",
]
