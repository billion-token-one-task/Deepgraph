"""PaperOrchestra full pipeline (Song et al., arXiv:2604.05018 section 4)."""


def run_paperorchestra_full(*args, **kwargs):
    """Lazy wrapper to avoid import cycles with prompt and writing-standard modules."""
    from agents.paperorchestra.full_pipeline import run_paperorchestra_full as _run

    return _run(*args, **kwargs)


__all__ = ["run_paperorchestra_full"]
