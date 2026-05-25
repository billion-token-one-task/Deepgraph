"""Tests for the LaTeX compile-error classifier and surgical fixes."""

from __future__ import annotations

import pytest

from agents.manuscript_compile_repair import (
    _drop_usepackage,
    _surrogate_apacite,
    _surgical_attempt,
    CompileErrorClassification,
    classify_log_error,
)


_LOG_MISSING_PACKAGE = r"""
This is pdfTeX, Version 3.14159265-2.6-1.40.21
( /usr/share/texlive/texmf-dist/tex/latex/base/article.cls
( /home/user/main.aux )
! LaTeX Error: File `cleveref.sty' not found.

Type X to quit or <RETURN> to proceed,
l.42 \usepackage
              {cleveref}
"""

_LOG_UNDEFINED_CS = r"""
! Undefined control sequence.
l.227 We follow \citeA
                {garcia2022}.
"""

_LOG_BRACE = r"""
! Missing $ inserted.
<inserted text>
                $
l.181 $\Delta = ||v_{seen} - v_{generated}||
                                            $
"""

_LOG_ENV = r"""
! LaTeX Error: Environment apacite undefined.

l.18 \begin{apacite}
"""


def test_classify_missing_package() -> None:
    cls = classify_log_error(_LOG_MISSING_PACKAGE)
    assert cls is not None
    assert cls.kind == "missing_package_file"
    assert cls.missing_pkg == "cleveref"


def test_classify_undefined_cs_finds_line() -> None:
    cls = classify_log_error(_LOG_UNDEFINED_CS)
    assert cls is not None
    assert cls.kind == "undefined_control_sequence"
    assert cls.error_line == 227


def test_classify_bad_char_brace() -> None:
    cls = classify_log_error(_LOG_BRACE)
    assert cls is not None
    assert cls.kind in {"bad_char_or_alignment", "brace_mismatch"}


def test_classify_undefined_environment() -> None:
    cls = classify_log_error(_LOG_ENV)
    assert cls is not None
    assert cls.kind == "missing_environment"
    assert cls.missing_env == "apacite"


def test_surgical_drops_apacite_and_rewrites_citeA() -> None:
    tex = (
        r"\documentclass{article}" + "\n"
        r"\usepackage{apacite}" + "\n"
        r"\begin{document}" + "\n"
        r"As shown by \citeA{garcia2022}, the result holds." + "\n"
        r"\end{document}"
    )
    cls = CompileErrorClassification(
        kind="missing_package_file",
        message="missing LaTeX package file: apacite.sty",
        missing_pkg="apacite",
    )
    out, notes = _surgical_attempt(tex, cls)
    assert r"\usepackage{apacite}" not in out
    assert r"\citeA" not in out
    assert r"\cite{garcia2022}" in out
    assert notes  # one or more notes recorded


def test_surgical_escapes_stray_chars() -> None:
    tex = "line1\nThis paper uses % for partition signal & accepts _input.\nline3"
    cls = CompileErrorClassification(
        kind="bad_char_or_alignment",
        message="Misplaced alignment tab character &",
        error_line=2,
    )
    out, notes = _surgical_attempt(tex, cls)
    assert "\\%" in out
    assert "\\&" in out
    assert "\\_" in out


def test_drop_usepackage_handles_options() -> None:
    tex = r"\usepackage[options]{wasysym}" + "\nbody"
    out, changed = _drop_usepackage(tex, "wasysym")
    assert changed
    assert "wasysym" not in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
