"""Tests for the DB-backed content-corpus audit (R5).

These tests build a tiny synthetic deepgraph.db so we never touch the real
project database.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agents.manuscript_corpus_content_audit import (
    audit_against_db_corpus,
    diff_against_neighbors,
    parse_section_skeleton,
)
from agents.manuscript_corpus_content_audit import NeighborPaperProfile, SectionSkeleton


def test_parse_section_skeleton_handles_numbered_headings() -> None:
    text = (
        "Authors and affiliation block\n\n"
        "1. INTRODUCTION\n"
        "Recent vision-language models have grown rapidly. We motivate a new approach.\n\n"
        "2. RELATED WORK\n"
        "Prior work on calibration includes Gao et al. and Smith et al.\n\n"
        "3. METHOD\n"
        "We propose a routing layer on top of frozen embeddings.\n\n"
        "4. EXPERIMENTS\n"
        "We evaluate on GSM8K, 2Wiki, HotpotQA. Results show gains.\n\n"
        "References\n[1] Garcia et al.\n"
    )
    sections = parse_section_skeleton(text)
    families = {s.family for s in sections}
    assert "intro" in families
    assert "related_work" in families
    assert "method" in families
    assert "experiments" in families
    assert all(s.word_count > 0 for s in sections)


def test_parse_section_skeleton_handles_latex_headings() -> None:
    text = (
        r"\section{Introduction}"
        "\n"
        "We motivate the problem of routing. " * 30
        + "\n\n"
        r"\section{Method}"
        "\n"
        "Our method routes embeddings. " * 30
    )
    sections = parse_section_skeleton(text)
    assert {s.family for s in sections} >= {"intro", "method"}


def test_diff_flags_thin_method_against_strong_neighbors() -> None:
    main_tex = (
        r"\section{Introduction}" + "\n"
        + "We motivate. " * 200
        + "\n"
        + r"\section{Method}" + "\n"
        + "Tiny method. " * 5
        + "\n"
        + r"\section{Experiments}" + "\n"
        + "We evaluate on benchmarks. " * 200
    )
    neighbor = NeighborPaperProfile(
        paper_id="x",
        title="Strong neighbor",
        sections=[
            SectionSkeleton(name="Introduction", family="intro", word_count=600),
            SectionSkeleton(name="Method", family="method", word_count=1200),
            SectionSkeleton(name="Experiments", family="experiments", word_count=900),
        ],
        total_words=2700,
        citation_marker_count=40,
    )
    diff = diff_against_neighbors(main_tex, [neighbor] * 4)
    kinds = {iss["kind"] for iss in diff["issues"]}
    assert "thin_family_method" in kinds


def test_audit_against_db_corpus_with_synthetic_db(tmp_path: Path) -> None:
    db_path = tmp_path / "tiny.db"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            full_text TEXT
        );
        CREATE TABLE paper_taxonomy (
            paper_id TEXT,
            node_id TEXT,
            PRIMARY KEY (paper_id, node_id)
        );
        CREATE TABLE deep_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_paper_ids TEXT,
            supporting_papers TEXT,
            source_node_ids TEXT
        );
        """
    )
    neighbor_text = (
        "1. INTRODUCTION\n"
        + ("We motivate. " * 100)
        + "\n2. METHOD\n"
        + ("Our routing layer is principled. " * 200)
        + "\n3. EXPERIMENTS\n"
        + ("We evaluate on benchmarks. " * 150)
    )
    con.execute("INSERT INTO papers VALUES (?,?,?)", ("p1", "Neighbor", neighbor_text))
    con.execute("INSERT INTO papers VALUES (?,?,?)", ("p2", "Neighbor2", neighbor_text))
    con.execute(
        "INSERT INTO deep_insights (source_paper_ids, supporting_papers, source_node_ids) "
        "VALUES (?,?,?)",
        (json.dumps(["p1", "p2"]), "[]", "[]"),
    )
    insight_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.commit()
    con.close()

    main_tex = (
        r"\section{Introduction}" + "\n"
        + "We motivate. " * 30
        + "\n"
        + r"\section{Method}" + "\n"
        + "Tiny. " * 5
        + "\n"
        + r"\section{Experiments}" + "\n"
        + "We evaluate. " * 30
    )
    report = audit_against_db_corpus(
        main_tex=main_tex,
        deep_insight_id=insight_id,
        db_path=db_path,
    )
    assert report["available"]
    assert "thin_family_method" in {iss["kind"] for iss in report["issues"]}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
