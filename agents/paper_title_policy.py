"""Paper title naming policy and lightweight normalization."""

from __future__ import annotations

import re
from typing import Any, Mapping


TITLE_NAMING_STANDARD_TEXT = """Paper title naming standard:
- Prefer one of three forms:
  1. Symbolic word: Descriptive subtitle. Example: Interstellar: Beyond-Text Communication via Latent Hidden States in Multi-Agent LLM Systems
  2. Symbolic phrase: Descriptive subtitle. Use this when one word would be too vague.
  3. Method acronym: Expansion subtitle. Example: Q-VAE: Q-Guided Value-Gradient Matching for Flow-Matching VLA Policies
- The text before the colon should be memorable and short: one symbolic word, a compact symbolic phrase, or a pronounceable acronym.
- The subtitle should say what the method does and where it applies; it should not be a full hypothesis sentence.
- Avoid bare claim titles such as "X as Y", "X improves Y", "A study of X", or "Towards X".
- Avoid slashes in the title; choose one term or use "and".
- Do not use claim-heavy words such as Certified, Optimal, Universal, Fixed-Point, or State-of-the-Art unless the proof or completed benchmark evidence directly supports them.
- Keep the whole title concise, usually 7-16 words after the colon."""


CLAIM_HEAVY_TERMS = (
    "certified",
    "optimal",
    "universal",
    "state-of-the-art",
    "sota",
    "fixed-point",
    "fixed point",
)

GENERIC_PREFIXES = (
    "a study of ",
    "towards ",
    "toward ",
    "on ",
)

SYMBOLIC_NAME_BY_KEYWORD = (
    (("latent", "hidden", "communication", "inter-agent"), "Interstellar"),
    (("consensus", "refinement", "answer distribution", "fixed-point", "fixed point"), "Attractor"),
    (("routing", "gate", "budget", "selective"), "Compass"),
    (("value", "gradient", "q-guided", "flow"), "Q-VAE"),
    (("graph", "structure", "topology"), "Lattice"),
    (("memory", "retrieval", "cache"), "Mnemonic"),
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _squash(value: str) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip()


def _title_case_phrase(value: str) -> str:
    keep_upper = {"LLM", "VLA", "VAE", "QA", "RL", "AI", "GPU", "CPU"}
    small_words = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "via", "with"}
    parts = re.split(r"(\s+|-)", value)
    words = []
    word_index = 0
    previous_separator = ""
    for raw in parts:
        if raw.isspace() or raw == "-":
            words.append(raw)
            previous_separator = raw
            continue
        if not raw:
            continue
        upper = raw.upper()
        lower = raw.lower()
        if upper in keep_upper:
            words.append(upper)
        elif lower in small_words and word_index > 0 and previous_separator != "-":
            words.append(lower)
        else:
            words.append(raw[:1].upper() + raw[1:])
        word_index += 1
        previous_separator = ""
    return "".join(words)


def _extract_acronym(value: str) -> str:
    match = re.search(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b|\b[A-Z]{2,8}\b", value or "")
    return match.group(0) if match else ""


def _symbolic_name(*, raw_title: str, method_name: str, claim: str) -> str:
    acronym = _extract_acronym(method_name) or _extract_acronym(raw_title)
    if acronym:
        return acronym
    corpus = " ".join([raw_title, method_name, claim]).lower()
    for keywords, name in SYMBOLIC_NAME_BY_KEYWORD:
        if any(keyword in corpus for keyword in keywords):
            return name
    clean = re.sub(r"[^A-Za-z0-9\s-]+", " ", method_name or raw_title)
    words = [w for w in clean.split() if len(w) >= 4]
    return _title_case_phrase(words[0]) if words else "Method"


def _evidence_ready(context: Mapping[str, Any] | None) -> bool:
    context = context or {}
    if context.get("full_benchmark_completed") is True:
        return True
    packet = context.get("result_packet") if isinstance(context.get("result_packet"), Mapping) else {}
    return bool(packet.get("full_benchmark_completed"))


def _canonical_subtitle_source(value: str) -> str:
    source = _squash(value)
    lower = source.lower()
    has_consensus_refinement = "consensus/refinement" in lower or ("consensus" in lower and "refinement" in lower)
    if has_consensus_refinement and "answer distribution" in lower:
        prefix = "Benchmark-conditioned " if "benchmark-conditioned" in lower else ""
        return f"{prefix}refinement for answer distribution consensus"
    return source


def _clean_subtitle(value: str, *, evidence_ready: bool) -> str:
    subtitle = _canonical_subtitle_source(value)
    for prefix in GENERIC_PREFIXES:
        if subtitle.lower().startswith(prefix):
            subtitle = subtitle[len(prefix):].strip()
            break
    subtitle = subtitle.replace("/", " and ")
    subtitle = re.sub(r"\bas an?\b", "for", subtitle, flags=re.I)
    if not evidence_ready:
        subtitle = re.sub(r"\b[Ff]ixed[- ]?[Pp]oint\b", "Refinement", subtitle)
        subtitle = re.sub(r"\b[Cc]ertified\b", "Auditable", subtitle)
        subtitle = re.sub(r"\b[Oo]ptimal\b", "Budget-Aware", subtitle)
        subtitle = re.sub(r"\b[Uu]niversal\b", "General-Purpose", subtitle)
        subtitle = re.sub(r"\b[Ss]tate-of-the-[Aa]rt\b|\bSOTA\b", "Strong-Baseline", subtitle)
    return _title_case_phrase(subtitle)


def normalize_paper_title(
    raw_title: str | None,
    *,
    method_name: str | None = None,
    claim: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Return a title that follows the symbolic/acronym + subtitle policy."""

    raw = _squash(raw_title or "")
    method = _squash(method_name or "")
    claim_text = _squash(claim or "")
    evidence_ready = _evidence_ready(context)

    if ":" in raw:
        prefix, subtitle = raw.split(":", 1)
        prefix = _squash(prefix)
        subtitle = _clean_subtitle(subtitle, evidence_ready=evidence_ready)
        if prefix and subtitle and len(prefix.split()) <= 4 and "/" not in prefix:
            return f"{prefix}: {subtitle}"

    source = _canonical_subtitle_source(raw or claim_text or method or "Method")
    if source == raw:
        source = re.sub(r"\s+as\s+.+$", "", source, flags=re.I)
    source = re.sub(r"\bimproves?\b.+$", "", source, flags=re.I).strip() or source
    subtitle = _clean_subtitle(source, evidence_ready=evidence_ready)
    name = _symbolic_name(raw_title=raw, method_name=method, claim=claim_text)
    if name.lower() in subtitle.lower().split()[:2]:
        subtitle = _clean_subtitle(claim_text or raw or method or "A New Method", evidence_ready=evidence_ready)
    return f"{name}: {subtitle}"

