"""Ground extracted claims/results against full paper text (cite-and-verify, no extra LLM)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Minimum quote length for strict matching (avoid trivial false positives)
MIN_QUOTE_CHARS = 12
# Below this score, grounding_status is not "verified"
WEAK_THRESHOLD = 0.72


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def locate_quote_span(full_text: str, quote: str) -> tuple[int, int] | None:
    """Return (start, end) in full_text if quote is found verbatim or with flexible whitespace."""
    if not full_text or not quote:
        return None
    q = quote.strip()
    if len(q) < MIN_QUOTE_CHARS:
        return None

    idx = full_text.find(q)
    if idx >= 0:
        return idx, idx + len(q)

    # Flexible: same words, any whitespace run (handles PDF line breaks)
    words = [w for w in re.split(r"\s+", q) if w]
    if len(words) < 4:
        return None
    try:
        pattern = r"\s+".join(re.escape(w) for w in words)
        m = re.search(pattern, full_text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.start(), m.end()
    except re.error:
        pass

    # Last resort: normalized single-space form
    nq = _norm_ws(q)
    if len(nq) < MIN_QUOTE_CHARS:
        return None
    nf = _norm_ws(full_text)
    j = nf.find(nq)
    if j < 0:
        return None
    # Map position back to full_text approximately: scan for same prefix in original
    prefix = nq[: min(40, len(nq))]
    k = full_text.find(prefix)
    if k >= 0:
        end = min(k + len(q) + 80, len(full_text))
        return k, end

    return None


def _fuzzy_best_window(full_text: str, claim_text: str, quote: str) -> tuple[float, int, int] | None:
    """Return (ratio, start, end) for best-matching window of similar length to quote."""
    if not full_text or not claim_text:
        return None
    ref = _norm_ws(quote or claim_text)
    if len(ref) < MIN_QUOTE_CHARS:
        return None
    window = len(ref) + 40
    best: tuple[float, int, int] | None = None
    step = max(30, window // 3)
    i = 0
    while i + MIN_QUOTE_CHARS <= len(full_text):
        chunk = full_text[i : i + window]
        r = SequenceMatcher(None, ref.lower(), _norm_ws(chunk).lower()).ratio()
        if not best or r > best[0]:
            best = (r, i, min(i + window, len(full_text)))
        i += step
    return best


def score_grounding(
    full_text: str,
    claim_text: str,
    source_quote: str | None,
) -> tuple[str, float, int | None, int | None]:
    """
    Returns (grounding_status, grounding_score, char_start, char_end).
    Status: verified | weak | unverified | no_quote
    """
    quote = (source_quote or "").strip()
    if not quote:
        return "no_quote", 0.0, None, None

    span = locate_quote_span(full_text, quote)
    if span:
        a, b = span
        return "verified", 1.0, a, b

    fb = _fuzzy_best_window(full_text, claim_text, quote)
    if fb:
        ratio, a, b = fb
        if ratio >= 0.88:
            return "weak", ratio, a, b
        if ratio >= WEAK_THRESHOLD:
            return "weak", ratio, a, b
        return "unverified", ratio, None, None

    return "unverified", 0.0, None, None


def score_grounding_in_sources(
    sources: dict[str, str],
    claim_text: str,
    source_quote: str | None,
) -> tuple[str, float, str | None, int | None, int | None]:
    """Search multiple text regions and return the best grounding result."""
    quote = (source_quote or "").strip()
    if not quote:
        return "no_quote", 0.0, None, None, None

    for source_name, text in sources.items():
        if not text:
            continue
        span = locate_quote_span(text, quote)
        if span:
            a, b = span
            return "verified", 1.0, source_name, a, b

    best_source = None
    best_result = None
    for source_name, text in sources.items():
        if not text:
            continue
        fb = _fuzzy_best_window(text, claim_text, quote)
        if not fb:
            continue
        if best_result is None or fb[0] > best_result[0]:
            best_result = fb
            best_source = source_name

    if best_result and best_source:
        ratio, a, b = best_result
        if ratio >= 0.88 or ratio >= WEAK_THRESHOLD:
            return "weak", ratio, best_source, a, b
        return "unverified", ratio, None, None, None

    return "unverified", 0.0, None, None, None


def apply_claim_grounding(claim: dict, full_text: str, appendix_text: str = "") -> dict:
    """Mutate claim dict with grounding_* fields (does not remove claim_text)."""
    status, score, source_field, cs, ce = score_grounding_in_sources(
        {"full_text": full_text, "appendix_text": appendix_text},
        claim.get("claim_text") or "",
        claim.get("source_quote"),
    )
    claim["grounding_status"] = status
    claim["grounding_score"] = round(score, 4)
    claim["grounding_source_field"] = source_field
    if cs is not None and ce is not None:
        claim["char_start"] = cs
        claim["char_end"] = ce
    else:
        claim["char_start"] = None
        claim["char_end"] = None
    return claim


def apply_result_grounding(row: dict, full_text: str, appendix_text: str = "") -> dict:
    """Mutate a results[] row with grounding fields."""
    ref_line = " ".join(
        str(x)
        for x in (
            row.get("method_name"),
            row.get("dataset_name"),
            row.get("metric_name"),
            row.get("metric_value"),
        )
        if x is not None and str(x) != ""
    )
    status, score, source_field, cs, ce = score_grounding_in_sources(
        {"full_text": full_text, "appendix_text": appendix_text},
        ref_line,
        row.get("source_quote"),
    )
    row["grounding_status"] = status
    row["grounding_score"] = round(score, 4)
    row["grounding_source_field"] = source_field
    if cs is not None and ce is not None:
        row["char_start"] = cs
        row["char_end"] = ce
    else:
        row["char_start"] = None
        row["char_end"] = None
    return row
