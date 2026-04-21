"""Parse GROBID TEI XML into plain text for downstream LLM extraction."""

from __future__ import annotations

import xml.etree.ElementTree as ET

# TEI P5 namespace used by GROBID
_TEI_NS = "http://www.tei-c.org/ns/1.0"


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def tei_xml_to_plaintext(tei_xml: str | bytes) -> str:
    """
    Extract readable body text from GROBID processFulltextDocument TEI XML.
    Preserves paragraph / heading breaks where possible.
    """
    if isinstance(tei_xml, bytes):
        data = tei_xml
    else:
        data = tei_xml.encode("utf-8")
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return ""

    body = root.find(f".//{{{_TEI_NS}}}body")
    if body is None:
        for el in root.iter():
            if _local_tag(el.tag) == "body":
                body = el
                break
    if body is None:
        return ""

    # Prefer block elements; avoid standalone <formula> (usually inside <p> and would duplicate)
    block_tags = frozenset({"p", "head", "figDesc", "item"})
    parts: list[str] = []
    for el in body.iter():
        if _local_tag(el.tag) not in block_tags:
            continue
        t = "".join(el.itertext()).strip()
        if t:
            parts.append(t)

    if parts:
        return "\n\n".join(parts)

    # Fallback: all text under body (single block)
    raw = "".join(body.itertext())
    return " ".join(raw.split()).strip()
