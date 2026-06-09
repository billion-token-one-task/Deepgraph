"""Paper text extraction.

Preferred chain for arXiv papers:
1. arXiv source / TeX
2. GROBID TEI from PDF
3. PyMuPDF PDF text fallback
"""
import gzip
import io
import logging
import re
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import httpx

from config import GROBID_BASE_URL, GROBID_REQUEST_TIMEOUT, PDF_CACHE_DIR, PDF_TEXT_BACKEND
from ingestion.grobid_tei import tei_xml_to_plaintext

logger = logging.getLogger(__name__)

MAX_MAIN_TEXT_CHARS = 80_000
MAX_APPENDIX_TEXT_CHARS = 200_000
MIN_SOURCE_TEXT_CHARS = 800
GROBID_UNAVAILABLE_COOLDOWN_SECONDS = 300
_grobid_unavailable_until = 0.0
APPENDIX_HEADING_RE = re.compile(
    r"^\s*(?:appendix(?:es)?|supplement(?:ary|al)?(?:\s+material)?|"
    r"(?:[a-z]|[ivxlcdm]+)[\.\)]?\s+appendix|appendix\s+[a-z0-9ivxlcdm]+)\s*$",
    flags=re.IGNORECASE,
)


def clean_extracted_text(text: str) -> str:
    """Normalize PDF text before storing it in SQL text columns."""
    return (text or "").replace("\x00", "")


def _safe_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", (arxiv_id or "").strip()).replace("/", "_")


def download_arxiv_source(arxiv_id: str) -> Path | None:
    """Download arXiv source package to cache. Returns None when unavailable."""
    if not arxiv_id:
        return None
    source_dir = PDF_CACHE_DIR / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_arxiv_id(arxiv_id)
    source_path = source_dir / f"{safe_id}.src"
    if source_path.exists() and source_path.stat().st_size > 0:
        return source_path

    source_id = re.sub(r"v\d+$", "", arxiv_id.strip())
    url = f"https://arxiv.org/e-print/{source_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "DeepGraph/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:
        logger.info("arXiv source unavailable for %s: %s", arxiv_id, exc)
        return None
    if not data or len(data) < 64:
        return None
    source_path.write_bytes(data)
    return source_path


def _strip_tex_comments(text: str) -> str:
    lines = []
    for raw in (text or "").splitlines():
        out = []
        escaped = False
        for ch in raw:
            if ch == "\\" and not escaped:
                escaped = True
                out.append(ch)
                continue
            if ch == "%" and not escaped:
                break
            out.append(ch)
            escaped = False
        lines.append("".join(out))
    return "\n".join(lines)


def _latex_to_plaintext(tex: str) -> str:
    """Best-effort LaTeX cleanup that preserves section/table/caption cues."""
    text = _strip_tex_comments(tex)
    text = re.sub(r"\\(input|include)\{[^}]+\}", "\n", text)
    text = re.sub(r"\\bibliography\{[^}]+\}", "\n", text)
    text = re.sub(r"\\begin\{(?:figure|table|algorithm|equation|align|gather)\*?\}", "\n", text)
    text = re.sub(r"\\end\{(?:figure|table|algorithm|equation|align|gather)\*?\}", "\n", text)
    text = re.sub(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^{}]*)\}", r"\n\n\2\n", text)
    text = re.sub(r"\\caption\{([^{}]*)\}", r"\nCaption: \1\n", text)
    text = re.sub(r"\\(title|author)\{([^{}]*)\}", r"\n\2\n", text)
    text = re.sub(r"\\cite[t|p|alp|alt|author|year]*\{[^}]*\}", "[citation]", text)
    text = re.sub(r"\\ref\{[^}]*\}", "[ref]", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\url\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{([^{}]*)\}\{([^{}]*)\}", r"\2 (\1)", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda m: m.group(1) or " ", text)
    text = text.replace("~", " ")
    text = text.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_").replace("\\#", "#")
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return clean_extracted_text(text).strip()


def _decode_bytes(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _tex_files_from_tar(data: bytes) -> list[tuple[str, str]]:
    out = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".tex"):
                continue
            if member.size > 2_000_000:
                continue
            handle = tf.extractfile(member)
            if not handle:
                continue
            out.append((member.name, _decode_bytes(handle.read())))
    return out


def _tex_files_from_zip(data: bytes) -> list[tuple[str, str]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".tex"):
                continue
            if info.file_size > 2_000_000:
                continue
            out.append((info.filename, _decode_bytes(zf.read(info))))
    return out


def _tex_files_from_source_blob(data: bytes) -> list[tuple[str, str]]:
    if not data:
        return []
    try:
        return _tex_files_from_tar(data)
    except (tarfile.TarError, EOFError, OSError):
        pass
    try:
        return _tex_files_from_zip(data)
    except (zipfile.BadZipFile, OSError):
        pass
    try:
        decompressed = gzip.decompress(data)
        text = _decode_bytes(decompressed)
    except (OSError, EOFError):
        text = _decode_bytes(data)
    if "\\documentclass" in text or "\\begin{document}" in text:
        return [("main.tex", text)]
    return []


def _ordered_tex_files(tex_files: list[tuple[str, str]]) -> list[tuple[str, str]]:
    def score(item: tuple[str, str]) -> tuple[int, str]:
        name, text = item
        lower_name = name.lower()
        main_bonus = 0 if ("main" in lower_name or "paper" in lower_name or "\\documentclass" in text) else 1
        return (main_bonus, lower_name)

    return sorted(tex_files, key=score)


def extract_text_arxiv_source(arxiv_id: str) -> str:
    """Extract plain text from arXiv source/TeX package."""
    source_path = download_arxiv_source(arxiv_id)
    if not source_path:
        return ""
    try:
        tex_files = _tex_files_from_source_blob(source_path.read_bytes())
    except OSError as exc:
        logger.info("Cannot read arXiv source for %s: %s", arxiv_id, exc)
        return ""
    if not tex_files:
        logger.info("No TeX files found in arXiv source for %s", arxiv_id)
        return ""

    parts = []
    for name, tex in _ordered_tex_files(tex_files)[:12]:
        plain = _latex_to_plaintext(tex)
        if plain:
            parts.append(f"\n\n# Source file: {name}\n{plain}")
    text = "\n".join(parts).strip()
    if len(text) < MIN_SOURCE_TEXT_CHARS:
        logger.info("arXiv source text short (%d chars) for %s", len(text), arxiv_id)
        return ""
    return clean_extracted_text(text)


def download_pdf(arxiv_id: str, pdf_url: str) -> Path:
    """Download PDF to cache. Returns path."""
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = arxiv_id.replace("/", "_")
    pdf_path = PDF_CACHE_DIR / f"{safe_id}.pdf"

    if pdf_path.exists():
        return pdf_path

    url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    req = urllib.request.Request(url, headers={"User-Agent": "DeepGraph/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(pdf_path, "wb") as f:
            f.write(resp.read())
    return pdf_path


def extract_text_grobid(pdf_path: Path) -> str:
    """
    Call GROBID processFulltextDocument and convert TEI XML to plain text.
    Returns empty string on failure (service down, timeout, bad XML).
    """
    base = (GROBID_BASE_URL or "").rstrip("/")
    if not base:
        return ""
    global _grobid_unavailable_until
    now = time.time()
    if now < _grobid_unavailable_until:
        logger.info("Skipping GROBID for %s; service in cooldown", pdf_path.name)
        return ""
    url = f"{base}/api/processFulltextDocument"
    try:
        data = pdf_path.read_bytes()
        files = {"input": (pdf_path.name, data, "application/pdf")}
        with httpx.Client(timeout=GROBID_REQUEST_TIMEOUT) as client:
            resp = client.post(url, files=files)
        if resp.status_code != 200:
            logger.warning(
                "GROBID HTTP %s for %s: %s",
                resp.status_code,
                pdf_path.name,
                (resp.text or "")[:200],
            )
            return ""
        tei = resp.text
        if not tei or "<TEI" not in tei[:2000]:
            logger.warning("GROBID returned non-TEI body for %s", pdf_path.name)
            return ""
        return tei_xml_to_plaintext(tei)
    except httpx.RequestError as e:
        _grobid_unavailable_until = time.time() + GROBID_UNAVAILABLE_COOLDOWN_SECONDS
        logger.warning("GROBID request failed for %s: %s", pdf_path.name, e)
        return ""
    except OSError as e:
        logger.warning("Cannot read PDF for GROBID %s: %s", pdf_path.name, e)
        return ""


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF (legacy text layer)."""
    try:
        import fitz  # pymupdf

        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        return ""
    except Exception:
        return ""


def extract_text(pdf_path: Path) -> str:
    """
    Full-text extraction for scientific papers.
    Backend from DEEPGRAPH_PDF_TEXT_BACKEND for PDF-only extraction:
    - auto/source_auto: GROBID first, then PyMuPDF if GROBID yields too little text
    - grobid: GROBID only
    - pymupdf: PyMuPDF only
    """
    backend = (PDF_TEXT_BACKEND or "auto").strip().lower()
    if backend not in ("auto", "source_auto", "source", "grobid", "pymupdf"):
        backend = "auto"

    if backend in ("pymupdf", "source"):
        return clean_extracted_text(extract_text_pymupdf(pdf_path))

    text = ""
    if backend in ("auto", "grobid"):
        text = extract_text_grobid(pdf_path)
        if backend == "grobid":
            return clean_extracted_text(text)
        # auto: fall back if GROBID missing or nearly empty
        if len(text.strip()) >= 500:
            return clean_extracted_text(text)
        logger.info(
            "GROBID text short (%d chars) for %s, falling back to PyMuPDF",
            len(text.strip()),
            pdf_path.name,
        )

    return clean_extracted_text(extract_text_pymupdf(pdf_path))


def split_main_and_appendix_text(text: str) -> tuple[str, str]:
    """Split extracted paper text into main body and appendix/supplement."""
    clean = clean_extracted_text(text).strip()
    if not clean:
        return "", ""

    lines = clean.splitlines()
    offset = 0
    appendix_start = None
    for line in lines:
        next_offset = offset + len(line) + 1
        stripped = line.strip()
        if stripped and len(stripped) <= 120 and APPENDIX_HEADING_RE.match(stripped):
            appendix_start = offset
            break
        offset = next_offset

    if appendix_start is not None:
        main_text = clean[:appendix_start].rstrip()
        appendix_text = clean[appendix_start:].lstrip()
    else:
        main_text = clean
        appendix_text = ""

    if len(main_text) > MAX_MAIN_TEXT_CHARS:
        overflow = main_text[MAX_MAIN_TEXT_CHARS:].lstrip()
        main_text = main_text[:MAX_MAIN_TEXT_CHARS]
        appendix_text = "\n\n".join(part for part in (overflow, appendix_text) if part).strip()

    if len(appendix_text) > MAX_APPENDIX_TEXT_CHARS:
        appendix_text = appendix_text[:MAX_APPENDIX_TEXT_CHARS]

    return main_text, appendix_text


def get_paper_text_parts(arxiv_id: str, pdf_url: str = "", abstract: str = "") -> tuple[str, str]:
    """Get paper text split into main body and appendix."""
    try:
        backend = (PDF_TEXT_BACKEND or "auto").strip().lower()
        if backend in ("auto", "source_auto", "source"):
            source_text = extract_text_arxiv_source(arxiv_id)
            if len(source_text.strip()) > 500:
                logger.info("Using arXiv source/TeX text for %s (%d chars)", arxiv_id, len(source_text))
                return split_main_and_appendix_text(source_text)
            if backend == "source":
                return abstract, ""
        pdf_path = download_pdf(arxiv_id, pdf_url)
        text = extract_text(pdf_path)
        if len(text) > 500:
            return split_main_and_appendix_text(text)
        return abstract, ""
    except Exception:
        return abstract, ""


def get_paper_text(arxiv_id: str, pdf_url: str = "", abstract: str = "") -> str:
    """Backward-compatible helper returning the main body text only."""
    main_text, _ = get_paper_text_parts(arxiv_id, pdf_url, abstract)
    return main_text
