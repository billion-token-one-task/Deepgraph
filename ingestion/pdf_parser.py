"""PDF download and text extraction (GROBID TEI preferred; PyMuPDF fallback)."""
import logging
import re
import urllib.request
from pathlib import Path

import httpx

from config import GROBID_BASE_URL, GROBID_REQUEST_TIMEOUT, PDF_CACHE_DIR, PDF_TEXT_BACKEND
from ingestion.grobid_tei import tei_xml_to_plaintext

logger = logging.getLogger(__name__)

MAX_MAIN_TEXT_CHARS = 80_000
MAX_APPENDIX_TEXT_CHARS = 200_000
APPENDIX_HEADING_RE = re.compile(
    r"^\s*(?:appendix(?:es)?|supplement(?:ary|al)?(?:\s+material)?|"
    r"(?:[a-z]|[ivxlcdm]+)[\.\)]?\s+appendix|appendix\s+[a-z0-9ivxlcdm]+)\s*$",
    flags=re.IGNORECASE,
)


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
    Backend from DEEPGRAPH_PDF_TEXT_BACKEND:
    - auto: GROBID first, then PyMuPDF if GROBID yields too little text
    - grobid: GROBID only
    - pymupdf: PyMuPDF only
    """
    backend = (PDF_TEXT_BACKEND or "auto").strip().lower()
    if backend not in ("auto", "grobid", "pymupdf"):
        backend = "auto"

    if backend == "pymupdf":
        return extract_text_pymupdf(pdf_path)

    text = ""
    if backend in ("auto", "grobid"):
        text = extract_text_grobid(pdf_path)
        if backend == "grobid":
            return text
        # auto: fall back if GROBID missing or nearly empty
        if len(text.strip()) >= 500:
            return text
        logger.info(
            "GROBID text short (%d chars) for %s, falling back to PyMuPDF",
            len(text.strip()),
            pdf_path.name,
        )

    return extract_text_pymupdf(pdf_path)


def split_main_and_appendix_text(text: str) -> tuple[str, str]:
    """Split extracted paper text into main body and appendix/supplement."""
    clean = (text or "").strip()
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
