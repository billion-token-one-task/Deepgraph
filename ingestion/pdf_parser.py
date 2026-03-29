"""PDF download and text extraction."""
import urllib.request
from pathlib import Path
from config import PDF_CACHE_DIR


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


def extract_text(pdf_path: Path) -> str:
    """Extract text from PDF using pymupdf."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        # Fallback: just return empty, will use abstract only
        return ""
    except Exception:
        return ""


def get_paper_text(arxiv_id: str, pdf_url: str = "", abstract: str = "") -> str:
    """Get full text of paper. Falls back to abstract if PDF fails."""
    try:
        pdf_path = download_pdf(arxiv_id, pdf_url)
        text = extract_text(pdf_path)
        if len(text) > 500:
            # Truncate to ~80K chars to fit in context
            return text[:80000]
        return abstract
    except Exception:
        return abstract
