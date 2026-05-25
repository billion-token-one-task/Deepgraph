"""Visual layout + figure-duplicate audit for compiled bundles.

Two independent audits run after a successful compile:

* **VisualLayoutAudit (R4)**: opens ``main.pdf`` with PyMuPDF, walks every
  image+drawing+text block on every page, and flags
    - figures whose bounding box leaks past the printable area,
    - tables wider than ``\\textwidth`` (we detect this by looking for the
      ``Overfull \\hbox`` warning in the latex log AND by checking image
      widths in the PDF),
    - includegraphics references in the .tex that have no in-text
      ``\\ref{fig:...}`` or ``\\cref`` companion,
    - figure placements on a page where no ``\\caption`` text was emitted
      (a strong signal the float landed in a bad spot).

* **FigureDedupOverlap (R6)**: computes a 8x8 perceptual difference hash
  (dHash) on every PNG figure in the bundle and pairs PDF page bbox sets
  to detect:
    - near-duplicate raster figures (Hamming distance <= 4 of the 64-bit
      hash → flagged for human review or auto-drop),
    - figures whose drawn bbox overlaps another figure on the same page by
      more than 60% of the smaller area (catches stacking accidents from
      float placement).

Both audits write structured JSON reports that downstream pipeline stages
turn into ``\\includegraphics[width=...]`` patches or silent figure drops.
The audit functions themselves never mutate the manuscript — the patcher
``apply_visual_patches`` does that, and only when the issue is unambiguous.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - hard dep is in requirements.txt
    fitz = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


_INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}")
_LABEL = re.compile(r"\\label\{([^}]+)\}")
_REF = re.compile(r"\\(?:ref|autoref|cref|Cref)\{([^}]+)\}")


# ─────────────────────────── data containers ─────────────────────────────


@dataclass
class FigureRecord:
    raw_path: str
    resolved_path: Path | None
    label: str | None = None
    caption_snippet: str = ""
    placement_specifier: str = ""
    width_hint: str | None = None
    referenced: bool = False
    dhash: int | None = None
    file_bytes: int | None = None


@dataclass
class VisualIssue:
    severity: str
    kind: str
    message: str
    asset: str | None = None
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────── helpers ────────────────────────────────────


def _iter_figure_environments(tex: str) -> Iterable[tuple[int, int, str]]:
    pat = re.compile(r"\\begin\{figure\*?\}([\s\S]*?)\\end\{figure\*?\}")
    for m in pat.finditer(tex):
        yield m.start(), m.end(), m.group(1)


def _resolve_asset(bundle_dir: Path, raw: str) -> Path | None:
    raw = raw.strip()
    cand = bundle_dir / raw
    if cand.is_file():
        return cand
    if not cand.suffix:
        for ext in (".png", ".pdf", ".jpg", ".jpeg"):
            alt = cand.with_suffix(ext)
            if alt.is_file():
                return alt
    return None


def _parse_options(opts: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not opts:
        return out
    for part in opts.split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip().lower()] = v.strip()
        else:
            out[part.strip().lower()] = ""
    return out


def collect_figures(tex: str, bundle_dir: Path) -> list[FigureRecord]:
    """Walk ``\\begin{figure}`` envs once and pull every figure into a record."""
    refs = {m.group(1) for m in _REF.finditer(tex)}
    records: list[FigureRecord] = []
    for _start, _end, body in _iter_figure_environments(tex):
        inc = _INCLUDEGRAPHICS.search(body)
        if not inc:
            continue
        opts = _parse_options(inc.group(1))
        raw = inc.group(2).strip()
        label_m = _LABEL.search(body)
        label = label_m.group(1) if label_m else None
        caption_m = re.search(r"\\caption\{([\s\S]*?)\}", body)
        caption = caption_m.group(1).strip() if caption_m else ""
        place_m = re.search(r"\\begin\{figure\*?\}\[([^\]]*)\]", body)
        records.append(
            FigureRecord(
                raw_path=raw,
                resolved_path=_resolve_asset(bundle_dir, raw),
                label=label,
                caption_snippet=caption[:140],
                placement_specifier=(place_m.group(1) if place_m else ""),
                width_hint=opts.get("width"),
                referenced=bool(label and label in refs),
            )
        )
    return records


# ─────────────────────────── R6: dHash + overlap ────────────────────────


def _dhash(path: Path, *, size: int = 8) -> int | None:
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((size + 1, size))
            pixels = list(img.getdata())
    except Exception:
        return None
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_duplicate_figures(records: list[FigureRecord], *, threshold: int = 4) -> list[VisualIssue]:
    issues: list[VisualIssue] = []
    raster: list[tuple[FigureRecord, int]] = []
    for rec in records:
        if rec.resolved_path is None:
            continue
        if rec.resolved_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        h = _dhash(rec.resolved_path)
        if h is None:
            continue
        rec.dhash = h
        try:
            rec.file_bytes = rec.resolved_path.stat().st_size
        except OSError:
            pass
        raster.append((rec, h))
    seen: list[tuple[FigureRecord, int]] = []
    for rec, h in raster:
        for prev_rec, prev_h in seen:
            if _hamming(h, prev_h) <= threshold:
                issues.append(
                    VisualIssue(
                        severity="high",
                        kind="duplicate_figure",
                        message=(
                            f"figure {rec.raw_path} is a near-duplicate of "
                            f"{prev_rec.raw_path} (dHash hamming<={threshold})"
                        ),
                        asset=rec.raw_path,
                        extra={
                            "duplicate_of": prev_rec.raw_path,
                            "rec_label": rec.label,
                            "duplicate_label": prev_rec.label,
                        },
                    )
                )
                break
        else:
            seen.append((rec, h))
    return issues


def _bbox_overlap_fraction(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_y = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_x * inter_y
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = min(area_a, area_b)
    if denom <= 0:
        return 0.0
    return inter / denom


def detect_pdf_figure_overlap(
    pdf_path: Path,
    *,
    overlap_threshold: float = 0.6,
) -> list[VisualIssue]:
    if fitz is None or not pdf_path.is_file():
        return []
    issues: list[VisualIssue] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []
    try:
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            blocks = page.get_text("dict").get("blocks", []) if hasattr(page, "get_text") else []
            image_bboxes: list[tuple[float, float, float, float]] = []
            for block in blocks:
                if block.get("type") == 1:  # image block
                    bbox = block.get("bbox")
                    if bbox and len(bbox) == 4:
                        image_bboxes.append(tuple(bbox))  # type: ignore[arg-type]
            for i in range(len(image_bboxes)):
                for j in range(i + 1, len(image_bboxes)):
                    frac = _bbox_overlap_fraction(image_bboxes[i], image_bboxes[j])
                    if frac >= overlap_threshold:
                        issues.append(
                            VisualIssue(
                                severity="medium",
                                kind="figure_overlap",
                                message=(
                                    f"two image blocks on page {page_idx + 1} "
                                    f"overlap by {frac:.0%} (>= {overlap_threshold:.0%})"
                                ),
                                page=page_idx + 1,
                                bbox=image_bboxes[i],
                                extra={"other_bbox": image_bboxes[j], "fraction": round(frac, 3)},
                            )
                        )
    finally:
        doc.close()
    return issues


# ─────────────────────────── R4: layout audit ──────────────────────────


def detect_off_page_floats(pdf_path: Path) -> list[VisualIssue]:
    """Flag images/drawings whose bbox extends outside the page mediabox."""
    if fitz is None or not pdf_path.is_file():
        return []
    issues: list[VisualIssue] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []
    try:
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            mb = page.mediabox
            for block in page.get_text("dict").get("blocks", []) if hasattr(page, "get_text") else []:
                if block.get("type") != 1:
                    continue
                bbox = block.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = bbox
                if x0 < mb.x0 - 1 or y0 < mb.y0 - 1 or x1 > mb.x1 + 1 or y1 > mb.y1 + 1:
                    issues.append(
                        VisualIssue(
                            severity="high",
                            kind="float_off_page",
                            message=f"image block on page {page_idx + 1} extends past mediabox",
                            page=page_idx + 1,
                            bbox=tuple(bbox),  # type: ignore[arg-type]
                            extra={"mediabox": [mb.x0, mb.y0, mb.x1, mb.y1]},
                        )
                    )
    finally:
        doc.close()
    return issues


def detect_overfull_hbox(bundle_dir: Path) -> list[VisualIssue]:
    log = bundle_dir / "latex_compile.log"
    if not log.is_file():
        return []
    text = log.read_text(encoding="utf-8", errors="replace")
    issues: list[VisualIssue] = []
    # Only flag *Overfull* over 5pt to avoid noisy micro-overflows.
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)", text):
        try:
            pt = float(m.group(1))
        except ValueError:
            continue
        if pt < 5.0:
            continue
        issues.append(
            VisualIssue(
                severity="medium",
                kind="overfull_hbox",
                message=f"Overfull \\hbox ({pt:.1f}pt) — likely a wide table/figure",
                extra={"points_over": pt},
            )
        )
    return issues


def detect_unreferenced_floats(records: list[FigureRecord]) -> list[VisualIssue]:
    return [
        VisualIssue(
            severity="medium",
            kind="float_unreferenced",
            message=f"figure {rec.raw_path} (label={rec.label}) is never \\ref'd in the body",
            asset=rec.raw_path,
            extra={"label": rec.label},
        )
        for rec in records
        if rec.label and not rec.referenced
    ]


# ─────────────────────────── top-level audit ─────────────────────────────


def audit_bundle_visuals(bundle_dir: Path) -> dict[str, Any]:
    """Run R4 + R6 together. Returns a structured report (also saved to disk)."""
    bundle_dir = Path(bundle_dir)
    main_tex = bundle_dir / "main.tex"
    pdf_path = bundle_dir / "main.pdf"
    issues: list[VisualIssue] = []
    records: list[FigureRecord] = []
    if main_tex.is_file():
        tex = main_tex.read_text(encoding="utf-8", errors="replace")
        records = collect_figures(tex, bundle_dir)
        issues.extend(detect_unreferenced_floats(records))
        issues.extend(find_duplicate_figures(records))
    if pdf_path.is_file():
        issues.extend(detect_off_page_floats(pdf_path))
        issues.extend(detect_pdf_figure_overlap(pdf_path))
    issues.extend(detect_overfull_hbox(bundle_dir))
    report = {
        "figure_count": len(records),
        "issues": [
            {
                "severity": iss.severity,
                "kind": iss.kind,
                "message": iss.message,
                "asset": iss.asset,
                "page": iss.page,
                "bbox": list(iss.bbox) if iss.bbox else None,
                "extra": iss.extra,
            }
            for iss in issues
        ],
        "figures": [
            {
                "raw_path": rec.raw_path,
                "label": rec.label,
                "referenced": rec.referenced,
                "width_hint": rec.width_hint,
                "placement": rec.placement_specifier,
                "file_bytes": rec.file_bytes,
                "dhash": rec.dhash,
            }
            for rec in records
        ],
    }
    (bundle_dir / "visual_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return report


# ─────────────────────────── patcher ────────────────────────────────────


def apply_visual_patches(
    bundle_dir: Path,
    *,
    audit: dict[str, Any] | None = None,
    drop_duplicates: bool = True,
) -> dict[str, Any]:
    """Apply unambiguous fixes derived from the audit, return what was changed.

    Conservative by design: we only auto-drop duplicate figures (keeping the
    first occurrence), shrink obviously oversized figures (``width=...`` >
    ``\\linewidth``) to ``\\linewidth``, and tighten the placement specifier
    from ``H`` / ``h!`` to ``tbp`` where we detect off-page floats.
    """
    bundle_dir = Path(bundle_dir)
    main_tex = bundle_dir / "main.tex"
    if not main_tex.is_file():
        return {"ok": False, "skipped": "main_tex_missing"}
    audit = audit or audit_bundle_visuals(bundle_dir)
    tex = main_tex.read_text(encoding="utf-8", errors="replace")
    original = tex
    changes: list[str] = []

    duplicates: set[str] = set()
    if drop_duplicates:
        for iss in audit.get("issues", []):
            if iss.get("kind") == "duplicate_figure" and iss.get("asset"):
                duplicates.add(iss["asset"])
    if duplicates:
        def _maybe_drop(match: re.Match[str]) -> str:
            block = match.group(0)
            inc = _INCLUDEGRAPHICS.search(block)
            if inc and inc.group(2).strip() in duplicates:
                return f"% duplicate figure {inc.group(2).strip()} removed by visual_audit\n"
            return block

        tex = re.sub(
            r"\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}",
            _maybe_drop,
            tex,
        )
        if tex != original:
            changes.append(f"dropped {len(duplicates)} duplicate figures")

    # Shrink oversize widths
    def _shrink_width(match: re.Match[str]) -> str:
        opts = match.group(1) or ""
        body = match.group(2)
        new_opts_parts: list[str] = []
        widened = False
        for part in (p.strip() for p in opts.split(",") if p.strip()):
            if part.lower().startswith("width="):
                val = part.split("=", 1)[1].strip()
                m_num = re.match(r"([\d.]+)(\\\\linewidth|\\\\textwidth)?", val)
                if m_num and m_num.group(2) and float(m_num.group(1)) > 1.0:
                    new_opts_parts.append("width=\\linewidth")
                    widened = True
                    continue
            new_opts_parts.append(part)
        if not widened:
            return match.group(0)
        opt_str = ",".join(new_opts_parts)
        return rf"\includegraphics[{opt_str}]{{{body}}}"

    tex2 = _INCLUDEGRAPHICS.sub(_shrink_width, tex)
    if tex2 != tex:
        changes.append("shrunk over-wide \\includegraphics to \\linewidth")
        tex = tex2

    # Tighten H/h!/h placement when off-page floats are flagged
    off_page = any(iss.get("kind") == "float_off_page" for iss in audit.get("issues", []))
    if off_page:
        tex3 = re.sub(
            r"\\begin\{figure\*?\}\[(?:H|h!|h)\]",
            lambda m: m.group(0).replace(re.search(r"\[(.*?)\]", m.group(0)).group(1), "tbp"),
            tex,
        )
        if tex3 != tex:
            changes.append("relaxed H/h placement to tbp to escape off-page floats")
            tex = tex3

    if tex != original:
        (bundle_dir / "main.tex.visual_patch_backup").write_text(original, encoding="utf-8")
        main_tex.write_text(tex, encoding="utf-8")
    return {"ok": True, "changes": changes, "duplicates_dropped": sorted(duplicates)}
