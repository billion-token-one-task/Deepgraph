# NeurIPS 2024 Template Assets

| Field | Value |
| --- | --- |
| **Venue** | NeurIPS 2024 (Conference on Neural Information Processing Systems) |
| **Source URL** | https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles |
| **License** | Permissive (NeurIPS author template — redistributable for submission preparation) |
| **Date snapshotted** | 2026-05-14 (during piece-rate Deepgraph #13 implementation) |
| **Redistribution allowed?** | Yes — assets ship with the conference template specifically to be reused by authors |

## Files

- `neurips2024.sty` — minimal stub of the official package that defines the
  `\documentclass` workflow used by NeurIPS papers. The full asset is not
  vendored here in this PR because the routing pipeline only needs the
  adapter contract (`copy_files` / `inject_preamble` / `normalize_source`)
  to compile correctly under a real TeX install. Replace this stub with the
  upstream `.sty` before any real submission build.

## Notes

NeurIPS papers are single-column with a compact 10pt body; `FormatLinter`
(issue #14) consumes `NeurIPS2024Adapter.column_layout == "single_column"`
to enforce the 4-subfigure-per-row grid density rule the user flagged
during D1 review.
