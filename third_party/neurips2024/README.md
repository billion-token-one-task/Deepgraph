# NeurIPS 2024 Template Assets

| Field | Value |
| --- | --- |
| **Venue** | NeurIPS 2024 (Conference on Neural Information Processing Systems) |
| **Source URL** | https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles |
| **License** | Permissive (NeurIPS author template — redistributable for submission preparation) |
| **Date snapshotted** | 2026-05-14 (during piece-rate Deepgraph #13 implementation) |
| **Redistribution allowed?** | Yes — assets ship with the conference template specifically to be reused by authors |

## Files

- `neurips_2024.sty` — official NeurIPS 2024 single-column style file
  vendored verbatim from the conference Style Files page. NeurIPS uses
  natbib defaults, so the adapter pairs this with `bibstyle_name =
  "unsrtnat"` for the canonical numeric inline citations
  (`Vaswani et al. [2017]`).

## Notes

NeurIPS papers are single-column with a compact 10pt body; `FormatLinter`
(issue #14) consumes `NeurIPS2024Adapter.column_layout == "single_column"`
to enforce the 4-subfigure-per-row grid density rule the user flagged
during D1 review.
