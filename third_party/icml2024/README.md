# ICML 2024 Template Assets

| Field | Value |
| --- | --- |
| **Venue** | ICML 2024 (International Conference on Machine Learning) |
| **Source URL** | https://icml.cc/Conferences/2024/StyleAuthorInstructions |
| **License** | Permissive (ICML author template — redistributable for submission preparation) |
| **Date snapshotted** | 2026-05-14 (during piece-rate Deepgraph #13 implementation) |
| **Redistribution allowed?** | Yes — assets ship with the conference template for author reuse |

## Files

- `icml2024.sty` — official ICML 2024 style file. It issues `\twocolumn`
  internally, so the adapter declares `column_layout = "two_column"`.
- `icml2024.bst` — official ICML bibstyle producing the conference's
  canonical parenthetical author-year citations.
- `fancyhdr.sty`, `algorithm.sty`, `algorithmic.sty` — vendored dependencies
  required by `icml2024.sty` so Tectonic can compile offline.

## Notes

ICML 2024 main body is two-column (driven by `icml2024.sty` itself).
`ICML2024Adapter.column_layout == "two_column"`.
