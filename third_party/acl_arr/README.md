# ACL ARR Template Assets

| Field | Value |
| --- | --- |
| **Venue** | ACL Rolling Review (shared NAACL / ACL / EMNLP submission stream) |
| **Source URL** | https://github.com/acl-org/acl-style-files |
| **License** | MIT (per upstream `LICENSE` file) |
| **Date snapshotted** | 2026-05-14 (during piece-rate Deepgraph #13 implementation) |
| **Redistribution allowed?** | Yes — MIT-licensed |

## Files

- `acl.sty` — official ACL two-column style file from the upstream
  `acl-style-files` repository.
- `acl_natbib.bst` — natbib bibstyle producing the conference's canonical
  parenthetical author-year citations (`(Vaswani et al., 2017)`).

## Notes

ACL papers are two-column. `ACLArrAdapter.column_layout == "two_column"`,
so `FormatLinter` will expect `\columnwidth` for in-column figures and
`figure*` for full-page-width spans.
