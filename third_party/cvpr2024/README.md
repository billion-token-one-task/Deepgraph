# CVPR 2024 Template Assets

| Field | Value |
| --- | --- |
| **Venue** | CVPR 2024 (Conference on Computer Vision and Pattern Recognition) |
| **Source URL** | https://github.com/cvpr-org/author-kit |
| **License** | Permissive author-kit license (redistributable for author submissions) |
| **Date snapshotted** | 2026-05-14 (during piece-rate Deepgraph #13 implementation) |
| **Redistribution allowed?** | Yes — assets ship with the conference template for author reuse |

## Files

- `cvpr.sty` — official CVPR two-column author-kit style file.
- `ieeenat_fullname.bst` — natbib variant of IEEE's `IEEEtran` bibstyle used
  by the CVPR author kit; produces the conference's canonical numeric
  citations (`[1]`, `[2, 3]`).

## Notes

CVPR papers are two-column. `CVPR2024Adapter.column_layout == "two_column"`.
The 8-page main body + unlimited references rule is encoded as
`max_pages = 8`.
