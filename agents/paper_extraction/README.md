# Paper Extraction Agent

Owns paper discovery, PDF parsing, structured extraction, grounding, and source completeness checks.

Primary legacy modules:

- `ingestion.arxiv_client`
- `ingestion.arxiv_ids`
- `ingestion.grobid_tei`
- `ingestion.pdf_parser`
- `agents.extraction_agent`
- `agents.claim_grounding`
- `agents.paper_completeness`
- `agents.reference_corpus_audit`

Configuration lives in `deepgraph.toml` under `profile`, `arxiv`, `grobid`, `pdf`, `llm`, and `paths`.

