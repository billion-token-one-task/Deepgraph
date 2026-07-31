# LLM caller inventory and grant boundary

Last reviewed: 2026-07-31 UTC.

This inventory is based on AST/text review only. No provider call or module
import was performed. `legacy` means the caller still invokes `call_llm` or
`call_llm_json`; it does not mean the call is approved for production.

| Surface | Current route | Intended authority | Status |
|---|---|---|---|
| experiment forge code scout | proposer role + forge/pilot ResourceGrant | idea/stage grant | adapted; CI pending |
| experiment scaffold | proposer role + forge/pilot ResourceGrant | idea/stage grant | adapted; CI pending |
| experiment-plan repair | proposer role + repair/design/pilot grant | idea/stage grant | adapted; CI pending |
| validation code iteration | proposer role + validation grant | run idea/stage grant | adapted; CI pending |
| reproduction repair | proposer role + validation grant | run idea/stage grant | adapted; CI pending |
| problem sharpening | deterministic persisted problem selection | no LLM authority | adapted; no pre-identity call |
| method invention/experiment design | proposer role + proposal ResourceGrant | stable `proposal_pending` deep-insight ID | adapted; CI pending |
| benchmark design agent | proposer role + forge/pilot ResourceGrant | idea/stage grant | adapted; CI pending |
| codebase scout direct/agentic entry point | legacy when called outside forge | idea-scoped grant | open; production forge bypasses it |
| PaperOrchestra manuscript revision | proposer role + manuscript grant | manuscript-allowed idea/run | adapted; CI pending |
| plain manuscript review | reviewer role + manuscript grant | recorded proposer route plus independent reviewer | adapted; CI pending |
| Tier-2 debate/refinement | evaluator/reviewer/proposer roles + proposal grant | stable proposal candidate | adapted; CI pending |
| legacy PaperOrchestra refinement-loop helper | legacy, no known generic call site | must be routed or removed before activation | open |
| paradigm/insight ranking outside the proposal path | legacy | agenda/candidate-scoped route | open |
| extraction/abstraction/reasoning/taxonomy/domain summary | legacy ingestion | bounded ingestion identity and grant contract | open |
| example CGGR plugin | legacy/non-production | explicit example-only policy | excluded from generic runtime |

## Pre-candidate authority decision

The implementation uses an honest minimal `deep_insights` row with status
`proposal_pending`, agenda/problem provenance and no fabricated method or
result. Frontier/portfolio may issue a `stage=proposal` grant to that stable
ID. Method/experiment generation then consumes the grant, and successful
storage promotes the same row in place. Without a grant, the row remains
pending and no LLM call occurs.

## Review rule

`rg`/AST inventory must be regenerated at the candidate commit. Any new
post-agenda legacy caller is a release blocker unless it is demonstrably
non-resource-consuming pure formatting with a separate integrity gate.
