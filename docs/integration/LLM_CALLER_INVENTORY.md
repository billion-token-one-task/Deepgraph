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
| problem sharpening/method invention/experiment design | legacy | a persisted pre-candidate identity and bounded proposal grant | open; do not invent a deep-insight ID |
| benchmark design agent | legacy | idea-scoped proposer/reviewer grant | open |
| codebase scout direct/agentic entry point | legacy when called outside forge | idea-scoped grant | open; production forge bypasses it |
| PaperOrchestra/refinement/plain manuscript review | legacy | manuscript/reviewer grant after `manuscript_allowed` | open |
| paradigm/tier-2 refinement/insight ranking | legacy | agenda/candidate-scoped role route | open |
| extraction/abstraction/reasoning/taxonomy/domain summary | legacy ingestion | bounded ingestion identity and grant contract | open |
| example CGGR plugin | legacy/non-production | explicit example-only policy | excluded from generic runtime |

## Unresolved pre-candidate authority

`ResourceGrant.idea_id` is mandatory and the existing runtime uses it as the
persisted `deep_insights.id`. Problem sharpening and method invention occur
before that row exists. Reusing a research-problem ID or inserting a fake
deep-insight would create false lineage. The remaining correct choices for
isolated design review are:

1. persist a first-class proposal candidate, make its ID the stable idea
   identity through promotion and execution; or
2. pre-allocate a minimal, honest deep-insight candidate row whose status and
   provenance explicitly say `proposal_pending`.

Until one schema/contract path is reviewed and migrated, those legacy calls are
not authorized as meta-harness-v1 high-cost production calls.

## Review rule

`rg`/AST inventory must be regenerated at the candidate commit. Any new
post-agenda legacy caller is a release blocker unless it is demonstrably
non-resource-consuming pure formatting with a separate integrity gate.
