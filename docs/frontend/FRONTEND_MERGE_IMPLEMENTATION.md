# Frontend merge v1: what was implemented

Branch: `feat/frontend-merge-v1`. Implements the merge agreed after the
discovery report (`FRONTEND_DISCOVERY_AND_RECOMMENDATION.md`): new frontend as
the base, old frontend as a parts source, plus the evidence-ladder wiring that
neither baseline had. No schema change, no migration, no service touched.

## Performance (the "loads very slowly" fix)

- Restored `web/stats_cache.py` verbatim from `prod-snapshot-20260621`
  (issue #34 fix dropped in the upgrade): `/api/stats` runs ~30 COUNT(*)
  scans and was executed synchronously per request, at first paint and every
  15s. It is now served from an in-process TTL cache with one background
  refresher; cold start returns `{"warming": true}` and the frontend keeps
  the previous numbers instead of rendering zeros. `main.py` prewarms the
  cache in a background thread at startup.
- D3 and marked were render-blocking `<head>` scripts from d3js.org and
  jsdelivr (unreachable/slow from China; page hung on them). Vendored into
  `web/static/vendor/{d3,marked}/` (as MathJax already was) and loaded with
  `defer`.
- Static assets now carry `?v=` cache-busting versions.

## Restored from the old frontend

- Global stats board: an always-visible 4-card row on the overview (Source
  Papers Analyzed / Paper Ideas Generated / Experiment Runs / Analysis
  Tokens) plus Analysis Tokens in the top bar. Remaining counts stay in the
  collapsed details section, joined by a new Agenda Tokens card
  (`agenda_tokens_total`, summed from `agenda_token_ledger`; kept separate
  from `tokens_consumed` because the ledgers may overlap - never summed).
- EN/ZH i18n: new `web/static/js/i18n.js` (same API as the old one:
  `window.t`, `dgI18n`, `data-i18n` attributes), EN/ZH switcher in the top
  bar, full chrome coverage of the new template, and translated dynamic
  vocabulary for badges, timeline kinds, and empty states. Deep
  render-function strings remain English (follow-up).
- Selection rationale: `GET /api/v1/agendas/<id>/selection` exposes
  `agenda_selections` (rationale, rejected candidates, scoring breakdown)
  and `idea_decision_packets` (decision + reason codes); rendered as a
  "Selection Rationale" card on the Process tab.
- A place to submit directions: "Propose direction" (sidebar > More) opens a
  form that POSTs `{confirmed: true, agenda: {direction, contact, keywords,
  goal, token_budget}}` to the existing token-gated
  `POST /api/meta-harness/v1/agendas` with `X-DeepGraph-Operator-Token`
  supplied by the user at submit time. No new auth mechanism; no execution
  controls.

## New: evidence ladder wired to the UI

- `web/provenance_routes.py`, read-only blueprint at `/api/v1`:
  - `GET /api/v1/agendas` - public agenda list (no submitter contact, no raw
    config), used by the frontend to establish its agenda scope;
  - `GET /api/v1/evidence_states?agenda_id=` - latest ladder state per run,
    rolled up per idea, with verdicts from `scientific_decision_records`;
  - `GET /api/v1/agendas/<id>/selection`;
  - `GET /api/v1/agendas/<id>/timeline` - merged chronological events from
    frontier packets (incl. refused gates), decision packets, resource
    grants, compute jobs (incl. failures), evidence transitions, decisions,
    outcomes.
  Every response goes through explicit field allowlists; absolute paths in
  free text are redacted to `<path>`; missing tables yield empty results.
- Two-register badges everywhere ideas/runs are listed: `RUN: <operational
  status>` and `EVIDENCE: <ladder state>` / `DECIDED: <verdict>` are separate
  badges, never merged. Default scientific state is grey "not assessed";
  refuted renders in purple (a valid outcome), red stays reserved for
  operational failure. The informal `hypothesis_verdict` chip was removed
  from cards in favor of the ladder verdict.
- Process tab (repurposed from the dead, unreachable experiments panel; now
  in the main nav): Process Timeline, Selection Rationale, Automation
  Services, Idea Experiments, Meta report.

## Fixed while merging

- Agenda scoping bug: the UI never sent `agenda_id`, so every scoped route
  (`/api/deep_insights`, `/api/generated_papers`, `/api/experiment_groups`,
  ...) returned 400 and tabs silently rendered their empty states. The
  frontend now resolves the active agenda from `/api/v1/agendas` at init and
  appends `agenda_id` centrally in the `api()` helper.
- Data leaks closed (all pre-existing):
  - `/api/automation` no longer returns `binary_path`, `workdir`, or
    `log_tail`/`log_error` (raw log bytes); sessions carry booleans/ages only;
  - `_workspace_payload` no longer emits `workspace_root`/`experiment_root`/
    `plan_root`/`paper_root`;
  - `_api_failure` returns a generic message plus a correlation id; raw
    exception text goes to the server log only;
  - a global `after_request` scrubber strips a denylist of path/log keys from
    every JSON response and redacts absolute-path substrings in string
    values; the SSE `/api/events` stream is scrubbed the same way.

## Tests

`tests/test_provenance_web.py`: provenance endpoints (allowlists, rollup,
path scrubbing, missing-table tolerance), stats cache route behavior
(warming -> snapshot, no per-request recompute), scrubber unit tests, an
`_api_failure` disclosure test, and a leak guard that walks every
parameterless GET route asserting no `/home/` paths, `log_tail`,
`workspace_root`, etc. in any response body.

Suite status: 525 passed / 74 failed; the 74 failures are pre-existing on
the branch base (verified identical with all these changes stashed) and lie
outside the changed surface. `tests/test_dataset_resolver.py`,
`test_paperorchestra_briefing.py`, `test_vnext_manuscript.py` fail to
collect on this machine for a pre-existing `scripts` import issue.

## Known follow-ups

- Deep dynamic strings inside older render functions are not yet
  i18n-covered.
- The Chinese-labeled duplicate renderer `renderExperimentGroups` (dead code)
  was left in place.
- Deployment: use the rollback-safe update procedure; nothing here touches
  the DB schema, so no migration step is needed.
