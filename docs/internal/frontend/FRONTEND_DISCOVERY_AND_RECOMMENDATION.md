# DeepGraph frontend: discovery and recommendation

Status: Phase 1 discovery only. No application code was changed, no service was
restarted, no migration was run, no database row was read or written.
All findings below come from read-only inspection of Git objects and of files
under `/home/billion-token/Deepgraph`.

Scope note: this report analyses the *frontend* (routes, templates, assets, API
shapes). The pipeline, agents, and database are treated as out of scope except
where they determine what the frontend can honestly display.

---

## 1. Baseline identification and evidence

### 1.1 The two baselines

| | Current baseline ("new") | Pre-upgrade baseline ("old") |
|---|---|---|
| Identifier | `master` = `origin/master` = `a6973b8` | `refs/archive/prod-snapshot-20260621` = `7d0b42a` |
| Equivalent refs | - | `web/` tree is byte-identical to `refs/archive/topic-gate-20260729` (`9d24d29`) |
| Entry template | `web/templates/index.html` (399 lines) | `web/templates/index.html` (553 lines) |
| Server | `web/app.py` (2114 lines) + `web/meta_harness_routes.py` (493) | `web/app.py` (1784) + `web/agenda_routes.py` (422) + `web/manuscript_routes.py` (213) + `web/stats_cache.py` (89) |
| Client JS | `web/static/js/app.js` (3898) | `web/static/js/app.js` (2681) + `i18n.js` (956) + `agenda.js` (157) + `manuscript_routing.js` (96) + `graph/{adapter,renderer,tooltip}.js` (641) |
| CSS | `web/static/css/style.css` (2534) | `web/static/css/style.css` (2101) |

### 1.2 Evidence for the old baseline

The service worktree `/home/billion-token/Deepgraph` is at Git HEAD
`7d0b42af8e8f061c3c16800c44224c110f3b94a0`, which is exactly
`refs/archive/prod-snapshot-20260621`.

Six frontend files that exist only in the old lineage are still present in the
service directory and are byte-identical (MD5) to that ref:

```
web/agenda_routes.py              9c1457954715e5b4901a546c53bdf344
web/manuscript_routes.py          72d6cfebac88c8a18e25cbab5b2c3d5b
web/stats_cache.py                f98acc8f5c758ac8becb2d9f66db1af4
web/static/js/agenda.js           3317337003c0fe1d23debd4d3274f771
web/static/js/i18n.js             dfa9e6ad4b5635a2c608eca350d66772
web/static/js/manuscript_routing.js  2e595ae3d537fe58d803cee9c0ea68db
```

These same six files are identical between `prod-snapshot-20260621` and
`topic-gate-20260729`, so the two archive refs are interchangeable as the old
frontend baseline. `git diff` between them over `web/` is empty.

### 1.3 Important finding: the old frontend is no longer deployed

This materially changes the framing of the task, which assumed the service
directory still shows the pre-upgrade UI. It does not.

The four core frontend files in the service directory are byte-identical to
current `master`:

```
web/app.py                 9e2568fb039b6cfd7dc81ad58979db0d   (both)
web/templates/index.html   b93a4abe505ee1a845f6a94004b9535e   (both)
web/static/css/style.css   4710746852d15087ca301936544dfc55   (both)
web/static/js/app.js       d9b48e8a49feff39bca51168322e6af1   (both)
web/meta_harness_routes.py 223cde44cd1f99d5f9338e6f80206d3a   (both)
```

Those files are owned by `root` with mtime 2026-08-02 20:23, whereas the rest of
the directory is owned by `billion-token`. The live service therefore already
serves the **new** frontend. Consequences:

- The pre-upgrade UI cannot be observed by running the service. It was
  reconstructed from Git only. This is stated as a limitation, not worked around.
- `agenda_routes.py`, `manuscript_routes.py`, `stats_cache.py`, `agenda.js`,
  `i18n.js`, `manuscript_routing.js` are now **dead code** in the service
  directory: the new `web/app.py` registers only `meta_harness_blueprint`
  (`web/app.py:28`) and never imports them. They are inert, not a live risk,
  but they are misleading to anyone auditing the deployment.

### 1.4 Ambiguity that cannot be resolved locally

- The task file names `3f9ddbf` as current master. Master has since advanced by
  five commits to `a6973b8`; the newest commit is the one that added the task
  file itself. The intervening commits (`06e7059` hide runtime topology,
  `dc5cedb` responsive dashboard, `f473645` restore research map,
  `54137bb` seed scoped harness research) are frontend-relevant and are included
  in this analysis. Confirm that analysing `a6973b8` rather than `3f9ddbf` is
  intended.
- Exact date the new frontend was copied into the service directory, and by whom,
  is not recoverable beyond the mtime. There is no deployment log in the repo.
- No screenshots or browser session were produced. A disposable local preview was
  not started, because the same conclusions are reachable from source and
  starting one risked port or database contention with the live service. If
  visual confirmation is required, that is a follow-up with an explicit port and
  a read-only database.

---

## 2. Comparison matrix

### 2.1 Navigation and page inventory

| Old nav (13 panels) | New nav (7 panels) | Assessment |
|---|---|---|
| Overview | Research (overview) | Changed: old was a wall of 9 stat cards; new leads with "Research outcomes" and a research map, and demotes counts into a `<details>`. Improvement for a first-time visitor. |
| Explore | Research Map | Retained, renamed. |
| Evidence | Evidence | Retained. Old also had an "Entity-Relation Network" card; new does not. Regression. |
| Papers (source papers) | -- | Removed. Source-paper corpus is no longer browsable. Regression for provenance. |
| Generated | Manuscripts | Improved: new has a two-pane reading desk (list + reader) rather than a flat list. |
| Insights | -- | Removed as a top-level panel. |
| Advanced > Paper Progress | -- | Removed. Regression for process transparency. |
| Advanced > Discoveries | Ideas | Retained, promoted to top level. |
| Advanced > Experiments | (panel exists, unreachable) | See 2.4. Regression. |
| Advanced > Feed | -- | Removed. |
| Advanced > Providers | -- | Removed. Correct: this was runtime topology. |
| Advanced > Agenda | -- | Removed. See 2.3. Correct removal, but nothing replaced it. |
| -- | More > Activity (Agent Office) | Added: pixel-art "Agent Office" workspace view. |

### 2.2 API surface

Old, removed wholesale in new:

- `/api/research_agenda` GET, POST; `/current`; `/<id>/insights`;
  `/<id>/selections`; `/<id>/resume`; `/select`; `/selection/latest`;
  `/selection/<id>`; `/selection/<id>/bench`; `/gate`; `/gate/latest`;
  `/review`; `/plan`; `/loop/<id>` (16 endpoints)
- `/api/manuscript/venues`, `/route`, `/route/<id>`, `/lint`, `/lint/<id>`,
  `/lint_run/<id>` (7 endpoints)
- `/api/token_usage`

New, added:

- `/api/meta-harness/v1/*`: `status`, `agendas`, `legacy-import`, `frontier`,
  `frontier/from-evidence-graph`, `portfolio/decide`, `grants`,
  `runs/<id>/attach-grant`, `runs/<id>/evidence-state`, `outcomes`,
  `ingestion/jobs`, `compute/colab/jobs`,
  `harness/candidates/<id>/evaluate` (13 endpoints)
- `/api/agent_office`, `/api/generated_papers`

### 2.3 Authorization: the largest real change

The old Agenda tab was, in plain terms, an unauthenticated operator control
panel published on the open web. Its markup contains buttons wired to
mutating endpoints with **no authentication anywhere in
`agenda_routes.py`** (no token check, no `abort(401/403)`, no session gate):

- "Run Selector + Dispatch" with a dispatch-mode selector
  (`auto` / `link existing` / `enqueue fresh` / `none`)
- "Upload agenda (YAML)" -> free-form textarea -> `POST /api/research_agenda`
- "Run Review", "Build Revision Plan", "Inspect Full Loop"
- Manuscript routing and LaTeX lint with a raw source textarea
- Raw JSON dumped into `<pre>` blocks for the loop inspection view

The new baseline removes all of it and replaces the mutation surface with a
single token-gated blueprint. `_require_operator()`
(`web/meta_harness_routes.py:68`) requires
`DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN` in the environment and a matching
`X-DeepGraph-Operator-Token` header, compared with `hmac.compare_digest`, and it
fails closed: if the env var is unset the API is disabled entirely.

Legacy mutating routes in `app.py` were audited individually rather than by
decorator. 15 of 17 POST routes now return `410` via
`_manual_api_removed_response()`; `/api/runtime-config` returns `404`
(`web/app.py:877`); `/api/insights/rank` returns `410` with a pointer to
`/api/meta-harness/portfolio/decide` (`web/app.py:1539`). No live unauthenticated
mutation route remains.

**This is a genuine and significant improvement and should be preserved.** The
problem is not that the old controls were removed; it is that nothing replaced
them, so product goal 1 (authorized people submit directions) currently has no
interface at all.

### 2.4 Regressions in the current frontend

1. **No submission workflow.** There is no UI path to propose a research
   direction, and no UI path to see the status of one you submitted. The
   `research_agendas` table has a `submitter` column and the meta-harness API can
   create agendas, but only via `curl` with an operator token.
2. **Unreachable Experiments panel.** `#tab-experiments` exists in
   `index.html:337-387` with three cards (Automation Services, Auto Research,
   Idea Experiments) but no `nav-item` has `data-tab="experiments"`. The panel is
   dead markup, and the underlying `/api/auto_research/*` and `/api/gpu/*`
   endpoints now return 410 anyway, so the cards would render errors if reached.
3. **Internationalisation deleted.** Old had `i18n.js` (956 lines), an EN/ZH
   switcher, and 133 `data-i18n` attributes. New has zero. Given a
   China-based audience this is a real accessibility regression.
4. **Provenance narrowed.** The source-paper library and the entity-relation
   network are gone. A reader can no longer walk from a result back to the source
   corpus in the UI.
5. **Process visibility narrowed.** Paper Progress and Feed are gone. What
   replaced them, Agent Office, is a stylised departmental animation rather than
   a chronological, inspectable record.
6. **Mixed-language leftovers.** After removing i18n, three `title` attributes
   remain hard-coded in Chinese (`index.html:178, 185, 192`) inside an
   otherwise English UI.

### 2.5 Strengths of the old frontend worth retaining

- Bilingual EN/ZH support.
- A named place where a research direction could be entered at all, even though
  the implementation was unsafe.
- Explicit exposure of *selection rationale*: `/api/research_agenda/selection/<id>`
  returned `rationale`, `rejected_candidates_json`, and `scoring_breakdown_json`.
  The new UI shows selected work but never why alternatives were rejected. This
  is the single most valuable thing the old UI had and the new one lost.
- A token-usage endpoint, giving readers a sense of cost.

### 2.6 Strengths of the new frontend worth retaining

- Fails closed on mutations; token-gated operator API.
- Outcome-first overview rather than counts-first.
- The manuscript reading desk.
- Runtime topology (Providers panel) removed from public view.
- Agenda-scoped queries: 11 routes require an explicit `agenda_id`. Note this is
  *scoping*, not *authorization* (see 4.2).

### 2.7 Failure, pause, uncertainty: both baselines are weak

This is the finding that matters most against the stated product goal.

The backend has a rigorous six-state evidence ladder in
`contracts/meta_harness.py:22`:

```
planned -> sanity_passed -> full_benchmark_complete
        -> evidence_audited -> scientifically_decided -> manuscript_allowed
```

with verdicts `supported` / `refuted` / `inconclusive`, content-hash gates at
each transition (`meta_harness/evidence_state.py`), and
`manuscript_allowed` requiring a `supported` verdict *plus* a reviewer approval
record plus a verdict hash.

**Neither frontend reads any of it.** Grepping `web/app.py` and
`web/meta_harness_routes.py` for the relevant tables:

| Table | Read by web layer |
|---|---|
| `evidence_state_transitions` | no |
| `scientific_decision_records` | no |
| `evidence_audit_records` | no |
| `reviewer_approval_records` | no |
| `benchmark_harness_jobs` | no |
| `research_problems` | no |
| `experimental_evidence_edges` | no |
| `agenda_resource_ledger` | no |
| `agenda_selections` | no |

Both old and new `app.js` carry a similar informal verdict vocabulary
(new: 36 `verdict`, 5 `confirmed`, 5 `refuted`, 2 `inconclusive`, 2 `limitation`;
old: 41 / 7 / 7 / 2 / 2), and neither mentions `evidence_state`,
`pending_review`, `paused`, or `failure_reason` at all.

The practical effect is exactly the risk the task names: an experiment whose
`compute_jobs_v1.status` reached `completed` is operationally finished, and the
UI has no vocabulary to distinguish that from `scientifically_decided` with a
`supported` verdict. The rigour exists in the database and stops at the API
boundary. Closing this gap is the core of the recommendation.

### 2.8 Shared weaknesses (unchanged between baselines)

- Both templates load D3, `marked`, and Google Fonts from third-party CDNs
  (`d3js.org`, `cdn.jsdelivr.net`, `fonts.googleapis.com`). For China-based
  readers this is a load-failure risk; the research map depends on D3, so the
  map breaks rather than degrades.
- Single-page app with no server-rendered routes and no deep links: every view is
  a tab in one `index.html`. Nothing is linkable, citable, or crawlable, which
  directly undercuts "let a community member understand what happened".

---

## 3. Recommended target product experience

Design principle: **the interface states what is known, how it was established,
and what is still unknown.** Operational status and scientific status are
rendered in two visually distinct registers and are never merged into one badge.

### 3.1 Information architecture

Replace the single-page tab shell with server-rendered, linkable routes.

```
/                          public overview
/agendas                   list of agendas (public: title, status, dates)
/agendas/<id>              agenda detail
/agendas/<id>/timeline     chronological process timeline
/directions/new            submit a direction        [authorized]
/directions/<id>           submission status         [submitter or operator]
/runs/<id>                 experiment / job detail
/evidence/<run_id>         evidence and artifact detail
/outcomes/<id>             result, failure, or limitation
/about/method              how to read this site: states, verdicts, caveats
```

`/about/method` is not filler. If the site publishes verdicts, it needs one page
defining what each state means and what it does not mean.

### 3.2 The two-register status model

Every run and every outcome carries two independent badges. They must never be
combined.

```
OPERATIONAL   planned | running | completed | failed | cancelled | timed out
SCIENTIFIC    not assessed | sanity passed | benchmark complete
              | audited | decided: supported / refuted / inconclusive
              | manuscript allowed
```

Rules:
- Default scientific state is `not assessed`, rendered in neutral grey.
- `completed` operational plus `not assessed` scientific must read as
  "the job finished; no scientific claim is being made". Never a green tick.
- Only `decided: supported`, backed by a `scientific_decision_records` row, may
  use affirmative language ("supported by evidence"). Never "proven", never
  "confirmed" without qualification.
- `refuted` and `inconclusive` are first-class outcomes shown with the same
  prominence as `supported`. A system that only surfaces successes is not
  trustworthy.

### 3.3 Text wireframes

**A. Public overview `/`**

```
+----------------------------------------------------------------+
| DeepGraph            [ Agendas ] [ Timeline ] [ How to read ]   |
+----------------------------------------------------------------+
| What this system does                                          |
| Two short paragraphs, plain language, no metrics.              |
| "Findings below are machine-generated and carry an explicit    |
|  evidence state. Read 'How to read this' before citing."       |
+----------------------------------------------------------------+
| Findings                                        [ all ]        |
|                                                                |
|  Supported (3)      Refuted (5)     Inconclusive (11)          |
|  ---------------------------------------------------------     |
|  > Method X improves Y on Z                                    |
|    SCIENTIFIC: decided - supported    OPERATIONAL: completed   |
|    baseline 0.412 -> 0.487  n=5 seeds  audited 2026-07-30      |
|    Limitations: single dataset; no held-out replication.       |
|                                                                |
|  > Method A does not transfer to B                             |
|    SCIENTIFIC: decided - refuted      OPERATIONAL: completed   |
+----------------------------------------------------------------+
| Work in progress                                               |
|  7 runs running - 2 failed - 1 paused awaiting human review    |
|  None of these support any claim yet.            [ timeline ]  |
+----------------------------------------------------------------+
| Active agendas                                                 |
|  #12 Latent communication   34 runs   2 supported   [ open ]   |
+----------------------------------------------------------------+
```

Counts such as "Reasoned Papers" belong below the fold or on the agenda page.
Leading with volume metrics implies productivity is the point; it is not.

**B. Submit a direction `/directions/new` [authorized]**

```
+----------------------------------------------------------------+
| Propose a research direction         signed in as: <handle>    |
+----------------------------------------------------------------+
| Question or direction *                                        |
| [ free text, 40-600 chars                                    ] |
|                                                                |
| Why it matters *                                               |
| [ free text                                                  ] |
|                                                                |
| What would count as evidence? *                                |
| [ free text: metric, baseline, threshold                     ] |
|   Required. A direction with no falsifiable target is not      |
|   admissible.                                                  |
|                                                                |
| Prefer / reject (optional)   [ tags ]  [ tags ]                |
| Suggested token budget (advisory)  [ ........ ]                |
|                                                                |
|  Note: submitting does not authorize compute. An operator      |
|  must issue a ResourceGrant before any run starts.             |
|                                        [ Cancel ] [ Submit ]   |
+----------------------------------------------------------------+
```

No YAML textarea. No dispatch-mode selector. No "run now" button. The old UI's
mistake was exposing execution controls; the fix is a proposal form whose output
is a queued record, not an action.

**C. Submission status `/directions/<id>`**

```
+----------------------------------------------------------------+
| Direction #47  "Does X transfer across domains?"               |
| STATUS: under review        submitted 2026-08-01 by <handle>   |
+----------------------------------------------------------------+
| [x] Submitted            2026-08-01                            |
| [x] Admissibility check  2026-08-01  passed                    |
| [>] Operator review      pending - no decision yet             |
| [ ] Agenda created                                             |
| [ ] Resource grant issued                                      |
| [ ] First run                                                  |
+----------------------------------------------------------------+
| Reviewer notes (visible to submitter)                          |
|  "Evidence target is too broad; specify a benchmark."          |
+----------------------------------------------------------------+
```

States must include `rejected` with a reason, and `withdrawn`. A queue that only
ever shows progress is dishonest.

**D. Agenda detail `/agendas/<id>`**

```
+----------------------------------------------------------------+
| Agenda #12  Latent communication                               |
| STATUS active   created 2026-06-02   origin: direction #31     |
+----------------------------------------------------------------+
| Scope                                                          |
|  Focus:  ...      Prefer: ...      Reject: ...                 |
|  Required output: ...                                          |
+----------------------------------------------------------------+
| Budget            tokens 4.2M / 10M     [======----]           |
|                   (operator view adds cost and GPU hours)      |
+----------------------------------------------------------------+
| Evidence ladder    planned 14 | sanity 9 | benchmark 6         |
|                    audited 4 | decided 3 | manuscript 1        |
+----------------------------------------------------------------+
| Candidates and selection                    [ full rationale ] |
|  SELECTED   #221 "cross-domain probe"      score 0.81          |
|    why: highest contribution delta, falsifiable in one run     |
|  REJECTED   #219 "scaling sweep"           score 0.54          |
|    why: obsolete_evidence - superseded by prior art            |
|  REJECTED   #223 "architecture search"     score 0.49          |
|    why: no minimum falsification experiment                    |
+----------------------------------------------------------------+
| Outcomes    3 supported   5 refuted   11 inconclusive   2 failed|
+----------------------------------------------------------------+
```

Showing rejected candidates *with reason codes* is what makes the system legible
rather than oracular. This restores the old UI's best feature in a readable form.

**E. Process timeline `/agendas/<id>/timeline`**

The chain the task specifies, rendered as one scrollable column, newest last.

```
direction -> agenda -> inputs/signals -> candidates + rationale
  -> resource authorization -> jobs/experiments -> artifacts/evidence
  -> result | failure | limitation | next step

+----------------------------------------------------------------+
| Filter: [x] decisions [x] runs [x] failures [ ] routine         |
+----------------------------------------------------------------+
| 06-02 10:14  DIRECTION      #31 submitted by <handle>          |
| 06-02 11:02  AGENDA         #12 created from direction #31     |
| 06-03 09:20  SIGNALS        frontier packet #88                |
|                             coverage: 41 papers, 6 benchmarks  |
|                             gate: allowed                      |
| 06-03 09:44  CANDIDATES     3 evaluated, 1 admitted            |
|                             reason codes: [contribution_delta] |
|                             -> rejected #219 obsolete_evidence |
| 06-03 10:01  AUTHORIZATION  grant #55 issued                   |
|                             stage pilot, cap 250k tok, T4      |
| 06-03 10:05  RUN            job #904 submitted (colab)         |
| 06-03 12:38  RUN            job #904 completed                 |
|                             OPERATIONAL only. No claim yet.    |
| 06-03 12:40  EVIDENCE       sanity_passed                      |
|                             artifacts sha256:9f2c... (3 files) |
| 06-04 08:10  EVIDENCE       full_benchmark_complete            |
| 06-04 09:55  BLOCKED        evidence_audited refused           |
|                             blockers: holdout_ref_missing      |
| 06-05 14:20  EVIDENCE       evidence_audited                   |
| 06-05 15:02  DECISION       scientifically_decided: supported  |
|                             baseline 0.412 -> 0.487            |
|                             verdict hash sha256:7a1e...        |
| 06-06 09:00  PAUSED         awaiting reviewer approval         |
+----------------------------------------------------------------+
```

Blocked transitions must appear. `evidence_state.py` already produces precise
blocker strings (`holdout_ref_missing`, `raw_artifacts_missing`,
`positive_evidence_decision_failed`); surfacing them is the cheapest possible
credibility win, because it shows the gate actually refusing things.

**F. Run detail `/runs/<id>` and evidence detail `/evidence/<run_id>`**

```
+----------------------------------------------------------------+
| Run #904   agenda #12   idea #221   grant #55                  |
|  OPERATIONAL: completed        SCIENTIFIC: decided - supported  |
+----------------------------------------------------------------+
| Authorization    grant #55, stage pilot                        |
|                  cap 250k tokens / 2.0 GPU-h                   |
|                  used 189k tokens / 1.4 GPU-h                  |
+----------------------------------------------------------------+
| Result           metric 0.487   baseline 0.412   delta +0.075  |
|                  p = 0.013   n = 5 seeds                       |
| Evidence ladder  planned > sanity > benchmark > audited >       |
|                  DECIDED > manuscript allowed                   |
| Audit            raw artifacts   sha256:9f2c...                |
|                  claim ledger    sha256:11ab...                |
|                  evaluator       eval-v3 / sha256:c4d0...      |
|                  holdout         hold-2026-07 / sha256:88fe... |
+----------------------------------------------------------------+
| Artifacts        metrics.json     412 KB   [ download ]        |
|                  training.log     2.1 MB   [ download ]        |
|                  (names and hashes only; no server paths)      |
+----------------------------------------------------------------+
| Limitations      single dataset; no independent replication;   |
|                  evaluator shares preprocessing with training. |
+----------------------------------------------------------------+
| Failures on the way to this result                             |
|  06-04 09:55  audit refused: holdout_ref_missing               |
|  06-03 11:10  attempt 1 timed out after 3600s                  |
+----------------------------------------------------------------+
```

The "Failures on the way" block is deliberate. A result with a visible failure
history is more trustworthy than one presented as a clean success.

### 3.4 Required explicit states

Every list and detail view must be able to render, with distinct styling:

| State | Meaning shown to reader | Source |
|---|---|---|
| Failed | operational failure, no claim | `compute_jobs_v1.status`, `failure_reason` |
| Blocked | a gate refused a transition | `evidence_state.py` blocker strings |
| Paused | awaiting reviewer approval | `reviewer_approval_records` absence |
| Budget-paused | grant exhausted | `resource_grants`, ledger |
| Pending review | human decision outstanding | agenda/direction status |
| Not assessed | job done, no scientific claim | absence of decision record |
| Inconclusive | assessed, no conclusion | verdict `inconclusive` |
| Refuted | hypothesis not supported | verdict `refuted` |
| Limitation | stated caveat on a result | outcome record |
| Missing evidence | expected artifact absent | audit record gaps |

An empty state must say *why* it is empty ("no runs have reached audit yet"),
never just "No data".

---

## 4. Public / private data policy

### 4.1 Field classification

| View | Public | Authorized (operator / submitter) | Never sent to browser |
|---|---|---|---|
| Overview | agenda titles, counts by verdict, decided outcomes, dates | -- | -- |
| Agenda | id, name, description, focus/prefer/reject, status, budget *percentage*, ladder counts | absolute token counts, cost USD, submitter identity, raw config | `raw_config_json` if it embeds credentials |
| Selection | admitted/rejected candidate titles, reason codes, score bands | full scoring breakdown, `rejected_candidates_json` raw | -- |
| Direction | question, why it matters, evidence target, status (once admitted) | reviewer notes, submitter contact | -- |
| Run | operational status, evidence state, timings, effect/baseline/p-value, artifact names + hashes | backend account ref, worker id, session ref, raw logs | `workspace_root`, `experiment_root`, `plan_root`, `paper_root`, `code_dir`, `command_tokens_json`, `binary_path`, `log_tail` |
| Evidence | all content hashes, evaluator ref, holdout ref | -- | raw holdout contents |
| Artifacts | filename, size, sha256, download for whitelisted types | -- | server filesystem paths |

Rule: content hashes are public (they are the point of the audit trail);
filesystem paths never are. A hash proves integrity without revealing layout.

### 4.2 Leaks that exist today and must be fixed

These are current-baseline findings, verified by reading the response builders.

1. **`GET /api/automation` is unauthenticated and returns server paths and raw
   logs.** `_automation_snapshot()` (`web/app.py:671`) includes
   `evoscientist` (`web/app.py:734`), whose builder returns
   `"binary_path": str(evosci_binary_path())` (`web/app.py:706`) and
   `"recent_sessions": sessions`. Each session comes from
   `_recent_evoscientist_sessions()` (`web/app.py:514`) and carries
   `"workdir": str(wd)` plus `log_tail`: the last 2000 bytes of
   `evoscientist.log` read verbatim (`web/app.py:551`), and `log_error` with a
   raw `OSError` string. Absolute paths and unfiltered log content are served to
   any anonymous visitor.
2. **`/api/experiment_groups` and `/api/experiment_groups/<id>` return four
   absolute paths.** `_workspace_payload()` (`web/app.py:321`) emits
   `workspace_root`, `experiment_root`, `plan_root`, `paper_root`, spread into
   the group payload at `web/app.py:486`.
3. **`agenda_id` is scoping, not authorization.** `_required_agenda_query_id()`
   (`web/app.py:62`) parses an integer from the query string and checks only that
   it is positive. Any caller can enumerate agendas by incrementing it. Where the
   data is genuinely public this is acceptable; where it gates workspace paths
   (item 2) it is not.
4. **`_api_failure()` returns raw exception text** to the client
   (`web/app.py:70`) and prints a full traceback to stdout. Exception strings from
   database errors can contain query fragments and schema names.

None of these are new to the upgrade, and none are exploitable for write access.
They are disclosure issues, and they are cheap to fix.

---

## 5. Implementation-readiness audit

### 5.1 What the schema already supports

The `0001_meta_harness_v1.sql` migration is, on inspection, close to a purpose
built provenance store for exactly the chain the task describes. Nothing in the
recommended design requires a new concept.

| Chain step | Existing tables |
|---|---|
| direction | `research_agendas` (`submitter`, `description`, `status`) |
| agenda | `research_agendas`, `agenda_token_ledger`, `agenda_resource_ledger` |
| inputs / signals | `frontier_packets` (coverage, prior art, counterevidence, gate reason codes), `agenda_signal_outcomes` |
| candidates + rationale | `idea_decision_packets` (`decision`, `reason_codes_json`, `revisit_condition_json`), `agenda_selections` (`rationale`, `rejected_candidates_json`, `scoring_breakdown_json`) |
| resource authorization | `resource_grants`, `resource_grant_usage_reservations` |
| jobs / experiments | `compute_jobs_v1`, `colab_work_requests_v1`, `benchmark_harness_jobs`, `scoped_ingestion_jobs_v1` |
| artifacts / evidence | `artifact_manifest_json` columns, `evidence_audit_records` |
| result / failure / limitation | `outcome_records`, `scientific_decision_records`, `evidence_state_transitions`, `failure_clusters` |
| human review | `reviewer_approval_records` |

`frontier_packets` even carries `minimum_falsification_experiment_json` and
`why_not_obsolete`, which are precisely the fields a sceptical reader wants.

### 5.2 The actual gap

The gap is **read APIs and server-rendered views**, not data. Nine of the most
provenance-relevant tables have zero references anywhere in the web layer
(section 2.7). No schema change is required for the recommended design.

Two additions are worth considering, and neither is required for phase 1:

- A `directions` concept distinct from `research_agendas`. Today a submitted
  direction and an approved agenda are the same row, so a rejected direction has
  nowhere to live. Smallest change: add `status` values (`proposed`,
  `rejected`, `withdrawn`) plus a nullable `review_note` column, rather than a
  new table. **Do not implement in phase 1.**
- A public/private flag per agenda, if some agendas must stay unlisted.
  Currently the only defence is that nothing renders them.

### 5.3 Phased plan

**Phase 0 - disclosure fixes (small, independent, ship first)**

- Files: `web/app.py`
- Remove `binary_path`, `workdir`, `log_tail`, `log_error` from
  `_recent_evoscientist_sessions()` and `evoscientist_status()`; replace with
  booleans and ages. Remove the four `*_root` paths from `_workspace_payload()`
  or gate them behind the operator token. Replace raw exception text in
  `_api_failure()` with a generic message plus a server-side correlation id.
- Also: delete the six dead files from the service directory *by redeploying*,
  not by editing in place. Out of scope for this task.
- Contract change: fields removed from two responses. Verify `app.js` does not
  read them first.
- Tests: extend `tests/test_web_app.py` with assertions that no response body
  matches `/home/` or `/api/automation` contains `log_tail`.

**Phase 1 - read APIs for the provenance chain**

- New file: `web/provenance_routes.py`, a read-only blueprint. Keeping it
  separate from `app.py` (2114 lines) avoids growing a file that is already hard
  to audit.
- Endpoints (all GET, all read-only):
  `/api/v1/agendas`, `/api/v1/agendas/<id>`, `/api/v1/agendas/<id>/timeline`,
  `/api/v1/agendas/<id>/selection`, `/api/v1/runs/<id>`,
  `/api/v1/runs/<id>/evidence`, `/api/v1/outcomes`.
- Each response passes through an explicit serializer with an allowlist of
  fields. No `SELECT *` into `jsonify`. This is the single most important
  implementation rule, and it is exactly how the current leaks arose.
- Tests: unit tests per serializer asserting the allowlist; a table-driven test
  that every endpoint's output contains no key in a denylist
  (`workspace_root`, `code_dir`, `log_tail`, `worker_id`, `account_ref`,
  `session_ref`, `binary_path`).

**Phase 2 - public views**

- Files: new `web/templates/overview.html`, `agenda.html`, `timeline.html`,
  `run.html`, `about_method.html`; new `web/static/css/provenance.css`;
  new `web/static/js/provenance.js`. Leave the existing SPA in place at `/app`
  during transition rather than rewriting `index.html` in one step.
- Implement the two-register badge component once, in CSS, and use it everywhere.
  If operational and scientific status can ever be rendered by the same
  component, they will eventually be conflated.
- Vendor D3 and `marked` locally (`web/static/vendor/`) as MathJax already is,
  removing the CDN dependency and the China load-failure risk.

**Phase 3 - direction submission**

- Files: `web/direction_routes.py`, `web/templates/direction_new.html`,
  `direction_status.html`.
- Requires a decision on authentication (see section 7). Until that is decided,
  this phase cannot start. It is deliberately last so that phases 0-2 deliver
  value without blocking on it.
- Every write path reuses `_require_operator()`-style fail-closed checks. No new
  authentication mechanism should be invented for this.

**Phase 4 - restore i18n**

- Port `i18n.js` from `refs/archive/prod-snapshot-20260621` to the new templates.
  Deferred because translating an interface that is still changing wastes effort,
  but it should not be dropped permanently.

### 5.4 Test strategy

- **Unit**: serializer allowlists; two-register badge state mapping, including
  the assertion that `completed` + no decision record renders as "not assessed".
- **Integration**: seed a fixture agenda through every ladder state including a
  blocked transition and a refuted verdict, then assert each view renders the
  correct state; assert unauthenticated access to submission routes returns 403;
  assert operator token absence disables writes entirely.
- **UI**: snapshot the timeline for a fixture with a failure, a pause, and a
  blocked transition. Snapshot an empty agenda to check empty states explain
  themselves.
- **Regression guard**: a test that walks every registered route, calls it
  unauthenticated, and fails if any response body matches server-path or
  log-content patterns. This is what would have caught the current leaks.
- Existing `tests/test_web_app.py` has 9 tests over 218 lines; it is a starting
  point, not adequate coverage for this surface.

### 5.5 Risks and dependencies

- **Highest risk: the honest UI shows less than the current one.** If most runs
  sit at `planned` or `not assessed`, an evidence-first overview will look emptier
  than today's stat wall. That is the correct outcome and the product owner should
  expect it. The temptation to backfill optimistic states must be resisted.
- Vocabulary drift: `outcome_records.verdict`, `agenda_signal_outcomes.verdict`,
  and `scientific_decision_records.verdict` are separate columns. If they can
  disagree, the UI needs a documented precedence rule. Only
  `scientific_decision_records` should drive a public claim.
- Data volume in the timeline is unknown; pagination may be needed. Not
  measurable without querying the database, which is out of scope here.
- Phase 3 blocks on an authentication decision that is not a frontend decision.

---

## 6. Summary of the recommendation

Keep what the upgrade got right: fail-closed mutations, a token-gated operator
API, no runtime topology in public, outcome-first framing.

Fix what it lost: there is no way to submit a direction, and no way to see why
anything was chosen.

Add what neither baseline ever had, and what the product goal actually requires:
a public, linkable, chronological provenance chain that reads the evidence ladder
already implemented in the backend, and that renders operational completion and
scientific confirmation as two separate, non-substitutable things.

The backend is ready. The database is ready. The frontend has not caught up.

---

## 7. Questions requiring product-owner decisions

1. **Authentication for submitters.** There is no user model, no session, and no
   login anywhere in either baseline. Options: (a) operator token only, submissions
   go through an operator; (b) a small allowlist of accounts with sessions;
   (c) an external identity provider. Phase 3 cannot start without this.
   Recommendation: (a) for the first release, because it needs no new
   infrastructure and matches how the system is actually used today.
2. **Default visibility.** Is every agenda public by default, or is publication an
   explicit per-agenda decision? This determines whether the public overview is
   opt-in or opt-out and changes the phase 1 serializers.
3. **How much of the budget is public?** Percentage consumed, absolute tokens, or
   cost in USD? Recommendation: percentage only in public views.
4. **Are rejected candidates public?** Showing them is the strongest
   trustworthiness signal in the design, but it publishes negative judgements
   about ideas, possibly including ones submitted by named people.
5. **Are raw artifacts downloadable, or only their hashes?** Hashes alone are
   safe; downloads are far more useful and much harder to make safe.
6. **Confirm the baseline commit.** Master has moved from `3f9ddbf` (named in the
   task) to `a6973b8`. This report analyses `a6973b8`.
7. **Should the old frontend be recoverable?** The live service already runs the
   new frontend and the six old files there are dead code. Confirm no rollback to
   the old UI is intended, so they can be removed at the next deployment.
8. **Is i18n a requirement or a nice-to-have?** It was deleted in the upgrade.
   Restoring it late is expensive; committing to it now changes how every new
   template is written.

---

## Appendix: verification commands used

All commands were read-only. No command wrote to `/home/billion-token/Deepgraph`,
touched the database, or contacted the network.

```
git for-each-ref
git ls-tree -r --name-only <ref> -- web/
git diff --stat refs/archive/prod-snapshot-20260621 refs/archive/topic-gate-20260729 -- web/
git show <ref>:<path>
md5sum <repo file> <service file>
git -C /home/billion-token/Deepgraph rev-parse HEAD
git -C /home/billion-token/Deepgraph status --porcelain
grep / sed over web/*.py, web/templates/, web/static/, db/migrations/, contracts/
```

Stopping here for review, as instructed. No implementation has begun.
