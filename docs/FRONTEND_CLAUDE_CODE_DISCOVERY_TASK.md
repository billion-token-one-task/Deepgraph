# Claude Code task: DeepGraph frontend discovery and recommendation

Read and follow this entire file before taking action. This is a **discovery
and product-design task**, not an implementation or deployment task.

## Objective

Compare the current DeepGraph frontend with the frontend that was deployed
before the master upgrade. Recommend a coherent target interface that does two
things well:

1. lets authorized people submit research/exploration directions; and
2. lets an interested community member understand what the system did, in what
   order, from which inputs, with what evidence, failures, limitations, and
   results.

Trustworthiness matters more than visual novelty. The interface must never make
an operational completion look like a scientifically confirmed result.

## Repository and runtime context

- Candidate repository: `/home/ec2-user/Deepgraph-meta-harness-v1`
- Current `master` / `origin/master` at task creation:
  `3f9ddbf4142e22926eea4a431b0b6bca0dfc7c88`
- Current service directory: `/home/billion-token/Deepgraph`
- Historical reference refs:
  - `refs/archive/prod-snapshot-20260621`
  - `refs/archive/koen-master-20260626`
  - `refs/archive/topic-gate-20260729`

The service directory is a long-lived historical snapshot with many pre-existing
uncommitted changes. It is evidence of the previously deployed frontend, not a
clean Git source. Do not use its Git `HEAD` alone to identify the old UI; compare
the actual templates, static assets, routes, and API calls.

The live service uses the original local PostgreSQL `deepgraph` database, which
has already completed the add-only schema migration. The database and service
are out of scope for this task.

## Non-negotiable safety boundaries

- Do not modify `/home/billion-token/Deepgraph`.
- Do not alter `.env`, systemd units, service processes, timers, database
  configuration, migrations, backups, or any database rows.
- Do not connect to a remote production database. Do not print secrets, tokens,
  passwords, or complete connection URLs.
- Do not delete databases, files, branches, tags, or refs.
- Do not reset, checkout, clean, merge, rebase, or otherwise rewrite the
  service worktree.
- Do not push, force-push, merge `master`, or change remote refs.
- Do not run a live deployment or restart `deepgraph-web.service`.

If a browser or screenshots are needed, use source inspection or a disposable
local preview only. Do not disturb the running service.

## Phase 1: required analysis

Perform only read-only analysis and create a written recommendation. Do not
change application code in this phase.

### 1. Establish the two frontend baselines

Identify the current-master frontend and the pre-upgrade deployed frontend by
examining, at minimum:

- Flask/web routes and their authorization boundaries;
- HTML templates;
- CSS and JavaScript assets;
- frontend API calls and response shapes;
- relevant frontend-related historical refs; and
- the actual deployed service assets, read-only.

Document exactly which files/revisions represent each baseline and any ambiguity
that cannot be resolved from local evidence.

### 2. Produce a comparison matrix

For each meaningful page, route, and workflow, compare:

- purpose and target audience;
- available actions and authorization behavior;
- information architecture and navigation;
- readability and ability to orient a first-time visitor;
- process transparency and provenance;
- treatment of failures, pauses, uncertainty, and missing evidence;
- regressions in the current frontend and strengths worth retaining from the
  earlier frontend;
- API/data dependencies and missing data.

Do not call an element an improvement merely because it is newer. Distinguish
observed facts from design inference.

### 3. Recommend a target product experience

Propose the smallest coherent interface that supports the two product goals.
It should include a text wireframe and an information architecture for at least:

1. a public overview page;
2. an authorized exploration-direction submission and status workflow;
3. an Agenda detail page;
4. a chronological work/process timeline;
5. task, experiment, artifact, and evidence detail views; and
6. explicit states for failure, pause, pending human review, uncertainty, and
   limitations.

The process view should make this chain understandable without exposing private
data:

`direction -> Agenda -> inputs/signals -> candidates and selection rationale ->
resource authorization -> jobs/experiments -> artifacts/evidence -> result,
failure, limitation, or next step`.

For every view, state which fields may be public, which require authorization,
and which must never be sent to the browser (for example secrets, private raw
materials, internal absolute paths, or unsafe operator controls).

### 4. Assess implementation readiness

Audit whether existing routes, APIs, and persisted schema can support the
recommended design. If a capability is absent, propose the smallest necessary
API or schema change, but do not implement it and do not run a migration.

Give a phased implementation plan with:

- exact files likely to change;
- API/data contract changes;
- authorization and privacy checks;
- test plan (unit, integration, UI); and
- risks, dependencies, and product decisions requiring approval.

## Deliverable

Create a single Markdown report at:

`docs/frontend/FRONTEND_DISCOVERY_AND_RECOMMENDATION.md`

The report must contain:

1. baseline identification and evidence;
2. comparison matrix;
3. recommended target UX and text wireframes;
4. public/private data policy;
5. implementation-readiness audit;
6. phased plan and test strategy; and
7. explicit questions that require product-owner decisions.

At the end, stop and wait for approval. Do not begin implementation in the same
task/session unless the product owner explicitly authorizes it after reviewing
the report.
