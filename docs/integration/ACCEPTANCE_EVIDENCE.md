# Master acceptance evidence

Decision: **REJECTED — not eligible to replace master**.

`Implemented` below means code/test material exists. It does not mean runtime
verified. Only isolated CI/canary evidence can change a `pending` item to
accepted. Record that evidence using
[ISOLATED_CI_EVIDENCE_TEMPLATE.md](ISOLATED_CI_EVIDENCE_TEMPLATE.md).

| # | Gate | Current evidence | Status |
|---|---|---|---|
| 1 | fixed results and caveat weakening removed | symbols removed; generic topic scan, integrity fixtures and 71-test policy lane passed | accepted in isolated policy lane |
| 2 | production backup starts; add-only repeatable migration | guarded additive SQL, checksum and twice-run test written | pending PostgreSQL restore |
| 3 | tests cannot touch production DB | test entry forcibly clears production DB URL; static audit and isolation test passed | accepted for unit/adapted entry; PostgreSQL enforcement pending |
| 4 | generation/consumption only inside agenda | selector/orchestrator/problem/idea/core queues require `agenda_id`; 154-mutation scope audit is clean | static/policy passed; PostgreSQL/fault CI pending |
| 5 | old backlog excluded | migration leaves scope null; explicit import ledger only | implemented, PostgreSQL pending |
| 6 | core objects carry correct `agenda_id` | migration/contracts/repositories and all literal legacy mutations are explicitly scoped | implemented, PostgreSQL CI pending |
| 7 | Frontier Gate rejects obsolete/duplicate | gate, persisted decision, API response and bypass prevention implemented | CI pending |
| 8 | pilot/GPU/full benchmark require grant | proposal/ingestion/post-agenda LLM roles and CPU/local/SSH/Colab durable admission are grant-scoped; PostgreSQL legacy GPU queue requires the compute identity | policy/static passed; PostgreSQL runtime pending |
| 9 | backend/LLM failures never complete/confirm | fail-closed route/backend contracts; synthetic targeted backend/router tests passed; aggregate validation-fault report has 5 stale failures | open |
| 10 | harness patch passes three suites | 71 pure policy tests passed; real held-in/held-out/canary bubblewrap evaluator passed; adapted legacy lane was 39 passed / 30 failed | rejected / legacy compatibility open |
| 11 | candidate cannot modify protected inputs/data | real bubblewrap held-in/held-out/canary plus protected-write, network and no-fallback negatives passed | accepted in isolated evaluator lane |
| 12 | restart resumes without duplicate | durable compute/Colab/ingestion claims, safe bind recovery, unknown-outcome quarantine and reconciliation-before-auto-research startup exist | PostgreSQL restart/fault evidence missing |
| 13 | predictions calibrate against outcomes | trusted assembler, non-success usage settlement, prediction errors and Brier/MAE/RMSE report implemented | no real OutcomeRecord sample |
| 14 | negative/zero/no-metric/compile failure cannot promote | scientific contract and fixtures passed in policy lane; legacy validation aggregate has stale failures | open |
| 15 | minimum Web/API/statistics compatible | count-only status and operator-authenticated mutation API added; temp SQLite app import/startup passed | PostgreSQL/API runtime pending |
| 16 | `7d0b42a` rollback rehearsed | exact immutable ref checked in detached worktree; temp SQLite rollback startup passed | isolated rehearsal passed; production rollback not run |

## Static evidence recorded in this session

- 260 Python files parsed by the final static AST audit;
- static topic/integrity/migration/secret audit passed at that checkpoint;
- migration dry plan reported 90 statements, 27,734 bytes, SHA-256
  `3b73e0647c5edfb13f82efbba79081b29f19734a4504a4327e4eabdbf06241f0`,
  and no destructive token;
- SQL AST audit found no definite placeholder mismatch across 836 countable
  literal calls; 114 dynamic calls remain review/CI scope;
- agenda mutation audit found 154 explicitly scoped literal mutations and no
  definite unscoped or dynamic mutation;
- scientific-state authority audit found two state-bearing SQL literals and
  zero unauthorized mutation locations;
- direct-LLM audit classified all 14 legacy direct calls with zero ingestion
  and zero unclassified calls; those 14 remain isolated legacy surfaces;
- no database or application was accessed by the static checks. A separate
  temp-SQLite startup smoke imported `main` and `web.app` successfully.
- agenda example JSON parsed successfully; the application startup smoke used
  the isolated Python 3.13 environment.

These values must be regenerated at the final candidate commit and included
with the commit hash; intermediate counts/checksums are not release evidence.

Implementation checkpoint for the new durable queues and isolated evaluator:
`724a3ed51fe4649a720c08fb0c213014eb9d236a`. It is local, unpushed and not an
acceptance identifier. The ingestion completion-truth follow-up is
`b17c7d110532197a7137a217ffa641b50486d295`, followed by the per-agenda
serialization guard `f2f6ea96d27673f077966b4bd2f278717393b0d9`; both are
also local and unpushed.

## Runtime evidence recorded in this session

- Frozen source candidate: `6851a991154906f11d8cfc247d22a5d5caa0a834`; the
  candidate tree hash before and after real evaluator execution was
  `b1c1e8ebfbc0607cc39bb617dad9d56fd949d214122a37dd70541bd634d9feab`.
- Pure policy/adapted-entry lane: `71 passed`; report SHA-256
  `78a9fb6e0724ec1267056778537914642069d95d03d68eb599bfc7e73e33b4d7`.
- Targeted synthetic backend/provider suite: `53 passed`; a broader fault
  collection was `55 passed, 5 failed`, with all five failures in stale
  validation-loop fairness/manifest expectations. The adapted legacy lane was
  `39 passed, 30 failed`, due removed APIs, required grant arguments and old
  fallback expectations; it was not weakened to make those tests pass.
- Real bubblewrap evaluator: held-in, held-out and canary all passed with
  evaluator hash `45bddd4cbd5eba5ba6a6377b765debcd4e67fa15fcab879bbb321c752d6c362d`;
  protected-write, network and missing-bwrap fallback negative checks passed.
- PostgreSQL integration tests were guarded and skipped. No local server,
  initdb/pg_ctl, or usable Docker daemon was available, so restore, migration
  twice, count preservation, concurrency, restart and PostgreSQL fault lanes
  remain unverified.
- Actual CPU/GPU/Colab canaries were not run. The Colab CLI was inspected only
  for contract/help availability; no OAuth, provider, production or remote
  credentials were used.
- The exact `7d0b42a` detached rollback worktree was clean and its isolated
  temp-SQLite application startup passed. No production rollback was run.
