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
| 9 | backend/LLM failures never complete/confirm | fail-closed route/backend contracts; refreshed synthetic fault lane is 60 passed; PostgreSQL/provider execution remains pending | policy/fault passed; PostgreSQL pending |
| 10 | harness patch passes three suites | 71 pure policy tests; validation-loop fairness/manifest lane is 22 passed; real held-in/held-out/canary bubblewrap evaluator passed; adapted legacy lane is 39 passed / 30 failed and individually classified as obsolete/new-contract mismatches | rejected / legacy classification recorded |
| 11 | candidate cannot modify protected inputs/data | prior real bubblewrap held-in/held-out/canary lane passed; post-fix rerun preserved candidate tree hash but bwrap failed to create its isolated network (`NETLINK_ROUTE ... Operation not permitted`); mocked contract and no-fallback negative passed | prior lane accepted; fresh rerun blocked by host |
| 12 | restart resumes without duplicate | durable compute/Colab/ingestion claims, safe bind recovery, unknown-outcome quarantine and reconciliation-before-auto-research startup exist | PostgreSQL restart/fault evidence missing |
| 13 | predictions calibrate against outcomes | trusted assembler, non-success usage settlement, prediction errors and Brier/MAE/RMSE report implemented | no real OutcomeRecord sample |
| 14 | negative/zero/no-metric/compile failure cannot promote | scientific contract and refreshed validation-loop fairness/manifest lane passed 22/22; PostgreSQL and full legacy runtime remain pending | policy/fault passed; PostgreSQL pending |
| 15 | minimum Web/API/statistics compatible | count-only status and operator-authenticated mutation API added; temp SQLite app import/startup passed | PostgreSQL/API runtime pending |
| 16 | `7d0b42a` rollback rehearsed | exact immutable ref checked in detached worktree; temp SQLite rollback startup passed | isolated rehearsal passed; production rollback not run |

## Static evidence recorded in this session

- Post-fix verification commit: `d3650fe0a2270eb265ef9dc40041b3ccab537efd`;
  Git tree `7e3183cd039cb7bace420355a6aad6b0a67f1358`, tracked-content tree
  SHA-256 `c9a2efac23e30abda6c9ab87242d76ecfa66d6679272404fbf27402d86db6114`.
  The source freeze commit `6851a991154906f11d8cfc247d22a5d5caa0a834` and
  `integration/meta-harness-v1@77e8ac0` remain unchanged.
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
- post-fix static report hashes: static
  `91c9dcc5af43b4b439191175a4dc6024fdb32e542de6e3a0d98d88416e8d564c`, scope
  `0f8de09da9f5d057f1c3141eb84a47ff1863a7a908623e9358e6123ed3491e7f`, SQL
  `99a7c03d4c21f759e7ecd0c563c6c9946f62a939e91633172b31304ac288e5db`, state
  `ab02a1636e736faa8fba8ec8aa00e3df8d0204857e5b74c4907557839d84acc5`, LLM
  `7ea25bcbf45728afb0c11edaf092dd32ecb28da53a045c2b6ce0b95ad751e5e4`, and
  migration `a6bb48b17dc02ac0269097f44d15c69129fe1a414fdbb0000ed663a0a34e7009`.
- no database or application was accessed by the static checks. A separate
  temp-SQLite startup smoke imported `main` and `web.app` successfully.
- agenda example JSON parsed successfully; the application startup smoke used
  the isolated Python 3.13 environment.

These values must be regenerated at the final candidate commit and included
with the commit hash; intermediate counts/checksums are not release evidence.

The adapted-legacy failure-by-failure decision is recorded in
[LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md). It is not a
waiver for removed unscoped, grantless, password-bearing or unlimited
contracts.

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
  `76c9e6d391f2e6d62b21e29a49b6fb56125a043adaf660d5fd9d391f55cd2669`.
- Targeted synthetic fault suite after the validation-loop guard repair:
  `60 passed` (report SHA-256
  `d41e3e25974b8dd8b9343e3b0b81cfe6a0a75088aac7017823f410a055d30f97`);
  the validation-loop subset is `22 passed` (report SHA-256
  `8c7f1fdad2f6e5c3a60fb237d75de6e9f9d84e96af3022b6b274f4b843168075`).
  The adapted legacy lane remains `39 passed, 30 failed`; every failure is
  classified in [LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md)
  and no safety contract was weakened.
- Real bubblewrap evaluator: held-in, held-out and canary all passed with
  evaluator hash `45bddd4cbd5eba5ba6a6377b765debcd4e67fa15fcab879bbb321c752d6c362d`;
  protected-write, network and missing-bwrap fallback negative checks passed.
- PostgreSQL integration tests were guarded and skipped in separate processes
  (migration 1 skipped, compute 4 skipped, evidence 1 skipped; report hashes
  `d8e17839871bd0d3b856105bc4462f330eb464d72898ee3d71a1be24ab3018f7`,
  `9bd69a10017890a84349f080bd2ffaea2a064772959dce88c1a9b7927bc6f98e`, and
  `f0578970c449ae746a699b3d622f1e6eca6ed1c223ad6c53898528d57055a27e`). No
  local server, initdb/pg_ctl, or usable Docker daemon was available, so
  restore, migration twice, count preservation, concurrency, restart and
  PostgreSQL fault lanes remain unverified.
- Actual CPU/GPU/Colab canaries were not run. The Colab CLI was inspected only
  for contract/help availability; no OAuth, provider, production or remote
  credentials were used.
- The exact `7d0b42a` detached rollback worktree was clean and its isolated
  temp-SQLite application startup passed. No production rollback was run.
