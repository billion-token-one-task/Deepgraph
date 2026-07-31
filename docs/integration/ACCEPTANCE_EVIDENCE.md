# Master acceptance evidence

Decision: **not eligible to replace master**.

`Implemented` below means code/test material exists. It does not mean runtime
verified. Only isolated CI/canary evidence can change a `pending` item to
accepted.

| # | Gate | Current evidence | Status |
|---|---|---|---|
| 1 | fixed results and caveat weakening removed | symbols removed; generic topic scan and integrity fixtures added | implemented, CI pending |
| 2 | production backup starts; add-only repeatable migration | guarded additive SQL, checksum and twice-run test written | pending PostgreSQL restore |
| 3 | tests cannot touch production DB | isolated URL/ack/name guards documented and coded | pending CI enforcement audit |
| 4 | generation/consumption only inside agenda | selector/orchestrator/problem/idea/core queues require `agenda_id`; 134-mutation scope audit is clean | implemented, PostgreSQL/fault CI pending |
| 5 | old backlog excluded | migration leaves scope null; explicit import ledger only | implemented, PostgreSQL pending |
| 6 | core objects carry correct `agenda_id` | migration/contracts/repositories and all literal legacy mutations are explicitly scoped | implemented, PostgreSQL CI pending |
| 7 | Frontier Gate rejects obsolete/duplicate | gate, persisted decision, API response and bypass prevention implemented | CI pending |
| 8 | pilot/GPU/full benchmark require grant | proposal identity, post-agenda LLM role routes and CPU/local/SSH durable compute admission are grant-scoped | partial; 10 pre-agenda ingestion LLM sites need bounded authority, Colab durable runtime and all isolated execution remain |
| 9 | backend/LLM failures never complete/confirm | fail-closed route/backend contracts; GPU validation failure now marks job failed; operational positive verdict is `supported` | fault CI pending |
| 10 | harness patch passes three suites | policy and RegressionReport require all three plus reviewer | no suite executed |
| 11 | candidate cannot modify protected inputs/data | path/environment/namespace policy implemented | isolation CI pending |
| 12 | restart resumes without duplicate | durable claim-before-submit, live-job reuse, unknown-outcome quarantine and per-agenda startup reconciliation exist | isolated crash evidence missing |
| 13 | predictions calibrate against outcomes | trusted assembler, non-success usage settlement, prediction errors and Brier/MAE/RMSE report implemented | no real outcomes/runtime evidence |
| 14 | negative/zero/no-metric/compile failure cannot promote | scientific contract, manuscript gate and fixtures implemented | CI pending |
| 15 | minimum Web/API/statistics compatible | count-only status and operator-authenticated mutation API added; legacy control POSTs return 410 and agenda-owned reads require scope | runtime/API CI pending |
| 16 | `7d0b42a` rollback rehearsed | immutable ref and runbook recorded | not rehearsed |

## Static evidence recorded in this session

- 258 Python files parsed by the broad AST pass and 248 by the release static
  audit at the latest checkpoint;
- static topic/integrity/migration/secret audit passed at that checkpoint;
- migration dry plan reported 84 statements, SHA-256
  `f0fcc7680ad211774d53d40179c34cf01044537d009407e5d58e6a74c7c862a2`,
  and no destructive token;
- SQL AST audit found no definite placeholder mismatch across 793 countable
  literal calls; 112 dynamic calls remain review/CI scope;
- agenda mutation audit found 138 explicitly scoped literal mutations and no
  definite unscoped or dynamic mutation;
- scientific-state authority audit found two state-bearing SQL literals and
  zero unauthorized mutation locations;
- direct-LLM audit classified all 24 legacy direct calls with zero
  unclassified; 10 pre-agenda ingestion calls remain an explicit authority
  blocker rather than accepted evidence;
- no database or application was accessed by those checks.
- agenda example JSON parsed successfully; `deepgraph.toml` was text-reviewed
  but not runtime-parsed because the host Python 3.9 has no TOML parser and
  dependency installation is forbidden.

These values must be regenerated at the final candidate commit and included
with the commit hash; intermediate counts/checksums are not release evidence.
