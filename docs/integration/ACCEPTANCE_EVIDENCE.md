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
| 4 | generation/consumption only inside agenda | selector/orchestrator/problem/idea/core queues require `agenda_id` | partial; exhaustive legacy call audit pending |
| 5 | old backlog excluded | migration leaves scope null; explicit import ledger only | implemented, PostgreSQL pending |
| 6 | core objects carry correct `agenda_id` | migration/contracts/repositories scoped, including evidence/harness artifacts | partial; legacy surface audit pending |
| 7 | Frontier Gate rejects obsolete/duplicate | gate, persisted decision, API response and bypass prevention implemented | CI pending |
| 8 | pilot/GPU/full benchmark require grant | durable compute plus forge scout/scaffold/repair and validation LLM repair paths are grant-scoped | partial; pre-idea LLM and legacy scheduler integration remain |
| 9 | backend/LLM failures never complete/confirm | fail-closed route/backend contracts; GPU validation failure now marks job failed; operational positive verdict is `supported` | fault CI pending |
| 10 | harness patch passes three suites | policy and RegressionReport require all three plus reviewer | no suite executed |
| 11 | candidate cannot modify protected inputs/data | path/environment/namespace policy implemented | isolation CI pending |
| 12 | restart resumes without duplicate | durable claim-before-submit, live-job reuse, unknown-outcome quarantine and guarded PostgreSQL test exist | startup wiring and isolated crash evidence missing |
| 13 | predictions calibrate against outcomes | trusted persistence assembler, prediction errors and Brier/MAE/RMSE report implemented | no real outcomes; failed-compute usage capture open |
| 14 | negative/zero/no-metric/compile failure cannot promote | scientific contract, manuscript gate and fixtures implemented | CI pending |
| 15 | minimum Web/API/statistics compatible | count-only status and operator-authenticated mutation API added | runtime/API CI pending |
| 16 | `7d0b42a` rollback rehearsed | immutable ref and runbook recorded | not rehearsed |

## Static evidence recorded in this session

- 239 Python files parsed successfully at the latest checkpoint;
- static topic/integrity/migration/secret audit passed at that checkpoint;
- migration dry plan reported 81 statements, SHA-256
  `dcdf8fcce3113a36f8c652b5f015135921b8541c68523e6b01cb576e0c8aecb9`,
  and no destructive token;
- SQL AST audit found no definite placeholder mismatch across 754 countable
  literal calls; 109 dynamic calls remain review/CI scope;
- no database or application was accessed by those checks.
- agenda example JSON parsed successfully; `deepgraph.toml` was text-reviewed
  but not runtime-parsed because the host Python 3.9 has no TOML parser and
  dependency installation is forbidden.

These values must be regenerated at the final candidate commit and included
with the commit hash; intermediate counts/checksums are not release evidence.
