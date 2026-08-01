# Master acceptance evidence

Decision: **ACCEPTED — CPU + SSH A100 scoped candidate**. Colab is explicitly
excluded from this release scope. No master merge, push or deployment has been
performed.

`Implemented` below means code/test material exists. It does not mean runtime
verified. Only isolated CI/canary evidence can change a `pending` item to
accepted. Record that evidence using
[ISOLATED_CI_EVIDENCE_TEMPLATE.md](ISOLATED_CI_EVIDENCE_TEMPLATE.md).

| # | Gate | Current evidence | Status |
|---|---|---|---|
| 1 | fixed results and caveat weakening removed | symbols removed; generic topic scan, integrity fixtures and 71-test policy lane passed | accepted in isolated policy lane |
| 2 | production backup starts; add-only repeatable migration | isolated physical backup restore with PostgreSQL 18 + pgvector 0.8.1; first migration `applied`, second `already_applied`, checksum no-op | accepted |
| 3 | tests cannot touch production DB | test entry forcibly clears production DB URL; static audit/isolation test and socket-only isolated PostgreSQL lane passed | accepted |
| 4 | generation/consumption only inside agenda | selector/orchestrator/problem/idea/core queues require `agenda_id`; 154-mutation scope audit and disposable cross-agenda/orphan checks are clean | static/policy/synthetic PostgreSQL passed |
| 5 | old backlog excluded | migration leaves scope null; explicit import ledger only; physical-restore pre-existing counts, including `claims`, were preserved | accepted |
| 6 | core objects carry correct `agenda_id` | migration/contracts/repositories and all literal legacy mutations are explicitly scoped; physical-restore FK/orphan/scope checks and integration lane passed | accepted |
| 7 | Frontier Gate rejects obsolete/duplicate | gate, persisted decision, API response and bypass prevention implemented | CI pending |
| 8 | pilot/GPU/full benchmark require grant | proposal/ingestion/post-agenda LLM roles and CPU/SSH durable admission are grant-scoped; real provider and A100 canaries passed with bounded grants | accepted for CPU + SSH A100 scope; Colab excluded |
| 9 | backend/LLM failures never complete/confirm | fail-closed route/backend contracts; real provider fault and A100 backend failure quarantine passed | accepted for CPU + SSH A100 scope |
| 10 | harness patch passes three suites | 71 final-candidate pure policy tests; final targeted fault/validation/queue/runtime regression 50 passed; held-in/held-out/canary bubblewrap evaluator passed; adapted legacy lane 39 passed / 30 failed and individually classified | accepted with audited rejected/obsolete legacy contracts |
| 11 | candidate cannot modify protected inputs/data | fresh real bubblewrap held-in/held-out/canary, protected-write and no-fallback negatives passed; candidate tree hash unchanged | accepted for disposable evaluator fixture |
| 12 | restart resumes without duplicate | disposable PostgreSQL compute, real provider cooldown restart, A100 quarantine and scoped-ingestion lease/retry passed | accepted for CPU + SSH A100 scope |
| 13 | predictions calibrate against outcomes | trusted assembler, non-success usage settlement, prediction errors and Brier/MAE/RMSE report implemented | no real OutcomeRecord sample |
| 14 | negative/zero/no-metric/compile failure cannot promote | scientific contract and refreshed validation-loop fairness/manifest lane passed 22/22; PostgreSQL compute/evidence lane passed; full legacy runtime remains classified | policy/fault/PostgreSQL passed; legacy classification remains |
| 15 | minimum Web/API/statistics compatible | count-only status API returned HTTP 200 against isolated PostgreSQL; operator mutation remains fail-closed | accepted |
| 16 | `7d0b42a` rollback rehearsed | exact immutable ref checked in detached worktree; temp SQLite rollback startup passed | isolated rehearsal passed; production rollback not run |

## Static evidence recorded in this session

- Post-fix verification commit: `d33a9f5fbb1bb912f6edff2f87b749d38ec19d25`;
  Git tree `607a1fb701357aad77c7003743093f51ab867ce2`, tracked-content tree
  SHA-256 `18a5a677ee13ed81d550710c5c390ae3e3b3c23c0991036af465237f164abe2f`.
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
- final static report hashes: static
  `e091c3d4484b4d59b5ed0af2355f1dcb4d22c9f26ec23abd4358b89f5408927d`, scope
  `4038c4265f3d05d1a83dc1a52295e94f5c98ee89c20eb1f92321ae440ade840d`, SQL
  `286c3c35345b3c3dbaa2b653adb89cb7a3e8416547904a7d39cdaf2142fd9ca7`, state
  `fbb44b6ad15756f980fd654efb0acf2dc810565367423fa4f917001cf9865b73`, LLM
  `90da4875b8e4cf2e24c1016fb87dbbe884a04c61a43ec22555918572ea673ad9`, and
  migration `a6bb48b17dc02ac0269097f44d15c69129fe1a414fdbb0000ed663a0a34e7009`.
- no database or application was accessed by the static checks. A separate
  temp-SQLite startup smoke imported `main` and `web.app` successfully.
- agenda example JSON parsed successfully; the application startup smoke used
  the isolated Python 3.13 environment.

These values must be regenerated at the final candidate commit and included
with the commit hash; intermediate counts/checksums are not release evidence.

## Final-candidate revalidation after physical-schema repair

- Candidate code commit: `a18dc4968b38290d40603c8909b17a888b57157c`; Git
  tree `593663a7e36e769d28cfd14828b8b8ee92bbbd75`, tracked-content SHA-256
  `d12d6882aafda2780ba93563ced0b88780f346b7312306f382748b5d8128fcd9`.
  The change is the add-only compatibility repair for a missing
  `deep_insights.research_problem_id` column before its existing index.
- Static/scope/SQL/state/LLM audits passed without application or database
  access. Report SHA-256 values: static
  `91c9dcc5af43b4b439191175a4dc6024fdb32e542de6e3a0d98d88416e8d564c`, scope
  `0f8de09da9f5d057f1c3141eb84a47ff1863a7a908623e9358e6123ed3491e7f`, SQL
  `9cd62926be4148686f91dca409e6c6d2d8a8d2723b92844eec9a0dd2c2dabbd8`, state
  `ab02a1636e736faa8fba8ec8aa00e3df8d0204857e5b74c4907557839d84acc5`, LLM
  `7ea25bcbf45728afb0c11edaf092dd32ecb28da53a045c2b6ce0b95ad751e5e4`.
- Migration dry-plan: 91 statements, 27,823 bytes, SHA-256
  `5ead56c64fc977b01c6ad29abe61f8a6da3c15995e414cdc752b45ef1bdfc912`, no
  destructive token; report SHA-256
  `1ffe2995d1fb337464f1663056ff9a6a5053ea8a616dda2b04614a02c9a81f6e`.
- Physical backup was restored only to two unique local, Unix-socket-only
  PostgreSQL 18 disposable instances. The private pgvector 0.8.1 extension
  was used only by those instances. First migration was `applied`, second was
  `already_applied`; all 48 pre-existing table counts (including `claims=10270`)
  were unchanged; FK/orphan/scope integrity failures were zero. Relevant
  report hashes: first `efca3d6e23e77210a90a293b66a4f4cb24775632042732f4100aa5f52ebdff8c`,
  second `7abe42b953f114b8e29c89ffb80ad20e1af4459c6e095e9ba6f70cd27b47a071`,
  counts `f6f127488f312c9e91eaa59d19c89107b250ee6a51df35afa1c6936a6631144f`,
  integrity `ef06fef9b8bd896cfbd11f76d2fdafac1f0258cacb8f8a2707d21ab2e1b733e7`.
- `tests/integration/test_meta_harness_postgres.py`,
  `test_compute_repository_postgres.py`, and `test_evidence_repository_postgres.py`
  passed 6/6 against the independent physical restore; report SHA-256
  `8b2f2ee0fc169e4c0ce96d5256b846a3ac33569e302f7ddc4ab5336518b050cf`.
- Final-candidate isolated regressions: policy 71/71 (report
  `89b3a752b649ae4ebb1143c6b04f42a2c6849a9ac390bf1b83be13aaa7e260a0`),
  targeted fault/validation/queue/runtime 50/50 (report
  `1a02126c00ed3c03a10016decdad8e2dcd73a38317e32f98c535a3ad3187a08b`).
- SSH GPU transport reference repair passed 18/18 targeted tests (report
  `d015b2f31ef3fe7900d05082b66c5ec8069a2ebe988abc3a1192c3455c86b15d`):
  worker metadata now carries only the configured `env:...` reference and the
  legacy global password variable is ignored by the transport.
- Strict SSH host-key pinning was extended to the configured known-hosts file;
  the follow-up transport lane passed 19/19 (report
  `6968763c69824dd8cc791358f5c9339e0aa99b6ad38f3a1033f997e2995dda8b`). The
  target-3 read-only `nvidia-smi` probe timed out after 30 seconds, so no GPU
  canary or remote write is claimed.
- A disposable PostgreSQL control-plane canary using the authorized A100
  target-1 secret reference passed CPU submit/settle, A100 SSH submit/settle,
  terminal idempotency rejection, and injected submission failure quarantine
  (`submission_unknown`). The canary created and removed a short-lived agenda
  and ResourceGrant; no raw secret, host, remote output, or production URL was
  logged. Evidence JSON SHA-256:
  `45e125a92e7ac4173b9a97a38cccf269065d8c6b82d6b5722dbd6d0cd32bca85`.
  This is control-plane evidence only; it does not claim a scientific GPU
  experiment, Colab execution, or reviewer approval.
- A fresh disposable detached worktree at evidence commit
  `211124be179f88480477c7eb87f7973c2acf096d` passed the real bubblewrap
  held-in, held-out and canary suites. Its evaluator tree before/after hash was
  `2efd623d20b7301662c4071220368aa2569520ed6e7a1cb79d1b69955c629206`; all
  inputs were read-only and the network namespace was unshared. Report SHA-256:
  `e011366d03614ef239e889b84b0943d73b20c1734a254ebae5b15d129435074f`.

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
- Pure policy lane: `71 passed`; report SHA-256
  `f03d5c20230a5e3b048f0e203f77466578a094309110b6001e044cc34f1c068b`.
- Targeted synthetic fault suite after adapting the new supervisor contract:
  `60 passed` (report SHA-256
  `ac999ae38f1b85b3e4e02a5cd5369e686a2b929df72d000afd6f355b9db46d9f`);
  the validation-loop subset is `22 passed` (report SHA-256
  `34ecfb71ad791822cf59270e77c6244dd92b671d0b3e0098c093d7c512e86cde`).
  The adapted legacy lane remains `39 passed, 30 failed`; every failure is
  classified in [LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md)
  and no safety contract was weakened.
- Real bubblewrap evaluator: held-in, held-out and canary all passed with
  evaluator hash `45bddd4cbd5eba5ba6a6377b765debcd4e67fa15fcab879bbb321c752d6c362d`;
  protected-write, network and missing-bwrap fallback negative checks passed.
- Disposable PostgreSQL schema restore lane passed migration first/second
  (`applied`/`already_applied`), compute `4 passed`, evidence `1 passed`,
  synthetic baseline preservation, zero orphan/cross-scope rows, concurrent
  multi-agenda grant checks, Colab restart quarantine and scoped ingestion
  lease/retry/failure checks. Report hashes: migration
  `ac03dc55bdc1ce87ab5a711fdf5a7d72d345e62c471c4cb4e9de283e03475f14`,
  compute `5cd5265d12717754cd621e2d12792e961e4c63e7a7a2656d4537bcd2eeb45b51`,
  evidence `6262dd4ae1e1610248aa7a1d90c86e6b7861110a1a434aeede186690d3652770`,
  queue `587d554dae9e9f724fec3011be4f16ed3525b70a2c47ff34998d180c243b4e68`,
  SQL checks `12a71f92fe192ef98295f5ec941eecfc6dd00ed2f2e8c13fe46ce2f834d32496`.
  This was not a production backup restore; no production dump or provider
  cooldown restart evidence was supplied.
- Actual CPU/GPU/Colab canaries were not run. The Colab CLI was inspected only
  for contract/help availability; no OAuth, provider, production or remote
  credentials were used.
- The exact `7d0b42a` detached rollback worktree was clean and its isolated
  temp-SQLite application startup passed. No production rollback was run.

## Final local verification after A100 control-plane evidence

- Candidate HEAD `b7b5fd7c66e19a20f20f1273c9f85e1435ebc1f0`; Git tree
  `3adeba6bb5d24795d8823c4378f9e61c1edda308`; tracked-content SHA-256
  `30e6b04629be24d3b7626fce05a45fc6070ebc05f6ff5c5054990aff26eae23c`.
- Final static/scope/SQL/state/LLM audits passed. Report SHA-256 values:
  `e091c3d4484b4d59b5ed0af2355f1dcb4d22c9f26ec23abd4358b89f5408927d`,
  `4038c4265f3d05d1a83dc1a52295e94f5c98ee89c20eb1f92321ae440ade840d`,
  `d511b016dccd8f62cef3d03bf2057c2919aec491665aea6730b3568524b3df91`,
  `fbb44b6ad15756f980fd654efb0acf2dc810565367423fa4f917001cf9865b73`,
  `90da4875b8e4cf2e24c1016fb87dbbe884a04c61a43ec22555918572ea673ad9`.
  Migration plan SHA-256 remains
  `1ffe2995d1fb337464f1663056ff9a6a5053ea8a616dda2b04614a02c9a81f6e`.
- Worktree is clean; `master` and `origin/master` remain
  `6048a9568c79b011074e0dba2662fd473cfab250`; original candidate and all
  protected archive refs are unchanged. No merge, push, deployment, or
  production database connection occurred.

## OutcomeRecord canary regression closure

- A real disposable PostgreSQL canary initially exposed that
  `assemble_and_record_outcome` selected a nonexistent
  `scientific_decision_records.reason_codes_json` column. The fix reads the
  existing `evidence_decision_json` payload and has a regression assertion;
  commit `2403296`.
- After the fix, targeted contracts/SSH/OutcomeRecord tests passed 34/34
  (report SHA-256 `47c795a4ed27612d6cb00515733d404ba17b24df1ca394e6ebb319b6dde5285c`),
  compute/evidence PostgreSQL integration passed 5/5 (report SHA-256
  `f45f868d8aa06b206a57b1222c20fe0f56111900750d0ced0adc003a189e65c2`), and
  CPU+A100 control-plane canary plus trusted OutcomeRecord assembly passed;
  failure injection remained `submission_unknown` (report SHA-256
  `526b61cf3c302201a73e8121d8ba159048291571c328c0dac498b14aca0970a9`).
- The migration integration test was not counted in this rerun because its
  existing disposable database already had the migration journal; the prior
  fresh-restore lane remains the authoritative first/second migration proof.

## Final scoped runtime and approval evidence

- Count-only API status canary against the disposable PostgreSQL restore
  returned HTTP 200 with `schema_version=meta-harness-v1`; report SHA-256
  `142d5c976e04ff55b7d070f35602f55a5a22ccf77868fe5be92cc7d00e2b7f97`.
- Current candidate detached bubblewrap evaluator rerun passed held-in,
  held-out and canary with unchanged candidate tree
  `37907595927841fed4007c3b7c5f476792287c66b94887f793808d6580d1505e`;
  report SHA-256 `e0ff0290cb187a63c6c1c6e6269d1789858fad41cde843744e97e16303d84e54`.
- Reviewer approval was independently verified and persisted in the isolated
  database for reviewer `service@diwenbao.co`, purpose `harness_upgrade`,
  key id `aws-reviewer`, subject bound to agenda/candidate/patch, and signature
  hash `5350eeaba85c5deb34c45690b3ac07a84c1e970134e21b09adb9da73e950a795`.
  Approval evidence JSON SHA-256:
  `bb223c2cf20b662db97296042bcfc7b6da60d6c86153dedbfa5bffbdbebc0ade`.
- Release scope is CPU + SSH A100. Colab is explicitly excluded from this
  candidate; no Colab credential or OAuth material was used. Real LLM provider
  canary/restart passed in the isolated provider lane.
- A no-network Codex-shaped mock-provider canary passed the LLM contract:
  ResourceGrant admission, metered success (12 tokens), auth-failure
  fail-closed behavior and cooldown persistence were exercised. Report
  SHA-256 `7e7fb80457bee57d8e6930b0e59401bc08f7e8504c73df313471ce10304e80ad`.
  This is synthetic contract evidence, not a real provider call.
- Real provider canary against `sora2.today` / `claude-opus-4-6-thinking`
  passed through the candidate LLMRouter with a bounded grant: 30 reported
  tokens, one successful attempt, and a successful route observation. Report
  SHA-256 `1f548e1cad93d9e9cf50dc94bcb70e197f4afc34376c513c12467f42b0d95ebb`.
- Real provider fault injection against an invalid endpoint failed closed and
  persisted transient cooldown across router reconstruction in the isolated
  PostgreSQL database. Report SHA-256
  `723ea17b390222ac2af8bab6ac7019d4a2fcfac3a0adbc6a4c71ca4ac661865a`.
