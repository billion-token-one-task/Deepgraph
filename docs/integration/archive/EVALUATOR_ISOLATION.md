# Harness evaluator isolation

This is an isolated-CI procedure, not permission to run on production. No
candidate evaluation has been executed on the current host.

## Boundary

`IsolatedEvaluatorRunner` invokes a configured bubblewrap-compatible binary
with:

- no inherited application secrets or database URL;
- an unshared network/process namespace;
- read-only candidate, evaluator and suite mounts;
- a dedicated writable output mount;
- operator-pinned evaluator and suite tree hashes;
- a candidate tree hash checked before and after execution.

If the isolation binary, production boundary, hash, result contract or output
manifest is missing, the evaluation fails closed. The candidate process cannot
issue reviewer approval.

## CI prerequisites

Use a throwaway clone, a disposable PostgreSQL namespace and dedicated sibling
directories for candidates, evaluators, held-out suites and output artifacts.
Set:

```text
DEEPGRAPH_HARNESS_PRODUCTION_PATH=<read-only production reference path>
DEEPGRAPH_HARNESS_PRODUCTION_DATABASE_NAMESPACE=<production namespace name>
DEEPGRAPH_HARNESS_CANDIDATE_ROOT=<throwaway candidate parent>
DEEPGRAPH_HARNESS_EVALUATOR_ROOT=<trusted evaluator parent>
DEEPGRAPH_HARNESS_HOLDOUT_ROOT=<trusted suite parent>
DEEPGRAPH_HARNESS_EVALUATOR_ARTIFACT_ROOT=<disposable output parent>
DEEPGRAPH_HARNESS_EVALUATOR_ISOLATION_BINARY=bwrap
```

The evaluator entrypoint must emit `/output/result.json` with status `passed`
or `failed`. Each suite request supplies reviewed full tree hashes. Run
held-in, held-out and canary against the same immutable patch, persist all
three `EvaluationRun` rows, then verify that approval remains
`awaiting_approval` until a separate signed reviewer envelope is supplied.

## Required negative checks

- remove or replace `bwrap`: the runner must refuse to execute;
- change one evaluator or suite byte: the pinned hash must fail;
- attempt writes to `/candidate` or `/suite`: the evaluator must fail;
- attempt network access: it must fail;
- add a symlink to an input or output: artifact certification must fail;
- mutate the candidate concurrently: before/after tree hashes must fail;
- reuse a suite row for another patch/agenda: repository scope checks must
  fail;
- omit the production path/database boundary: the API must return a failure.

Archive the command policy, candidate commit/tree hash, patch hash, evaluator
hash, suite hashes, output manifest hashes and PostgreSQL row IDs. Do not
archive secrets, raw database URLs or reviewer signatures.
