# Local commit record and future split guidance

No push, remote reference, branch deletion, deployment or master replacement
was performed.

The current local implementation was intentionally checkpointed as:

```text
c25e63c feat: build controlled meta-harness-v1 candidate
```

This commit contains code, additive migration, tests, configuration and the
topic-plugin moves. Rename review paired all 30 removed generic topic paths
with plugin destinations (96–100% similarity for modified moves).

The integration documents and root changelog are a separate local
documentation commit. These two commits are review checkpoints, not release
approval.

The continuation hardening slice was checkpointed as:

```text
2cccc7a feat: harden meta-harness control plane
```

It adds evidence-graph Frontier assembly, stable proposal identity, signed
reviewer approval, durable failure usage/startup recovery, expanded role
routing, and explicit agenda scoping for legacy mutation paths. It is also a
local review checkpoint, not a release or deployment approval.

## Recommended review split before a future PR

If reviewers require smaller commits before publication, recreate them on a
new local review branch only after approval; do not rewrite a shared remote
branch. Suggested semantic groups are:

1. audit baseline and schema/state dictionaries;
2. scientific integrity and `plugins/examples/cggr` boundary;
3. bounded Agenda and additive PostgreSQL migration;
4. Frontier/portfolio/ResourceGrant/OutcomeRecord;
5. durable LLM/ComputeBackend and evidence authority;
6. Harness Evolution, tests, runbooks and acceptance material.

Before any rewrite, rerun only approved static checks on this host. Run pytest
and PostgreSQL validation later in isolated CI. Do not push until explicit
user approval and operator/deployment-quiescence verification.
