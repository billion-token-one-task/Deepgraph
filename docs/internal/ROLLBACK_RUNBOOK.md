# Rollback and production-reference runbook

Production `local/snapshot-20260621@7d0b42a` remains unchanged and is the
behavioral/rollback reference. This document does not authorize a deployment,
branch replacement, service restart, or production database write.

## Database rollback

Migration `0001` is add-only and has no down script. Rollback is:

1. stop only the isolated candidate service;
2. preserve non-sensitive failure logs and schema/count evidence;
3. discard the migrated disposable database;
4. restore the pre-migration backup into a new isolated database;
5. verify the original table counts and application version.

Never emulate rollback with `DROP`, `DELETE`, column removal, or in-place
rewrites.

## Version rollback rehearsal

On an isolated machine, create a separate clone/worktree at the immutable
production object:

```bash
git clone --no-checkout https://github.com/billion-token-one-task/Deepgraph.git \
  Deepgraph-rollback-rehearsal
git -C Deepgraph-rollback-rehearsal fetch \
  '<read-only-local-object-source>' \
  7d0b42af8e8f061c3c16800c44224c110f3b94a0
git -C Deepgraph-rollback-rehearsal checkout --detach \
  7d0b42af8e8f061c3c16800c44224c110f3b94a0
```

Use a restored pre-migration database and non-production configuration. Prove
startup, count-only health, and rollback timing there. Do not run these
commands in `/home/billion-token/Deepgraph`.

## Future production rollback trigger

If deployment is eventually approved, define before it:

- immutable deployed and rollback commits;
- backup ID and restore command owner;
- maximum tolerated rollback time;
- service owner and single-operator window;
- traffic drain/restart commands;
- verification counts and health checks.

None of those production actions were exercised or approved in this work.
