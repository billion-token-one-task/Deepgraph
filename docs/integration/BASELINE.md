# meta-harness-v1 integration baseline

Date: 2026-07-30 UTC

## Safety envelope

- Production worktree: `/home/billion-token/Deepgraph` (read-only reference).
- Candidate worktree: `/home/ec2-user/Deepgraph-meta-harness-v1`.
- Candidate branch: `integration/meta-harness-v1`.
- No production checkout, configuration, service, database, or worktree mutation is permitted.
- This host must not run pytest, import-smoke checks, application startup,
  migrations, builds, dependency installation, or CPU/GPU/SSH/Colab experiments.
- Production database access is limited to `information_schema` and
  `SELECT COUNT`; this session did not open a production database connection.
- Remote mutation is forbidden until operator identity, deployment quiescence,
  acceptance, and explicit user approval are all established.

Initial host observations:

| Check | Observation |
|---|---|
| `uptime` | load average `1.69, 1.21, 0.89` (below stop threshold 3) |
| disk | root volume 53% used, about 96 GB available |
| `/tmp` | about 2.9 GB available |
| process view | no high-load task visible in the restricted process namespace |
| production tracked status | clean on `local/snapshot-20260621`; untracked names intentionally not enumerated |

The process namespace is restricted, so absence of a visible deployment is not
proof of single-operator control. That acceptance precondition remains open.

## Immutable source objects

These are local refs only. Nothing was pushed and no remote ref was created.

| Local ref | Object |
|---|---|
| `refs/archive/prod-snapshot-20260621` | `7d0b42af8e8f061c3c16800c44224c110f3b94a0` |
| `refs/archive/koen-master-20260626` | `6048a9568c79b011074e0dba2662fd473cfab250` |
| `refs/archive/topic-gate-20260729` | `9d24d29c6a7d1017301ffa9c36ff9b4b3dfae88d` |

Production parent: `4f78f828704567f4210b8628973d4a0e6ba62868`.
`git merge-base` between production and GitHub master exits 1 with no result.
Semantic porting is therefore mandatory; merge/cherry-pick of the large source
commits is prohibited.

Proposed remote archive commands, **not executed**:

```bash
git push origin \
  refs/archive/prod-snapshot-20260621:refs/archive/prod-snapshot-20260621 \
  refs/archive/koen-master-20260626:refs/archive/koen-master-20260626 \
  refs/archive/topic-gate-20260729:refs/archive/topic-gate-20260729
```

Remote archive creation requires explicit approval and a second read-only
verification of all three object IDs immediately before execution.

## Measured deviations from the supplied approximation

The candidate clone reports 537 changed paths from GitHub master to the
production snapshot, with 49,826 insertions and 50,749 deletions when diffed
in that direction. The supplied “about 539” and inverse insertion/deletion
figures are compatible with a different direction or snapshotting time, but
the measured values above are used by this integration.

The production Agenda behavior also contains two contracts that are explicitly
rejected for v1:

1. non-positive token budgets disable the cap;
2. untagged legacy insights may enter an agenda through keyword matching.

meta-harness-v1 requires positive hard caps and `explicit_import` before any
legacy backlog record can become agenda-scoped.
