# Self-heal watchdog runbook

## Why this exists

The previously installed watchdog (`/usr/local/bin/deepgraph-selfheal.sh`, not
source controlled) judged the application hung whenever core output tables had
no fresh row for 45 minutes, and restarted `deepgraph-web.service` after a
30-minute cooldown.

That signal is only valid when the system is supposed to be producing output.
With `DEEPGRAPH_AUTO_RESEARCH_ENABLED=false` and
`DEEPGRAPH_AUTO_PIPELINE_ENABLED=0`, "no new rows" is the designed state, so the
watchdog restarted a healthy service roughly every 31 minutes while
`NRestarts=0` proved the application had never crashed.

## Deployment drift

The installed script had **no source-controlled equivalent** at session start.
That is recorded drift, not an accepted pattern. The source-owned replacement is:

| Concern | Source file |
| --- | --- |
| Decision (pure, unit tested) | `orchestrator/selfheal_policy.py` |
| Signal collection and action | `scripts/deepgraph_selfheal.py` |
| systemd unit | `deploy/deepgraph-selfheal.service` |
| systemd timer | `deploy/deepgraph-selfheal.timer` |
| Tests | `tests/test_selfheal_policy.py` |

Do not edit the installed copy. Change the source, run the tests, then deploy
through the `1-selfheal` batch of `deploy/manifest/recovery_2026-08-03.spec.json`
(pinned SHA256 set in `deploy/manifest/recovery_2026-08-03.json`). The
previously referenced `deploy/manifest/selfheal_v2.json` never existed in this
repo (2026-08-06 audit).

## Decision rules

Evaluated in order; exactly one action per tick.

1. **Web process not running** -> hold (`hold_process_not_running_systemd_owns_recovery`).
   `Restart=always` owns that case; two supervisors cause restart storms.
2. **Health probe failed >= threshold (default 3 consecutive)** -> restart
   (`restart_health_probe_failed`). This is the only signal that can restart on
   its own, because it is the only positive proof of a wedged process.
3. **Health probe failed below threshold** -> hold (`hold_health_failure_below_threshold`).
4. **Autonomy disabled** -> hold (`hold_autonomy_disabled_no_output_expected`).
5. **Agenda awaiting authority** (portfolio/grant decision pending, no running
   job) -> hold (`hold_awaiting_authority`).
6. **No admitted work** (no active ResourceGrant and no running job) -> hold
   (`hold_no_active_work_no_output_expected`).
7. **Provider/credit outage in the log** -> hold
   (`hold_provider_or_credit_issue_restart_cannot_fix`).
8. **Output freshness unavailable** -> hold (`hold_output_freshness_unavailable`).
   Unknown never means failure.
9. **Output fresh** -> hold (`hold_output_fresh`).
10. Otherwise: autonomy on, work admitted, provider healthy, process alive, no
    output past the stall window -> restart (`restart_expected_output_stalled`).

Every restart path is rate limited in one place; a suppressed restart is logged
as `hold_restart_cooldown_active` with the suppressed reason.

## Safety properties

- Status/count-only SQL. No paper text, claim, insight, or user row is read.
- No database URL, password, or host address is printed. The password reaches
  `psql` through the process environment only.
- Any unobservable signal (database unreadable, probe transport error, missing
  env file) degrades to "do nothing".

## Verification

Source, no privileges needed:

```
python3 -m pytest tests/test_selfheal_policy.py -q
```

On the host, read-only, after deployment:

```
sudo /usr/local/bin/deepgraph-selfheal.py --dry-run --json
```

`--dry-run` collects every signal, prints the decision, restarts nothing, and
does not write the cooldown state.

The paused-system acceptance check is seven consecutive quiet ticks
(70 minutes) with the timer active and no restart:

```
systemctl show deepgraph-web.service -p NRestarts -p ActiveEnterTimestamp
grep decision= /var/log/deepgraph-selfheal.log | tail -10
```

Expected: `ActiveEnterTimestamp` unchanged across the window and every logged
decision `hold_*`.

## Rollback

1. `sudo systemctl stop deepgraph-selfheal.timer`
2. Restore the backed-up `/usr/local/bin/deepgraph-selfheal.sh` and the previous
   unit files from the manifest's rollback artifacts (SHA256 recorded there).
3. `sudo systemctl daemon-reload && sudo systemctl start deepgraph-selfheal.timer`

Stopping the timer alone is a safe intermediate state: the web service keeps its
own `Restart=always` recovery.
