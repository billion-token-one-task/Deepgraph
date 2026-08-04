"""Fail-safe self-heal decision policy.

The installed watchdog used to treat "no fresh research output" as proof that
the application had hung. That signal is only valid when the system is
*supposed* to be producing output. With autonomous research deliberately
disabled, or with an Agenda parked behind Frontier / portfolio / grant /
reviewer / budget / provider authority, the absence of output is the designed
state, and restarting the web service destroys in-flight work for nothing.

This module holds the whole decision as a pure function so it can be unit
tested without systemd, a database, or a network. The runner
(``scripts/deepgraph_selfheal.py``) only collects signals and applies the
returned action.

Design rules:

* restart only for a *proven* failure (health probe failing repeatedly);
* never restart on an unknown or unavailable signal (fail safe = do nothing);
* never restart merely because output is absent when no work is expected;
* treat a database transaction abandoned past the reclaim window as proof in
  its own right -- that half-dead state serves HTTP 200 and produces no output,
  so neither the health probe nor output freshness can see it;
* every decision carries a stable, operator-safe reason code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HEALTH_OK = "ok"
HEALTH_FAILED = "failed"
HEALTH_UNKNOWN = "unknown"
HEALTH_STATES = {HEALTH_OK, HEALTH_FAILED, HEALTH_UNKNOWN}

ACTION_RESTART = "restart"
ACTION_HOLD = "hold"

# Reason codes are part of the operator contract: they are logged, alerted on
# and asserted in tests. Do not rename one without updating the runbook.
REASON_PROCESS_NOT_RUNNING = "hold_process_not_running_systemd_owns_recovery"
REASON_STARTUP_GRACE = "hold_within_startup_grace"
REASON_MAINTENANCE = "hold_maintenance_mode"
REASON_HEALTH_UNKNOWN = "hold_health_probe_unavailable"
REASON_HEALTH_OK = "hold_health_ok"
REASON_HEALTH_FLAPPING = "hold_health_failure_below_threshold"
REASON_RESTART_HEALTH = "restart_health_probe_failed"
REASON_RESTART_OUTPUT_STALLED = "restart_expected_output_stalled"
REASON_RESTART_DB_TRANSACTION_STALLED = "restart_db_idle_in_transaction_stalled"
REASON_AUTONOMY_DISABLED = "hold_autonomy_disabled_no_output_expected"
REASON_NO_WORK_EXPECTED = "hold_no_active_work_no_output_expected"
REASON_AWAITING_AUTHORITY = "hold_awaiting_authority"
REASON_PROVIDER_ISSUE = "hold_provider_or_credit_issue_restart_cannot_fix"
REASON_OUTPUT_AGE_UNKNOWN = "hold_output_freshness_unavailable"
REASON_OUTPUT_FRESH = "hold_output_fresh"
REASON_COOLDOWN = "hold_restart_cooldown_active"


class SelfHealPolicyError(ValueError):
    """Raised when a caller hands the policy an unusable signal set."""


@dataclass(frozen=True)
class SelfHealPolicy:
    """Thresholds. Defaults are intentionally conservative."""

    stall_seconds: int = 45 * 60
    cooldown_seconds: int = 30 * 60
    health_failure_threshold: int = 3
    # A service that is still starting is not a service that has failed. This
    # host takes over a minute to finish its startup backfills, and a probe
    # that ignored that restarted it mid-startup once a minute, forever.
    startup_grace_seconds: int = 180
    # A transaction left open by a dead worker holds its row locks forever and
    # wedges every other writer while the HTTP probe still answers 200. The
    # server-side idle_in_transaction_session_timeout (10 min by default) is the
    # first line of defence; this threshold sits above it so PostgreSQL always
    # gets the first, cheaper attempt at reclaiming the session.
    idle_transaction_seconds: int = 15 * 60

    def validate(self) -> None:
        if self.stall_seconds <= 0:
            raise SelfHealPolicyError("stall_seconds must be positive")
        if self.cooldown_seconds <= 0:
            raise SelfHealPolicyError("cooldown_seconds must be positive")
        if self.health_failure_threshold <= 0:
            raise SelfHealPolicyError("health_failure_threshold must be positive")
        if self.startup_grace_seconds < 0:
            raise SelfHealPolicyError("startup_grace_seconds cannot be negative")
        if self.idle_transaction_seconds <= 0:
            raise SelfHealPolicyError("idle_transaction_seconds must be positive")


@dataclass(frozen=True)
class SelfHealSignals:
    """Everything the policy is allowed to look at.

    ``None`` always means "not observable right now" and must never be
    interpreted as a failure.
    """

    web_process_running: bool = True
    health_status: str = HEALTH_UNKNOWN
    health_consecutive_failures: int = 0
    # Autonomy: is the system configured to produce research output at all?
    auto_research_enabled: bool = False
    auto_pipeline_enabled: bool = False
    # Work: is at least one unit of scoped work actually admitted right now?
    active_resource_grants: int = 0
    running_jobs: int = 0
    # Authority: is every candidate parked behind a human/gate decision?
    awaiting_authority: bool = False
    awaiting_authority_reasons: tuple[str, ...] = ()
    # Freshness of core research output, seconds. None = query unavailable.
    output_age_seconds: int | None = None
    # Age of the oldest session sitting "idle in transaction", seconds.
    # None = query unavailable. This is the half-dead signal output freshness
    # cannot see: the process is alive and serving, but its workers are parked
    # on locks held by a transaction nobody is going to finish.
    max_idle_transaction_seconds: int | None = None
    idle_transaction_sessions: int = 0
    # Provider/credit outage seen in the operator log. A restart cannot fix it.
    provider_issue: bool = False
    # Seconds since the watchdog last restarted the service. None = never.
    seconds_since_last_restart: int | None = None
    # Seconds since the unit entered the active state. None = not observable.
    service_uptime_seconds: int | None = None
    # An operator has declared a maintenance window; the watchdog stands down.
    maintenance_mode: bool = False

    def validate(self) -> None:
        if self.health_status not in HEALTH_STATES:
            raise SelfHealPolicyError("invalid health_status")
        if self.health_consecutive_failures < 0:
            raise SelfHealPolicyError("health_consecutive_failures cannot be negative")
        if self.active_resource_grants < 0 or self.running_jobs < 0:
            raise SelfHealPolicyError("work counters cannot be negative")
        if self.output_age_seconds is not None and self.output_age_seconds < 0:
            raise SelfHealPolicyError("output_age_seconds cannot be negative")
        if (
            self.max_idle_transaction_seconds is not None
            and self.max_idle_transaction_seconds < 0
        ):
            raise SelfHealPolicyError("max_idle_transaction_seconds cannot be negative")
        if self.idle_transaction_sessions < 0:
            raise SelfHealPolicyError("idle_transaction_sessions cannot be negative")
        if (
            self.seconds_since_last_restart is not None
            and self.seconds_since_last_restart < 0
        ):
            raise SelfHealPolicyError("seconds_since_last_restart cannot be negative")
        if self.service_uptime_seconds is not None and self.service_uptime_seconds < 0:
            raise SelfHealPolicyError("service_uptime_seconds cannot be negative")

    @property
    def autonomy_enabled(self) -> bool:
        return bool(self.auto_research_enabled or self.auto_pipeline_enabled)

    @property
    def work_admitted(self) -> bool:
        return bool(self.active_resource_grants > 0 or self.running_jobs > 0)


@dataclass(frozen=True)
class SelfHealDecision:
    action: str
    reason_code: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def should_restart(self) -> bool:
        return self.action == ACTION_RESTART


def _cooldown_blocks(
    signals: SelfHealSignals,
    policy: SelfHealPolicy,
) -> bool:
    if signals.seconds_since_last_restart is None:
        return False
    return signals.seconds_since_last_restart < policy.cooldown_seconds


def _restart(
    reason: str,
    signals: SelfHealSignals,
    policy: SelfHealPolicy,
    details: dict[str, Any],
) -> SelfHealDecision:
    """Rate-limit every restart path through one place."""
    if _cooldown_blocks(signals, policy):
        return SelfHealDecision(
            ACTION_HOLD,
            REASON_COOLDOWN,
            {
                "suppressed_reason": reason,
                "seconds_since_last_restart": signals.seconds_since_last_restart,
                "cooldown_seconds": policy.cooldown_seconds,
                **details,
            },
        )
    return SelfHealDecision(ACTION_RESTART, reason, details)


def decide(
    signals: SelfHealSignals,
    *,
    policy: SelfHealPolicy | None = None,
) -> SelfHealDecision:
    """Return the single action the watchdog is allowed to take this tick."""
    active_policy = policy or SelfHealPolicy()
    active_policy.validate()
    signals.validate()

    # 0. An operator-declared maintenance window outranks every signal: a
    #    deployment stops the service on purpose, and a watchdog that races the
    #    operator is worse than no watchdog.
    if signals.maintenance_mode:
        return SelfHealDecision(ACTION_HOLD, REASON_MAINTENANCE, {})

    # 1. A dead process belongs to systemd's Restart= directive. Two supervisors
    #    fighting over the same unit is how restart storms start.
    if not signals.web_process_running:
        return SelfHealDecision(
            ACTION_HOLD,
            REASON_PROCESS_NOT_RUNNING,
            {"owner": "systemd"},
        )

    # 2. Still starting is not failed. Without this, a probe faster than the
    #    startup time is a guaranteed restart loop -- which is exactly what
    #    happened on this host with a one-minute external healthcheck.
    if (
        signals.service_uptime_seconds is not None
        and signals.service_uptime_seconds < active_policy.startup_grace_seconds
    ):
        return SelfHealDecision(
            ACTION_HOLD,
            REASON_STARTUP_GRACE,
            {
                "service_uptime_seconds": signals.service_uptime_seconds,
                "startup_grace_seconds": active_policy.startup_grace_seconds,
            },
        )

    # 3. A failing health probe is the only *positive* proof of a wedged
    #    process, so it is the only signal allowed to restart on its own.
    if signals.health_status == HEALTH_FAILED:
        if signals.health_consecutive_failures >= active_policy.health_failure_threshold:
            return _restart(
                REASON_RESTART_HEALTH,
                signals,
                active_policy,
                {
                    "health_consecutive_failures": signals.health_consecutive_failures,
                    "health_failure_threshold": active_policy.health_failure_threshold,
                },
            )
        return SelfHealDecision(
            ACTION_HOLD,
            REASON_HEALTH_FLAPPING,
            {
                "health_consecutive_failures": signals.health_consecutive_failures,
                "health_failure_threshold": active_policy.health_failure_threshold,
            },
        )

    # 4. A transaction abandoned past the reclaim window is direct evidence of a
    #    wedged writer, not an inference from missing output. It is checked
    #    ahead of the autonomy branches on purpose: a stuck transaction is a
    #    defect whether or not the system was asked to produce anything, and it
    #    is invisible to both the HTTP probe and output freshness. An
    #    unobservable age falls through rather than holding, so it can never
    #    mask the output-stall path below.
    if (
        signals.max_idle_transaction_seconds is not None
        and signals.max_idle_transaction_seconds > active_policy.idle_transaction_seconds
    ):
        return _restart(
            REASON_RESTART_DB_TRANSACTION_STALLED,
            signals,
            active_policy,
            {
                "max_idle_transaction_seconds": signals.max_idle_transaction_seconds,
                "idle_transaction_seconds": active_policy.idle_transaction_seconds,
                "idle_transaction_sessions": signals.idle_transaction_sessions,
            },
        )

    # 5. Output freshness is a *derived* signal. It only means anything when the
    #    system was actually asked to produce output. Every branch below is a
    #    reason the old watchdog was wrong.
    if not signals.autonomy_enabled:
        return SelfHealDecision(
            ACTION_HOLD,
            REASON_AUTONOMY_DISABLED,
            {
                "auto_research_enabled": signals.auto_research_enabled,
                "auto_pipeline_enabled": signals.auto_pipeline_enabled,
            },
        )
    if signals.awaiting_authority:
        return SelfHealDecision(
            ACTION_HOLD,
            REASON_AWAITING_AUTHORITY,
            {"awaiting": list(signals.awaiting_authority_reasons)},
        )
    if not signals.work_admitted:
        return SelfHealDecision(
            ACTION_HOLD,
            REASON_NO_WORK_EXPECTED,
            {
                "active_resource_grants": signals.active_resource_grants,
                "running_jobs": signals.running_jobs,
            },
        )
    if signals.provider_issue:
        return SelfHealDecision(
            ACTION_HOLD,
            REASON_PROVIDER_ISSUE,
            {"operator_action": "top_up_or_reroute_provider"},
        )
    if signals.output_age_seconds is None:
        return SelfHealDecision(ACTION_HOLD, REASON_OUTPUT_AGE_UNKNOWN, {})
    if signals.output_age_seconds <= active_policy.stall_seconds:
        return SelfHealDecision(
            ACTION_HOLD,
            REASON_OUTPUT_FRESH,
            {"output_age_seconds": signals.output_age_seconds},
        )

    # 6. Admitted work, autonomy on, provider healthy, process alive, and still
    #    no output for longer than the stall window: that is a real hang.
    return _restart(
        REASON_RESTART_OUTPUT_STALLED,
        signals,
        active_policy,
        {
            "output_age_seconds": signals.output_age_seconds,
            "stall_seconds": active_policy.stall_seconds,
            "active_resource_grants": signals.active_resource_grants,
            "running_jobs": signals.running_jobs,
        },
    )


# Health-probe bookkeeping is kept here so the runner stays I/O only.
def next_consecutive_failures(previous: int, *, health_status: str) -> int:
    """Count consecutive failures; an unknown probe never resets or advances."""
    if health_status not in HEALTH_STATES:
        raise SelfHealPolicyError("invalid health_status")
    if health_status == HEALTH_OK:
        return 0
    if health_status == HEALTH_FAILED:
        return max(0, int(previous)) + 1
    return max(0, int(previous))
