"""The artifact archive must survive the VM dying between exec and download.

Runs 163 and 166 both finished their compute with returncode 0 and lost every
artifact because /content was already gone when collection started. The
archive therefore also rides the exec stdout stream as base64; these tests pin
the split/restore round trip.
"""

import base64

from meta_harness.backends.colab_cli import (
    _EMBED_BEGIN,
    _EMBED_END,
    _strip_embedded_archive,
)


def _stream(data: bytes, *, declared_len: int | None = None, chunk: int = 8) -> str:
    payload = base64.b64encode(data).decode()
    lines = [payload[i:i + chunk] for i in range(0, len(payload), chunk)]
    declared = len(payload) if declared_len is None else declared_len
    return (
        "experiment output line\n"
        f"{_EMBED_BEGIN}{declared}\n" + "\n".join(lines) + f"\n{_EMBED_END}\n"
    )


def test_round_trip_restores_bytes_and_cleans_stdout():
    data = b"tar.gz bytes \x00\x01\x02" * 100
    cleaned, restored = _strip_embedded_archive(_stream(data))
    assert restored == data
    assert _EMBED_BEGIN not in cleaned
    assert "experiment output line" in cleaned


def test_absent_marker_passes_through():
    cleaned, restored = _strip_embedded_archive("plain output only")
    assert restored is None
    assert cleaned == "plain output only"


def test_truncated_payload_is_rejected_not_misdecoded():
    data = b"payload" * 50
    stream = _stream(data, declared_len=len(base64.b64encode(data)) + 4)
    cleaned, restored = _strip_embedded_archive(stream)
    assert restored is None
    assert _EMBED_BEGIN not in cleaned


def test_missing_end_marker_keeps_stdout_untouched():
    stream = f"output\n{_EMBED_BEGIN}12\nabcd"
    cleaned, restored = _strip_embedded_archive(stream)
    assert restored is None
    assert cleaned == stream


def _account(ref, priority=100, quota=8.0):
    from meta_harness.compute import ColabAccount

    return ColabAccount(
        account_ref=ref,
        credential_ref=f"env:CRED_{ref}",
        isolated_home=f"/tmp/{ref}",
        oauth_store=f"/tmp/{ref}/token.json",
        session_namespace=ref,
        quota_gpu_hours=quota,
        priority=priority,
    )


def test_pool_prefers_lower_priority_then_least_used():
    from meta_harness.backends.colab_cli import ColabAccountPool

    pool = ColabAccountPool([_account("slow", priority=100), _account("fast", priority=0)])
    picked = pool.acquire(0.5)
    assert picked.account_ref == "fast"
    pool.release(picked, 0.5)
    assert pool.acquire(0.5).account_ref == "fast"  # still preferred while quota lasts


def test_provision_failures_cool_an_account_off_and_success_clears_it():
    from meta_harness.backends.colab_cli import ColabAccountPool, ColabCLIError

    fast, slow = _account("fast", priority=0), _account("slow", priority=100)
    pool = ColabAccountPool([fast, slow])
    for _ in range(2):
        pool.record_provision_failure(fast)
    picked = pool.acquire(0.5)
    assert picked.account_ref == "slow"  # cooling account is skipped, not fatal
    pool.release(picked, 0.5)
    pool.record_provision_success(fast)
    assert pool.acquire(0.5).account_ref == "fast"


def test_all_accounts_cooling_reports_the_distinct_reason():
    from meta_harness.backends.colab_cli import ColabAccountPool, ColabCLIError
    import pytest as _pytest

    only = _account("only", priority=0)
    pool = ColabAccountPool([only])
    for _ in range(2):
        pool.record_provision_failure(only)
    with _pytest.raises(ColabCLIError, match="cooling off"):
        pool.acquire(0.5)


def test_manifest_priority_zero_survives_parsing(tmp_path, monkeypatch):
    """Priority 0 is the fastest lane, and zero is falsy: it must not default."""
    import json as _json

    from meta_harness.backends.colab_durable import load_colab_accounts

    key = tmp_path / "key.pem"
    key.write_text("x")
    manifest = tmp_path / "accounts.json"
    manifest.write_text(_json.dumps([
        {
            "account_ref": "fast", "credential_ref": "env:CRED",
            "isolated_home": str(tmp_path), "oauth_store": str(key),
            "session_namespace": "ns", "quota_gpu_hours": 8, "priority": 0,
            "transport": "ssh", "ssh_target": "u@h", "ssh_key_path": str(key),
        },
        {
            "account_ref": "slow", "credential_ref": "env:CRED2",
            "isolated_home": str(tmp_path), "oauth_store": str(key),
            "session_namespace": "ns2", "quota_gpu_hours": 8,
        },
    ]))
    monkeypatch.setenv("DG_TEST_MANIFEST", str(manifest))
    accounts = load_colab_accounts("env:DG_TEST_MANIFEST")
    by_ref = {a.account_ref: a for a in accounts}
    assert by_ref["fast"].priority == 0
    assert by_ref["slow"].priority == 100


def test_long_stages_prefer_a_dedicated_host_but_fall_back():
    from meta_harness.backends.colab_cli import ColabAccountPool
    from meta_harness.compute import ColabAccount

    def acct(ref, transport="colab", priority=50):
        return ColabAccount(
            account_ref=ref, credential_ref=f"env:C_{ref}",
            isolated_home=f"/tmp/{ref}", oauth_store=f"/tmp/{ref}/k",
            session_namespace=ref, quota_gpu_hours=8.0, priority=priority,
            transport=transport,
            ssh_target="u@h" if transport == "ssh" else "",
            ssh_key_path=f"/tmp/{ref}/k" if transport == "ssh" else "",
        )

    notebook, host = acct("nb", priority=0), acct("host", "ssh", priority=90)
    pool = ColabAccountPool([notebook, host])
    # an hour-long stage takes the dedicated host even though it ranks worse
    picked = pool.acquire(1.0, stage="full_benchmark")
    assert picked.account_ref == "host"
    # a pilot still follows plain priority
    assert pool.acquire(1.0, stage="pilot").account_ref == "nb"
    # with the host busy, a long stage falls back rather than failing
    pool2 = ColabAccountPool([acct("nb2", priority=0)])
    assert pool2.acquire(1.0, stage="evidence_audit").account_ref == "nb2"


# --- durable provisioning cooldown -------------------------------------------
# The cooldown counter lived in process memory keyed on time.monotonic().
# Deploy restarts on 2026-08-19 wiped it, colab-pro-2 got fresh chances within
# two minutes and refused each one, and every refusal burned a whole
# experiment run out of the M2 acceptance window (runs 188 and 190).

_REFUSAL = "transport:ColabCLIError:colab provision failed: TooManyAssignments"
_LOST_SESSION = "transport:ColabCLIError:Colab output omitted the return-code sentinel"


def _history(n, *, marker, status="failed", age_seconds=60):
    from datetime import datetime, timedelta, timezone

    when = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return [
        {"status": status, "failure_reason": marker, "created_at": when}
        for _ in range(n)
    ]


def _cooling(rows):
    from unittest import mock

    from meta_harness.backends import colab_cli

    with mock.patch.object(colab_cli.db, "fetchall", return_value=rows):
        return colab_cli.durable_provision_cooldown("colab-pro-2")


def test_two_consecutive_refusals_cool_the_account():
    assert _cooling(_history(2, marker=_REFUSAL)) is True


def test_the_wait_doubles_with_each_further_refusal():
    # colab-pro-2's record: three refusals in a row, each landing right after
    # the previous flat-hour cooldown expired, each costing an experiment run
    from meta_harness.backends import colab_cli

    base = colab_cli._PROVISION_COOLDOWN_SECONDS
    # streak of 2 -> one base window: expired at 1.5x base
    assert _cooling(_history(2, marker=_REFUSAL, age_seconds=base * 1.5)) is False
    # streak of 3 -> two base windows: still cooling at 1.5x base
    assert _cooling(_history(3, marker=_REFUSAL, age_seconds=base * 1.5)) is True
    # streak of 4 -> four base windows
    assert _cooling(_history(4, marker=_REFUSAL, age_seconds=base * 3.5)) is True


def test_the_wait_is_capped():
    from meta_harness.backends import colab_cli

    cap = colab_cli._PROVISION_COOLDOWN_CAP_SECONDS
    assert _cooling(_history(10, marker=_REFUSAL, age_seconds=cap + 60)) is False


def test_any_intervening_success_ends_the_streak():
    rows = _history(4, marker=_REFUSAL)
    rows[2] = {
        "status": "succeeded",
        "failure_reason": None,
        "created_at": rows[2]["created_at"],
    }
    # only the two newest refusals count, so it is back to one base window
    from meta_harness.backends import colab_cli

    base = colab_cli._PROVISION_COOLDOWN_SECONDS
    rows_recent = [dict(r) for r in rows]
    assert _cooling(rows_recent) is True  # inside the base window
    for r in rows_recent[:2]:
        r["created_at"] = r["created_at"].replace(year=2020)
    assert _cooling(rows_recent) is False


def test_a_recent_success_clears_the_cooldown():
    rows = _history(2, marker=_REFUSAL)
    rows[0] = {
        "status": "succeeded",
        "failure_reason": None,
        "created_at": rows[0]["created_at"],
    }
    assert _cooling(rows) is False


def test_an_expired_window_no_longer_cools():
    assert _cooling(_history(2, marker=_REFUSAL, age_seconds=7200)) is False


def test_a_lost_session_is_not_a_provisioning_refusal():
    # losing a notebook VM says nothing about quota; it must not park the lane
    assert _cooling(_history(2, marker=_LOST_SESSION)) is False


def test_an_unreadable_history_fails_open():
    from unittest import mock

    from meta_harness.backends import colab_cli

    with mock.patch.object(
        colab_cli.db, "fetchall", side_effect=RuntimeError("no db")
    ):
        assert colab_cli.durable_provision_cooldown("colab-pro-2") is False


# --- capacity is a condition of the pool, not a failure of the work ----------
# acquire() is only reached deep inside the executor, so a full pool surfaced
# as an exception AFTER the request was claimed. The worker treats any
# exception as losing control of its claim, so it failed the request and
# requeued it straight back into the same full pool. Request 65 burned ten
# attempts in three minutes that way on 2026-08-19, and run 189's audit --
# already at full_benchmark_complete -- could not proceed.


def _pool(*, cooling=(), busy=()):
    from meta_harness.backends import colab_cli
    from meta_harness.compute import ColabAccount

    accounts = [
        ColabAccount(
            account_ref=ref,
            credential_ref=f"env:{ref}",
            isolated_home=f"/home/{ref}",
            oauth_store=f"/home/{ref}/token.json",
            session_namespace=ref,
            quota_gpu_hours=8,
        )
        for ref in ("lane-a", "lane-b")
    ]
    pool = colab_cli.ColabAccountPool(accounts)
    for ref in busy:
        pool._active[ref] = 1
    return pool, set(cooling)


def _with_cooling(cooling):
    from unittest import mock

    from meta_harness.backends import colab_cli

    return mock.patch.object(
        colab_cli,
        "durable_provision_cooldown",
        side_effect=lambda ref: ref in cooling,
    )


def test_capacity_probe_sees_a_free_lane():
    pool, cooling = _pool(busy=("lane-a",))
    with _with_cooling(cooling):
        assert pool.has_capacity() is True


def test_capacity_probe_reports_a_full_pool():
    pool, cooling = _pool(busy=("lane-a", "lane-b"))
    with _with_cooling(cooling):
        assert pool.has_capacity() is False


def test_capacity_probe_counts_a_cooling_lane_as_unavailable():
    pool, cooling = _pool(busy=("lane-a",), cooling=("lane-b",))
    with _with_cooling(cooling):
        assert pool.has_capacity() is False


def test_the_probe_does_not_claim_anything():
    pool, cooling = _pool()
    with _with_cooling(cooling):
        before = dict(pool._active)
        pool.has_capacity()
        assert pool._active == before


def test_a_full_pool_raises_the_distinct_capacity_error():
    import pytest as _pytest

    from meta_harness.backends.colab_cli import ColabCapacityUnavailable

    pool, cooling = _pool(busy=("lane-a", "lane-b"))
    with _with_cooling(cooling):
        with _pytest.raises(ColabCapacityUnavailable):
            pool.acquire(1.0)


def test_the_worker_reaches_the_real_pool_not_a_raw_tuple():
    """Pin the path from the configured backend to the ColabAccountPool.

    Three objects in this chain expose an `.accounts` attribute and only one
    of them is the pool: ColabGPUBackend.accounts and
    DurableColabTransport.accounts are both raw tuples, while the pool lives
    on ColabCLIExecutor. The first version of the capacity guard reached for
    the backend's, raised AttributeError into a bare `except`, and turned
    itself into a silent no-op -- the fail-open-and-hide-it shape the guard
    exists to prevent.
    """
    import inspect

    from orchestrator import colab_worker

    source = inspect.getsource(colab_worker.run_one)
    assert "_transport.executor.accounts" in source
    # and the failure of the probe must be reported, never swallowed silently
    assert "capacity probe unavailable" in source


# --- route work to a lane that can actually host it --------------------------
# A hosted notebook session is rebuilt from scratch each time, so a package
# installed into one does not survive to the next; only the dedicated host can
# hold a pre-provisioned runtime. Work declaring anything beyond the base
# therefore has exactly one lane that can run it. idea 175 declared POT, drew
# colab-pro-2 and then colab-pro, and lost a run to exit 78 each time while
# aws-g5-1 sat ready (2026-08-20).


def _reqs(tmp_path, *lines):
    (tmp_path / "requirements.txt").write_text("\n".join(lines), encoding="utf-8")
    return tmp_path


def test_base_only_requirements_need_no_dedicated_lane(tmp_path):
    from meta_harness.backends.colab_cli import _requires_provisioned_runtime

    d = _reqs(tmp_path, "torch>=2.2.0", "transformers>=4.44.0", "datasets", "accelerate")
    assert _requires_provisioned_runtime(d) is False


def test_the_exact_declaration_that_lost_two_runs_wants_a_dedicated_lane(tmp_path):
    from meta_harness.backends.colab_cli import _requires_provisioned_runtime

    d = _reqs(
        tmp_path,
        "torch>=2.2.0",
        "transformers>=4.44.0",
        "accelerate>=0.30.0",
        "networkx>=3.0",
        "scipy>=1.11.0",
        "POT>=0.9.0",
    )
    assert _requires_provisioned_runtime(d) is True


def test_comments_and_blanks_are_ignored(tmp_path):
    from meta_harness.backends.colab_cli import _requires_provisioned_runtime

    d = _reqs(tmp_path, "# a comment", "", "torch", "   ")
    assert _requires_provisioned_runtime(d) is False


def test_a_missing_requirements_file_does_not_force_a_lane(tmp_path):
    from meta_harness.backends.colab_cli import _requires_provisioned_runtime

    assert _requires_provisioned_runtime(tmp_path) is False


def test_the_submit_path_passes_the_requirement(tmp_path):
    import inspect

    from meta_harness.backends import colab_cli

    source = inspect.getsource(colab_cli.ColabCLIExecutor.run_request)
    assert "require_dedicated=needs_runtime" in source
    # and says which lane it chose, so a surprise is one log line away
    assert "[COLAB] route request=" in source


def test_a_hard_requirement_never_falls_back(tmp_path):
    """Waiting costs minutes; the fallback costs a run and a grant."""
    from unittest import mock

    from meta_harness.backends import colab_cli
    from meta_harness.compute import ColabAccount

    accounts = [
        ColabAccount(
            account_ref="colab-pro",
            credential_ref="env:A",
            isolated_home="/h/a",
            oauth_store="/h/a/t.json",
            session_namespace="a",
            quota_gpu_hours=8,
        ),
        ColabAccount(
            account_ref="aws-g5-1",
            credential_ref="env:B",
            isolated_home="/h/b",
            oauth_store="/h/b/k.pem",
            session_namespace="b",
            quota_gpu_hours=24,
            transport="ssh",
            ssh_target="user@host.example",
            ssh_key_path="/h/b/k.pem",
        ),
    ]
    pool = colab_cli.ColabAccountPool(accounts)
    with mock.patch.object(colab_cli, "durable_provision_cooldown", return_value=False):
        # dedicated free -> it is chosen
        assert [a.account_ref for a in pool._eligible_locked(1.0, "pilot", True)] == [
            "aws-g5-1"
        ]
        # dedicated busy -> NOTHING is eligible, rather than a doomed notebook lane
        pool._active["aws-g5-1"] = 1
        assert pool._eligible_locked(1.0, "pilot", True) == []
        # without the requirement the notebook lane is still usable
        assert [a.account_ref for a in pool._eligible_locked(1.0, "pilot", False)] == [
            "colab-pro"
        ]


def test_long_stages_still_only_prefer(tmp_path):
    # a T4 CAN run a long stage, just less reliably -- that stays a preference
    import inspect

    from meta_harness.backends import colab_cli

    source = inspect.getsource(colab_cli.ColabAccountPool._eligible_locked)
    assert "if stage in _LONG_RUNNING_STAGES and dedicated:" in source
