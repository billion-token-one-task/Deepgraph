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
