from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SIGNERS_FILE = REPO_ROOT / "cla-signers.json"
CLA_FILE = REPO_ROOT / "CLA.md"


def normalize_github_login(value: str) -> str:
    return value.strip().lower()


def read_pull_request_author() -> str:
    author = os.environ.get("PR_AUTHOR", "").strip()
    if author:
        return normalize_github_login(author)

    event_path = os.environ.get("GITHUB_EVENT_PATH", "").strip()
    if event_path:
        event_data = json.loads(Path(event_path).read_text(encoding="utf-8"))
        author = event_data.get("pull_request", {}).get("user", {}).get("login", "")
        if author:
            return normalize_github_login(author)

    raise RuntimeError("Unable to determine the pull request author.")


def read_signed_usernames() -> set[str]:
    if not SIGNERS_FILE.exists():
        raise RuntimeError(f"Missing signer list: {SIGNERS_FILE.name}")

    data = json.loads(SIGNERS_FILE.read_text(encoding="utf-8"))
    signers = data.get("signers", [])
    if not isinstance(signers, list):
        raise RuntimeError(f"Invalid signer list in {SIGNERS_FILE.name}: 'signers' must be a list.")

    usernames: set[str] = set()
    for entry in signers:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid signer entry in {SIGNERS_FILE.name}: each signer must be an object.")
        github_login = entry.get("github", "")
        if github_login:
            usernames.add(normalize_github_login(str(github_login)))

    return usernames


def write_step_summary(message: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not summary_path:
        return
    Path(summary_path).write_text(message + "\n", encoding="utf-8")


def main() -> int:
    author = read_pull_request_author()
    if author.endswith("[bot]"):
        message = f"CLA check skipped for bot account: {author}"
        print(message)
        write_step_summary(message)
        return 0

    signed_usernames = read_signed_usernames()
    if author in signed_usernames:
        message = f"CLA check passed for @{author}."
        print(message)
        write_step_summary(message)
        return 0

    failure_message = "\n".join(
        [
            f"CLA check failed for @{author}.",
            f"Read {CLA_FILE.name}, comment on the pull request with the required agreement text,",
            f"and ask a maintainer to add @{author} to {SIGNERS_FILE.name}.",
        ]
    )
    print(failure_message)
    write_step_summary(failure_message)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        error_message = f"CLA check could not run: {exc}"
        print(error_message)
        write_step_summary(error_message)
        sys.exit(1)
