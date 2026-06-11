"""Watch research_agendas/inbox/ for user direction submissions (.yaml).

For each new YAML file:
- reject files larger than MAX_SUBMISSION_BYTES without reading their content,
- parse it with agents.direction_intake (deterministic, no LLM),
- persist the mapped agenda via agents.agenda_loader (direct DB write — works
  whether or not the web app is running),
- write <name>.echo.json (what the system understood, for the submitter),
- move the file to inbox/processed/ on success or inbox/failed/ (plus
  <name>.error.txt) on failure.

Usage:
    python3 -m scripts.agenda_inbox_watcher --once          # single scan
    python3 -m scripts.agenda_inbox_watcher                 # poll every 60s
    python3 -m scripts.agenda_inbox_watcher --interval 30

See deploy/agenda-inbox-watcher.example for a systemd timer / cron setup.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INBOX = REPO_ROOT / "research_agendas" / "inbox"

# Direction files are short YAML documents; anything bigger is malformed or
# abusive. Oversized files are quarantined without their content being read.
MAX_SUBMISSION_BYTES = 256 * 1024


def _unique_target(directory: Path, name: str) -> Path:
    """Destination path inside directory; suffix a timestamp on collision."""
    target = directory / name
    if not target.exists():
        return target
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    p = Path(name)
    return directory / f"{p.stem}.{stamp}{p.suffix}"


def process_file(path: Path, processed_dir: Path, failed_dir: Path) -> dict:
    """Process one submission file. Returns a small result dict for logging."""
    from agents import agenda_loader
    from agents.direction_intake import DirectionParseError, parse_direction_yaml

    try:
        size = path.stat().st_size
        if size > MAX_SUBMISSION_BYTES:
            raise DirectionParseError(
                f"file too large: {size} bytes (limit {MAX_SUBMISSION_BYTES}); "
                "content not read"
            )
        text = path.read_text(encoding="utf-8")
        agenda, echo = parse_direction_yaml(text)
        agenda_id = agenda_loader.save_agenda(agenda)
        echo["agenda_id"] = agenda_id
        target = _unique_target(processed_dir, path.name)
        echo_path = target.parent / (target.name + ".echo.json")
        echo_path.write_text(
            json.dumps(echo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(path), str(target))
        print(f"[INBOX] OK {path.name} -> agenda #{agenda_id} ({agenda.name})", flush=True)
        return {"file": path.name, "status": "ok", "agenda_id": agenda_id}
    except (DirectionParseError, OSError, UnicodeDecodeError) as exc:
        reason = str(exc)
    except Exception as exc:  # noqa: BLE001 - keep the watcher alive
        reason = f"{type(exc).__name__}: {exc}"

    target = _unique_target(failed_dir, path.name)
    try:
        shutil.move(str(path), str(target))
        (target.parent / (target.name + ".error.txt")).write_text(
            reason + "\n", encoding="utf-8"
        )
    except OSError as move_err:
        print(f"[INBOX] Could not quarantine {path.name}: {move_err}", flush=True)
    print(f"[INBOX] FAILED {path.name}: {reason}", flush=True)
    return {"file": path.name, "status": "failed", "error": reason}


def scan_inbox(inbox: Path) -> list[dict]:
    """One scan pass over the inbox. Creates the directory layout if missing."""
    processed_dir = inbox / "processed"
    failed_dir = inbox / "failed"
    for d in (inbox, processed_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    results = []
    for path in sorted(inbox.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".yaml", ".yml"):
            continue
        results.append(process_file(path, processed_dir, failed_dir))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--once", action="store_true", help="scan once and exit")
    parser.add_argument(
        "--interval", type=int, default=60, help="poll interval in seconds (default 60)"
    )
    parser.add_argument(
        "--inbox", type=Path, default=DEFAULT_INBOX, help="inbox directory to watch"
    )
    args = parser.parse_args(argv)

    from db import database as db

    db.init_db()

    if args.once:
        results = scan_inbox(args.inbox)
        print(f"[INBOX] Scan done: {len(results)} file(s) handled", flush=True)
        return 0

    print(
        f"[INBOX] Watching {args.inbox} every {args.interval}s (Ctrl-C to stop)",
        flush=True,
    )
    while True:
        scan_inbox(args.inbox)
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
