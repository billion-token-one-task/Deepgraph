"""Demo runner for the agenda-driven autonomous research loop (issue #9).

PR #10 step 7 of the reproduction recipe (and the corresponding section
of the merged commit message) cites ``python scripts/demo_agenda_loop.py``
as the one-liner for exercising the full agenda → selection → real
benchmark → evidence gate → manuscript → reviewer → revision-plan chain.

That filename did not exist on disk — the actual logic lives in
``scripts/build_agenda_loop_acceptance.py``, which also writes
``artifacts/agenda_loop_acceptance.json`` as a side effect. To keep the
documented command working without forcing every reader to know the
internal builder name, this module is a thin alias that delegates to the
builder's ``main()`` and forwards its exit code.

Both invocations are now valid:

    python scripts/demo_agenda_loop.py
    python -m scripts.build_agenda_loop_acceptance

The wrapper inherits the builder's environment contract — set
``DEEPGRAPH_DB_PATH`` to point at an isolated SQLite file so the demo
does not stomp on a developer's working database. See the builder's
module docstring for the recommended invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``python scripts/demo_agenda_loop.py`` work as documented in the
# PR #10 reproduction recipe. Without this insertion, that direct
# invocation crashes with ``ModuleNotFoundError: No module named
# 'scripts'`` because the repository root is not on sys.path when the
# script is executed by path. ``python -m scripts.demo_agenda_loop``
# already works because ``-m`` puts CWD on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_agenda_loop_acceptance import main as _build_main  # noqa: E402


def main() -> int:
    """Delegate to the acceptance-bundle builder.

    Returns its exit code so callers (verify_acceptance.sh, CI, etc.)
    see ``0`` on success and a non-zero value when the underlying loop
    fails any gate.
    """
    return _build_main()


if __name__ == "__main__":
    sys.exit(main())
