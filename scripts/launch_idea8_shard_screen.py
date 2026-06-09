#!/usr/bin/env python3
"""Create a robust launch.sh for an idea8 shard and start it in screen."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:120] or "shard"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--name", default="")
    parser.add_argument("--screen-name", default="")
    parser.add_argument("--repo", type=Path, default=Path('/root/hk/Deepgraph'))
    parser.add_argument("--out-root", type=Path, default=Path('/root/deepgraph_ideas/idea_8/experiments/main/shards'))
    args = parser.parse_args()

    name = args.name or "__".join([
        _safe_slug(args.model),
        _safe_slug(args.dataset),
        f"seed{args.seed}",
        _safe_slug(args.methods),
    ])
    screen_name = args.screen_name or _safe_slug('idea8_' + name)[:70]
    workdir = args.out_root / name
    workdir.mkdir(parents=True, exist_ok=True)
    launch = workdir / 'launch.sh'
    cmd = [
        'python3', '-u', 'scripts/run_idea8_benchmark_shard.py',
        '--model', args.model,
        '--dataset', args.dataset,
        '--seed', str(args.seed),
        '--methods', args.methods,
        '--gpu', str(args.gpu),
        '--timeout', str(args.timeout),
        '--max-examples', str(args.max_examples),
        '--name', name,
    ]
    launch.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        f'cd {shlex.quote(str(args.repo))}\n'
        f'exec {" ".join(shlex.quote(part) for part in cmd)} > {shlex.quote(str(workdir / "launcher.log"))} 2>&1\n',
        encoding='utf-8',
    )
    launch.chmod(0o755)
    subprocess.run(['screen', '-dmS', screen_name, str(launch)], check=True)
    print(json.dumps({
        'screen_name': screen_name,
        'workdir': str(workdir),
        'launch': str(launch),
        'launcher_log': str(workdir / 'launcher.log'),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
