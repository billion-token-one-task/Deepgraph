"""Measure production code size the way a reader pays for it: Python tokens.

usage: python -I size.py <tree>                 totals for one tree
       python -I size.py <base_tree> <head_tree>   per-file change, largest first

Counts every token except comments, line breaks and indentation, so joining
lines or deleting blank lines changes nothing; only removing code does. Tests,
third_party/ and tools/ are not production code and are not counted.
"""
from __future__ import annotations

import io
import subprocess
import sys
import tokenize
from pathlib import Path

SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
        tokenize.ENCODING, tokenize.ENDMARKER}


def measure(tree: Path) -> dict[str, tuple[int, int]]:
    files = subprocess.run(["git", "-C", str(tree), "ls-files", "*.py"], capture_output=True, text=True, check=False).stdout.split()
    sizes = {}
    for f in files:
        if f.startswith(("tests/", "third_party/", "tools/")) or not (tree / f).exists():
            continue
        src = (tree / f).read_text(encoding="utf-8")
        tokens = sum(1 for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type not in SKIP)
        lines = sum(1 for ln in src.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))
        sizes[f] = (tokens, lines)
    return sizes


a = measure(Path(sys.argv[1]).resolve())
if len(sys.argv) == 2:
    print(f"{len(a)} files, {sum(t for t, _ in a.values())} tokens, {sum(n for _, n in a.values())} lines")
    sys.exit(0)
b = measure(Path(sys.argv[2]).resolve())
rows = []
for f in sorted(set(a) | set(b)):
    ta, la = a.get(f, (0, 0))
    tb, lb = b.get(f, (0, 0))
    if (ta, la) != (tb, lb):
        rows.append((tb - ta, lb - la, f))
rows.sort()
ta, tb = sum(t for t, _ in a.values()), sum(t for t, _ in b.values())
la, lb = sum(n for _, n in a.values()), sum(n for _, n in b.values())
print("## Code size\n")
print(f"tokens {ta} -> {tb} ({tb - ta:+d}, {(tb - ta) / ta * 100:+.2f}%); lines {la} -> {lb} ({lb - la:+d})\n")
if rows:
    print("| file | tokens | lines |\n|---|---|---|")
    for dt, dl, f in rows[:30]:
        print(f"| {f} | {dt:+d} | {dl:+d} |")
