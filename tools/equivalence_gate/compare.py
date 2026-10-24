"""Decide whether a change kept behaviour: compare two gate.py runs (base, head).

usage: python -I compare.py <base_out> <head_out>      exit 0 = equivalent, 1 = not

Prints a Markdown report (CI puts it in the job summary). Every rule is
one-directional: the head may gain passing tests, routes or symbols, never lose
one, and nothing that worked on the base may start failing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
base = json.loads(Path(sys.argv[1], "gate.json").read_text())
head = json.loads(Path(sys.argv[2], "gate.json").read_text())
problems: list[str] = []
lines: list[str] = []


def check(name: str, lost: list, detail: str) -> None:
    lines.append(f"| {name} | {detail} | {'FAIL' if lost else 'ok'} |")
    if lost:
        problems.append(f"**{name}**: " + ", ".join(f"`{x}`" for x in lost[:20]) + (" ..." if len(lost) > 20 else ""))


bt, ht = base["tests"], head["tests"]
lost = sorted(set(bt["passed"]) - set(ht["passed"]))
check("tests", lost, f"{len(bt['passed'])} -> {len(ht['passed'])} passed, {len(ht['failing'])} failing")

lost = sorted(set(head["imports"]["failing"]) - set(base["imports"]["failing"]))
check("imports", lost, f"{len(base['imports']['modules'])} -> {len(head['imports']['modules'])} modules")

if isinstance(base["routes"], list) and isinstance(head["routes"], list):
    check("routes", sorted(set(base["routes"]) - set(head["routes"])), f"{len(base['routes'])} -> {len(head['routes'])}")
else:
    check("routes", ["route table unreadable"], "-")

check("undefined names", sorted(set(head["names"]) - set(base["names"])), f"{len(base['names'])} -> {len(head['names'])}")
check("entry scripts", sorted(k for k, v in head["entry"].items() if v != "ok" and base["entry"].get(k) == "ok"),
      f"{len(head['entry'])} scripts")
check("outside symbols", sorted(k for k, v in head["symbols"].items() if v != "ok" and base["symbols"].get(k) == "ok"),
      f"{len(head['symbols'])} symbols")

if "schema" in base and "schema" in head:
    if "error" in head["schema"] and "error" not in base["schema"]:
        check("schema", [head["schema"]["error"][:200]], "bootstrap")
    elif "error" not in base["schema"]:
        diff = sorted(set(map(str, (base["schema"]["columns"] or []) + (base["schema"]["indexes"] or [])))
                      ^ set(map(str, (head["schema"]["columns"] or []) + (head["schema"]["indexes"] or []))))
        check("schema", diff, f"{len(base['schema']['columns'] or [])} columns")

if "pg_tests" in base and "pg_tests" in head:
    lost = []
    for f, b in base["pg_tests"].items():
        h = head["pg_tests"].get(f, {})
        lost += sorted(set(b.get("passed", [])) - set(h.get("passed", [])))
    check("PostgreSQL tests", lost, f"{len(base['pg_tests'])} files")

if "dir" in base.get("api", {}) and "dir" in head.get("api", {}):
    mask = Path(sys.argv[2], "api_mask.json")
    subprocess.run([sys.executable, "-I", str(HERE / "api_snapshot.py"), "mask", base["api"]["dir"],
                    base["api"]["mask_dir"], str(mask)], capture_output=True, text=True, check=True)
    p = subprocess.run([sys.executable, "-I", str(HERE / "api_snapshot.py"), "diff", base["api"]["dir"],
                        head["api"]["dir"], str(mask)], capture_output=True, text=True, check=False)
    differ = [ln.strip() for ln in p.stdout.splitlines()[1:] if ln.startswith("  /")]
    first = (p.stdout.splitlines() or ["?"])[0]
    check("API responses", differ, first.replace("[diff] ", ""))
elif "api" in head:
    check("API responses", [str(head["api"].get("error", "missing"))[:200]], "snapshot")

print("## Equivalence gate\n")
print(f"base `{base['commit'][:7]}` -> head `{head['commit'][:7]}`\n")
print("| check | numbers | result |\n|---|---|---|")
print("\n".join(lines))
print("\n**RESULT: " + ("PASS**" if not problems else "FAIL**"))
for p in problems:
    print(f"- {p}")
sys.exit(1 if problems else 0)
