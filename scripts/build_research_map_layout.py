#!/usr/bin/env python3
"""Pre-compute the research map's node coordinates.

The overview map used to run a force-directed layout in the browser. With a
few thousand taxonomy nodes that is a long task on the critical path, and it
produces a different picture on every load, so nothing about the map could be
relied on. The layout is data, but it is data that only changes when the set
of research directions changes -- which is rarely -- so it is computed here,
once, and committed.

The page draws; it does not solve. Node *sizes* stay dynamic (area tracks the
paper count, which moves daily) and are computed at render time from the API;
only the positions come from this file.

Usage:
    python scripts/build_research_map_layout.py [--slots N] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "web" / "static" / "data" / "research-map-layout.json"

# The viewBox the SVG declares. Coordinates below are in these units, so the
# map scales with its container without any of them being recomputed.
VIEW_W, VIEW_H = 480, 320
CENTER_X, CENTER_Y = 196, 156
CENTER_R = 42

# Kept well clear of the viewBox edges. A satellite can be 27 units across and
# carries a centred label wider than itself, so the ring has to stop short of
# the frame or the outermost names get sliced off. The page truncates long
# direction names in the SVG; the full name is in the table beside it.
RADIUS_X, RADIUS_Y = 150, 106

# Satellites start upper-right and run clockwise, so the largest direction --
# the list is sorted by paper count -- lands in the reading corner.
START_ANGLE_DEG = -62.0


def build_layout(slot_count: int) -> dict:
    if slot_count < 1:
        raise ValueError("slot_count must be at least 1")
    # One extra seat for the "other directions" summary node, which is always
    # drawn last and always dashed.
    total = slot_count + 1
    positions = []
    for index in range(total):
        angle = math.radians(START_ANGLE_DEG + index * (360.0 / total))
        positions.append({
            "x": round(CENTER_X + RADIUS_X * math.cos(angle), 1),
            "y": round(CENTER_Y + RADIUS_Y * math.sin(angle), 1),
        })
    return {
        "generator": "scripts/build_research_map_layout.py",
        "viewBox": f"0 0 {VIEW_W} {VIEW_H}",
        "center": {"x": CENTER_X, "y": CENTER_Y, "r": CENTER_R},
        "slots": positions[:slot_count],
        "other": positions[slot_count],
        # Satellite area is proportional to the paper count, clamped so the
        # smallest direction stays legible and the largest stays inside the
        # frame. The page interpolates between these on sqrt(papers).
        "radius": {"min": 10, "max": 27},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=7,
                        help="named directions drawn as satellites (default: 7)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    layout = build_layout(args.slots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({args.slots} slots + 1 summary node)")


if __name__ == "__main__":
    main()
