#!/usr/bin/env python3
"""Mutation gate for TF_GLM_OVN_MULTI_FILL's scheduling and descriptor decisions.

Each entry below perturbs one decision in cuda_lanes_fill.plan / admits or cuda_prompt_pieces.fits and
must make test-multifill-host FAIL (a non-zero exit). A surviving mutant means the host tests do not pin
that decision, so this script exits non-zero and names it. The anchor for each mutant must occur exactly
once, so a drifted anchor is reported rather than silently skipped.

Usage: python3 tools/mutants_multifill.py   (run from the worktree root)
Env:   ZIG=<path to zig.exe>                 (default: Adam's 0.17.0 toolchain)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIG = os.environ.get("ZIG", r"C:\Users\Adam\tools\zig-x86_64-windows-0.17.0\zig.exe")
STEP = "test-multifill-host"

FILL = "zig/src/families/glm/cuda_lanes_fill.zig"
PIECES = "zig/src/families/glm/cuda_prompt_pieces.zig"

# (file, anchor, mutation, what the mutation breaks). One decision each.
MUTANTS = [
    (FILL, "ends[order[j - 1]] - at[order[j - 1]] > ends[i] - a",
           "ends[order[j - 1]] - at[order[j - 1]] < ends[i] - a",
           "plan: longest next piece first instead of shortest"),
    (FILL, "ends[order[j - 1]] - at[order[j - 1]] > ends[i] - a",
           "ends[order[j - 1]] - at[order[j - 1]] >= ends[i] - a",
           "plan: ties reorder instead of keeping arrival order"),
    (FILL, "if (k > 0 and rows + r > engine.prompt_rows) break;",
           "if (k > 0 and rows + r >= engine.prompt_rows) break;",
           "plan: the prompt_rows cap drops a row that still fits"),
    (FILL, "if (a >= len) continue;",
           "if (a > len) continue;",
           "plan: a finished stream joins the pass as a zero-row piece"),
    (FILL, "ends[i] = cut(a, len, mk);",
           "ends[i] = cut(a, len, &.{});",
           "plan: a piece runs past a mark instead of cutting at it"),
    (FILL, "rows <= engine.prompt_rows",
           "rows < engine.prompt_rows",
           "admits: a full-width prompt is sent to a solo fill"),
    (FILL, "s.images.len == 0",
           "true",
           "admits: an image prompt joins a batched pass"),
    (PIECES, "p.start + p.rows > p.len",
             "p.start + p.rows >= p.len",
             "fits: a piece that ends its prompt is rejected"),
    (PIECES, "if (r.at != p.start or r.at == 0) return false;",
             "if (r.at != p.start) return false;",
             "fits: a resume from position 0 is accepted"),
]


def build() -> int:
    r = subprocess.run([ZIG, "build", STEP], cwd=ROOT, capture_output=True, text=True)
    return r.returncode


def main() -> int:
    print(f"baseline: {STEP} on the unmutated tree ...", flush=True)
    if build() != 0:
        print("FAIL: the baseline does not pass; fix the tests before mutating", file=sys.stderr)
        return 1
    print("baseline passes; applying mutants\n", flush=True)
    survived, broken = [], []
    for rel, find, repl, note in MUTANTS:
        path = ROOT / rel
        src = path.read_bytes()
        n = src.count(find.encode())
        if n != 1:
            print(f"BROKEN ANCHOR ({n} matches) {rel}: {note}", file=sys.stderr)
            broken.append(note)
            continue
        path.write_bytes(src.replace(find.encode(), repl.encode(), 1))
        try:
            code = build()
        finally:
            path.write_bytes(src)
        if code == 0:
            print(f"SURVIVED  {note}")
            survived.append(note)
        else:
            print(f"killed    {note}")
    print()
    if broken or survived:
        if broken:
            print(f"FAIL: {len(broken)} anchor(s) no longer match the source", file=sys.stderr)
        if survived:
            print(f"FAIL: {len(survived)} mutant(s) survived (the tests do not pin them)", file=sys.stderr)
        return 1
    print(f"OK: all {len(MUTANTS)} mutants killed by {STEP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
