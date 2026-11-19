"""Baseline solver: base-B Horner evaluation with shared-prefix reuse.

For each puzzle and each base B, keep digit registers 1..B-1 and B, then build
every target as ((d1*B + d2)*B + d3)... . Targets of equal digit-length are
processed in sorted order so a common leading-digit prefix with the previous
target is reused instead of recomputed (this is what helps the 'shared'
family). The best B per puzzle is kept.

Usage: python3 solve_horner.py PUZZLES_CSV OUT_CSV
"""
import csv
import json
import sys

from score import load_puzzles, run_program

NREG = 48


def digits_of(t, B):
    ds = []
    while t:
        ds.append(t % B)
        t //= B
    return ds[::-1]


def build(p, B):
    start, limit = p["start"], p["limit"]
    if B > limit:
        return None
    targets = sorted(set(p["targets"]) - set(start))
    prog = []
    # --- registers -------------------------------------------------------
    # digit_reg[d] for d in 1..B (B itself stored as 'digit' B)
    digit_reg = {}
    for r, v in enumerate(start):
        if 1 <= v <= B and v not in digit_reg:
            digit_reg[v] = r
    reserved = set(digit_reg.values())
    free = [r for r in range(NREG) if r not in reserved]
    # start registers not holding a kept digit are free once '1' exists
    def alloc():
        return free.pop(0)
    if 1 not in digit_reg:
        r = next(r for r in range(len(start), NREG) if r not in reserved) if len(start) < NREG else None
        free.remove(r)
        prog.append(["/", 0, 0, r])
        digit_reg[1] = r
    for d in range(2, B + 1):
        if d not in digit_reg:
            r = alloc()
            prog.append(["+", digit_reg[d - 1], digit_reg[1], r])
            digit_reg[d] = r
    maxlen = max((len(digits_of(t, B)) for t in targets), default=0)
    if len(free) < max(maxlen - 1, 0):
        return None
    depth_reg = free[: max(maxlen - 1, 0)]
    # --- targets --------------------------------------------------------
    covered = set(start) | set(range(1, B + 1))
    prev = None  # digits of previous target (same length)
    for t in sorted(targets, key=lambda x: (len(digits_of(x, B)), x)):
        if t in covered:
            continue
        ds = digits_of(t, B)
        k = 0
        if prev is not None and len(prev) == len(ds):
            while k < len(ds) and prev[k] == ds[k]:
                k += 1
        # prefix of length max(k,1) is available: depth 1 is a digit register
        start_depth = max(k, 1)
        cur = digit_reg[ds[0]] if start_depth == 1 else depth_reg[start_depth - 2]
        val = digits_of_val = None
        for depth in range(start_depth, len(ds)):
            dst = depth_reg[depth - 1]
            prog.append(["*", cur, digit_reg[B], dst])
            if ds[depth]:
                prog.append(["+", dst, digit_reg[ds[depth]], dst])
            cur = dst
        covered.add(t)
        prev = ds
    return prog


def solve(p):
    best = None
    for B in range(2, 41):
        try:
            prog = build(p, B)
        except (StopIteration, ValueError, IndexError, KeyError):
            continue
        if prog is None:
            continue
        if best is None or len(prog) < len(best):
            ok, _ = run_program(p, prog)
            if ok:
                best = prog
    return best


if __name__ == "__main__":
    puzzles = load_puzzles(sys.argv[1])
    out = sys.argv[2]
    total = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in sorted(puzzles):
            prog = solve(puzzles[pid])
            if prog is None:
                raise SystemExit(f"no program for {pid}")
            total += len(prog)
            w.writerow([pid, json.dumps(prog, separators=(",", ":"))])
    print("total instructions", total)
