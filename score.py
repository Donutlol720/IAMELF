"""Local validator / scorer for Kaggle Santa 2026 - The Elf Calculator.

Usage:
    python3 score.py PUZZLES_CSV SUBMISSION_CSV [--verbose]

Implements the calculator rules from the competition Data page exactly:
  - 48 registers; starting_values fill registers 0..n-1, the rest are undefined
  - [op, i, j, d]: both sources read before write; sources must be defined
  - results must be positive integers <= value_limit
  - "-" only when a > b; "/" only when b divides a exactly
  - a target is covered once it appears in any register (incl. at the start)
  - <= 4096 instructions, <= 131072 UTF-8 bytes per program cell
"""
import csv
import json
import sys

MAX_INSTR = 4096
MAX_CELL_BYTES = 131072
csv.field_size_limit(10**9)


def load_puzzles(path):
    puzzles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            puzzles[row["id"]] = {
                "start": json.loads(row["starting_values"]),
                "targets": json.loads(row["targets"]),
                "limit": int(row["value_limit"]),
                "registers": int(row["registers"]),
                **{k: row[k] for k in ("family", "design_tier", "lower_bound") if k in row},
            }
    return puzzles


def run_program(puzzle, program):
    """Return (ok, message). Raises nothing; all errors are reported in message."""
    nreg = puzzle["registers"]
    limit = puzzle["limit"]
    if not isinstance(program, list):
        return False, "program is not a JSON array"
    if len(program) > MAX_INSTR:
        return False, f"{len(program)} instructions > {MAX_INSTR}"
    regs = [None] * nreg
    for k, v in enumerate(puzzle["start"]):
        regs[k] = v
    remaining = set(puzzle["targets"]) - set(puzzle["start"])
    for step, ins in enumerate(program):
        if not (isinstance(ins, list) and len(ins) == 4):
            return False, f"step {step}: malformed instruction {ins!r}"
        op, i, j, d = ins
        if op not in ("+", "-", "*", "/"):
            return False, f"step {step}: bad operator {op!r}"
        for r in (i, j, d):
            if type(r) is not int or not (0 <= r < nreg):
                return False, f"step {step}: bad register {r!r}"
        a, b = regs[i], regs[j]
        if a is None or b is None:
            return False, f"step {step}: undefined source register"
        if op == "+":
            res = a + b
        elif op == "-":
            if a <= b:
                return False, f"step {step}: subtraction {a}-{b} not positive"
            res = a - b
        elif op == "*":
            res = a * b
        else:
            if a % b:
                return False, f"step {step}: inexact division {a}/{b}"
            res = a // b
        if res > limit:
            return False, f"step {step}: value {res} exceeds limit {limit}"
        regs[d] = res
        remaining.discard(res)
    if remaining:
        return False, f"{len(remaining)} targets not covered, e.g. {min(remaining)}"
    return True, "ok"


def score_submission(puzzles, sub_path, verbose=False):
    seen = {}
    with open(sub_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ["id", "program"]:
            raise SystemExit(f"bad header {header}")
        for row in reader:
            pid, cell = row
            if pid in seen:
                raise SystemExit(f"duplicate id {pid}")
            if len(cell.encode("utf-8")) > MAX_CELL_BYTES:
                raise SystemExit(f"{pid}: cell exceeds {MAX_CELL_BYTES} bytes")
            seen[pid] = json.loads(cell)
    missing = set(puzzles) - set(seen)
    extra = set(seen) - set(puzzles)
    if missing or extra:
        raise SystemExit(f"missing ids: {sorted(missing)[:5]} extra ids: {sorted(extra)[:5]}")
    total = 0
    bad = []
    per = {}
    for pid, prog in seen.items():
        ok, msg = run_program(puzzles[pid], prog)
        if not ok:
            bad.append((pid, msg))
        per[pid] = len(prog)
        total += len(prog)
    if verbose:
        for pid in sorted(per):
            print(pid, per[pid])
    return total, bad, per


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    puzzles = load_puzzles(sys.argv[1])
    total, bad, _ = score_submission(puzzles, sys.argv[2], "--verbose" in sys.argv)
    if bad:
        print(f"INVALID: {len(bad)} programs fail")
        for pid, msg in bad[:20]:
            print(" ", pid, msg)
        sys.exit(1)
    print(f"VALID. Score = {total}")
