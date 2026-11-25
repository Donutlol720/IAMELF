"""Hidden-atom detector for Santa 2026 puzzles.

Many puzzles are generated as  t = m * a_i * a_j + e  (small m, small e) for a
handful of hidden "atoms" a_i (typically 5-7 digits). Detection:

  1. squares:  for each target t and small multiplier m, x = t / m is close to
     a perfect square r^2  ->  r is a candidate atom
  2. products: with known atoms A, x / a is close to an integer c  ->  c is a
     candidate atom
Candidates need support from >= MIN_HITS distinct targets; repeat step 2
until no new atoms appear.

    python3 atoms.py PUZZLES_CSV [ids...]     # report atoms per puzzle
"""
import csv
import json
import math
import sys
from collections import defaultdict

MAX_M = 256
MIN_HITS = 2


def _tol(x):
    return max(64, x >> 27)  # relative ~7e-9


def _score(hits):
    """Support of a candidate: distinct targets, small multipliers count more."""
    best = {}
    for t, m in hits:
        best[t] = max(best.get(t, 0), 1.0 / (1 + math.log2(m)))
    return sum(best.values())


def detect_atoms(targets, max_m=MAX_M, top=8):
    """Ranked atom candidates (best first)."""
    targets = sorted(set(targets))
    hits = defaultdict(set)
    for t in targets:
        for m in range(1, max_m + 1):
            x = t // m
            if x < 10**6:
                break
            r = math.isqrt(x)
            for c in (r, r + 1):
                if c >= 1000 and abs(x - c * c) <= _tol(x):
                    hits[c].add((t, m))
    score = {c: _score(h) for c, h in hits.items() if len({t for t, _ in h}) >= MIN_HITS}
    seeds = _dedupe(score)[:4]
    prod = defaultdict(set)
    for t in targets:
        for m in range(1, 65):
            x = t // m
            if x < 10**6:
                break
            tol = _tol(x)
            for a in seeds:
                if a < 256 * tol:
                    continue
                c = (x + a // 2) // a
                if c >= 1000 and abs(x - a * c) <= tol:
                    prod[c].add((t, m))
    for c, h in prod.items():
        if len({t for t, _ in h}) >= MIN_HITS:
            score[c] = max(score.get(c, 0), _score(h))
    return _dedupe(score)[:top]


def _dedupe(score):
    """Neighbouring candidates (c, c+1) come from the same atom: keep the best scored."""
    out = []
    for c in sorted(score, key=lambda c: -score[c]):
        if not any(abs(c - o) <= 2 for o in out):
            out.append(c)
    return out


def main():
    csv.field_size_limit(10**9)
    rows = list(csv.DictReader(open(sys.argv[1])))
    want = set(sys.argv[2:])
    n_with = 0
    for r in rows:
        if want and r["id"] not in want:
            continue
        a = detect_atoms(json.loads(r["targets"]))
        n_with += bool(a)
        print(r["id"], a)
    print("puzzles with atoms:", n_with)


if __name__ == "__main__":
    main()
