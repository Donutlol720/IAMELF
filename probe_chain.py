# Test: is each target reachable by a short single-accumulator chain over the starting values?
import sys, time
from score import load_puzzles
Q = load_puzzles('/Users/ericmin/Downloads/santa-2026/practice_puzzles.csv')

def search(v, S, depth, maxS):
    # can v be produced in <= depth ops where each op is acc op s (s in S) and initial acc in S
    if v in S: return 0
    if depth == 0: return None
    if v > maxS ** (depth + 1): return None
    best = None
    for s in S:
        if v % s == 0:
            r = search(v // s, S, depth - 1, maxS)
            if r is not None and (best is None or r + 1 < best): best = r + 1
    if best is not None: return best
    if depth >= 1 and v <= maxS ** depth * 2:
        for s in S:
            for w in (v - s, v + s):
                if w > 0:
                    r = search(w, S, depth - 1, maxS)
                    if r is not None: return r + 1
    return None

for fam in ('multiplicative', 'shared', 'mixed', 'independent'):
    ps = [p for p in Q.values() if p['family'] == fam and p['design_tier'] == 'hard'][:4]
    hits = tot = 0
    t0 = time.time()
    for p in ps:
        S = set(p['start']); maxS = max(S)
        for t in p['targets'][:8]:
            tot += 1
            r = search(t, S, 9, maxS)
            if r is not None: hits += 1
    print(fam, f'{hits}/{tot} targets found as short acc-chains over starts', f'{time.time()-t0:.1f}s')
