import sys, collections
from score import load_puzzles
Q = load_puzzles('/Users/ericmin/Downloads/santa-2026/practice_puzzles.csv')
sys.setrecursionlimit(10000)
def search(v, S, depth, maxS, path):
    if v in S: return [('start', v)]
    if depth == 0 or v > maxS ** (depth + 1): return None
    for s in sorted(S, reverse=True):
        if v % s == 0:
            r = search(v // s, S, depth - 1, maxS, path)
            if r is not None: return r + [('*', s)]
    if v <= maxS ** depth * 2:
        for s in S:
            for w, op in ((v - s, '+'), (v + s, '-')):
                if w > 0:
                    r = search(w, S, depth - 1, maxS, path)
                    if r is not None: return r + [(op, s)]
    return None
p = Q['practice_0001']; S=set(p['start'])
print('start', p['start'])
lens=collections.Counter()
for t in p['targets']:
    for d in range(1, 12):
        r = search(t, S, d, max(S), None)
        if r: break
    print(t, d if r else None, r)
