import sys, csv, json, collections
from score import load_puzzles, run_program
csv.field_size_limit(10**9)
Q = load_puzzles('/Users/ericmin/Downloads/santa-2026/practice_puzzles.csv')
for path in sys.argv[1:]:
    agg = collections.defaultdict(lambda: [0, 0]); bad = 0
    for row in csv.DictReader(open(path)):
        p = Q[row['id']]; prog = json.loads(row['program'])
        if not run_program(p, prog)[0]: bad += 1
        a = agg[p['family']]; a[0] += 1; a[1] += len(prog)
    print(path, 'invalid', bad, ' '.join(f'{k}:{c/n:.1f}' for k, (n, c) in sorted(agg.items())))
