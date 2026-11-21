import math, statistics as st, collections, sys
import solve_beam as sb
from score import load_puzzles
Q = load_puzzles('/Users/ericmin/Downloads/santa-2026/practice_puzzles.csv')
orig = sb.SEARCH
log = []
def wrapped(t, values, limit, beam, rate):
    r = orig(t, values, limit, beam, rate)
    if r: log.append((t, len(r[1]) + (1 if r[0][0]=='one' else 0), len(values)))
    return r
sb.SEARCH = wrapped
beam = int(sys.argv[1]) if len(sys.argv) > 1 else 100
for fam in ('independent', 'mixed', 'shared', 'multiplicative'):
    ps = [p for p in Q.values() if p['family']==fam and p['design_tier']=='hard'][:8]
    log.clear(); setups=[]
    for p in ps:
        m = sb.Machine(p); sb.build_pool(m, 16, 20, 100, 50000, 7); setups.append(len(m.prog))
        sb.run_config(p, 16, 20, 100, 50000, 6, beam, 5.0)
    bits = [math.log2(t) for t,_,_ in log]; ops=[c for _,c,_ in log]
    P = st.mean(n for *_,n in log)
    # ideal single-acc chain: first op from ~2P^2 one-op values, then log2(4P) bits per op
    ideal = [max(1, math.ceil((b - math.log2(2*P*P)) / math.log2(4*P)) + 1) for b in bits]
    print(f'{fam:15} setup {st.mean(setups):.0f}  targets {len(log)}  avg bits {st.mean(bits):.1f}  ops/target {st.mean(ops):.2f}  bits/op {sum(bits)/sum(ops):.2f}  ideal-chain ops/target {st.mean(ideal):.2f}')
