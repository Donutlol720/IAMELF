import sys, math, statistics as st
import solve_beam as sb
from score import load_puzzles
Q = load_puzzles('/Users/ericmin/Downloads/santa-2026/practice_puzzles.csv')
ps = [p for p in Q.values() if p['family'] in ('independent','mixed') and p['design_tier']=='hard'][:10]
configs = [(16,20,100,50000,6),(16,20,200,1000000,6),(12,26,100,100000,4),(20,22,64,20000,3),(16,26,100,1000000,2),(24,20,200,10**7,2),(12,30,500,10**8,2)]
for cfg in configs:
    setups=[]; bpo=[]; tot=0
    for p in ps:
        m = sb.Machine(p); bp = sb.build_pool(m, cfg[0], cfg[1], cfg[2], cfg[3], cfg[4]+1)
        setups.append(len(m.prog))
        prog = sb.run_config(p, *cfg, 12, 5.0)
        tot += len(prog)
        bits = sum(math.log2(t) for t in set(p['targets'])-set(p['start']))
        bpo.append(bits/(len(prog)-len(m.prog)))
    print(cfg, 'setup', st.mean(setups), 'bits/op', round(st.mean(bpo),2), 'avg total', tot/len(ps))
