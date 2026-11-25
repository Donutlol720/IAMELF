"""Rip-up-and-reroute optimizer for Santa 2026 - The Elf Calculator.

Takes an existing valid submission and shrinks each program by treating it as a
DAG of values rather than a list of instructions:

  1. parse the program into nodes  value -> (op, a, b)  (dedupe, drop dead code)
  2. for every node v: remove v together with its *exclusive* ancestor cone C
     (ancestors used by nothing outside C, and not targets), then search for a
     cheaper way to produce v using ANY other value in the program
     (reroute.c: 1-op, 2-op chain, 3-op pair x OP y, 3-op chain)
  3. accept if the new route needs fewer new nodes than |C|; repeat passes
  4. re-schedule (topological, keeping original order where possible) and
     register-allocate into 48 registers; validate with score.run_program

Build the C core once:
    clang -O3 -shared -fPIC -o reroute_core.so reroute.c
Usage:
    python3 reroute.py PUZZLES_CSV IN_CSV OUT_CSV [--passes N] [--workers W] [--limit N]
                       [--plateau] [--atoms]
  --plateau  also accept equal-cost reroutes (randomised) on alternate passes
  --atoms    also rebuild each puzzle from scratch around hidden-atom candidates
             (atoms.py) and keep the rebuild when it is shorter
"""
import csv
import ctypes
import heapq
import json
import os
import random
import sys
from multiprocessing import Pool

from score import load_puzzles, run_program

csv.field_size_limit(10**9)
NREG = 48
OPS = "+-*/"
HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(os.path.join(HERE, "reroute_core.so"))
        u64p = ctypes.POINTER(ctypes.c_uint64)
        _LIB.reroute.argtypes = [ctypes.c_uint64, u64p, ctypes.c_int, u64p, ctypes.c_int,
                                 ctypes.c_uint64, ctypes.c_int, u64p]
        _LIB.reroute.restype = ctypes.c_int
    return _LIB


def calc(op, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a // b


# ----------------------------------------------------------------------------
# program <-> DAG
# ----------------------------------------------------------------------------
def parse(p, prog):
    """Return (nodes, pos): nodes[v] = None for starts else (op, a, b)."""
    regs = [None] * NREG
    nodes, pos = {}, {}
    for k, v in enumerate(p["start"]):
        regs[k] = v
        nodes.setdefault(v, None)
        pos.setdefault(v, -1.0)
    for idx, (op, i, j, d) in enumerate(prog):
        a, b = regs[i], regs[j]
        r = calc(op, a, b)
        regs[d] = r
        if r not in nodes:
            nodes[r] = (op, a, b)
            pos[r] = float(idx)
    prune(nodes, set(p["targets"]))
    for v in list(pos):
        if v not in nodes:
            del pos[v]
    return nodes, pos


def prune(nodes, targets):
    need = set()
    stack = [v for v in targets if v in nodes]
    while stack:
        v = stack.pop()
        if v in need:
            continue
        need.add(v)
        d = nodes[v]
        if d is not None:
            stack.extend((d[1], d[2]))
    for v in list(nodes):
        if v not in need and nodes[v] is not None:
            del nodes[v]


def users_of(nodes):
    u = {v: set() for v in nodes}
    for v, d in nodes.items():
        if d is not None:
            u[d[1]].add(v)
            u[d[2]].add(v)
    return u


def schedule(p, nodes, pos):
    """Topological order by desired position, then 48-register allocation.
    Returns program list or None if registers run out."""
    deps = {v: ({d[1], d[2]} if d is not None else set()) for v, d in nodes.items()}
    users = users_of(nodes)
    indeg = {v: len(deps[v]) for v in nodes}
    heap = [(pos[v], v) for v in nodes if indeg[v] == 0 and nodes[v] is not None]
    for v in nodes:
        if nodes[v] is None:
            for w in users[v]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    heapq.heappush(heap, (pos[w], w))
    heap = list(set(heap))
    heapq.heapify(heap)
    order, seen = [], set()
    while heap:
        _, v = heapq.heappop(heap)
        if v in seen:
            continue
        seen.add(v)
        order.append(v)
        for w in users[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(heap, (pos[w], w))
    if len(order) != sum(1 for d in nodes.values() if d is not None):
        return None
    last = {}
    for k, v in enumerate(order):
        _, a, b = nodes[v]
        last[a] = k
        last[b] = k
    reg_of, free = {}, []
    for k, v in enumerate(p["start"]):
        if v in nodes and v in last and v not in reg_of:
            reg_of[v] = k
        else:
            free.append(k)
    free.extend(range(len(p["start"]), NREG))
    prog = []
    for k, v in enumerate(order):
        op, a, b = nodes[v]
        ia, ib = reg_of[a], reg_of[b]
        for x in {a, b}:
            if last.get(x) == k:
                free.append(reg_of.pop(x))
        if not free:
            return None
        d = free.pop()
        prog.append([op, ia, ib, d])
        if v in last:
            reg_of[v] = d
        else:
            free.append(d)
    return prog


# ----------------------------------------------------------------------------
# optimisation
# ----------------------------------------------------------------------------
def cone(v, nodes, users, targets):
    C = {v}
    stack = [nodes[v][1], nodes[v][2]]
    while stack:
        x = stack.pop()
        if x in C or nodes[x] is None or x in targets:
            continue
        if users[x] <= C:
            C.add(x)
            stack.extend((nodes[x][1], nodes[x][2]))
    return C


def descendants(v, users):
    D, stack = set(), list(users[v])
    while stack:
        x = stack.pop()
        if x not in D:
            D.add(x)
            stack.extend(users[x])
    return D


def one_pass(p, nodes, pos, out_buf, plateau=False, rng=None):
    L = lib()
    targets = set(p["targets"])
    limit = p["limit"]
    gained = 0
    for v in sorted((v for v in nodes if nodes[v] is not None), key=lambda v: pos[v]):
        if v not in nodes or nodes[v] is None:
            continue
        users = users_of(nodes)
        C = cone(v, nodes, users, targets)
        maxcost = min(3, len(C) - (0 if plateau else 1))
        if maxcost < 1:
            continue
        D = descendants(v, users)
        S = [x for x in nodes if x not in C and x not in D]
        if rng:
            rng.shuffle(S)
        Sa = (ctypes.c_uint64 * len(S))(*S)
        Dl = list(D | {v})
        Da = (ctypes.c_uint64 * len(Dl))(*Dl)
        c = L.reroute(v, Sa, len(S), Da, len(Dl), limit, maxcost, out_buf)
        if c < 0 or c > len(C) or (c == len(C) and not plateau):
            continue
        if c == len(C) and {out_buf[4 * k] for k in range(c)} == C:
            continue
        new = [(out_buf[4 * k], OPS[out_buf[4 * k + 1]], out_buf[4 * k + 2], out_buf[4 * k + 3]) for k in range(c)]
        trial = dict(nodes)
        tpos = dict(pos)
        for x in C:
            del trial[x]
        base = pos[v]
        for k, (val, op, a, b) in enumerate(new):
            trial[val] = (op, a, b)
            tpos[val] = base - (c - k) * 1e-3
        if schedule(p, trial, tpos) is None:
            continue
        nodes.clear(); nodes.update(trial)
        pos.clear(); pos.update({x: tpos[x] for x in nodes})
        gained += len(C) - c
    return gained


def preds(t, s, limit):
    """Values u with t = u OP s or t = s OP u, as (u, (op, first, second))."""
    out = []
    if t > s:
        out.append((t - s, ("+", None, s)))
    if t + s <= limit:
        out.append((t + s, ("-", None, s)))
    if s > t:
        out.append((s - t, ("-", s, None)))
    if s > 1 and t % s == 0:
        out.append((t // s, ("*", None, s)))
    return out


def build_value(p, nodes, v, out_buf, small=60):
    """Cheapest new nodes (<= 4) that produce v from the current DAG, or None."""
    L = lib()
    S = list(nodes)
    Sa = (ctypes.c_uint64 * len(S))(*S)
    Da = (ctypes.c_uint64 * 1)(v)
    c = L.reroute(v, Sa, len(S), Da, 1, p["limit"], 3, out_buf)
    if c > 0:
        return [(out_buf[4 * k], OPS[out_buf[4 * k + 1]], out_buf[4 * k + 2], out_buf[4 * k + 3]) for k in range(c)]
    for s in sorted(S)[:small]:
        for u, (op, a, b) in preds(v, s, p["limit"]):
            if u in nodes:
                continue
            Da = (ctypes.c_uint64 * 2)(u, v)
            c = L.reroute(u, Sa, len(S), Da, 2, p["limit"], 3, out_buf)
            if c > 0:
                new = [(out_buf[4 * k], OPS[out_buf[4 * k + 1]], out_buf[4 * k + 2], out_buf[4 * k + 3]) for k in range(c)]
                return new + [(v, op, u if a is None else a, u if b is None else b)]
    return None


def deep_build(p, nodes, v, beam=200, rate=5.0):
    """Longer route to v via solve_beam's C beam search (elfsearch.so), as new nodes."""
    import solve_beam as sb
    res = sb.search_c(v, {x: 0 for x in nodes}, p["limit"], beam, rate)
    if res is None:
        return None
    start, chain = res
    new = []

    def node(val, op, a, b):
        if val not in nodes and all(val != n[0] for n in new):
            new.append((val, op, a, b))
        return val

    if start[0] == "reg":
        acc = start[1]
    elif start[0] == "one":
        op, a, b = start[1]
        acc = node(calc(op, a, b), op, a, b)
    else:
        (ox, ax, bx), (oy, ay, by), comb = start[1], start[2], start[3]
        x = node(calc(ox, ax, bx), ox, ax, bx)
        y = node(calc(oy, ay, by), oy, ay, by)
        if comb == 4:
            acc = node(x - y, "-", x, y)
        else:
            op = "+-*/"[comb]
            acc = node(calc(op, y, x), op, y, x)
    for op, s in chain:
        if op.startswith("T"):
            dec = sb.decompose_one(s, set(nodes))
            if dec is None:
                return None
            node(s, *dec)
            op = op[1]
        if op == "r-":
            acc = node(s - acc, "-", s, acc)
        else:
            acc = node(calc(op, acc, s), op, acc, s)
    if acc != v:
        return None
    return new


def evaluate(p, nodes, pos):
    trial = dict(nodes)
    prune(trial, set(p["targets"]))
    prog = schedule(p, trial, {x: pos[x] for x in trial})
    if prog is None or not run_program(p, prog)[0]:
        return None
    return prog


def f1_map(nodes, limit):
    vs = list(nodes)
    f1 = {}
    for i, a in enumerate(vs):
        for b in vs[i:]:
            hi, lo = (a, b) if a >= b else (b, a)
            for val, d in ((a + b, ("+", a, b)), (a * b, ("*", a, b)), (hi - lo, ("-", hi, lo)),
                           (hi // lo if hi % lo == 0 else 0, ("/", hi, lo))):
                if 0 < val <= limit and val not in nodes and val not in f1:
                    f1[val] = d
    return f1


def structured(p, nodes, f1, atoms_in, t):
    """Route t = m * (a_i * a_j +- d) +- e. Returns new-node list or None."""
    limit = p["limit"]
    best = None

    def cheap(v, new):
        if v in nodes or any(v == n[0] for n in new):
            return True
        if v in f1:
            new.append((v,) + f1[v])
            return True
        return False

    for i, a in enumerate(atoms_in):
        for b in atoms_in[i:]:
            P = a * b
            if P > limit or P == 0:
                continue
            q = t // P
            for m in (q - 1, q, q + 1):
                if m < 1:
                    continue
                for X in ({t // m, (t + m - 1) // m, P} if m > 1 else {t, P}):
                    if X <= 0 or m * X > limit:
                        continue
                    new = []
                    if not cheap(P, new):
                        new.append((P, "*", a, b))
                    if X != P:
                        d = abs(X - P)
                        if not cheap(d, new):
                            continue
                        new.append((X, "+" if X > P else "-", P, d) if X > P else (X, "-", P, d))
                    Y = X
                    if m > 1:
                        if not cheap(m, new):
                            continue
                        Y = m * X
                        new.append((Y, "*", X, m))
                    if Y != t:
                        e = abs(t - Y)
                        if not cheap(e, new):
                            continue
                        new.append((t, "+", Y, e) if t > Y else (t, "-", Y, e))
                    # drop duplicates / nodes that already exist (keep order)
                    seen, out = set(), []
                    for n in new:
                        if n[0] not in nodes and n[0] not in seen:
                            seen.add(n[0])
                            out.append(n)
                    if best is None or len(out) < len(best):
                        best = out
    return best


def construct(p, old_nodes, old_pos, atom_list, out_buf, use_struct=True):
    """Fresh DAG: starts + atoms, then greedily add the cheapest remaining target
    (<= 3 new nodes via reroute, <= 4 via build_value, else its old ancestry)."""
    L = lib()
    limit = p["limit"]
    nodes = {v: None for v in p["start"]}
    order = []

    def add(new):
        for val, op, a, b in new:
            if val not in nodes:
                nodes[val] = (op, a, b)
                order.append(val)

    for a in atom_list:
        if a not in nodes:
            new = build_value(p, nodes, a, out_buf) or deep_build(p, nodes, a)
            if new:
                add(new)
    remaining = [t for t in set(p["targets"]) if t not in nodes]
    while remaining:
        S = list(nodes)
        Sa = (ctypes.c_uint64 * len(S))(*S)
        best = None
        for t in remaining:
            Da = (ctypes.c_uint64 * 1)(t)
            c = L.reroute(t, Sa, len(S), Da, 1, limit, 3 if best is None else best[0] - 1, out_buf)
            if c > 0 and (best is None or c < best[0]):
                best = (c, [(out_buf[4 * k], OPS[out_buf[4 * k + 1]], out_buf[4 * k + 2], out_buf[4 * k + 3]) for k in range(c)])
                if c == 1:
                    break
        if (best is None or best[0] >= 3) and atom_list and use_struct:
            f1 = f1_map(nodes, limit)
            ain = [a for a in atom_list if a in nodes]
            for t in remaining:
                new = structured(p, nodes, f1, ain, t)
                if new and (best is None or len(new) < best[0]):
                    best = (len(new), new)
        if best is None:
            for t in remaining:
                new = build_value(p, nodes, t, out_buf, small=20)
                if new and (best is None or len(new) < best[0]):
                    best = (len(new), new)
        if best is None:
            for t in remaining:
                new = deep_build(p, nodes, t)
                if new and (best is None or len(new) < best[0]):
                    best = (len(new), new)
        if best is None or best[0] > 4:
            # fall back to the old program's ancestry of the target with fewest missing nodes
            def missing(t):
                need, st = [], [t]
                seen = set()
                while st:
                    x = st.pop()
                    if x in nodes or x in seen:
                        continue
                    seen.add(x)
                    need.append(x)
                    st.extend((old_nodes[x][1], old_nodes[x][2]))
                return need
            t = min(remaining, key=lambda t: len(missing(t)))
            need = missing(t)
            if best is None or len(need) < best[0]:
                # topological: old program order is a valid order
                need.sort(key=lambda x: old_pos[x])
                best = (len(need), [(x,) + old_nodes[x] for x in need])
        add(best[1])
        remaining = [t for t in remaining if t not in nodes]
    pos = {v: -1.0 for v in p["start"]}
    pos.update({v: float(k) for k, v in enumerate(order)})
    return nodes, pos


def optimise(args):
    p, prog, passes, plateau, atoms = args
    nodes, pos = parse(p, prog)
    best = schedule(p, nodes, pos)
    if best is None or not run_program(p, best)[0] or len(best) > len(prog):
        best = prog
    out_buf = (ctypes.c_uint64 * 16)()
    rng = random.Random(hash(p["targets"][0]) & 0xFFFF) if plateau else None
    if atoms:
        from atoms import detect_atoms
        cands = detect_atoms(p["targets"])
        if cands:
            r2 = rng or random.Random(1)
            for k, st in ((len(cands), True), (len(cands), False), (4, True), (4, False)):
                cn, cp = construct(p, nodes, pos, cands[:k], out_buf, use_struct=st)
                for it in range(4):
                    one_pass(p, cn, cp, out_buf, plateau=it % 2 == 1, rng=r2)
                prog = evaluate(p, cn, cp)
                if prog is not None and len(prog) < len(best):
                    best = prog
                    prune(cn, set(p["targets"]))
                    nodes, pos = cn, {x: cp[x] for x in cn}
    idle = 0
    for it in range(passes):
        g = one_pass(p, nodes, pos, out_buf, plateau=plateau and it % 2 == 1, rng=rng)
        cand = schedule(p, nodes, pos)
        if cand is not None and run_program(p, cand)[0] and len(cand) < len(best):
            best = cand
            idle = 0
        else:
            idle += 1
        if (g == 0 and not plateau) or idle >= 6:
            break
    return best


def main():
    a = sys.argv[1:]
    puzzles_csv, in_csv, out_csv = a[:3]
    passes = int(a[a.index("--passes") + 1]) if "--passes" in a else 6
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else os.cpu_count() - 1
    lim = int(a[a.index("--limit") + 1]) if "--limit" in a else None
    plateau = "--plateau" in a
    atoms = "--atoms" in a
    P = load_puzzles(puzzles_csv)
    with open(in_csv, newline="") as f:
        base = {r["id"]: json.loads(r["program"]) for r in csv.DictReader(f)}
    ids = sorted(base)[:lim] if lim else sorted(base)
    with Pool(workers) as pool:
        res = pool.map(optimise, [(P[i], base[i], passes, plateau, atoms) for i in ids], chunksize=1)
    out = dict(base)
    for i, prog in zip(ids, res):
        out[i] = prog
    before = sum(len(base[i]) for i in ids)
    after = sum(len(out[i]) for i in ids)
    print(f"{len(ids)} puzzles: {before} -> {after}  (total {sum(len(v) for v in out.values())})")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for i in sorted(out):
            w.writerow([i, json.dumps(out[i], separators=(",", ":"))])


if __name__ == "__main__":
    main()
