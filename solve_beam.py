"""Search solver for Santa 2026 - The Elf Calculator.

Register layout per puzzle
  pool : fixed values built once (starts, small digits 1..k, multipliers)
  keep : ring of recently produced targets (reused as operands -> 'shared' clusters)
  acc  : one accumulator; after a target is finished the acc slot joins the keep
         ring and the oldest keep slot becomes the next acc (no copy cost)

Each target is produced by a single-accumulator chain
    acc = a OP b ; acc = acc OP r ; ...      (r = any live register)
found by a backward beam search from the target:
    macro move   v -> (v - d) / m     m = register, d = offset of 0/1/2 register terms
    plain move   v -> v - r, v + r, r - v
    terminals    v in registers (cost 0) or v = r1 OP r2 (cost 1)

The search runs in C (elfsearch.c -> elfsearch.so, loaded with ctypes). Build it once:
    clang -O3 -shared -fPIC -o elfsearch.so elfsearch.c -lm

Usage: python3 solve_beam.py PUZZLES_CSV OUT_CSV [--limit N] [--workers W] [--beam W]
                             [--rate R] [--grid] [--py]
  --grid  try 32 pool layouts per puzzle instead of 8
  --py    use the original pure-Python search (slow; for cross-checking)
  --flags F  C-core options (add together): 1 = cost-2 finish, 2 = pair finish,
             4 = temp-register multipliers, 8 = temp-register offsets;
             add K*256 to apply 2/4 to the best K states per round (default 8)
"""
import csv
import json
import math
import sys
from multiprocessing import Pool

from score import load_puzzles, run_program

NREG = 48


# ----------------------------------------------------------------------------
# program builder with value tracking
# ----------------------------------------------------------------------------
class Machine:
    def __init__(self, p):
        self.limit = p["limit"]
        self.reg = [None] * NREG
        for i, v in enumerate(p["start"]):
            self.reg[i] = v
        self.prog = []
        self.targets = set(p["targets"])
        self.covered = set(v for v in p["start"] if v in self.targets)

    def emit(self, op, i, j, d):
        a, b = self.reg[i], self.reg[j]
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == "*":
            r = a * b
        else:
            assert a % b == 0
            r = a // b
        assert 0 < r <= self.limit, (op, a, b, r, self.limit)
        self.reg[d] = r
        self.prog.append([op, i, j, d])
        if r in self.targets:
            self.covered.add(r)
        return r

    def where(self, v, exclude=()):
        for i, x in enumerate(self.reg):
            if x == v and i not in exclude:
                return i
        return None


# ----------------------------------------------------------------------------
# pool construction
# ----------------------------------------------------------------------------
def build_pool(m, k, n_mult, mult_lo, mult_hi, reserve):
    """Fill registers with digits 1..k and n_mult multipliers. Returns pool regs."""
    limit = m.limit
    nstart = sum(1 for v in m.reg if v is not None)
    free = [i for i in range(NREG) if m.reg[i] is None]
    vals = {}
    for i, v in enumerate(m.reg):
        if v is not None and v not in vals:
            vals[v] = i
    budget = NREG - reserve  # registers usable by pool (incl. starts)

    def new_reg():
        if free:
            return free.pop(0)
        return None

    def have_room():
        return len(vals) < budget and free

    # 1
    if 1 not in vals and 1 <= limit and have_room():
        r = new_reg()
        m.emit("/", 0, 0, r)
        vals[1] = r
    # digits 2..k : prefer x+y from existing, fall back to (d-1)+1
    for d in range(2, k + 1):
        if d > limit or not have_room():
            break
        if d in vals:
            continue
        src = None
        for a in sorted(vals):
            if a >= d:
                break
            if d - a in vals:
                src = ("+", vals[a], vals[d - a])
                break
        if src is None:
            for a in sorted(vals, reverse=True):
                if a > d and a - d in vals:
                    src = ("-", vals[a], vals[a - d])
                    break
        if src is None:
            continue
        r = new_reg()
        m.emit(src[0], src[1], src[2], r)
        vals[d] = r
    # multipliers: geometric spread in [mult_lo, mult_hi], each one op from existing
    if n_mult > 0 and have_room():
        wanted = [mult_lo * (mult_hi / mult_lo) ** (i / max(n_mult - 1, 1)) for i in range(n_mult)]
        for w in wanted:
            if not have_room():
                break
            best = None
            keys = sorted(vals)
            for a in keys:
                for b in keys:
                    if b > a:
                        break
                    for op, x in (("*", a * b), ("+", a + b)):
                        if x in vals or x > limit or x < 2:
                            continue
                        err = abs(math.log(x / w))
                        if best is None or err < best[0]:
                            best = (err, op, vals[a], vals[b], x)
            if best is None:
                break
            r = new_reg()
            m.emit(best[1], best[2], best[3], r)
            vals[best[4]] = r
    return set(vals.values())


# ----------------------------------------------------------------------------
# backward beam search for one target
# ----------------------------------------------------------------------------
def make_tables(values, limit):
    """values: dict value->reg. Build offset table (cost 1/2) and 1-op table."""
    vs = sorted(values)
    off1 = {}
    for s in vs:
        off1.setdefault(s, ((+1, s),))
        off1.setdefault(-s, ((-1, s),))
    off2 = {}
    for i, a in enumerate(vs):
        for b in vs[i:]:
            for d, terms in ((a + b, ((+1, a), (+1, b))), (-(a + b), ((-1, a), (-1, b))),
                             (a - b, ((+1, a), (-1, b))), (b - a, ((+1, b), (-1, a)))):
                if d and d not in off1 and d not in off2:
                    off2[d] = terms
    one = {}
    for i, a in enumerate(vs):
        for b in vs[i:]:
            for x, ins in ((a + b, ("+", a, b)), (a * b, ("*", a, b)),
                           (b - a, ("-", b, a)), (b // a if b % a == 0 else 0, ("/", b, a))):
                if 0 < x <= limit and x not in values and x not in one:
                    one[x] = ins
    return off1, off2, one


def apply_terms(x, terms, limit):
    """Forward: x then apply offset terms; choose order to stay in (0, limit]. Returns ordered terms or None."""
    if len(terms) == 1:
        y = x + terms[0][0] * terms[0][1]
        return terms if 0 < y <= limit else None
    for order in (terms, terms[::-1]):
        y = x
        ok = True
        for sgn, s in order:
            y += sgn * s
            if not (0 < y <= limit):
                ok = False
                break
        if ok:
            return order
    return None


def search(t, values, tables, limit, beam, rate):
    """Return (cost, start_spec, forward_ops) or None.
    start_spec: ('reg', v) or ('one', (op, a, b)); forward_ops: list of (op, operand_value)
    where op in '+','-','*','r-' (r- means operand - acc)."""
    off1, off2, one = tables
    vals_sorted = sorted(values)
    mults = [s for s in vals_sorted if s >= 2]
    best = None  # (cost, v_terminal, chain)

    def term_cost(v):
        if v in values:
            return 0
        if v in one:
            return 1
        return None

    # state: v -> (cost, parent_v, move)   move = list of forward ops from parent's predecessor
    seen = {t: 0}
    frontier = [(t, 0, ())]  # (v, cost, chain-backward as tuple of forward-op lists)
    lg = math.log2
    rounds = 0
    while frontier and rounds < 60:
        rounds += 1
        children = {}
        for v, c, chain in frontier:
            tc = term_cost(v)
            if tc is not None:
                if best is None or c + tc < best[0]:
                    best = (c + tc, v, chain)
                continue
            if best is not None and c + 1 >= best[0]:
                continue

            def add_child(u, cc, ops):
                if u <= 0 or u > limit:
                    return
                if seen.get(u, 1 << 30) <= cc:
                    return
                old = children.get(u)
                if old is None or old[0] > cc:
                    children[u] = (cc, ops + chain)

            # macro moves: v = u*m + d
            for mm in mults:
                if mm > v:
                    break
                r = v % mm
                for d in (r, r - mm) if r else (0,):
                    if d == 0:
                        add_child(v // mm, c + 1, (("*", mm),))
                        continue
                    for tab, extra in ((off1, 1), (off2, 2)):
                        terms = tab.get(d)
                        if terms is None:
                            continue
                        u = (v - d) // mm
                        if u <= 0:
                            continue
                        x = u * mm
                        if x > limit:
                            continue
                        order = apply_terms(x, terms, limit)
                        if order is None:
                            continue
                        ops = (("*", mm),) + tuple(("+" if sg > 0 else "-", s) for sg, s in order)
                        add_child(u, c + 1 + extra, ops)
                        break
            # plain additive moves
            for s in vals_sorted:
                if s < v:
                    add_child(v - s, c + 1, (("+", s),))
                elif s > v:
                    add_child(s - v, c + 1, (("r-", s),))
                if v + s <= limit:
                    pass  # v + s rarely helps; skip to keep branching low
        if not children:
            break
        items = []
        for u, (cc, ch) in children.items():
            seen[u] = cc
            items.append((cc + lg(u) / rate, u, cc, ch))
        items.sort()
        frontier = [(u, cc, ch) for _, u, cc, ch in items[:beam]]
        if best is not None and items[0][0] >= best[0]:
            break
    return best


def search_py(t, values, limit, beam, rate):
    tables = make_tables(values, limit)
    res = search(t, values, tables, limit, beam, rate)
    if res is None:
        return None
    _, v0, chain = res
    start = ("reg", v0) if v0 in values else ("one", tables[2][v0])
    return start, list(chain)


# ----------------------------------------------------------------------------
# C core (elfsearch.c) via ctypes
# ----------------------------------------------------------------------------
_LIB = None
MAX_ROUNDS = 60
_COPS = {0: "+", 1: "-", 2: "*", 3: "r-", 4: "/", 5: "T*", 6: "T+", 7: "T-"}


def decompose_one(x, vals):
    """Find (op, a, b) with a op b == x, a and b register values."""
    vs = sorted(vals)
    for a in vs:
        for b in vs:
            if a + b == x:
                return ("+", a, b)
            if a * b == x:
                return ("*", a, b)
            if a - b == x:
                return ("-", a, b)
            if a % b == 0 and a // b == x:
                return ("/", a, b)
    return None
_CSTART = {0: "+", 1: "-", 2: "*", 3: "/"}


def _lib():
    global _LIB
    if _LIB is None:
        import ctypes
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        lib = ctypes.CDLL(os.path.join(here, "elfsearch.so"))
        u64 = ctypes.c_uint64
        lib.elf_search.restype = ctypes.c_int
        lib.elf_search.argtypes = [
            u64, ctypes.POINTER(u64), ctypes.c_int, u64, ctypes.c_int, ctypes.c_double, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(u64),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(u64), ctypes.c_int,
        ]
        _LIB = (lib, ctypes)
    return _LIB


FLAGS = 0  # bit0 cost-2 finish, bit1 pair finish; (K << 8) = pair tries per round


def search_c(t, values, limit, beam, rate):
    lib, ct = _lib()
    vs = list(values)
    arr = (ct.c_uint64 * len(vs))(*vs)
    si = (ct.c_int * 8)()
    sv = (ct.c_uint64 * 8)()
    MAXO = 512
    ops = (ct.c_int * MAXO)()
    args = (ct.c_uint64 * MAXO)()
    n = lib.elf_search(t, arr, len(vs), limit, beam, rate, MAX_ROUNDS, FLAGS, si, sv, ops, args, MAXO)
    if n < 0:
        return None
    if si[0] == 0:
        start = ("reg", sv[0])
    elif si[0] == 1:
        start = ("one", (_CSTART[si[1]], sv[0], sv[1]))
    else:
        start = ("pair", (_CSTART[si[1]], sv[0], sv[1]), (_CSTART[si[2]], sv[2], sv[3]), si[3])
    return start, [(_COPS[ops[i]], args[i]) for i in range(n)]


SEARCH = search_c


# ----------------------------------------------------------------------------
# per-puzzle driver
# ----------------------------------------------------------------------------
def run_config(p, k, n_mult, mult_lo, mult_hi, keep, beam, rate):
    m = Machine(p)
    reserve = keep + 1
    pool_regs = build_pool(m, k, n_mult, mult_lo, mult_hi, reserve)
    limit = m.limit
    # ring registers: anything not in pool
    ring = [i for i in range(NREG) if i not in pool_regs]
    ring_pos = 0
    todo = sorted(set(p["targets"]) - m.covered)
    for t in todo:
        if t in m.covered:
            continue
        values = {}
        for i, v in enumerate(m.reg):
            if v is not None and v not in values:
                values[v] = i
        res = SEARCH(t, values, limit, beam, rate)
        if res is None:
            return None
        start, chain = res
        # operands needed by this chain must survive: pick a ring slot not holding one
        tbuild = {}
        for op, s in chain:
            if op.startswith("T") and s not in tbuild:
                dec = decompose_one(s, values)
                if dec is None:
                    return None
                tbuild[s] = dec
        needed = {s for op, s in chain if not op.startswith("T")}
        for dec in tbuild.values():
            needed.update(dec[1:])
        if start[0] == "reg":
            needed.add(start[1])
        elif start[0] == "one":
            needed.update(start[1][1:])
        else:
            needed.update(start[1][1:])
            needed.update(start[2][1:])

        def pick_slot(exclude):
            nonlocal ring_pos
            for step in range(len(ring)):
                cand = ring[(ring_pos + step) % len(ring)]
                if cand in exclude:
                    continue
                v_c = m.reg[cand]
                if v_c is None or v_c not in needed or m.where(v_c, exclude=(cand,) + tuple(exclude)) is not None:
                    ring_pos += step + 1
                    return cand
            return None
        acc = pick_slot(())
        if acc is None:
            return None
        temp = None
        if start[0] == "pair" or tbuild:
            temp = pick_slot((acc,))
            if temp is None:
                return None
        busy = (acc,) if temp is None else (acc, temp)

        def reg_of(v):
            return m.where(v, exclude=busy)
        if start[0] == "reg":
            src = reg_of(start[1])
            if src is None:
                return None
        elif start[0] == "pair":
            (ox, ax, bx), (oy, ay, by), comb = start[1], start[2], start[3]
            rx = [reg_of(ax), reg_of(bx)]
            if None in rx:
                return None
            m.emit(ox, rx[0], rx[1], temp)
            ry = [reg_of(ay), reg_of(by)]
            if None in ry:
                return None
            m.emit(oy, ry[0], ry[1], acc)
            if comb == 4:
                m.emit("-", temp, acc, acc)
            else:
                m.emit("+-*/"[comb], acc, temp, acc)
            src = acc
        else:
            op, a, b = start[1]
            ra, rb = reg_of(a), reg_of(b)
            if ra is None or rb is None:
                return None
            m.emit(op, ra, rb, acc)
            src = acc
        for op, s in chain:
            if op.startswith("T"):
                top, ta, tb = tbuild[s]
                ra, rb = reg_of(ta), reg_of(tb)
                if ra is None or rb is None:
                    return None
                m.emit(top, ra, rb, temp)
                m.emit(op[1], src, temp, acc)
                src = acc
                continue
            rs = reg_of(s)
            if rs is None:
                return None
            if op == "r-":
                m.emit("-", rs, src, acc)
            else:
                m.emit(op, src, rs, acc)
            src = acc
        if src != acc or m.reg[acc] != t:
            # chain of length 0 from a register value: t already in registers
            if t not in m.covered:
                return None
    if not set(p["targets"]) <= m.covered:
        return None
    return m.prog


CONFIGS = [
    # k, n_mult, mult_lo, mult_hi, keep
    (16, 20, 50, 5000, 6),
    (16, 20, 100, 50000, 6),
    (10, 26, 30, 3000, 6),
    (6, 34, 8, 2000, 6),
    (20, 22, 64, 20000, 3),
    (16, 14, 50, 5000, 14),
    (8, 10, 20, 500, 20),
    (4, 4, 8, 64, 30),
]

# wider sweep, enabled with --grid
GRID = CONFIGS + [
    (k, nm, lo, hi, keep)
    for k, nm in ((8, 30), (12, 26), (20, 18), (24, 14))
    for lo, hi in ((30, 3000), (64, 30000), (100, 200000))
    for keep in (3, 8)
]


def solve(args):
    global SEARCH, FLAGS
    pid, p, beam, rate, engine, grid, flags = args
    SEARCH = search_py if engine == "py" else search_c
    FLAGS = flags
    from solve_horner import solve as horner
    best = horner(p)
    for k, nm, lo, hi, keep in (GRID if grid else CONFIGS):
        try:
            prog = run_config(p, k, nm, lo, hi, keep, beam, rate)
        except AssertionError:
            prog = None
        if prog is not None and (best is None or len(prog) < len(best)):
            ok, msg = run_program(p, prog)
            if ok:
                best = prog
    return pid, best


def main():
    puzzles = load_puzzles(sys.argv[1])
    out = sys.argv[2]
    argv = sys.argv
    lim = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 8
    beam = int(argv[argv.index("--beam") + 1]) if "--beam" in argv else 12
    rate = float(argv[argv.index("--rate") + 1]) if "--rate" in argv else 5.0
    ids = sorted(puzzles)
    if lim:
        ids = ids[:lim]
    engine = "py" if "--py" in argv else "c"
    grid = "--grid" in argv
    flags = int(argv[argv.index("--flags") + 1], 0) if "--flags" in argv else 0
    jobs = [(pid, puzzles[pid], beam, rate, engine, grid, flags) for pid in ids]
    results = {}
    with Pool(workers) as pool:
        for pid, prog in pool.imap_unordered(solve, jobs, chunksize=1):
            results[pid] = prog
    total = sum(len(v) for v in results.values())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in ids:
            w.writerow([pid, json.dumps(results[pid], separators=(",", ":"))])
    print("puzzles", len(ids), "total instructions", total)


if __name__ == "__main__":
    main()
