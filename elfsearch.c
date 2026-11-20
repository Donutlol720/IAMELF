/* Backward beam search core for Santa 2026 - The Elf Calculator.
 * C port of solve_beam.search(). Build:
 *   clang -O3 -shared -fPIC -o elfsearch.so elfsearch.c -lm
 *
 * Forward op codes in the returned chain (acc = accumulator, s = register value):
 *   0: acc = acc + s    1: acc = acc - s    2: acc = acc * s    3: acc = s - acc
 * Start: type 0 -> acc chain starts from register value start_a
 *        type 1 -> acc = start_a OP start_b first, OP code in start_op:
 *                  0 '+', 1 '-', 2 '*', 3 '/'
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned __int128 u128;

/* ---------------------------------------------------------------- hash map */
typedef struct {
    u64 *keys;
    int *vals;
    unsigned *stamp;
    unsigned cur;
    size_t cap, n;
} HMap;

static inline u64 mix(u64 k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33; return k;
}

static void hm_alloc(HMap *h, size_t cap) {
    h->keys = malloc(cap * sizeof(u64));
    h->vals = malloc(cap * sizeof(int));
    h->stamp = calloc(cap, sizeof(unsigned));
    h->cap = cap; h->cur = 1; h->n = 0;
}
static void hm_free(HMap *h) { free(h->keys); free(h->vals); free(h->stamp); }
static void hm_clear(HMap *h) {
    h->n = 0;
    if (++h->cur == 0) { memset(h->stamp, 0, h->cap * sizeof(unsigned)); h->cur = 1; }
}
static int *hm_get(HMap *h, u64 k) {
    size_t m = h->cap - 1, i = mix(k) & m;
    while (h->stamp[i] == h->cur) {
        if (h->keys[i] == k) return &h->vals[i];
        i = (i + 1) & m;
    }
    return NULL;
}
static int *hm_put(HMap *h, u64 k, int init, int *isnew);
static void hm_grow(HMap *h) {
    HMap g; hm_alloc(&g, h->cap * 2);
    for (size_t i = 0; i < h->cap; i++)
        if (h->stamp[i] == h->cur) { int nw; *hm_put(&g, h->keys[i], h->vals[i], &nw) = h->vals[i]; }
    hm_free(h); *h = g;
}
static int *hm_put(HMap *h, u64 k, int init, int *isnew) {
    if ((h->n + 1) * 2 > h->cap) hm_grow(h);
    size_t m = h->cap - 1, i = mix(k) & m;
    while (h->stamp[i] == h->cur) {
        if (h->keys[i] == k) { *isnew = 0; return &h->vals[i]; }
        i = (i + 1) & m;
    }
    h->stamp[i] = h->cur; h->keys[i] = k; h->vals[i] = init; h->n++;
    *isnew = 1; return &h->vals[i];
}

/* ------------------------------------------------------------------- state */
#define MAXOPS 3
typedef struct {
    u64 v;
    int cost;
    int parent;
    unsigned char nops;
    unsigned char op[MAXOPS];
    u64 arg[MAXOPS];
} Node;

typedef struct { double f; int idx; } Key;

static int cmp_key(const void *a, const void *b) {
    double x = ((const Key *)a)->f, y = ((const Key *)b)->f;
    if (x < y) return -1;
    if (x > y) return 1;
    u64 va = (u64)((const Key *)a)->idx, vb = (u64)((const Key *)b)->idx;
    return va < vb ? -1 : va > vb;
}

/* offset table entries */
typedef struct { signed char sg[2]; u64 s[2]; unsigned char n; } Off;
/* one-op table entries */
typedef struct { unsigned char op; u64 a, b, x; } One;

static HMap OFF, ONE, SEEN, CH;
static Off *offs; static size_t noffs, capoffs;
static One *ones; static size_t nones, capones;
static Node *store; static size_t nstore, capstore;
static Node *child; static size_t nchild, capchild;
static Key *keys; static size_t capkeys;
static int inited = 0;

static void ensure_init(void) {
    if (inited) return;
    hm_alloc(&OFF, 1 << 14); hm_alloc(&ONE, 1 << 14);
    hm_alloc(&SEEN, 1 << 16); hm_alloc(&CH, 1 << 16);
    capoffs = 8192; offs = malloc(capoffs * sizeof(Off));
    capones = 8192; ones = malloc(capones * sizeof(One));
    capstore = 1 << 14; store = malloc(capstore * sizeof(Node));
    capchild = 1 << 14; child = malloc(capchild * sizeof(Node));
    capkeys = 1 << 14; keys = malloc(capkeys * sizeof(Key));
    inited = 1;
}

static void off_add(i64 d, int n, int sg0, u64 s0, int sg1, u64 s1) {
    if (d == 0) return;
    int isnew;
    int *slot = hm_put(&OFF, (u64)d, (int)noffs, &isnew);
    if (!isnew) return; /* first (cheapest) wins */
    if (noffs == capoffs) { capoffs *= 2; offs = realloc(offs, capoffs * sizeof(Off)); }
    Off *o = &offs[noffs++];
    o->n = (unsigned char)n; o->sg[0] = (signed char)sg0; o->s[0] = s0; o->sg[1] = (signed char)sg1; o->s[1] = s1;
    (void)slot;
}

static void one_add(u64 x, int op, u64 a, u64 b, u64 limit, HMap *vals) {
    if (x == 0 || x > limit) return;
    if (hm_get(vals, x)) return;
    int isnew;
    hm_put(&ONE, x, (int)nones, &isnew);
    if (!isnew) return;
    if (nones == capones) { capones *= 2; ones = realloc(ones, capones * sizeof(One)); }
    One *o = &ones[nones++];
    o->op = (unsigned char)op; o->a = a; o->b = b; o->x = x;
}

static int cmp_u64(const void *a, const void *b) {
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return x < y ? -1 : x > y;
}

static void push_child(u64 u, int cc, int parent, int nops, const unsigned char *op, const u64 *arg, u64 limit) {
    if (u == 0 || u > limit) return;
    int *sv = hm_get(&SEEN, u);
    if (sv && *sv <= cc) return;
    int isnew;
    int *slot = hm_put(&CH, u, (int)nchild, &isnew);
    Node *nd;
    if (isnew) {
        if (nchild == capchild) { capchild *= 2; child = realloc(child, capchild * sizeof(Node)); }
        nd = &child[nchild++];
    } else {
        nd = &child[*slot];
        if (nd->cost <= cc) return;
    }
    nd->v = u; nd->cost = cc; nd->parent = parent; nd->nops = (unsigned char)nops;
    for (int k = 0; k < nops; k++) { nd->op[k] = op[k]; nd->arg[k] = arg[k]; }
}

/* order two offset terms so that every intermediate stays in (0, limit].
   returns 1 and writes ordered terms, or 0 */
static int order_terms(u64 x, const Off *o, u64 limit, int *sg, u64 *s) {
    if (o->n == 1) {
        u128 y = o->sg[0] > 0 ? (u128)x + o->s[0] : (u128)x - o->s[0];
        if (o->sg[0] < 0 && x <= o->s[0]) return 0;
        if (y == 0 || y > limit) return 0;
        sg[0] = o->sg[0]; s[0] = o->s[0];
        return 1;
    }
    for (int r = 0; r < 2; r++) {
        int a = r, b = 1 - r;
        i64 sga = o->sg[a], sgb = o->sg[b];
        /* step 1 */
        u64 y;
        if (sga > 0) { u128 t = (u128)x + o->s[a]; if (t > limit) continue; y = (u64)t; }
        else { if (x <= o->s[a]) continue; y = x - o->s[a]; }
        if (sgb > 0) { u128 t = (u128)y + o->s[b]; if (t > limit) continue; }
        else { if (y <= o->s[b]) continue; }
        sg[0] = (int)sga; s[0] = o->s[a]; sg[1] = (int)sgb; s[1] = o->s[b];
        return 2;
    }
    return 0;
}

/* main entry ------------------------------------------------------------- */
/* flags: bit0 = cost-2 finish (one pool op away from a register / one-op value)
 *        bit1 = pair finish (X in temp, Y in acc, combine; cost 3), tried on the
 *               best (flags >> 8) states of each round (default 8)
 * sinfo/sval describe the start:
 *   sinfo[0]=0 reg   : sval[0] = register value
 *   sinfo[0]=1 one   : acc = sval[0] OP(sinfo[1]) sval[1]
 *   sinfo[0]=2 pair  : temp = sval[0] OP(sinfo[1]) sval[1]; acc = sval[2] OP(sinfo[2]) sval[3];
 *                      then combine sinfo[3]: 0 acc+temp, 1 acc-temp, 2 acc*temp, 3 acc/temp, 4 temp-acc
 * chain op codes: 0 '+', 1 '-', 2 '*', 3 's - acc', 4 '/'
 *   5 acc*T, 6 acc+T, 7 acc-T  where T = arg is a one-op value built in the temp register
 *   (flag 4: temp multipliers on the best K states per round; flag 8: temp offsets) */
static int term_cost(HMap *VAL, u64 u) {
    if (hm_get(VAL, u)) return 0;
    if (hm_get(&ONE, u)) return 1;
    return -1;
}

int elf_search(u64 t, const u64 *vals_in, int nv, u64 limit, int beam, double rate, int max_rounds,
               int flags, int *sinfo, u64 *sval,
               int *out_ops, u64 *out_args, int max_out) {
    ensure_init();
    /* register values: sorted unique */
    u64 vals[64];
    int n = 0;
    for (int i = 0; i < nv && n < 64; i++) vals[n++] = vals_in[i];
    qsort(vals, n, sizeof(u64), cmp_u64);
    int m2 = 0;
    for (int i = 0; i < n; i++) if (i == 0 || vals[i] != vals[i - 1]) vals[m2++] = vals[i];
    n = m2;
    HMap VAL; hm_alloc(&VAL, 256);
    for (int i = 0; i < n; i++) { int nw; hm_put(&VAL, vals[i], i, &nw); }
    u64 mults[64]; int nm = 0;
    for (int i = 0; i < n; i++) if (vals[i] >= 2) mults[nm++] = vals[i];

    /* offset table: 1-term entries first so they win */
    hm_clear(&OFF); noffs = 0;
    for (int i = 0; i < n; i++) {
        if (vals[i] <= (u64)INT64_MAX) {
            off_add((i64)vals[i], 1, +1, vals[i], 0, 0);
            off_add(-(i64)vals[i], 1, -1, vals[i], 0, 0);
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            u64 a = vals[i], b = vals[j];
            if (a + b < a || a + b > (u64)INT64_MAX) continue;
            off_add((i64)(a + b), 2, +1, a, +1, b);
            off_add(-(i64)(a + b), 2, -1, a, -1, b);
            off_add((i64)a - (i64)b, 2, +1, a, -1, b);
            off_add((i64)b - (i64)a, 2, +1, b, -1, a);
        }
    /* one-op table */
    hm_clear(&ONE); nones = 0;
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            u64 a = vals[i], b = vals[j];
            u128 s = (u128)a + b;
            if (s <= limit) one_add((u64)s, 0, a, b, limit, &VAL);
            u128 p = (u128)a * b;
            if (p <= limit) one_add((u64)p, 2, a, b, limit, &VAL);
            if (b > a) one_add(b - a, 1, b, a, limit, &VAL);
            if (b % a == 0) one_add(b / a, 3, b, a, limit, &VAL);
        }

    hm_clear(&SEEN);
    nstore = 0;
    store[nstore].v = t; store[nstore].cost = 0; store[nstore].parent = -1; store[nstore].nops = 0;
    nstore++;
    { int nw; *hm_put(&SEEN, t, 0, &nw) = 0; }
    int fr_lo = 0, fr_hi = 1; /* frontier = store[fr_lo..fr_hi) */
    int best_cost = 1 << 30, best_node = -1, best_term = -1; /* term: 0 reg, 1 one, 2 pair */
    u64 best_base = 0;            /* terminal value for reg/one */
    int best_xop = -1; u64 best_xarg = 0; /* extra first op (cost-2 finish) */
    int best_px = -1, best_py = -1, best_comb = -1;
    int pair_k = (flags >> 8) ? (flags >> 8) : 8;

    for (int round = 0; round < max_rounds && fr_lo < fr_hi; round++) {
        hm_clear(&CH); nchild = 0;
        for (int fi = fr_lo; fi < fr_hi; fi++) {
            Node *nd = &store[fi];
            u64 v = nd->v; int c = nd->cost;
            int tc = term_cost(&VAL, v);
            if (tc >= 0) {
                if (c + tc < best_cost) {
                    best_cost = c + tc; best_node = fi; best_term = tc; best_base = v; best_xop = -1;
                }
                continue;
            }
            /* cost-2 finish: v = u OP s with u a register value or one-op value */
            if ((flags & 1) && c + 2 < best_cost) {
                for (int k = 0; k < n; k++) {
                    u64 s = vals[k];
                    u64 us[5]; int ok[5] = {0, 0, 0, 0, 0};
                    if (s < v) { us[0] = v - s; ok[0] = 1; }                         /* acc + s */
                    if ((u128)v + s <= limit) { us[1] = v + s; ok[1] = 1; }          /* acc - s */
                    if (s >= 2 && v % s == 0) { us[2] = v / s; ok[2] = 1; }          /* acc * s */
                    if (s > v) { us[3] = s - v; ok[3] = 1; }                         /* s - acc */
                    if (s >= 2 && (u128)v * s <= limit) { us[4] = v * s; ok[4] = 1; } /* acc / s */
                    for (int q = 0; q < 5; q++) {
                        if (!ok[q]) continue;
                        int tu = term_cost(&VAL, us[q]);
                        if (tu < 0) continue;
                        if (c + 1 + tu < best_cost) {
                            best_cost = c + 1 + tu; best_node = fi; best_term = tu; best_base = us[q];
                            best_xop = q; best_xarg = s;
                        }
                    }
                }
            }
            /* pair finish: v = Y comb X, X and Y both one-op values (needs a temp register) */
            if ((flags & 2) && fi - fr_lo < pair_k && c + 3 < best_cost) {
                for (size_t xi = 0; xi < nones && c + 3 < best_cost; xi++) {
                    u64 X = ones[xi].x;
                    u64 ys[5]; int ok[5] = {0, 0, 0, 0, 0};
                    if (X < v) { ys[0] = v - X; ok[0] = 1; }
                    if ((u128)v + X <= limit) { ys[1] = v + X; ok[1] = 1; }
                    if (X >= 2 && v % X == 0) { ys[2] = v / X; ok[2] = 1; }
                    if (X >= 2 && (u128)v * X <= limit) { ys[3] = v * X; ok[3] = 1; }
                    if (X > v) { ys[4] = X - v; ok[4] = 1; }
                    for (int q = 0; q < 5; q++) {
                        if (!ok[q]) continue;
                        int *yi = hm_get(&ONE, ys[q]);
                        if (!yi) continue;
                        best_cost = c + 3; best_node = fi; best_term = 2;
                        best_px = (int)xi; best_py = *yi; best_comb = q; best_xop = -1;
                        break;
                    }
                }
            }
            if (c + 1 >= best_cost) continue;
            unsigned char op[MAXOPS]; u64 arg[MAXOPS];
            /* macro moves */
            for (int k = 0; k < nm; k++) {
                u64 mm = mults[k];
                if (mm > v) break;
                u64 r = v % mm;
                if (r == 0) {
                    op[0] = 2; arg[0] = mm;
                    push_child(v / mm, c + 1, fi, 1, op, arg, limit);
                    continue;
                }
                i64 ds[2] = {(i64)r, (i64)r - (i64)mm};
                for (int di = 0; di < 2; di++) {
                    i64 d = ds[di];
                    int *oi = hm_get(&OFF, (u64)d);
                    if (d >= 0 && (u64)d >= v) continue;
                    if (!oi) {
                        /* offset built in the temp register (flag 8): acc*mm then acc +/- T */
                        if (!(flags & 8)) continue;
                        u64 ad = d >= 0 ? (u64)d : (u64)(-d);
                        if (!hm_get(&ONE, ad)) continue;
                        u128 vd2 = d >= 0 ? (u128)v - ad : (u128)v + ad;
                        u64 u2 = (u64)(vd2 / mm);
                        if (u2 == 0) continue;
                        u128 x2 = (u128)u2 * mm;
                        if (x2 > limit) continue;
                        op[0] = 2; arg[0] = mm;
                        op[1] = d >= 0 ? 6 : 7; arg[1] = ad;
                        push_child(u2, c + 3, fi, 2, op, arg, limit);
                        continue;
                    }
                    const Off *o = &offs[*oi];
                    /* u = (v - d) / mm */
                    u128 vd = d >= 0 ? (u128)v - (u64)d : (u128)v + (u64)(-d);
                    u64 u = (u64)(vd / mm);
                    if (u == 0) continue;
                    u128 x = (u128)u * mm;
                    if (x > limit) continue;
                    int sg[2]; u64 s[2];
                    int nt = order_terms((u64)x, o, limit, sg, s);
                    if (!nt) continue;
                    op[0] = 2; arg[0] = mm;
                    for (int q = 0; q < nt; q++) { op[1 + q] = sg[q] > 0 ? 0 : 1; arg[1 + q] = s[q]; }
                    push_child(u, c + 1 + o->n, fi, 1 + nt, op, arg, limit);
                }
            }
            /* temp multipliers (flag 4): v = u*T + d, T a one-op value built in the temp register */
            if ((flags & 4) && fi - fr_lo < pair_k) {
                for (size_t xi = 0; xi < nones; xi++) {
                    u64 X = ones[xi].x;
                    if (X < 2 || X > v || hm_get(&VAL, X)) continue;
                    u64 r = v % X;
                    if (r == 0) {
                        op[0] = 5; arg[0] = X;
                        push_child(v / X, c + 2, fi, 1, op, arg, limit);
                        continue;
                    }
                    i64 ds[2] = {(i64)r, (i64)r - (i64)X};
                    for (int di = 0; di < 2; di++) {
                        i64 d = ds[di];
                        int *oi = hm_get(&OFF, (u64)d);
                        if (!oi) continue;
                        const Off *o = &offs[*oi];
                        u128 vd = d >= 0 ? (u128)v - (u64)d : (u128)v + (u64)(-d);
                        u64 u = (u64)(vd / X);
                        if (u == 0) continue;
                        u128 x = (u128)u * X;
                        if (x > limit) continue;
                        int sg[2]; u64 s[2];
                        int nt = order_terms((u64)x, o, limit, sg, s);
                        if (!nt) continue;
                        op[0] = 5; arg[0] = X;
                        for (int q = 0; q < nt; q++) { op[1 + q] = sg[q] > 0 ? 0 : 1; arg[1 + q] = s[q]; }
                        push_child(u, c + 2 + o->n, fi, 1 + nt, op, arg, limit);
                    }
                }
            }
            /* plain additive moves */
            for (int k = 0; k < n; k++) {
                u64 s = vals[k];
                if (s < v) { op[0] = 0; arg[0] = s; push_child(v - s, c + 1, fi, 1, op, arg, limit); }
                else if (s > v) { op[0] = 3; arg[0] = s; push_child(s - v, c + 1, fi, 1, op, arg, limit); }
            }
        }
        if (nchild == 0) break;
        if (nchild > capkeys) { while (capkeys < nchild) capkeys *= 2; keys = realloc(keys, capkeys * sizeof(Key)); }
        for (size_t i = 0; i < nchild; i++) {
            int nw;
            *hm_put(&SEEN, child[i].v, child[i].cost, &nw) = child[i].cost;
            keys[i].f = child[i].cost + log2((double)child[i].v) / rate;
            keys[i].idx = (int)i;
        }
        qsort(keys, nchild, sizeof(Key), cmp_key);
        size_t take = nchild < (size_t)beam ? nchild : (size_t)beam;
        if (nstore + take > capstore) { while (capstore < nstore + take) capstore *= 2; store = realloc(store, capstore * sizeof(Node)); }
        fr_lo = (int)nstore;
        for (size_t i = 0; i < take; i++) store[nstore++] = child[keys[i].idx];
        fr_hi = (int)nstore;
        if (best_node >= 0 && keys[0].f >= best_cost) break;
    }
    /* frontier of the final round may still contain terminals */
    for (int fi = fr_lo; fi < fr_hi; fi++) {
        Node *nd = &store[fi];
        int tc = term_cost(&VAL, nd->v);
        if (tc >= 0 && nd->cost + tc < best_cost) {
            best_cost = nd->cost + tc; best_node = fi; best_term = tc; best_base = nd->v; best_xop = -1;
        }
    }
    if (best_node < 0) { hm_free(&VAL); return -1; }
    sinfo[0] = best_term;
    if (best_term == 0) { sval[0] = best_base; }
    else if (best_term == 1) {
        int *oi = hm_get(&ONE, best_base);
        sinfo[1] = ones[*oi].op; sval[0] = ones[*oi].a; sval[1] = ones[*oi].b;
    } else {
        sinfo[1] = ones[best_px].op; sval[0] = ones[best_px].a; sval[1] = ones[best_px].b;
        sinfo[2] = ones[best_py].op; sval[2] = ones[best_py].a; sval[3] = ones[best_py].b;
        /* q: 0 v=Y+X, 1 v=Y-X, 2 v=Y*X, 3 v=Y/X, 4 v=X-Y  (matches combine codes) */
        sinfo[3] = best_comb;
    }
    int no = 0;
    if (best_xop >= 0) {
        /* map q index of the cost-2 finish to chain op codes */
        static const int qmap[5] = {0, 1, 2, 3, 4};
        out_ops[no] = qmap[best_xop]; out_args[no] = best_xarg; no++;
    }
    for (int idx = best_node; idx >= 0 && store[idx].parent >= 0; idx = store[idx].parent) {
        Node *nd = &store[idx];
        for (int k = 0; k < nd->nops; k++) {
            if (no >= max_out) { hm_free(&VAL); return -2; }
            out_ops[no] = nd->op[k]; out_args[no] = nd->arg[k]; no++;
        }
    }
    hm_free(&VAL);
    /* return number of forward ops; cost = no + start_type */
    return no;
}
