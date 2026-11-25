// Reroute search core for reroute.py.
// Given available values S, forbidden values D and a target t, find the
// cheapest set of new values (<= maxcost, at most 3) that produces t.
// Build: clang -O3 -shared -fPIC -o reroute_core.so reroute.c
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;

typedef struct { u64 key; u64 a, b; int op; int cost; } Ent;   // cost 0 = in S, 1 = one op from S
static Ent *tab = 0; static size_t cap = 0, mask = 0;
static u64 *dset = 0; static size_t dcap = 0, dmask = 0;

static inline u64 mix(u64 k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k; }

static void tab_reset(size_t need) {
    size_t c = 1024; while (c < need * 2) c <<= 1;
    if (c > cap) { free(tab); tab = malloc(c * sizeof(Ent)); cap = c; }
    mask = c - 1; memset(tab, 0, (mask + 1) * sizeof(Ent));
}
static inline Ent *tab_get(u64 k) {
    size_t h = mix(k) & mask;
    while (tab[h].key) { if (tab[h].key == k) return &tab[h]; h = (h + 1) & mask; }
    return 0;
}
static inline void tab_put(u64 k, int op, u64 a, u64 b, int cost) {
    size_t h = mix(k) & mask;
    while (tab[h].key) { if (tab[h].key == k) return; h = (h + 1) & mask; }
    tab[h].key = k; tab[h].op = op; tab[h].a = a; tab[h].b = b; tab[h].cost = cost;
}
static void d_reset(int n, const u64 *D) {
    size_t c = 64; while (c < (size_t)n * 2 + 2) c <<= 1;
    if (c > dcap) { free(dset); dset = malloc(c * sizeof(u64)); dcap = c; }
    dmask = c - 1; memset(dset, 0, c * sizeof(u64));
    for (int i = 0; i < n; i++) { size_t h = mix(D[i]) & dmask; while (dset[h] && dset[h] != D[i]) h = (h + 1) & dmask; dset[h] = D[i]; }
}
static inline int in_d(u64 k) { size_t h = mix(k) & dmask; while (dset[h]) { if (dset[h] == k) return 1; h = (h + 1) & dmask; } return 0; }

// ops: 0 '+', 1 '-', 2 '*', 3 '/'
static inline int apply(int op, u64 a, u64 b, u64 lim, u64 *r) {
    switch (op) {
    case 0: if (a > lim - b) return 0; *r = a + b; return 1;
    case 1: if (a <= b) return 0; *r = a - b; return 1;
    case 2: if (b && a > lim / b) return 0; *r = a * b; return *r <= lim && *r > 0;
    default: if (!b || a % b) return 0; *r = a / b; return *r > 0;
    }
}
// predecessors u such that t = u OP s or t = s OP u. Writes (u, op, ua_is_u) triples.
typedef struct { u64 u; int op; int u_first; } Inv;
static int invs(u64 t, u64 s, u64 lim, Inv *out) {
    int n = 0;
    if (t > s) out[n++] = (Inv){t - s, 0, 1};                 // u + s
    if (t <= lim - s) out[n++] = (Inv){t + s, 1, 1};          // u - s
    if (s > t) out[n++] = (Inv){s - t, 1, 0};                 // s - u
    if (s > 1 && t % s == 0) out[n++] = (Inv){t / s, 2, 1};   // u * s
    if (s > 1 && t <= lim / s) out[n++] = (Inv){t * s, 3, 1}; // u / s
    if (t > 1 && s % t == 0 && s / t > 1) out[n++] = (Inv){s / t, 3, 0}; // s / u
    return n;
}

// out: up to 3 nodes, each (value, op, a, b), in topological order. Returns cost or -1.
static int nout; static u64 *O;
static void emit(u64 v, int op, u64 a, u64 b) { O[nout * 4] = v; O[nout * 4 + 1] = op; O[nout * 4 + 2] = a; O[nout * 4 + 3] = b; nout++; }
static void emit_ent(u64 v) { Ent *e = tab_get(v); if (e && e->cost == 1) emit(v, e->op, e->a, e->b); }

int reroute(u64 t, const u64 *S, int nS, const u64 *D, int nD, u64 lim, int maxcost, u64 *out) {
    O = out; nout = 0;
    if (maxcost < 1) return -1;
    tab_reset((size_t)nS * nS * 2 + nS + 16);
    d_reset(nD, D);
    for (int i = 0; i < nS; i++) tab_put(S[i], -1, 0, 0, 0);
    for (int i = 0; i < nS; i++) for (int j = 0; j < nS; j++) for (int op = 0; op < 4; op++) {
        if ((op == 0 || op == 2) && j < i) continue;
        u64 r; if (!apply(op, S[i], S[j], lim, &r)) continue;
        if (in_d(r)) continue;
        tab_put(r, op, S[i], S[j], 1);
    }
    Ent *e = tab_get(t);
    if (e && e->cost == 1) { emit(t, e->op, e->a, e->b); return 1; }
    if (maxcost < 2) return -1;
    Inv iv[6];
    for (int i = 0; i < nS; i++) {
        int n = invs(t, S[i], lim, iv);
        for (int k = 0; k < n; k++) {
            Ent *f = tab_get(iv[k].u);
            if (f && f->cost == 1) {
                emit_ent(iv[k].u);
                if (iv[k].u_first) emit(t, iv[k].op, iv[k].u, S[i]); else emit(t, iv[k].op, S[i], iv[k].u);
                return 2;
            }
        }
    }
    if (maxcost < 3) return -1;
    // pair: t = x OP y, x,y in F1 (both new) -- y in S is covered by cost 2
    for (size_t h = 0; h <= mask; h++) {
        if (!tab[h].key || tab[h].cost != 1) continue;
        u64 x = tab[h].key;
        int n = invs(t, x, lim, iv);
        for (int k = 0; k < n; k++) {
            Ent *f = tab_get(iv[k].u);
            if (f && f->cost == 1 && iv[k].u != x) {
                emit_ent(x); emit_ent(iv[k].u);
                if (iv[k].u_first) emit(t, iv[k].op, iv[k].u, x); else emit(t, iv[k].op, x, iv[k].u);
                return 3;
            }
        }
    }
    // chain: t = u1 OP s, u1 = u2 OP s2, u2 in F1
    Inv iv2[6];
    for (int i = 0; i < nS; i++) {
        int n = invs(t, S[i], lim, iv);
        for (int k = 0; k < n; k++) {
            u64 u1 = iv[k].u;
            if (in_d(u1) || tab_get(u1)) continue;
            for (int j = 0; j < nS; j++) {
                int m = invs(u1, S[j], lim, iv2);
                for (int q = 0; q < m; q++) {
                    Ent *f = tab_get(iv2[q].u);
                    if (f && f->cost == 1) {
                        emit_ent(iv2[q].u);
                        if (iv2[q].u_first) emit(u1, iv2[q].op, iv2[q].u, S[j]); else emit(u1, iv2[q].op, S[j], iv2[q].u);
                        if (iv[k].u_first) emit(t, iv[k].op, u1, S[i]); else emit(t, iv[k].op, S[i], u1);
                        return 3;
                    }
                }
            }
        }
    }
    return -1;
}
