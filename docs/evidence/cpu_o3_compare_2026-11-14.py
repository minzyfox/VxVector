"""Compare vxvector_kmeans_v2 at -O0 and -O3, bit for bit, over every option."""
import ctypes as c, itertools, json, sys
import numpy as np
o0, o3 = (c.CDLL(p) for p in sys.argv[1:3])
for lib in (o0, o3): lib.vxvector_kmeans_v2.restype = c.c_int32
P = lambda a, t: a.ctypes.data_as(c.POINTER(t))
def run(lib, x, roots, leaves, factor, iters, restarts, metric, spherical, seed):
    n, d = x.shape; nodes = np.zeros((1 + roots + leaves, d), np.float32)
    parents = np.zeros(1 + roots + leaves, np.int32); assign = np.zeros(n, np.int32)
    cur = np.zeros((leaves, d), np.float32); sums = np.zeros_like(cur); counts = np.zeros(leaves, np.int32)
    s = max(1, min(n, leaves * factor)); slots = 1
    while slots < 2 * s: slots *= 2
    rows = np.zeros(s, np.int64); table = np.zeros(slots, np.int64)
    st = lib.vxvector_kmeans_v2(P(x, c.c_float), n, d, roots, leaves, factor, iters, restarts, metric, spherical, seed,
        P(nodes, c.c_float), P(parents, c.c_int32), P(assign, c.c_int32), P(cur, c.c_float), P(sums, c.c_float),
        P(counts, c.c_int32), P(rows, c.c_int64), P(table, c.c_int64))
    return st, nodes, parents, assign
rng = np.random.default_rng(11); cases = same = 0
for (n, d), metric, spherical, restarts, roots, factor, seed in itertools.product(
        [(500, 3), (2000, 37), (1500, 128)], [0, 1, 2], [0, 1], [1, 3], [0, 4], [4, 1000], [1, 99]):
    x = (rng.normal(0, 5, (n, d)) * rng.uniform(0.1, 10, d)).astype(np.float32)
    args = (x, roots, 16, factor, 6, restarts, metric, spherical, seed)
    a, b = run(o0, *args), run(o3, *args)
    ok = a[0] == b[0] and all(np.array_equal(p.view(np.uint32) if p.dtype == np.float32 else p,
                                             q.view(np.uint32) if q.dtype == np.float32 else q) for p, q in zip(a[1:], b[1:]))
    cases += 1; same += ok
    if not ok: print(json.dumps(dict(n=n, d=d, metric=metric, spherical=spherical, restarts=restarts, roots=roots, factor=factor, seed=seed)))
print(json.dumps(dict(cases=cases, identical=same)))
