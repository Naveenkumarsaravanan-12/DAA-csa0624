import random, time, json, csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from daa_algorithms import *

OUT = "/home/claude/work/data"
os.makedirs(OUT, exist_ok=True)
R = {}                      # all results
random.seed(2026)


def write_csv(name, header, rows):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def tmin(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


# =========================================================== Q1
def make_datasets(n):
    base = random.sample(range(1, 10 * n), n)
    nsorted = int(0.95 * n)
    srt = sorted(random.sample(range(1, 10 * n), nsorted))
    new = random.sample(range(1, 10 * n), n - nsorted)
    scattered = sorted(random.sample(range(1, 10 * n), n))
    for pos in random.sample(range(n), n // 20):     # 5% of positions overwritten by random scores
        scattered[pos] = random.randint(1, 10 * n)
    return {
        "Random": base,
        "Already sorted": sorted(base),
        "Reverse sorted": sorted(base, reverse=True),
        "95% sorted (5% new orders appended)": srt + new,
        "95% sorted (5% scattered)": scattered,
        "All equal scores (edge case)": [500] * n,
    }


n1 = 1000
q1 = []
for name, data in make_datasets(n1).items():
    for alg, fn in (("Selection", selection_sort), ("Bubble", bubble_sort), ("Insertion", insertion_sort)):
        out, c, s = fn(data)
        assert out == sorted(data)
        t = tmin(lambda: fn(data))
        q1.append([name, alg, c, s, round(t * 1000, 3)])
R["q1"] = q1
write_csv("q1_sorting_n1000", ["dataset", "algorithm", "comparisons", "swaps_or_moves", "time_ms"], q1)

# predicted values (formulas) for the deterministic cases
q1_scale = []
for n in (250, 500, 1000, 2000):
    d = make_datasets(n)["95% sorted (5% new orders appended)"]
    for alg, fn in (("Selection", selection_sort), ("Bubble", bubble_sort), ("Insertion", insertion_sort)):
        out, c, s = fn(d)
        q1_scale.append([n, alg, c, s])
R["q1_scale"] = q1_scale
write_csv("q1_scaling_95pct", ["n", "algorithm", "comparisons", "swaps_or_moves"], q1_scale)

# =========================================================== Q2
def make_records(n):
    ids = random.sample(range(1, 10 * n + 1000), n)
    return [(f"T-{i}", f"TECH-{random.randint(1,200)}", f"{random.randint(0,23):02d}:{random.randint(0,59):02d}",
             f"Bay-{random.randint(1,12)}") for i in ids]


q2 = []
tcmp = None
for n in (10, 100, 1000, 10000, 100000):
    recs = make_records(n)
    first = recs[0][0]
    mid = recs[n // 2][0]
    last = recs[-1][0]
    _, c_first = sequential_search(recs, first)
    _, c_mid = sequential_search(recs, mid)
    _, c_last = sequential_search(recs, last)
    _, c_miss = sequential_search(recs, "T-NOTISSUED")
    sample = recs if n <= 10000 else random.sample(recs, 5000)
    avg = sum(sequential_search(recs, r[0])[1] for r in sample) / len(sample)
    reps = max(1, 200000 // n)
    t = tmin(lambda: [sequential_search_fast(recs, "T-NOTISSUED") for _ in range(reps)], 3) / reps
    q2.append([n, c_first, c_mid, c_last, round(avg, 1), c_miss, round(t * 1e6, 3), round(t / n * 1e9, 1)])
    if n == 100000:
        tcmp = t / n
R["q2"] = q2
R["q2_tcmp_ns"] = tcmp * 1e9
write_csv("q2_sequential_search", ["n", "cmp_first", "cmp_middle", "cmp_last", "cmp_avg_successful",
                                   "cmp_unsuccessful", "time_miss_us", "ns_per_comparison"], q2)

# =========================================================== Q3
def log_text(lines, seed=7):
    rnd = random.Random(seed)
    ev = ["Accepted password for alice from 10.0.0.%d port 22 ssh2",
          "Failed password for invalid user admin from 10.0.0.%d port 22 ssh2",
          "Failed password for invalid user test from 10.0.0.%d port 22 ssh2",
          "Failed password for root from 10.0.0.%d port 22 ssh2",
          "Session opened for user bob by (uid=0) src 10.0.0.%d",
          "Connection closed by 10.0.0.%d port 22 [preauth]"]
    wts = [30, 25, 15, 3, 20, 7]
    out = []
    for i in range(lines):
        e = rnd.choices(ev, wts)[0] % rnd.randint(1, 250)
        out.append(f"Nov 16 10:{i % 60:02d}:{i % 60:02d} srv{rnd.randint(1,9)} sshd[{rnd.randint(1000,9999)}]: {e}\n")
    return "".join(out)


n3, m3 = 10000, 100
tests3 = {
    "T1 Best-case (mismatch at 1st char)": ("a" * n3, "b" + "a" * (m3 - 1)),
    "T2 Worst-case (mismatch at last char)": ("a" * n3, "a" * (m3 - 1) + "b"),
}
lt = log_text(1200)
tests3["T3 Realistic SSH log, pattern 'Failed password for root'"] = (lt, "Failed password for root")
q3 = []
for name, (txt, pat) in tests3.items():
    hits, c = brute_force_match_all(txt, pat)
    align = len(txt) - len(pat) + 1
    q3.append([name, len(txt), len(pat), align, len(hits), c, round(c / align, 3)])
R["q3"] = q3
write_csv("q3_string_matching", ["test", "n", "m", "alignments", "matches", "char_comparisons",
                                 "comparisons_per_alignment"], q3)
# effect of pattern length (worst case) and of text length
q3b = []
for n in (1000, 2000, 4000, 8000):
    for m in (10, 50, 100):
        _, cb = brute_force_match_all("a" * n, "b" + "a" * (m - 1))
        _, cw = brute_force_match_all("a" * n, "a" * (m - 1) + "b")
        q3b.append([n, m, cb, cw, (n - m + 1) * m])
R["q3b"] = q3b
write_csv("q3_scaling", ["n", "m", "best_comparisons", "worst_comparisons", "formula_(n-m+1)*m"], q3b)

# =========================================================== Q4
q4 = []
for n in (100, 250, 500, 1000, 2000, 4000, 6000):
    pts = [(random.uniform(0, 10000), random.uniform(0, 10000)) for _ in range(n)]
    pair, d, pairs, dists = closest_pair_bruteforce(pts)
    t = tmin(lambda: closest_pair_bruteforce(pts), 1 if n >= 2000 else 3)
    q4.append([n, n * (n - 1) // 2, pairs, dists, round(t, 4)])
    assert pairs == n * (n - 1) // 2
R["q4"] = q4
write_csv("q4_closest_pair", ["n", "theory_pairs", "pairs_examined", "distance_calcs", "time_s"], q4)

# =========================================================== Q5
def hull_ok(points):
    h, t, e = convex_hull_bruteforce(points)
    ref = convex_hull_reference(points)
    return set(h) == set(ref), len(h)


# set A: boundary + interior
rnd = random.Random(11)
setA = [(rnd.randint(0, 40), rnd.randint(0, 40)) for _ in range(30)]
setA += [(2, 3), (38, 5), (35, 38), (4, 36)]            # force a wide boundary
# set B: designed to challenge the rule
setB = [(0, 0), (10, 0), (10, 10), (0, 10),              # square corners
        (5, 0), (10, 5), (5, 10), (0, 5),                # collinear points ON the edges
        (5, 5), (5, 1), (9, 9),                          # interior points (one very close to edge/corner)
        (10, 10), (0, 0)]                                # duplicate points
hullA, tA, edA = convex_hull_bruteforce(setA)
hullB, tB, edB = convex_hull_bruteforce(setB)
R["q5_sets"] = {"A": [setA, hullA, tA], "B": [setB, hullB, tB]}
okA = set(hullA) == set(convex_hull_reference(setA))
okB = set(hullB) == set(convex_hull_reference(setB))
# extra degenerate tests
deg = {
    "All points collinear": [(0, 0), (1, 1), (2, 2), (3, 3), (5, 5)],
    "Two distinct points": [(1, 1), (4, 2)],
    "Triangle + interior": [(0, 0), (10, 0), (5, 9), (5, 3), (4, 2)],
}
deg_res = {k: convex_hull_bruteforce(v)[0] for k, v in deg.items()}
R["q5_deg"] = {k: v for k, v in deg_res.items()}
# randomised verification against reference (many integer grid sets -> lots of collinear points)
fails = 0
trials = 600
r2 = random.Random(5)
for _ in range(trials):
    k = r2.randint(3, 25)
    pts = [(r2.randint(0, 6), r2.randint(0, 6)) for _ in range(k)]
    if len(set(pts)) < 3:
        continue
    h, _, _ = convex_hull_bruteforce(pts)
    ref = convex_hull_reference(pts)
    if len(set(ref)) >= 3 and (set(h) != set(ref) or len(h) != len(ref)):
        fails += 1
    elif len(set(ref)) < 3 and set(h) != set(ref):
        fails += 1
R["q5_verify"] = [trials, fails, okA, okB]

# growth
q5 = []
for n in (10, 20, 40, 80, 160):
    r3 = random.Random(n)
    rand_pts = list({(r3.randint(0, 100000), r3.randint(0, 100000)) for _ in range(n)})
    par = [(i, i * i) for i in range(n)]                 # convex position: all points on hull
    h1, t1, _ = convex_hull_bruteforce(rand_pts)
    h2, t2, _ = convex_hull_bruteforce(par)
    q5.append([n, len(h1), t1, len(h2), t2, n * (n - 1) * (n - 2) // 2])
R["q5"] = q5
write_csv("q5_hull_growth", ["n", "hull_size_random", "tests_random", "hull_size_convex_position",
                             "tests_convex_position", "upper_bound_n(n-1)(n-2)/2"], q5)

# figure: hull visualisation
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, (title, pts, hull) in zip(axs, (("Set A: random boundary + interior (n=%d)" % len(set(setA)), setA, hullA),
                                        ("Set B: collinear edge points, duplicates, near-edge interior", setB, hullB))):
    hs = set(hull)
    ux = list(set(pts))
    ax.scatter([p[0] for p in ux if p not in hs], [p[1] for p in ux if p not in hs], c="#6b7280", s=28, label="interior / non-extreme")
    ax.scatter([p[0] for p in hull], [p[1] for p in hull], c="#dc2626", s=45, zorder=3, label="hull vertex")
    cyc = hull + [hull[0]]
    ax.plot([p[0] for p in cyc], [p[1] for p in cyc], c="#2563eb", lw=1.8)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_q5_hull.png", dpi=170)
plt.close()

# =========================================================== Q6
def make_instance(n, seed, correlated=False):
    r = random.Random(seed)
    w = [r.randint(5, 40) for _ in range(n)]
    if correlated:
        v = [x + 10 for x in w]                           # strongly correlated: hardest case for bounds
    else:
        v = [r.randint(10, 100) for _ in range(n)]
    cap = int(0.4 * sum(w))
    return w, v, cap


# correctness instance (5 packages) - all 32 subsets listed
w5, v5, c5 = [12, 7, 11, 8, 9], [24, 13, 23, 15, 16], 26
subsets = []
for mask in range(32):
    ws = sum(w5[i] for i in range(5) if mask >> i & 1)
    vs = sum(v5[i] for i in range(5) if mask >> i & 1)
    subsets.append((mask, [i + 1 for i in range(5) if mask >> i & 1], ws, vs, ws <= c5))
bv, bm, tot, feas = exhaustive_knapsack(w5, v5, c5)
R["q6_small"] = [w5, v5, c5, subsets, bv, bm, tot, feas, dp_knapsack(w5, v5, c5)]

# random verification vs DP
bad = 0
for s in range(200):
    n = random.Random(s).randint(1, 12)
    w, v, c = make_instance(n, s)
    a = exhaustive_knapsack(w, v, c)[0]
    b = dp_knapsack(w, v, c)
    d = pruned_search(w, v, c, "C")[0]
    e = pruned_search(w, v, c, "B")[0]
    f = pruned_search(w, v, c, "A")[0]
    if not (a == b == d == e == f):
        bad += 1
R["q6_verify_bad"] = bad

q6 = []
for n in (8, 10, 12, 14, 16, 18, 20):
    w, v, c = make_instance(n, 100 + n)
    t0 = time.perf_counter()
    bv, bm, tot, feas = exhaustive_knapsack(w, v, c)
    t = time.perf_counter() - t0
    q6.append([n, tot, 2 ** n, feas, round(feas / tot, 3), bv, round(t, 3)])
    assert bv == dp_knapsack(w, v, c)
R["q6"] = q6
write_csv("q6_exhaustive", ["n", "candidates_examined", "2^n", "feasible", "feasible_fraction",
                            "best_value", "time_s"], q6)

# =========================================================== Q7 (redesign of Q6)
q7 = []
for n in (8, 10, 12, 14, 16, 18, 20):
    w, v, c = make_instance(n, 100 + n)
    row = [n, 2 ** n]
    for mode in "ABC":
        t0 = time.perf_counter()
        best, nodes = pruned_search(w, v, c, mode)
        t = time.perf_counter() - t0
        assert best == dp_knapsack(w, v, c)
        row += [nodes, round(t, 4)]
    q7.append(row)
R["q7"] = q7
write_csv("q7_redesign_vs_original", ["n", "original_candidates", "A_nodes", "A_time_s", "B_nodes", "B_time_s",
                                      "C_nodes", "C_time_s"], q7)

# redesign beyond exhaustive reach (random and hard instances) - average over 5 seeds
q7b = []
for n in (20, 25, 30, 35, 40, 50):
    ns, ts = [], []
    for sd in range(5):
        w, v, c = make_instance(n, 500 + 10 * n + sd)
        t0 = time.perf_counter()
        best, nodes = pruned_search(w, v, c, "C")
        ts.append(time.perf_counter() - t0)
        ns.append(nodes)
        assert best == dp_knapsack(w, v, c)
    q7b.append([n, round(sum(ns) / 5), max(ns), round(sum(ts) / 5, 4), round(max(ts), 4)])
R["q7b"] = q7b
write_csv("q7_redesign_larger_random", ["n", "mean_nodes", "max_nodes", "mean_time_s", "max_time_s"], q7b)

q7c = []
for n in (16, 20, 24, 28):
    ns, ts = [], []
    for sd in range(3):
        w, v, c = make_instance(n, 900 + 10 * n + sd, correlated=True)
        t0 = time.perf_counter()
        best, nodes = pruned_search(w, v, c, "C")
        ts.append(time.perf_counter() - t0)
        ns.append(nodes)
        assert best == dp_knapsack(w, v, c)
    q7c.append([n, 2 ** n, round(sum(ns) / 3), round(sum(ts) / 3, 4)])
R["q7c"] = q7c
write_csv("q7_redesign_correlated_hard", ["n", "original_candidates", "mean_nodes_C", "mean_time_s_C"], q7c)

# ------------------------------------------------------------- figures
# Q1 scaling
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for alg, mk in (("Selection", "o-"), ("Bubble", "s-"), ("Insertion", "^-")):
    xs = [r[0] for r in q1_scale if r[1] == alg]
    axs[0].plot(xs, [r[2] for r in q1_scale if r[1] == alg], mk, label=alg)
    axs[1].plot(xs, [r[3] for r in q1_scale if r[1] == alg], mk, label=alg)
axs[0].set_title("Comparisons (95% sorted data)"); axs[1].set_title("Swaps / movements (95% sorted data)")
for a in axs:
    a.set_xlabel("n (orders)"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_q1.png", dpi=170); plt.close()

# Q2
fig, ax = plt.subplots(figsize=(5, 3.4))
ns = [r[0] for r in q2]
ax.loglog(ns, [r[2] for r in q2], "o-", label="successful, middle")
ax.loglog(ns, [r[3] for r in q2], "s-", label="successful, last (worst)")
ax.loglog(ns, [r[1] for r in q2], "^-", label="successful, first (best)")
ax.loglog(ns, [r[5] for r in q2], "x--", label="unsuccessful")
ax.set_xlabel("list size n"); ax.set_ylabel("comparisons"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_q2.png", dpi=170); plt.close()

# Q3
fig, ax = plt.subplots(figsize=(5, 3.4))
for m in (10, 50, 100):
    ax.plot([r[0] for r in q3b if r[1] == m], [r[3] for r in q3b if r[1] == m], "o-", label=f"worst, m={m}")
    ax.plot([r[0] for r in q3b if r[1] == m], [r[2] for r in q3b if r[1] == m], "--", label=f"best, m={m}")
ax.set_xlabel("text length n"); ax.set_ylabel("character comparisons"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_q3.png", dpi=170); plt.close()

# Q4
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
ns = [r[0] for r in q4]
axs[0].plot(ns, [r[2] for r in q4], "o-", label="measured pairs")
axs[0].plot(ns, [r[1] for r in q4], "k--", label="n(n-1)/2")
axs[0].set_xlabel("n drones"); axs[0].set_ylabel("pairs examined"); axs[0].legend(); axs[0].grid(alpha=.3)
axs[1].plot(ns, [r[4] for r in q4], "s-", c="#dc2626")
axs[1].axhline(1.0, c="gray", ls=":"); axs[1].text(ns[0], 1.05, "1 s acceptability limit", fontsize=7)
axs[1].set_xlabel("n drones"); axs[1].set_ylabel("time (s)"); axs[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_q4.png", dpi=170); plt.close()

# Q5 growth
fig, ax = plt.subplots(figsize=(5, 3.4))
ns = [r[0] for r in q5]
ax.loglog(ns, [r[2] for r in q5], "o-", label="random points")
ax.loglog(ns, [r[4] for r in q5], "s-", label="all points on hull")
ax.loglog(ns, [r[5] for r in q5], "k--", label="n(n-1)(n-2)/2")
ax.set_xlabel("n points"); ax.set_ylabel("orientation tests"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_q5_growth.png", dpi=170); plt.close()

# Q6
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
ns = [r[0] for r in q6]
axs[0].semilogy(ns, [r[1] for r in q6], "o-", label="candidates examined (measured)")
axs[0].semilogy(ns, [2 ** n for n in ns], "k--", label="2^n")
axs[0].set_xlabel("n packages"); axs[0].legend(fontsize=7); axs[0].grid(alpha=.3, which="both")
axs[1].semilogy(ns, [r[6] for r in q6], "s-", c="#dc2626")
axs[1].set_xlabel("n packages"); axs[1].set_ylabel("time (s)"); axs[1].grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_q6.png", dpi=170); plt.close()

# Q7
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ns = [r[0] for r in q7]
ax.semilogy(ns, [r[1] for r in q7], "k--", label="Original: 2^n")
ax.semilogy(ns, [r[2] for r in q7], "o-", label="A: feasibility pruning")
ax.semilogy(ns, [r[4] for r in q7], "s-", label="B: + remaining-value bound")
ax.semilogy(ns, [r[6] for r in q7], "^-", label="C: + density order + fractional bound")
ax.set_xlabel("n packages"); ax.set_ylabel("candidates / nodes examined"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_q7.png", dpi=170); plt.close()

json.dump(R, open(f"{OUT}/all_results.json", "w"), indent=1, default=str)
print("done")
