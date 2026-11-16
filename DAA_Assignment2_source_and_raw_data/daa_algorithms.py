"""
DAA Assignment 2 (CO2) - algorithm implementations.
Every algorithm returns its result together with operation counters, because
operation counts (not wall-clock time) are the primary experimental evidence.
"""
from itertools import combinations


# ----------------------------------------------------------------- Q1: sorting
def selection_sort(data):
    a = list(data)
    n = len(a)
    comps = swaps = 0
    for i in range(n - 1):
        m = i
        for j in range(i + 1, n):
            comps += 1
            if a[j] < a[m]:
                m = j
        if m != i:                      # swap only when needed
            a[i], a[m] = a[m], a[i]
            swaps += 1
    return a, comps, swaps


def bubble_sort(data):
    """Bubble sort with the standard early-exit flag (stop when a pass makes no swap)."""
    a = list(data)
    n = len(a)
    comps = swaps = 0
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            comps += 1
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swaps += 1
                swapped = True
        if not swapped:
            break
    return a, comps, swaps


def insertion_sort(data):
    """Returns sorted list, comparisons, and element movements (shifts)."""
    a = list(data)
    comps = moves = 0
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0:
            comps += 1
            if a[j] > key:
                a[j + 1] = a[j]
                moves += 1
                j -= 1
            else:
                break
        a[j + 1] = key
    return a, comps, moves


# ------------------------------------------------------- Q2: sequential search
def sequential_search(records, tool_id):
    """records: list of (tool_id, technician_id, issue_time, bay). Returns (index, comparisons)."""
    comps = 0
    for idx, rec in enumerate(records):
        comps += 1
        if rec[0] == tool_id:
            return idx, comps
    return -1, comps


def sequential_search_fast(records, tool_id):      # no counter - used for timing only
    for idx, rec in enumerate(records):
        if rec[0] == tool_id:
            return idx
    return -1


# ------------------------------------------------- Q3: brute-force string match
def brute_force_match(text, pattern):
    """Returns (first match index or -1, number of character comparisons)."""
    n, m = len(text), len(pattern)
    comps = 0
    for i in range(n - m + 1):
        j = 0
        while j < m:
            comps += 1
            if text[i + j] != pattern[j]:
                break
            j += 1
        if j == m:
            return i, comps
    return -1, comps


def brute_force_match_all(text, pattern):
    """Scans the WHOLE text (all occurrences) - used so that the work does not depend on where the first hit is."""
    n, m = len(text), len(pattern)
    comps = 0
    hits = []
    for i in range(n - m + 1):
        j = 0
        while j < m:
            comps += 1
            if text[i + j] != pattern[j]:
                break
            j += 1
        if j == m:
            hits.append(i)
    return hits, comps


# ------------------------------------------------------- Q4: brute-force closest pair
def closest_pair_bruteforce(pts):
    """Examines each unordered pair (i<j) exactly once. Squared distance avoids sqrt (same ordering)."""
    n = len(pts)
    best = float("inf")
    best_pair = None
    pairs = dists = 0
    for i in range(n - 1):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            pairs += 1
            dx = xi - pts[j][0]
            dy = yi - pts[j][1]
            d2 = dx * dx + dy * dy
            dists += 1
            if d2 < best:
                best = d2
                best_pair = (i, j)
    return best_pair, best ** 0.5, pairs, dists


# ------------------------------------------------------- Q5: brute-force convex hull
def cross(p, q, r):
    """Signed area test: >0 if r is to the LEFT of directed line p->q, <0 right, 0 collinear."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def convex_hull_bruteforce(points):
    """
    Geometric principle -> procedure:
      for every unordered pair (Pi,Pj):
          for every other point Pk compute s = cross(Pi,Pj,Pk)
          s>0 -> left side, s<0 -> right side, s==0 -> on the line
              (on the line but outside segment PiPj => reject: the pair is not extreme)
          if both left and right points exist => reject (early exit)
          else accept; orient edge so that all points are on its left (counter-clockwise)
      chain the directed edges: next[start] -> ... back to start.
    Returns (hull vertices in CCW order, number of orientation tests, accepted edge list).
    """
    pts = sorted(set(map(tuple, points)))          # remove duplicate points
    n = len(pts)
    if n <= 2:
        return pts, 0, []
    tests = 0
    edges = {}
    for a in range(n - 1):
        for b in range(a + 1, n):
            p, q = pts[a], pts[b]
            has_left = has_right = False
            ok = True
            for k in range(n):
                if k == a or k == b:
                    continue
                r = pts[k]
                tests += 1
                s = cross(p, q, r)
                if s > 0:
                    has_left = True
                elif s < 0:
                    has_right = True
                else:  # collinear: must lie strictly inside segment pq, otherwise p,q are not both extreme
                    if not (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
                            min(p[1], q[1]) <= r[1] <= max(p[1], q[1])):
                        ok = False
                        break
                if has_left and has_right:
                    ok = False
                    break
            if not ok:
                continue
            if has_left and not has_right:
                edges[p] = q                        # all points on the left of p->q
            elif has_right and not has_left:
                edges[q] = p                        # all points on the left of q->p
            else:                                   # all points collinear: hull is a segment
                return [p, q], tests, [(p, q)]
    start = min(edges)                              # lowest-leftmost vertex
    hull = [start]
    cur = edges[start]
    while cur != start:
        hull.append(cur)
        cur = edges[cur]
    return hull, tests, list(edges.items())


def convex_hull_reference(points):
    """Independent reference (Andrew's monotone chain) used ONLY to verify the brute-force result."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# ------------------------------------------------------- Q6: exhaustive search (0/1 selection)
def exhaustive_knapsack(weights, values, cap):
    """
    Representation: a candidate is an n-bit mask; bit i = 1 means package i is loaded.
    Generation: all masks 0 .. 2^n - 1.  Feasibility: total weight <= cap.
    Evaluation: total value.  Retention: keep the best feasible mask.
    Returns best_value, best_mask, candidates_examined, feasible_count.
    """
    n = len(weights)
    best_v, best_mask = -1, 0
    feasible = 0
    total = 1 << n
    for mask in range(total):
        w = v = 0
        for i in range(n):
            if (mask >> i) & 1:
                w += weights[i]
                v += values[i]
        if w <= cap:
            feasible += 1
            if v > best_v:
                best_v, best_mask = v, mask
    return best_v, best_mask, total, feasible


def dp_knapsack(weights, values, cap):
    """Independent verifier (dynamic programming) - used only to check optimum values."""
    best = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):
            best[c] = max(best[c], best[c - w] + v)
    return best[cap]


# ---- Q7 redesign: incremental depth-first search with pruning
def pruned_search(weights, values, cap, mode="C"):
    """
    mode "A": prune only when partial load already exceeds capacity (feasibility pruning)
    mode "B": A + prune if value so far + sum of ALL remaining values cannot beat the best
    mode "C": A + items ordered by value/weight, optimistic bound = fractional fill of leftover capacity
    Returns best_value, nodes_visited (each recursive call = one partial candidate examined).
    """
    n = len(weights)
    items = list(zip(weights, values))
    if mode == "C":
        items.sort(key=lambda t: t[1] / t[0], reverse=True)
    W = [t[0] for t in items]
    V = [t[1] for t in items]
    suffix_v = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_v[i] = suffix_v[i + 1] + V[i]
    best = [0]
    nodes = [0]

    def bound_c(i, w, v):
        room = cap - w
        b = v
        for k in range(i, n):
            if W[k] <= room:
                room -= W[k]
                b += V[k]
            else:
                b += V[k] * room / W[k]
                break
        return b

    def dfs(i, w, v):
        nodes[0] += 1
        if v > best[0]:
            best[0] = v
        if i == n:
            return
        if mode == "B" and v + suffix_v[i] <= best[0]:
            return
        if mode == "C" and bound_c(i, w, v) <= best[0]:
            return
        if w + W[i] <= cap:                  # include item i (only if still feasible)
            dfs(i + 1, w + W[i], v + V[i])
        dfs(i + 1, w, v)                     # exclude item i

    dfs(0, 0, 0)
    return best[0], nodes[0]
