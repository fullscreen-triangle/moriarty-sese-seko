#!/usr/bin/env python3
# =====================================================================
#  Figure panels for "Propagation Mechanics of an Embedded Agent".
#  4 panels, each 4 charts in a row, white background, minimal text,
#  at least one 3D chart per panel. Resting cuts computed exactly by
#  a self-contained Edmonds-Karp max-flow / min-cut backend.
#  No text/conceptual/table charts.
# =====================================================================
import os
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

RNG = np.random.default_rng(20260719)
FLOOR = 1.0
TOL = 1e-9
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 9,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
})
BLUE, GREEN, ORANGE, PURPLE, RED = (
    "#2563eb", "#16a34a", "#ea9a09", "#7c3aed", "#dc2626")


# ---- exact max-flow / min-cut ----
def min_cut(cap, s, t):
    n = cap.shape[0]
    res = cap.astype(float).copy()
    while True:
        parent = [-1] * n; parent[s] = s; q = deque([s])
        while q:
            u = q.popleft()
            for v in range(n):
                if parent[v] == -1 and res[u, v] > TOL:
                    parent[v] = u
                    if v == t:
                        q.clear(); break
                    q.append(v)
        if parent[t] == -1:
            break
        pf = np.inf; v = t
        while v != s:
            pf = min(pf, res[parent[v], v]); v = parent[v]
        v = t
        while v != s:
            u = parent[v]; res[u, v] -= pf; res[v, u] += pf; v = u
    seen = [False] * n; q = deque([s]); seen[s] = True
    while q:
        u = q.popleft()
        for v in range(n):
            if not seen[v] and res[u, v] > TOL:
                seen[v] = True; q.append(v)
    val = sum(cap[i, j] for i in range(n) if seen[i]
              for j in range(n) if not seen[j])
    return val, [v for v in range(n) if seen[v]]


def random_world(n_items, density, wmin=FLOOR, wspread=4.0):
    n = n_items + 1
    W = np.zeros((n, n))
    for v in range(1, n):
        W[0, v] = W[v, 0] = wmin + RNG.random() * wspread
    for i in range(1, n):
        for j in range(i + 1, n):
            if RNG.random() < density:
                W[i, j] = W[j, i] = wmin + RNG.random() * wspread
    return W


def sep_cost(W, v):
    return min_cut(W, s=v, t=0)


def enumerate_paths(W, src, dst, max_len):
    n = W.shape[0]; paths = []

    def dfs(u, vis, path):
        if len(path) > max_len:
            return
        if u == dst:
            paths.append(list(path)); return
        for v in range(n):
            if W[u, v] > TOL and v not in vis:
                vis.add(v); path.append(v); dfs(v, vis, path)
                path.pop(); vis.discard(v)

    dfs(src, {src}, [src]); return paths


def panel_header(fig, letters):
    for ax, L in zip(fig.axes, letters):
        ax.set_title(L, loc="left", fontweight="bold", fontsize=11, x=-0.02)


def finish(fig, name):
    fig.subplots_adjust(left=0.045, right=0.965, bottom=0.14, top=0.9, wspace=0.42)
    fig.savefig(os.path.join(FIGDIR, name), dpi=150)
    plt.close(fig)


# =====================================================================
#  PANEL 1 : The floor and recall-is-search
# =====================================================================
def panel1():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    # (A) separation cost per item vs graph size, on/above floor
    sizes, sigmas = [], []
    for _ in range(80):
        ni = int(RNG.integers(3, 9))
        W = random_world(ni, RNG.uniform(0.3, 0.8))
        for v in range(1, ni + 1):
            val, _ = sep_cost(W, v)
            sizes.append(ni + RNG.uniform(-0.3, 0.3)); sigmas.append(val)
    ax1.scatter(sizes, sigmas, s=8, c=BLUE, alpha=0.4)
    ax1.axhline(FLOOR, color=RED, lw=1.6)
    ax1.set_xlabel("world size (items)")
    ax1.set_ylabel("separation cost $\\sigma(v)$")

    # (B) histogram sigma/floor, all >= 1
    sg = np.array(sigmas)
    ax2.hist(sg / FLOOR, bins=40, color=GREEN, alpha=0.85)
    ax2.axvline(1.0, color=RED, lw=1.6)
    ax2.set_xlabel("$\\sigma(v)/\\beta$"); ax2.set_ylabel("count")

    # (C) recall = search: report requires >=1 committed act (cost) ; 0-act = nothing
    acts = np.arange(0, 12)
    yield_ = (acts >= 1).astype(int)
    ax3.step(acts, yield_, where="post", color=BLUE, lw=2)
    ax3.fill_between(acts, 0, yield_, step="post", color=GREEN, alpha=0.15)
    ax3.scatter([0], [0], s=60, facecolors="none", edgecolors=RED, lw=1.6)
    ax3.set_xlabel("committed acts in search")
    ax3.set_ylabel("report produced")
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(["none", "yes"])

    # (D) 3D: realised floor over (world size, density)
    NS = np.arange(3, 10); DS = np.linspace(0.2, 0.9, 18)
    NN, DD = np.meshgrid(NS, DS)
    ZZ = np.zeros_like(NN, dtype=float)
    for i in range(NN.shape[0]):
        for j in range(NN.shape[1]):
            W = random_world(int(NN[i, j]), DD[i, j])
            ZZ[i, j] = min(sep_cost(W, v)[0] for v in range(1, int(NN[i, j]) + 1))
    ax4.plot_surface(NN, DD, ZZ, cmap=cm.viridis, edgecolor="none", alpha=0.95)
    ax4.set_xlabel("items"); ax4.set_ylabel("density")
    ax4.set_zlabel("realised floor", labelpad=2); ax4.view_init(elev=24, azim=-58)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_1.png")


# =====================================================================
#  PANEL 2 : Path opacity and the monotone record
# =====================================================================
def panel2():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax4 = fig.add_subplot(1, 4, 4)

    # (A) number of distinct interiors between same endpoints vs graph density
    dens = np.linspace(0.3, 0.95, 30); ninteriors = []
    for d in dens:
        vals = []
        for _ in range(8):
            ni = 6; W = random_world(ni, d)
            paths = enumerate_paths(W, 1, ni, max_len=ni + 1)
            vals.append(len(set(tuple(p[1:-1]) for p in paths)))
        ninteriors.append(np.mean(vals))
    ax1.plot(dens, ninteriors, color=BLUE, lw=2, marker="o", ms=3)
    ax1.axhline(1, color=RED, lw=1.2, ls="--")
    ax1.set_xlabel("world density")
    ax1.set_ylabel("distinct interiors (same endpoints)")

    # (B) endpoints do not determine interior: interior length spread per endpoint pair
    W = random_world(7, 0.8)
    paths = enumerate_paths(W, 1, 7, max_len=8)
    lens = [len(p) - 2 for p in paths]
    ax2.hist(lens, bins=range(0, max(lens) + 2), color=GREEN, alpha=0.85,
             align="left", rwidth=0.8)
    ax2.set_xlabel("interior length (same seed & terminus)")
    ax2.set_ylabel("number of searches")

    # (C) 3D: several same-endpoint search walks with distinct interiors
    #      embed items on a ring; draw walks in (x, y, step)
    ni = 7
    ang = np.linspace(0, 2 * np.pi, ni + 1, endpoint=False)
    xy = np.c_[np.cos(ang), np.sin(ang)]              # positions incl medium at 0
    xy[0] = [0, 0]                                     # medium centre
    picks = paths[:4] if len(paths) >= 4 else paths
    cols = [BLUE, GREEN, ORANGE, PURPLE]
    for p, col in zip(picks, cols):
        px = xy[p, 0]; py = xy[p, 1]; pz = np.arange(len(p))
        ax3.plot(px, py, pz, color=col, lw=1.8, marker="o", ms=3)
    ax3.scatter(xy[1, 0], xy[1, 1], 0, color=RED, s=45)     # seed
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    ax3.set_zlabel("step", labelpad=2); ax3.view_init(elev=22, azim=-60)

    # (D) monotone committed record; revisited positions never return as states
    rec = np.arange(1, 46); pos = RNG.integers(0, 6, size=45)
    ax4.plot(rec, pos, color=BLUE, lw=0.8, alpha=0.5)
    ax4.scatter(rec, pos, s=16, c=rec, cmap="plasma")
    rv = np.where(pos == pos[0])[0]
    ax4.scatter(rec[rv], pos[rv], s=60, facecolors="none", edgecolors=RED, lw=1.4)
    ax4.set_xlabel("committed record $M$"); ax4.set_ylabel("position")

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_2.png")


# =====================================================================
#  PANEL 3 : Receiver-relative divergence; falsity as a relation
# =====================================================================
def panel3():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    # (A) same event: sparse-agent sigma vs dense-agent sigma (off diagonal)
    va, vb = [], []
    for _ in range(150):
        ni = int(RNG.integers(3, 7))
        Wa = random_world(ni, 0.3); Wb = random_world(ni, 0.85)
        v = int(RNG.integers(1, ni + 1))
        va.append(sep_cost(Wa, v)[0]); vb.append(sep_cost(Wb, v)[0])
    ax1.scatter(va, vb, s=14, c=BLUE, alpha=0.5)
    lim = [0, max(max(va), max(vb)) * 1.05]
    ax1.plot(lim, lim, color=RED, lw=1.3, ls="--")
    ax1.set_xlabel("sparse agent $\\sigma(v)$")
    ax1.set_ylabel("dense agent $\\sigma(v)$")

    # (B) both registrations correct (>= floor) though different
    diffs = np.abs(np.array(va) - np.array(vb))
    ax2.hist(diffs, bins=30, color=ORANGE, alpha=0.85)
    ax2.axvline(0, color=RED, lw=1.4)
    ax2.set_xlabel("|registration difference| across agents")
    ax2.set_ylabel("count")

    # (C) falsity as relation: agree/disagree of account pairs under a standard
    #      matrix of set-equality between agents' resting-cut sides
    ni = 6
    agents = [random_world(ni, RNG.uniform(0.25, 0.9)) for _ in range(8)]
    v = 3
    sides = [frozenset(sep_cost(A, v)[1]) for A in agents]
    M = np.array([[1 if sides[i] == sides[j] else 0
                   for j in range(8)] for i in range(8)])
    ax3.imshow(M, cmap="Greens", vmin=0, vmax=1)
    ax3.set_xlabel("agent (standard)"); ax3.set_ylabel("agent (account)")
    ax3.set_xticks(range(8)); ax3.set_yticks(range(8))

    # (D) 3D: sigma of an event over (sparse density, dense density)
    D1 = np.linspace(0.2, 0.9, 16); D2 = np.linspace(0.2, 0.9, 16)
    DD1, DD2 = np.meshgrid(D1, D2)
    Z = np.zeros_like(DD1)
    for i in range(DD1.shape[0]):
        for j in range(DD1.shape[1]):
            Wa = random_world(5, DD1[i, j])
            Z[i, j] = sep_cost(Wa, 2)[0]
    ax4.plot_surface(DD1, DD2, Z, cmap=cm.viridis, edgecolor="none", alpha=0.95)
    ax4.set_xlabel("density"); ax4.set_ylabel("density (2)")
    ax4.set_zlabel("$\\sigma$(event)", labelpad=2); ax4.view_init(elev=24, azim=-58)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_3.png")


# =====================================================================
#  PANEL 4 : Catalytic drift and forced inquiry
# =====================================================================
def panel4():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    # (A) surviving fraction prod g_i along a relay chain, several gain levels
    ns = np.arange(1, 15)
    for g, col in zip([0.6, 0.75, 0.9], [BLUE, GREEN, ORANGE]):
        ax1.plot(ns, g ** ns, color=col, lw=1.9, marker="o", ms=3,
                 label=f"$g_i={g}$")
    ax1.set_xlabel("relays $n$"); ax1.set_ylabel("surviving fraction $\\prod g_i$")
    ax1.legend(fontsize=7, frameon=False)

    # (B) drift 1 - prod g_i for random chains
    drifts = []
    for _ in range(400):
        n = int(RNG.integers(2, 12)); g = RNG.uniform(0.3, 1.0, n)
        drifts.append(1 - np.prod(g))
    ax2.hist(drifts, bins=30, color=PURPLE, alpha=0.85)
    ax2.set_xlabel("account drift $1-\\prod g_i$"); ax2.set_ylabel("count")

    # (C) forced inquiry: residual gap over an exchange, act while D>0, closes at 0
    T = 30
    gap = np.maximum(0, 3 * np.exp(-np.arange(T) / 6.0) + RNG.uniform(-0.1, 0.1, T))
    gap[-4:] = 0.0
    forced = (gap > 1e-9).astype(int)
    ax3.plot(np.arange(T), gap, color=BLUE, lw=2, label="residual gap $D$")
    ax3.fill_between(np.arange(T), 0, gap, where=gap > 0, color=ORANGE, alpha=0.15)
    ax3.scatter(np.where(forced)[0], gap[forced == 1], s=16, c=RED,
                label="act forced")
    ax3.axhline(0, color=GREEN, lw=1.4, ls="--", label="closes ($D=0$)")
    ax3.set_xlabel("exchange turn"); ax3.set_ylabel("residual gap $D$")
    ax3.legend(fontsize=7, frameon=False)

    # (D) 3D: drift surface over (chain length, mean gain)
    NN, GG = np.meshgrid(np.arange(1, 15), np.linspace(0.4, 0.98, 30))
    FF = 1 - GG ** NN
    ax4.plot_surface(NN, GG, FF, cmap=cm.viridis, edgecolor="none", alpha=0.95)
    ax4.set_xlabel("relays $n$"); ax4.set_ylabel("gain $g$")
    ax4.set_zlabel("drift", labelpad=2); ax4.view_init(elev=26, azim=-60)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_4.png")


def main():
    panel1(); panel2(); panel3(); panel4()
    print("[propagation] wrote panel_1..4.png to", FIGDIR)


if __name__ == "__main__":
    main()
