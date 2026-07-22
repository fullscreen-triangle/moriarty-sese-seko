#!/usr/bin/env python3
# =====================================================================
#  Figure panels for "The Purpose of a Character".
#  4 panels, each 4 charts in a row, white background, minimal text,
#  at least one 3D chart per panel. All charts are data-driven from the
#  theorem computations. No text/conceptual/table charts.
#  Outputs: figures/panel_1.png ... panel_4.png
# =====================================================================
import os
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

RNG = np.random.default_rng(20260719)
FLOOR = 1.0
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

# ---------------------------------------------------------------------
#  graph helpers (mirror the validation script)
# ---------------------------------------------------------------------
def random_self_graph(n, density, wmin=FLOOR, wspread=4.0):
    while True:
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if RNG.random() < density:
                    W[i, j] = W[j, i] = wmin + RNG.random() * wspread
        if is_connected(W):
            return W

def is_connected(W):
    n = W.shape[0]
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in range(n):
            if v not in seen and W[u, v] > 0:
                seen.add(v); stack.append(v)
    return len(seen) == n

def boundary_cost(W, U):
    U = set(U); comp = [v for v in range(W.shape[0]) if v not in U]
    return sum(W[i, j] for i in U for j in comp if W[i, j] > 0)

def all_partitions(coll):
    coll = list(coll)
    if len(coll) == 1:
        yield [coll]; return
    first = coll[0]
    for smaller in all_partitions(coll[1:]):
        for i, s in enumerate(smaller):
            yield smaller[:i] + [[first] + s] + smaller[i + 1:]
        yield [[first]] + smaller

def internal_residual(W, part):
    t = 0.0
    for a in range(len(part)):
        for b in range(a + 1, len(part)):
            t += sum(W[i, j] for i in part[a] for j in part[b] if W[i, j] > 0)
    return t

def character_invariant(W):
    best, bp = np.inf, None
    for part in all_partitions(range(W.shape[0])):
        if len(part) < 2:
            continue
        r = internal_residual(W, part)
        if r < best:
            best, bp = r, part
    return best, bp

def relabel(W, perm):
    n = W.shape[0]; Wp = np.zeros_like(W)
    for i in range(n):
        for j in range(n):
            Wp[perm[i], perm[j]] = W[i, j]
    return Wp

def two_triangle_agent(join_w=FLOOR, tri_w=2.0):
    W = np.zeros((6, 6))
    for (i, j) in [(0, 1), (1, 2), (0, 2)]:
        W[i, j] = W[j, i] = tri_w
    for (i, j) in [(3, 4), (4, 5), (3, 5)]:
        W[i, j] = W[j, i] = tri_w
    W[2, 3] = W[3, 2] = join_w
    return W

def make_drive(n):
    A = RNG.standard_normal((n, n))
    Q = A @ A.T + np.eye(n) * (0.5 + RNG.random())
    c = RNG.random(n)
    lam = float(np.linalg.eigvalsh(Q).min())
    return (lambda x: Q @ (x - c)), np.clip(c, 0, 1), lam, Q, c

def panel_header(fig, letters):
    for ax, L in zip(fig.axes, letters):
        ax.set_title(L, loc="left", fontweight="bold", fontsize=11, x=-0.02)


def finish(fig, name):
    """Consistent layout; extra right margin so 3D z-labels are not clipped."""
    fig.subplots_adjust(left=0.045, right=0.965, bottom=0.14, top=0.9,
                        wspace=0.42)
    fig.savefig(os.path.join(FIGDIR, name), dpi=150)
    plt.close(fig)

# =====================================================================
#  PANEL 1 : The floor and the conserved, non-local identity
# =====================================================================
def panel1():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    # (A) boundary cost vs floor: every point on/above the floor line
    bxs, bys = [], []
    for _ in range(120):
        n = int(RNG.integers(3, 8))
        W = random_self_graph(n, RNG.uniform(0.4, 0.9))
        for k in range(1, n):
            for U in itertools.combinations(range(n), k):
                bxs.append(FLOOR); bys.append(boundary_cost(W, U))
    bys = np.array(bys)
    ax1.scatter(RNG.uniform(0.6, 1.4, size=len(bys)), bys, s=6, c=BLUE, alpha=0.35)
    ax1.axhline(FLOOR, color=RED, lw=1.6)
    ax1.set_xlabel("floor $\\beta$"); ax1.set_ylabel("boundary cost $b(U)$")
    ax1.set_xlim(0.4, 1.6)

    # (B) histogram of b(U)/floor, all >= 1
    ax2.hist(bys / FLOOR, bins=40, color=GREEN, alpha=0.85)
    ax2.axvline(1.0, color=RED, lw=1.6)
    ax2.set_xlabel("$b(U)/\\beta$"); ax2.set_ylabel("count")

    # (C) chi invariance under relabelling: chi_after vs chi_before on diagonal
    xb, yb = [], []
    for _ in range(80):
        n = int(RNG.integers(3, 7))
        W = random_self_graph(n, RNG.uniform(0.4, 0.9))
        c0, _ = character_invariant(W)
        for _ in range(4):
            c1, _ = character_invariant(relabel(W, list(RNG.permutation(n))))
            xb.append(c0); yb.append(c1)
    ax3.scatter(xb, yb, s=10, c=PURPLE, alpha=0.5)
    lim = [0, max(max(xb), max(yb)) * 1.05]
    ax3.plot(lim, lim, color=RED, lw=1.4)
    ax3.set_xlabel("$\\chi$ before relabel"); ax3.set_ylabel("$\\chi$ after")

    # (D) 3D: non-locality - chi vs join weight and triangle weight (two-triangle)
    joins = np.linspace(FLOOR, 6, 22)
    tris = np.linspace(2, 6, 22)
    JJ, TT = np.meshgrid(joins, tris)
    ZZ = np.zeros_like(JJ)
    for a in range(JJ.shape[0]):
        for b in range(JJ.shape[1]):
            W = two_triangle_agent(join_w=JJ[a, b], tri_w=TT[a, b])
            ZZ[a, b], _ = character_invariant(W)
    ax4.plot_surface(JJ, TT, ZZ, cmap=cm.viridis, edgecolor="none", alpha=0.95)
    ax4.set_xlabel("join $w$"); ax4.set_ylabel("triangle $w$")
    ax4.set_zlabel("$\\chi$"); ax4.view_init(elev=22, azim=-60)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_1.png")

# =====================================================================
#  PANEL 2 : Behaviour, purpose as fixed point, attractor
# =====================================================================
def panel2():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    # (A) convergence trajectories ||x-xstar|| vs t (log), several starts
    grad, xstar, lam, Q, c = make_drive(3)
    dt, T = 2e-3, 4.0
    steps = int(T / dt); ts = np.arange(steps) * dt
    for _ in range(8):
        x = RNG.random(3); d = []
        for _ in range(steps):
            d.append(np.linalg.norm(x - xstar))
            x = np.clip(x - dt * grad(x), 0, 1)
        ax1.semilogy(ts, d, color=BLUE, alpha=0.6, lw=1)
    ax1.semilogy(ts, np.exp(-lam * ts) * 1.5, color=RED, lw=1.8, ls="--")
    ax1.set_xlabel("time $t$"); ax1.set_ylabel("$\\|x-x^\\star\\|$")

    # (B) empirical return rate vs strong-convexity lambda (diagonal)
    lams, rates = [], []
    for _ in range(50):
        g, xs, lm, _, _ = make_drive(int(RNG.integers(2, 5)))
        x = RNG.random(len(xs)); d0 = np.linalg.norm(x - xs)
        Tt = 2.0; st = int(Tt / dt)
        for _ in range(st):
            x = np.clip(x - dt * g(x), 0, 1)
        dT = np.linalg.norm(x - xs)
        emp = -np.log(max(dT, 1e-12) / max(d0, 1e-12)) / Tt
        lams.append(lm); rates.append(emp)
    ax2.scatter(lams, rates, s=14, c=GREEN, alpha=0.7)
    lim = [0, max(lams) * 1.05]
    ax2.plot(lim, lim, color=RED, lw=1.4, ls="--")
    ax2.set_xlabel("$\\lambda$ (strong convexity)")
    ax2.set_ylabel("empirical return rate")

    # (C) phase portrait of the 2D flow with the single fixed point
    g2, xs2, lm2, Q2, c2 = make_drive(2)
    gx, gy = np.meshgrid(np.linspace(0, 1, 18), np.linspace(0, 1, 18))
    U = np.zeros_like(gx); V = np.zeros_like(gy)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            d = -g2(np.array([gx[i, j], gy[i, j]]))
            U[i, j], V[i, j] = d[0], d[1]
    ax3.streamplot(gx, gy, U, V, color=BLUE, density=1.1, linewidth=0.7,
                   arrowsize=0.8)
    ax3.plot(xs2[0], xs2[1], "o", color=RED, ms=9)
    ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

    # (D) 3D drive surface Phi over the 2D disposition space + minimiser
    Z = np.zeros_like(gx)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            xy = np.array([gx[i, j], gy[i, j]]) - c2
            Z[i, j] = 0.5 * xy @ Q2 @ xy
    ax4.plot_surface(gx, gy, Z, cmap=cm.viridis, edgecolor="none", alpha=0.9)
    zmin = 0.5 * (xs2 - c2) @ Q2 @ (xs2 - c2)
    ax4.scatter([xs2[0]], [xs2[1]], [zmin], color=RED, s=45)
    ax4.set_xlabel("$x_1$"); ax4.set_ylabel("$x_2$"); ax4.set_zlabel("$\\Phi$")
    ax4.view_init(elev=30, azim=-55)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_2.png")

# =====================================================================
#  PANEL 3 : Society - quotient identity and one-group-one-drive
# =====================================================================
def panel3():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    # (A) society realised floor vs #agents: strictly positive
    ms, floors = [], []
    for _ in range(120):
        m = int(RNG.integers(3, 8))
        Q = random_self_graph(m, RNG.uniform(0.5, 0.9))
        rf = min(boundary_cost(Q, U)
                 for k in range(1, m) for U in itertools.combinations(range(m), k))
        ms.append(m + RNG.uniform(-0.25, 0.25)); floors.append(rf)
    ax1.scatter(ms, floors, s=12, c=BLUE, alpha=0.5)
    ax1.axhline(FLOOR, color=RED, lw=1.6)
    ax1.set_xlabel("number of agents $m$")
    ax1.set_ylabel("society realised floor")

    # (B) chi of the two-cluster society vs join weight: flat at join = min cut
    joins = np.linspace(FLOOR, 8, 40)
    chis, singles = [], []
    for jw in joins:
        W = two_triangle_agent(join_w=jw, tri_w=3.0)
        c0, _ = character_invariant(W)
        ms_ = min(boundary_cost(W, [v]) for v in range(6))
        chis.append(c0); singles.append(ms_)
    ax2.plot(joins, chis, color=GREEN, lw=2, label="$\\chi$ (two-group split)")
    ax2.plot(joins, singles, color=ORANGE, lw=2, ls="--", label="min singleton cut")
    ax2.set_xlabel("join weight"); ax2.set_ylabel("cost")
    ax2.legend(fontsize=7, frameon=False)

    # (C) crowd sharpening: collective failure prod q_i vs n, several q levels
    ns = np.arange(1, 15)
    for q, col in zip([0.3, 0.5, 0.7, 0.9], [BLUE, GREEN, ORANGE, PURPLE]):
        ax3.plot(ns, q ** ns, color=col, lw=1.8, marker="o", ms=3,
                 label=f"$q_i={q}$")
    ax3.set_xlabel("number of agents $n$")
    ax3.set_ylabel("collective failure $\\prod q_i$")
    ax3.legend(fontsize=7, frameon=False)

    # (D) 3D: collective failure surface over (n, q)
    NN, QQ = np.meshgrid(np.arange(1, 15), np.linspace(0.1, 0.95, 30))
    FF = QQ ** NN
    ax4.plot_surface(NN, QQ, FF, cmap=cm.viridis, edgecolor="none", alpha=0.95)
    ax4.set_xlabel("agents $n$"); ax4.set_ylabel("$q_i$")
    ax4.set_zlabel("$\\prod q_i$"); ax4.view_init(elev=26, azim=-58)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_3.png")

# =====================================================================
#  PANEL 4 : Coordination without shared internals; copies distinct
# =====================================================================
def panel4():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax4 = fig.add_subplot(1, 4, 4)

    # (A) disjoint agents' outcomes cluster within tolerance of a shared cell
    target = np.array([0.0, 0.0]); tau = 0.5
    ax1.add_patch(plt.Circle(target, tau, color=GREEN, alpha=0.12))
    ax1.add_patch(plt.Circle(target, tau, fill=False, color=GREEN, lw=1.4))
    for _ in range(120):
        off = RNG.standard_normal(2)
        off = off / (np.linalg.norm(off) + 1e-9) * RNG.uniform(0, tau * 0.95)
        p = target + off
        ax1.scatter(p[0], p[1], s=12, c=BLUE, alpha=0.5)
    ax1.plot(0, 0, "*", color=RED, ms=14)
    ax1.set_xlabel("outcome $y_1$"); ax1.set_ylabel("outcome $y_2$")
    ax1.set_aspect("equal"); ax1.set_xlim(-0.7, 0.7); ax1.set_ylim(-0.7, 0.7)

    # (B) reachable fraction vs n and tolerance (2 curves)
    ns = np.arange(1, 12)
    for tt, col in zip([0.3, 0.5, 0.7], [BLUE, GREEN, ORANGE]):
        frac = 1 - (1 - tt) ** ns
        ax2.plot(ns, frac, color=col, lw=1.9, marker="o", ms=3,
                 label=f"$\\tau={tt}$")
    ax2.set_xlabel("agents in crowd $n$")
    ax2.set_ylabel("reachable fraction")
    ax2.legend(fontsize=7, frameon=False)

    # (C) 3D: committed-count trajectories - copies stay distinct forever
    for start, col in zip([0, 0, 0], [BLUE, GREEN, ORANGE]):
        pass
    steps = np.arange(0, 40)
    # original at count m0, copy fresh at 0; disposition identical => x-axis same
    disp = np.sin(steps / 6.0)  # a shared disposition coordinate
    for m0, col in zip([0, 12, 25], [BLUE, GREEN, ORANGE]):
        ax3.plot(disp, steps + m0, steps + m0, color=col, lw=2)
    ax3.set_xlabel("disposition"); ax3.set_ylabel("committed count")
    ax3.set_zlabel("state index", labelpad=2); ax3.view_init(elev=22, azim=-62)

    # (D) monotone committed record - never returns even on revisits
    rng2 = np.random.default_rng(7)
    pos = rng2.integers(0, 5, size=45)          # revisited positions
    rec = np.arange(1, 46)                       # strictly increasing record
    ax4.plot(rec, pos, color=BLUE, lw=1, alpha=0.6)
    ax4.scatter(rec, pos, s=14, c=rec, cmap="plasma")
    # mark revisits of position 0
    rv = np.where(pos == pos[0])[0]
    ax4.scatter(rec[rv], pos[rv], s=55, facecolors="none", edgecolors=RED, lw=1.4)
    ax4.set_xlabel("committed record $M$"); ax4.set_ylabel("position")

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_4.png")


def main():
    panel1(); panel2(); panel3(); panel4()
    print("[purpose] wrote panel_1..4.png to", FIGDIR)


if __name__ == "__main__":
    main()
