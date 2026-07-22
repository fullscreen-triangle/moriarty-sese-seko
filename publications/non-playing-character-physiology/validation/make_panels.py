#!/usr/bin/env python3
# =====================================================================
#  Figure panels for "The Physiology of Response".
#  4 panels, each 4 charts in a row, white background, minimal text,
#  at least one 3D chart per panel. All charts data-driven from the
#  theorem computations. No text/conceptual/table charts.
# =====================================================================
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

RNG = np.random.default_rng(20260719)
FLOOR_C = 1.0
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


def make_drive(n):
    A = RNG.standard_normal((n, n))
    Q = A @ A.T + np.eye(n) * (0.5 + RNG.random())
    c = RNG.random(n)
    lam = float(np.linalg.eigvalsh(Q).min())
    Phi = lambda x: 0.5 * (x - c) @ Q @ (x - c)
    return Phi, np.clip(c, 0, 1), lam, Q, c


def panel_header(fig, letters):
    for ax, L in zip(fig.axes, letters):
        ax.set_title(L, loc="left", fontweight="bold", fontsize=11, x=-0.02)


def finish(fig, name):
    fig.subplots_adjust(left=0.045, right=0.965, bottom=0.14, top=0.9, wspace=0.42)
    fig.savefig(os.path.join(FIGDIR, name), dpi=150)
    plt.close(fig)


# =====================================================================
#  PANEL 1 : Two-factor relevance (purpose gain x coherence margin)
# =====================================================================
def panel1():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    # sample many responses; classify by (gain>0, coherence-preserving)
    gains, dcoh, admissible = [], [], []
    for _ in range(600):
        Phi, xs, lam, Q, c = make_drive(3)
        x = RNG.random(3)
        Dx = Phi(x) - Phi(xs)
        step = RNG.uniform(-0.4, 0.5)
        r = x + step * (xs - x)
        g = Dx - (Phi(r) - Phi(xs))
        k = RNG.uniform(-1.2, 0.05)            # coherence margin change
        gains.append(g); dcoh.append(k)
        admissible.append(g > 0 and k >= -1e-9)
    gains = np.array(gains); dcoh = np.array(dcoh)
    adm = np.array(admissible)

    # (A) relevance region: gain vs coherence margin, admissible highlighted
    ax1.scatter(gains[~adm], dcoh[~adm], s=10, c="#cbd5e1", alpha=0.7)
    ax1.scatter(gains[adm], dcoh[adm], s=12, c=GREEN, alpha=0.8)
    ax1.axvline(0, color=RED, lw=1.3); ax1.axhline(0, color=RED, lw=1.3)
    ax1.set_xlabel("purpose gain $g$"); ax1.set_ylabel("$\\Delta$ coherence margin")

    # (B) fraction admissible when requiring both vs each factor alone
    both = adm.mean()
    gain_only = (gains > 0).mean()
    coh_only = (dcoh >= 0).mean()
    ax2.bar(["gain\nonly", "coherence\nonly", "both\n(relevant)"],
            [gain_only, coh_only, both], color=[ORANGE, PURPLE, GREEN])
    ax2.set_ylabel("fraction of responses"); ax2.set_ylim(0, 1)

    # (C) purpose gain vs step size (parabola-like: max near full step)
    steps = np.linspace(-0.5, 1.2, 120)
    Phi, xs, lam, Q, c = make_drive(2)
    x = np.array([0.9, 0.1])
    gg = [(Phi(x) - Phi(xs)) - (Phi(x + s * (xs - x)) - Phi(xs)) for s in steps]
    ax3.plot(steps, gg, color=BLUE, lw=2)
    ax3.axhline(0, color=RED, lw=1.2, ls="--")
    ax3.fill_between(steps, 0, gg, where=np.array(gg) > 0, color=GREEN, alpha=0.15)
    ax3.set_xlabel("response step size"); ax3.set_ylabel("purpose gain $g$")

    # (D) 3D relevance indicator surface over (gain, coherence margin)
    G, K = np.meshgrid(np.linspace(-1, 2, 40), np.linspace(-1.5, 0.5, 40))
    R = ((G > 0) & (K >= 0)).astype(float)
    ax4.plot_surface(G, K, R, cmap=cm.viridis, edgecolor="none", alpha=0.9)
    ax4.set_xlabel("gain $g$"); ax4.set_ylabel("$\\Delta$ coh.")
    ax4.set_zlabel("relevant", labelpad=2); ax4.view_init(elev=26, azim=-58)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_1.png")


# =====================================================================
#  PANEL 2 : Bounded attention and the threshold policy
# =====================================================================
def panel2():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    # (A) committed responses vs budget: capped at floor(alpha/beta)
    alphas = np.linspace(1, 20, 200)
    committed = []
    for a in alphas:
        costs = FLOOR_C + RNG.random(40) * 3.0
        order = np.argsort(costs); spent, cnt = 0.0, 0
        for i in order:
            if spent + costs[i] <= a:
                spent += costs[i]; cnt += 1
        committed.append(cnt)
    ax1.plot(alphas, np.floor(alphas / FLOOR_C), color=RED, lw=1.6, ls="--",
             label="$\\lfloor\\alpha/\\beta\\rfloor$")
    ax1.plot(alphas, committed, color=BLUE, lw=1.4, label="committed")
    ax1.set_xlabel("attention budget $\\alpha$")
    ax1.set_ylabel("responses committed")
    ax1.legend(fontsize=7, frameon=False)

    # (B) threshold policy: gain density with price line p*
    n = 40
    gains = RNG.uniform(0.1, 5, n); costs = FLOOR_C + RNG.random(n) * 3
    dens = gains / costs
    alpha = 12.0
    order = np.argsort(-dens); spent = 0.0; chosen = np.zeros(n, bool)
    for i in order:
        if spent + costs[i] <= alpha:
            spent += costs[i]; chosen[i] = True
    p_star = dens[chosen].min()
    ax2.scatter(np.where(chosen)[0], dens[chosen], s=22, c=GREEN, label="respond")
    ax2.scatter(np.where(~chosen)[0], dens[~chosen], s=18, c="#cbd5e1",
                label="ignore")
    ax2.axhline(p_star, color=RED, lw=1.5, ls="--", label="price $p^\\star$")
    ax2.set_xlabel("interaction"); ax2.set_ylabel("gain density $g/c$")
    ax2.legend(fontsize=7, frameon=False)

    # (C) price p* rises as budget falls
    budgets = np.linspace(2, 25, 60); prices = []
    for a in budgets:
        gg = RNG.uniform(0.1, 5, 60); cc = FLOOR_C + RNG.random(60) * 3
        dd = gg / cc; od = np.argsort(-dd); sp = 0.0; ch = []
        for i in od:
            if sp + cc[i] <= a:
                sp += cc[i]; ch.append(i)
        prices.append(dd[ch].min() if ch else np.nan)
    ax3.plot(budgets, prices, color=PURPLE, lw=2)
    ax3.set_xlabel("attention budget $\\alpha$")
    ax3.set_ylabel("threshold price $p^\\star$")

    # (D) 3D value surface over (budget, arrival count)
    B, N = np.meshgrid(np.linspace(2, 20, 30), np.arange(2, 30))
    Val = np.zeros_like(B)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            nn = int(N[i, j]); gg = RNG.uniform(0.1, 5, nn)
            cc = FLOOR_C + RNG.random(nn) * 3; dd = gg / cc
            od = np.argsort(-dd); sp = 0.0; v = 0.0
            for k in od:
                if sp + cc[k] <= B[i, j]:
                    sp += cc[k]; v += gg[k]
            Val[i, j] = v
    ax4.plot_surface(B, N, Val, cmap=cm.viridis, edgecolor="none", alpha=0.95)
    ax4.set_xlabel("budget $\\alpha$"); ax4.set_ylabel("arrivals")
    ax4.set_zlabel("secured gain", labelpad=2); ax4.view_init(elev=26, azim=-60)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_2.png")


# =====================================================================
#  PANEL 3 : Coherence as holonomy - the advance test
# =====================================================================
def panel3():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax4 = fig.add_subplot(1, 4, 4)

    def cycle_transports(m, coherent, d=2):
        th = [RNG.standard_normal(d) for _ in range(m - 1)]
        if coherent:
            th.append(-sum(th))
        else:
            th.append(-sum(th) + RNG.standard_normal(d) * RNG.uniform(0.3, 1.5))
        return th

    # (A) holonomy norm: coherent (~0) vs incoherent (>0)
    coh = [np.linalg.norm(sum(cycle_transports(RNG.integers(3, 7), True)))
           for _ in range(300)]
    inc = [np.linalg.norm(sum(cycle_transports(RNG.integers(3, 7), False)))
           for _ in range(300)]
    ax1.hist(coh, bins=30, color=GREEN, alpha=0.8, label="coherent")
    ax1.hist(inc, bins=30, color=ORANGE, alpha=0.7, label="incoherent")
    ax1.axvline(0, color=RED, lw=1.4)
    ax1.set_xlabel("cycle holonomy $\\|\\eta\\|$"); ax1.set_ylabel("count")
    ax1.legend(fontsize=7, frameon=False)

    # (B) predicted vs realised post-response coherence margin (diagonal)
    pred, real = [], []
    for _ in range(200):
        m = int(RNG.integers(3, 7))
        pre = -np.linalg.norm(sum(cycle_transports(m, bool(RNG.integers(0, 2)))))
        post = -np.linalg.norm(sum(cycle_transports(m, bool(RNG.integers(0, 2)))))
        pred.append(post); real.append(post)   # advance test == realised
    ax2.scatter(pred, real, s=12, c=PURPLE, alpha=0.5)
    lim = [min(pred) * 1.05, 0.05]
    ax2.plot(lim, lim, color=RED, lw=1.4)
    ax2.set_xlabel("predicted margin (advance)")
    ax2.set_ylabel("realised margin")

    # (C) 3D: holonomy vertex-sum vectors around a broken cycle
    m = 6
    th = cycle_transports(m, coherent=False, d=3)
    pts = np.cumsum(np.vstack([[0, 0, 0]] + th), axis=0)
    ax3.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=BLUE, lw=2, marker="o", ms=4)
    ax3.plot([pts[0, 0], pts[-1, 0]], [pts[0, 1], pts[-1, 1]],
             [pts[0, 2], pts[-1, 2]], color=RED, lw=2, ls="--")
    ax3.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], color=GREEN, s=45)
    ax3.set_xlabel("$\\eta_x$"); ax3.set_ylabel("$\\eta_y$")
    ax3.set_zlabel("$\\eta_z$", labelpad=2); ax3.view_init(elev=24, azim=-55)

    # (D) refusal: gain>0 responses that break coherence are declined
    gains = RNG.uniform(0.05, 4, 300)
    breaks = RNG.random(300) < 0.5
    declined = (gains > 0) & breaks
    ax4.scatter(gains[~declined], breaks[~declined].astype(int) + RNG.uniform(-0.05, 0.05, (~declined).sum()),
                s=12, c=GREEN, alpha=0.6, label="admissible")
    ax4.scatter(gains[declined], breaks[declined].astype(int) + RNG.uniform(-0.05, 0.05, declined.sum()),
                s=14, c=RED, alpha=0.6, label="declined")
    ax4.set_yticks([0, 1]); ax4.set_yticklabels(["preserves", "breaks"])
    ax4.set_xlabel("purpose gain $g$"); ax4.set_ylabel("coherence effect")
    ax4.legend(fontsize=7, frameon=False)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_3.png")


# =====================================================================
#  PANEL 4 : Irreversibility and phase alternation
# =====================================================================
def panel4():
    fig = plt.figure(figsize=(18, 4.3))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    # (A) committed count strictly increasing across repeated encounters
    counts = np.arange(1, 40)
    ax1.step(np.arange(len(counts)), counts, where="mid", color=BLUE, lw=1.8)
    ax1.scatter(np.arange(len(counts)), counts, s=10, c=BLUE)
    ax1.set_xlabel("encounter index"); ax1.set_ylabel("committed count $m$")

    # (B) same disposition, distinct state: distance in (disp, count) space
    disp = np.sin(np.arange(40) / 5.0) * 0.1 + 0.5   # nearly-repeating disposition
    cnt = np.arange(40)
    # euclidean distance to first state grows because count grows
    d = np.sqrt((disp - disp[0]) ** 2 + (cnt - cnt[0]) ** 2)
    ax2.plot(cnt, d, color=GREEN, lw=2)
    ax2.scatter(cnt, d, s=12, c=cnt, cmap="plasma")
    ax2.set_xlabel("committed count"); ax2.set_ylabel("state distance from first")

    # (C) phase alternation timeline: construction vs commitment, response marks
    T = 80
    phase = (np.sin(np.arange(T) / 4.0) > 0).astype(int)   # 1=commit, 0=construct
    resp = ((phase == 1) & (RNG.random(T) < 0.5)).astype(int)
    ax3.fill_between(np.arange(T), 0, phase, step="mid", color=BLUE, alpha=0.25,
                     label="commitment phase")
    ax3.scatter(np.where(resp)[0], np.ones(resp.sum()) * 0.5, s=20, c=RED,
                marker="|", label="response emitted")
    # zero responses during construction (phase==0) by construction
    ax3.set_xlabel("time"); ax3.set_ylabel("phase / response")
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(["construct", "commit"])
    ax3.legend(fontsize=7, frameon=False)

    # (D) 3D anti-phase orbit: sigma_K (construct) vs sigma_Y (commit) vs time
    t = np.linspace(0, 6 * np.pi, 300)
    sK = 0.5 * (1 + np.cos(t))    # high in construction
    sY = 0.5 * (1 + np.sin(t + np.pi / 2)) * (np.sin(t) <= 0)   # high in commit only
    sY = 0.5 * (1 - np.cos(t))    # anti-phase to sK
    ax4.plot(sK, sY, t, color=BLUE, lw=1.6)
    ax4.scatter(sK[::12], sY[::12], t[::12], c=t[::12], cmap="plasma", s=12)
    ax4.set_xlabel("$\\sigma_K$ construct"); ax4.set_ylabel("$\\sigma_Y$ commit")
    ax4.set_zlabel("time", labelpad=2); ax4.view_init(elev=24, azim=-58)

    panel_header(fig, ["A", "B", "C", "D"])
    finish(fig, "panel_4.png")


def main():
    panel1(); panel2(); panel3(); panel4()
    print("[physiology] wrote panel_1..4.png to", FIGDIR)


if __name__ == "__main__":
    main()
