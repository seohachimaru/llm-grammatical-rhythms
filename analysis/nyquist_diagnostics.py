"""Diagnostic figures of component pooling at 0.5 unit^-1 (Nyquist) and 0.25 unit^-1.

Reads the cached activations; writes intro-meeting-*.png.

Run:
    uv run --with numpy --with matplotlib -- python analysis/nyquist_diagnostics.py \
        --activations <OUTPUT_DIR>/activations
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = pathlib.Path(__file__).resolve().parent / "figures"
MODELS = ["gpt2-zh", "rwkv7"]
LABEL = {"gpt2-zh": "GPT-2 (Transformer)", "rwkv7": "RWKV-7 (RNN)"}
CORPORA = ["random", "phrase", "phrase+sentence", "reversed", "english"]

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 160})


def bins(P):
    """Target DFT bins: sentence at P/4, phrase at P/2 (Nyquist)."""
    return {"0.25": P // 4, "0.5": P // 2}


def zscore(x, P, which):
    """z of the target bin against all other bins, excluding DC and the other target."""
    ks = bins(P)
    k = ks[which]
    X = np.abs(np.fft.rfft(x - x.mean()))
    mask = np.ones(len(X), bool)
    mask[0] = False
    for name, other in ks.items():
        if name != which:
            mask[other] = False
    mask[k] = False
    return (X[k] - X[mask].mean()) / X[mask].std()


def load(act, model, corpus):
    return np.load(act / model / f"{corpus}.npy")


def standardise(A):
    """Per-component |H| standardised over positions; drops constant components."""
    L, d, P = A.shape
    H = np.abs(A).reshape(L * d, P)
    mu = H.mean(axis=1, keepdims=True)
    sd = H.std(axis=1, keepdims=True)
    keep = sd[:, 0] > 1e-8
    return (H[keep] - mu[keep]) / sd[keep], sd[keep][:, 0]


def panel_single_vs_aggregate(ax, act):
    """Panel A: z at 0.5 of the aggregate vs the component with the largest mean |H|."""
    labels, agg, best = [], [], []
    for m in MODELS:
        A = load(act, m, "phrase+sentence")
        L, d, P = A.shape
        agg.append(zscore(np.sqrt((A ** 2).sum(axis=(0, 1))), P, "0.5"))
        mag = np.abs(A).mean(axis=2)
        l, i = np.unravel_index(np.argmax(mag), mag.shape)
        best.append(zscore(np.abs(A[l, i, :]), P, "0.5"))
        labels.append(f"{LABEL[m]}\nbest = (L{l}, d{i})")
    x = np.arange(len(MODELS))
    ax.bar(x - 0.19, agg, 0.38, label="all components pooled", color="#4C72B0")
    ax.bar(x + 0.19, best, 0.38, label="single largest component", color="#DD8452")
    for xi, (a, b) in enumerate(zip(agg, best)):
        ax.text(xi - 0.19, a + 1.5, f"{a:.1f}", ha="center", fontsize=7)
        ax.text(xi + 0.19, b + 1.5, f"{b:.1f}", ha="center", fontsize=7)
    ax.axhline(3, color="crimson", ls="--", lw=0.9)
    ax.text(1.45, 4.5, "z = 3", color="crimson", fontsize=7, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("z at the phrase rate (0.5 unit$^{-1}$)")
    ax.set_title("A  One component vs the whole aggregate", loc="left", fontweight="bold")
    ax.legend(fontsize=7, frameon=False)


def panel_weighting(ax, act):
    """Panel B: z at both rates under three pooling schemes."""
    schemes = ["raw Frobenius\n(paper)", "standardised\n+ Frobenius", "standardised\n+ mean"]
    colours = ["#4C72B0", "#55A868", "#C44E52"]
    groups, vals = [], {s: [] for s in schemes}
    for m in MODELS:
        A = load(act, m, "phrase+sentence")
        L, d, P = A.shape
        S, _ = standardise(A)
        series = [np.sqrt((A ** 2).sum(axis=(0, 1))),
                  np.sqrt((S ** 2).sum(axis=0)),
                  S.mean(axis=0)]
        for rate in ["0.25", "0.5"]:
            nice = "sentence\n0.25" if rate == "0.25" else "phrase\n0.5"
            groups.append(f"{LABEL[m].split(' ')[0]}\n{nice}")
            for s, ser in zip(schemes, series):
                vals[s].append(zscore(ser, P, rate))
    x = np.arange(len(groups))
    for j, (s, c) in enumerate(zip(schemes, colours)):
        off = (j - 1) * 0.27
        ax.bar(x + off, vals[s], 0.27, label=s, color=c)
        for xi, v in zip(x + off, vals[s]):
            ax.text(xi, v + (1.2 if v >= 0 else -3.0), f"{v:.1f}",
                    ha="center", fontsize=6)
    ax.axhline(3, color="crimson", ls="--", lw=0.9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=6.5)
    ax.set_ylabel("z")
    ax.set_title("B  z under three pooling schemes", loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False, ncol=3)


def panel_coherence(ax, act):
    """Panel C: phase concentration across components, equal vs magnitude weighted."""
    groups, eq, mg = [], [], []
    for m in MODELS:
        A = load(act, m, "phrase+sentence")
        P = A.shape[2]
        S, sd = standardise(A)
        G = np.fft.rfft(S, axis=1)
        for rate in ["0.25", "0.5"]:
            g = G[:, bins(P)[rate]]
            eq.append(np.abs(g.sum()) / np.abs(g).sum())
            mg.append(np.abs((sd * g).sum()) / (sd * np.abs(g)).sum())
            nice = "sentence\n0.25" if rate == "0.25" else "phrase\n0.5\n(Nyquist)"
            groups.append(f"{LABEL[m].split(' ')[0]}\n{nice}")
    x = np.arange(len(groups))
    ax.bar(x - 0.19, eq, 0.38, label="equal weight", color="#8172B3")
    ax.bar(x + 0.19, mg, 0.38, label="magnitude weight (paper)", color="#937860")
    for xi, (a, b) in enumerate(zip(eq, mg)):
        ax.text(xi - 0.19, a + 0.02, f"{a:.3f}", ha="center", fontsize=6)
        ax.text(xi + 0.19, b + 0.02, f"{b:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=6.5)
    ax.set_ylabel("phase concentration $R$  (1 = all aligned)")
    ax.set_ylim(0, 0.95)
    ax.set_title("C  Phase concentration across components", loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False)


def panel_phase_hist(ax, act):
    """Panel D: amplitude-weighted histogram of component phase at both rates (GPT-2)."""
    A = load(act, "gpt2-zh", "phrase+sentence")
    P = A.shape[2]
    S, _ = standardise(A)
    G = np.fft.rfft(S, axis=1)
    for rate, colour, lab in [("0.25", "#C44E52", "sentence 0.25 (ordinary bin)"),
                              ("0.5", "#4C72B0", "phrase 0.5 (Nyquist)")]:
        g = G[:, bins(P)[rate]]
        ax.hist(np.angle(g), bins=72, weights=np.abs(g), density=True,
                alpha=0.6, color=colour, label=lab)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])
    ax.set_xlabel("component phase at the target bin")
    ax.set_ylabel("amplitude-weighted density")
    ax.set_title("D  GPT-2: component phase at the target bins", loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False)


def panel_selectivity(ax, act):
    """Panel E: z at 0.5 of selected GPT-2 layer-34 dimensions, per corpus."""
    dims = [501, 599, 731, 102, 330, 704]
    M = np.zeros((len(CORPORA), len(dims)))
    for r, c in enumerate(CORPORA):
        A = load(act, "gpt2-zh", c)
        P = A.shape[2]
        for j, dd in enumerate(dims):
            M[r, j] = zscore(np.abs(A[34, dd, :]), P, "0.5")
    im = ax.imshow(M, cmap="RdBu_r", vmin=-45, vmax=45, aspect="auto")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"d{d}" for d in dims], fontsize=7)
    ax.set_yticks(range(len(CORPORA)))
    ax.set_yticklabels(CORPORA, fontsize=7)
    for r in range(len(CORPORA)):
        for j in range(len(dims)):
            ax.text(j, r, f"{M[r, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[r, j]) > 22 else "black")
    ax.grid(False)
    ax.set_title("E  GPT-2 layer 34, selected dimensions: z at 0.5",
                 loc="left", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("z", fontsize=7)


def panel_layer_phase(axes, act):
    """Panels F, G: per-layer phase at 0.25; within-layer, across-layer and global concentration."""
    ax_ph, ax_bar = axes
    store = {}
    for m, colour in zip(MODELS, ["#4C72B0", "#C44E52"]):
        A = load(act, m, "phrase+sentence")
        L, d, P = A.shape
        H = np.abs(A)
        mu = H.mean(axis=2, keepdims=True)
        sd = H.std(axis=2, keepdims=True)
        S = np.where(sd > 1e-8, (H - mu) / np.where(sd > 1e-8, sd, 1), 0.0)
        g = np.fft.rfft(S, axis=2)[:, :, bins(P)["0.25"]]      # (L, d) complex
        per_layer = g.sum(axis=1)
        phase = np.degrees(np.unwrap(np.angle(per_layer)))
        phase -= phase[0]
        ax_ph.plot(np.arange(L) / (L - 1), phase, "o-", ms=2.5, lw=1.2,
                   color=colour, label=f"{LABEL[m]}  ({L} layers)")
        within = np.mean([np.abs(g[l].sum()) / np.abs(g[l]).sum() for l in range(L)])
        across = np.abs(per_layer.sum()) / np.abs(per_layer).sum()
        store[m] = (within, across, np.abs(g.sum()) / np.abs(g).sum())
    ax_ph.axhline(0, color="k", lw=0.6)
    ax_ph.set_xlabel("relative depth  (0 = first layer, 1 = last)")
    ax_ph.set_ylabel("phase at 0.25, unwrapped (deg)")
    ax_ph.set_title("F  Sentence-rate phase by depth", loc="left", fontweight="bold")
    ax_ph.legend(fontsize=6.5, frameon=False)

    x = np.arange(2)
    w = 0.26
    names = ["within layer\n(across dims)", "across layers", "global"]
    for j, (nm, c) in enumerate(zip(names, ["#8172B3", "#55A868", "#937860"])):
        vals = [store[m][j] for m in MODELS]
        ax_bar.bar(x + (j - 1) * w, vals, w, label=nm, color=c)
        for xi, v in zip(x + (j - 1) * w, vals):
            ax_bar.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([LABEL[m].split(" ")[0] for m in MODELS], fontsize=7)
    ax_bar.set_ylabel("phase concentration $R$ at 0.25")
    ax_bar.set_ylim(0, 0.82)
    ax_bar.set_title("G  Phase concentration at 0.25", loc="left", fontweight="bold")
    ax_bar.legend(fontsize=6.5, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activations", type=pathlib.Path, required=True,
                        help="directory holding <model>/<corpus>.npy")
    parser.add_argument("--out", type=pathlib.Path, default=FIG_DIR,
                        help="figure directory (default: analysis/figures)")
    args = parser.parse_args()
    act, out = args.activations, args.out
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    panel_layer_phase(axes, act)
    fig.tight_layout()
    fig.savefig(out / "intro-meeting-layer-phase.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    panel_single_vs_aggregate(axes[0, 0], act)
    panel_weighting(axes[0, 1], act)
    panel_coherence(axes[1, 0], act)
    panel_phase_hist(axes[1, 1], act)
    fig.tight_layout()
    fig.savefig(out / "intro-meeting-aggregate-diagnostics.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    panel_selectivity(ax, act)
    fig.tight_layout()
    fig.savefig(out / "intro-meeting-dimension-selectivity.png", bbox_inches="tight")
    plt.close(fig)

    print("wrote:", out / "intro-meeting-layer-phase.png")
    print("wrote:", out / "intro-meeting-aggregate-diagnostics.png")
    print("wrote:", out / "intro-meeting-dimension-selectivity.png")


if __name__ == "__main__":
    main()
