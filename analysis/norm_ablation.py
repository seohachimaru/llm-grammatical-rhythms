"""Component ablation of the Frobenius-norm signal x(u) = ||H_u||_F.

Ranks the L x d components four ways, removes or keeps the top-k for every k,
and rescores z at 0.25 and 0.5 unit^-1. Writes norm_ablation_results.json and
macroscopic-ablation-*.png.

Run:
    uv run --with numpy --with matplotlib -- python analysis/norm_ablation.py \
        --activations <OUTPUT_DIR>/activations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT_JSON = Path(__file__).resolve().parent / "norm_ablation_results.json"

SENTENCE_FREQ = 0.25
PHRASE_FREQ = 0.50
Z_THRESHOLD = 3.0
SENTENCE_LEN = 4

MODELS = ("gpt2-zh", "rwkv7")
MODEL_LABEL = {"gpt2-zh": "GPT-2", "rwkv7": "RWKV-7"}
ABLATION_CORPORA = ("phrase+sentence", "phrase", "reversed")
ALL_CORPORA = ("random", "phrase", "phrase+sentence", "reversed", "english")

RANKINGS = ("magnitude", "rawpower", "sqpower", "aligned")
RANKING_LABEL = {
    "magnitude": "mean |component|",
    "rawpower": "spectral power (raw)",
    "sqpower": "spectral power (squared)",
    "aligned": "aligned contribution",
}

# Aggregate z-scores (0.25, 0.5) quoted in the manuscript; checked by validate().
PUBLISHED = {
    ("gpt2-zh", "phrase+sentence"): (9.0, 11.8),
    ("gpt2-zh", "phrase"): (None, 11.1),
    ("gpt2-zh", "random"): (-0.3, -1.4),
    ("gpt2-zh", "english"): (1.3, 2.0),
    ("rwkv7", "phrase+sentence"): (14.2, 56.7),
    ("rwkv7", "phrase"): (None, 44.0),
    ("rwkv7", "english"): (8.6, 11.5),
}


# --------------------------------------------------------------------------
# Observable and z-score, as defined in notebook.ipynb.
# --------------------------------------------------------------------------


def global_signal(activations):
    """Frobenius norm over layers and hidden dims: x(u), one value per position."""
    return np.sqrt(np.sum(activations.astype(np.float64) ** 2, axis=(0, 1)))


def spectrum(signal):
    return np.fft.rfftfreq(len(signal), d=1.0), np.abs(np.fft.rfft(signal))


def z_at(freqs, mags, target, exclude=()):
    index = int(np.argmin(np.abs(freqs - target)))
    drop = {0, index, *(int(np.argmin(np.abs(freqs - f))) for f in exclude)}
    noise = np.array([mags[i] for i in range(1, len(mags)) if i not in drop])
    if noise.size == 0 or noise.std() == 0:
        return 0.0
    return float((mags[index] - noise.mean()) / noise.std())


def target_z_scores(signal):
    freqs, mags = spectrum(signal)
    return (
        z_at(freqs, mags, SENTENCE_FREQ, exclude=(PHRASE_FREQ,)),
        z_at(freqs, mags, PHRASE_FREQ, exclude=(SENTENCE_FREQ,)),
    )


# --------------------------------------------------------------------------
# Vectorised z-scoring: same definition, evaluated for many signals at once.
# --------------------------------------------------------------------------


def bin_indices(n_positions, target, exclude=()):
    """(target bin index, boolean mask of the noise bins) for length n."""
    freqs = np.fft.rfftfreq(n_positions, d=1.0)
    index = int(np.argmin(np.abs(freqs - target)))
    drop = {0, index, *(int(np.argmin(np.abs(freqs - f))) for f in exclude)}
    mask = np.ones(len(freqs), dtype=bool)
    mask[list(drop)] = False
    return index, mask


def z_many(signals, target_index, noise_mask):
    """z-score at one bin for signals of shape [n_signals, positions] (ddof=0, as z_at)."""
    mags = np.abs(np.fft.rfft(signals, axis=1))
    peak = mags[:, target_index]
    noise = mags[:, noise_mask]
    sd = noise.std(axis=1)
    z = np.zeros(len(signals))
    ok = sd > 0
    z[ok] = (peak[ok] - noise[ok].mean(axis=1)) / sd[ok]
    return z


# --------------------------------------------------------------------------
# Component rankings
# --------------------------------------------------------------------------


def rank_scores(components, squared, target, ranking):
    """Score every component under one ranking. Higher = ablate first.

    `components` is [C, P] (layer-major flattening of [L, d, P]);
    `squared` is components**2.
    """
    n_positions = components.shape[1]
    if ranking == "magnitude":
        return np.abs(components).mean(axis=1)

    index, _ = bin_indices(n_positions, target)
    if ranking == "rawpower":
        return np.abs(np.fft.rfft(components, axis=1)[:, index])

    per_component = np.fft.rfft(squared, axis=1)[:, index]
    if ranking == "sqpower":
        return np.abs(per_component)
    if ranking == "aligned":
        total = per_component.sum()
        if total == 0:
            return np.abs(per_component)
        return np.real(per_component * np.conj(total)) / np.abs(total)
    raise ValueError(ranking)


# --------------------------------------------------------------------------
# The ablation itself
# --------------------------------------------------------------------------


def ablation_curves(squared, order, target_index, noise_mask, chunk=2048):
    """(z after removing the top-k, z keeping only the top-k), for every k = 1..C."""
    n_components, n_positions = squared.shape
    total = squared.sum(axis=0)
    kept = np.cumsum(squared[order], axis=0)  # [C, P]: sum of the top-k

    z_removed = np.empty(n_components)
    z_retained = np.empty(n_components)
    for start in range(0, n_components, chunk):
        stop = min(start + chunk, n_components)
        block = kept[start:stop]
        z_retained[start:stop] = z_many(
            np.sqrt(block), target_index, noise_mask
        )
        z_removed[start:stop] = z_many(
            np.sqrt(np.maximum(total - block, 0.0)), target_index, noise_mask
        )
    return z_removed, z_retained


def first_at_or_above(curve, threshold=Z_THRESHOLD):
    hits = np.nonzero(curve >= threshold)[0]
    return int(hits[0] + 1) if hits.size else None


def curve_summary(curve, threshold=Z_THRESHOLD):
    """Min, max, final value and threshold crossings of an ablation curve."""
    below = np.nonzero(curve < threshold)[0]
    return {
        "z_min": float(curve.min()),
        "k_at_z_min": int(np.argmin(curve) + 1),
        "z_max": float(curve.max()),
        "k_at_z_max": int(np.argmax(curve) + 1),
        "z_final": float(curve[-1]),
        "first_k_below": None if below.size == 0 else int(below[0] + 1),
        "last_k_below": None if below.size == 0 else int(below[-1] + 1),
        "fraction_k_above_threshold": float(np.mean(curve >= threshold)),
    }


def random_subset_control(squared, target_index, noise_mask, k_grid,
                          n_draws=20, seed=0):
    """Removal / retention z for random subsets of each size in k_grid.

    Returns (removed, retained), each {str(k): [mean, sd]} over n_draws draws.
    """
    rng = np.random.default_rng(seed)
    n_components, n_positions = squared.shape
    total = squared.sum(axis=0)
    removed, retained = {}, {}
    for k in k_grid:
        if k > n_components:
            continue
        kept_stack = np.empty((n_draws, n_positions))
        for draw in range(n_draws):
            picks = rng.choice(n_components, size=k, replace=False)
            kept_stack[draw] = squared[picks].sum(axis=0)
        z_ret = z_many(np.sqrt(kept_stack), target_index, noise_mask)
        z_rem = z_many(np.sqrt(np.maximum(total - kept_stack, 0.0)),
                       target_index, noise_mask)
        retained[str(k)] = [float(z_ret.mean()), float(z_ret.std(ddof=1))]
        removed[str(k)] = [float(z_rem.mean()), float(z_rem.std(ddof=1))]
    return removed, retained


def component_locations(order, n_dims, top=100):
    """Where the top-ranked components live, as (layer, dim) pairs."""
    head = order[:top]
    layers = head // n_dims
    dims = head % n_dims
    layer_counts = np.bincount(layers)
    return {
        "top": int(top),
        "distinct_layers": int(len(np.unique(layers))),
        "distinct_dims": int(len(np.unique(dims))),
        "modal_layer": int(np.argmax(layer_counts)),
        "modal_layer_count": int(layer_counts.max()),
        "layer_histogram": layer_counts.tolist(),
        "first_ten": [[int(l), int(d)] for l, d in zip(layers[:10], dims[:10])],
    }


def squared_signal_z(squared, target, other):
    """z at the target bin of x(u)^2 rather than x(u)."""
    freqs, mags = spectrum(squared.sum(axis=0))
    return z_at(freqs, mags, target, exclude=(other,))


def interference_diagnostic(squared, order, target, ks=(1, 10, 100, 1000, 3000)):
    """Top-k vs remaining components: their DFT coefficients of x(u)^2 at the target bin.

    At the Nyquist bin the coefficients are real, so the phase difference is 0 or 180 deg.
    """
    n_positions = squared.shape[1]
    index, _ = bin_indices(n_positions, target)
    ranked = squared[order]
    total = np.fft.rfft(ranked.sum(axis=0))[index]
    out = {}
    for k in ks:
        if k >= len(ranked):
            continue
        top = np.fft.rfft(ranked[:k].sum(axis=0))[index]
        rest = total - top
        magnitudes = abs(top) + abs(rest)
        out[str(k)] = {
            "abs_top": float(abs(top)),
            "abs_rest": float(abs(rest)),
            "abs_total": float(abs(total)),
            # 1.0 = perfectly in phase, 0.0 = complete cancellation.
            "coherence": float(abs(total) / magnitudes) if magnitudes else 0.0,
            "phase_difference_deg": float(
                np.degrees(np.abs(np.angle(top / rest))) if rest != 0 else 0.0
            ),
        }
    return out


def participation_ratio(squared):
    """Participation ratio of g_c = H[c,u]^2, averaged over positions.

    PR = (sum_c g_c)^2 / sum_c g_c^2.
    """
    numerator = squared.sum(axis=0) ** 2
    denominator = (squared ** 2).sum(axis=0)
    return float(np.mean(numerator / denominator))


def position_domination(activations, signal):
    """x(u) at the first and largest positions; z with the leading positions dropped.

    With one position dropped, 0.25 and 0.5 are not exact DFT bins.
    """
    share = signal ** 2 / (signal ** 2).sum()
    drop_sentence = global_signal(activations[:, :, SENTENCE_LEN:])
    drop_one = global_signal(activations[:, :, 1:])
    return {
        "first_over_median": float(signal[0] / np.median(signal)),
        "max_over_median": float(signal.max() / np.median(signal)),
        "argmax_position": int(np.argmax(signal)),
        "max_power_share": float(share.max()),
        "first_power_share": float(share[0]),
        "z_full": list(target_z_scores(signal)),
        "z_drop_first_sentence": list(target_z_scores(drop_sentence)),
        "z_drop_first_position": list(target_z_scores(drop_one)),
    }


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------


def validate(act_dir):
    """Check the aggregate z-scores against PUBLISHED; exit on mismatch."""
    print("=" * 78)
    print("STEP 1  validation against the published Results")
    print("=" * 78)
    print(f"{'model':10s} {'corpus':18s} {'z(0.25)':>9s} {'pub':>7s} "
          f"{'z(0.50)':>9s} {'pub':>7s}")
    failures = []
    for model in MODELS:
        for corpus in ALL_CORPORA:
            activations = np.load(act_dir / model / f"{corpus}.npy")
            z_sentence, z_phrase = target_z_scores(global_signal(activations))
            published = PUBLISHED.get((model, corpus), (None, None))
            for measured, quoted in zip((z_sentence, z_phrase), published):
                if quoted is not None and abs(round(measured, 1) - quoted) > 0.05:
                    failures.append((model, corpus, measured, quoted))
            fmt = lambda v: f"{v:7.1f}" if v is not None else "      -"  # noqa: E731
            print(f"{model:10s} {corpus:18s} {z_sentence:9.2f} "
                  f"{fmt(published[0])} {z_phrase:9.2f} {fmt(published[1])}")
            del activations
    if failures:
        raise SystemExit(f"validation failed: {failures}")
    print("\nall quoted values reproduce to the printed precision.\n")


def run(act_dir, fig_dir):
    validate(act_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = {"removal": {}, "retention": {}, "meta": {}, "position": {}}
    curves = {}

    print("=" * 78)
    print("STEP 2  component ablation")
    print("=" * 78)

    for model in MODELS:
        for corpus in ABLATION_CORPORA:
            activations = np.load(act_dir / model / f"{corpus}.npy").astype(np.float64)
            n_layers, n_dims, n_positions = activations.shape
            n_components = n_layers * n_dims
            components = activations.reshape(n_components, n_positions)
            squared = components ** 2
            signal = np.sqrt(squared.sum(axis=0))

            key = f"{model}/{corpus}"
            results["meta"][key] = {
                "n_layers": n_layers,
                "n_dims": n_dims,
                "n_components": n_components,
                "n_positions": n_positions,
                "z_baseline": list(target_z_scores(signal)),
                "z_baseline_squared_signal": [
                    squared_signal_z(squared, SENTENCE_FREQ, PHRASE_FREQ),
                    squared_signal_z(squared, PHRASE_FREQ, SENTENCE_FREQ),
                ],
                "participation_ratio": participation_ratio(squared),
                "coefficient_of_variation": float(signal.std() / signal.mean()),
            }
            results["position"][key] = position_domination(activations, signal)
            print(f"\n--- {MODEL_LABEL[model]} / {corpus}  "
                  f"C={n_components} ({n_layers}x{n_dims}), P={n_positions}")
            print(f"    baseline z = {results['meta'][key]['z_baseline'][0]:.2f} "
                  f"(0.25), {results['meta'][key]['z_baseline'][1]:.2f} (0.5); "
                  f"participation ratio = "
                  f"{results['meta'][key]['participation_ratio']:.0f}")

            # Phrase-only corpus: phrasal target only.
            targets = ((PHRASE_FREQ, "0.5"),) if corpus == "phrase" else (
                (SENTENCE_FREQ, "0.25"), (PHRASE_FREQ, "0.5")
            )
            for target, target_label in targets:
                other = PHRASE_FREQ if target == SENTENCE_FREQ else SENTENCE_FREQ
                target_index, noise_mask = bin_indices(
                    n_positions, target, exclude=(other,)
                )
                for ranking in RANKINGS:
                    scores = rank_scores(components, squared, target, ranking)
                    order = np.argsort(-scores, kind="stable")
                    z_removed, z_retained = ablation_curves(
                        squared, order, target_index, noise_mask
                    )
                    cell = f"{key}@{target_label}/{ranking}"
                    curves[cell] = (z_removed, z_retained)

                    k_grid = [k for k in (1, 3, 10, 30, 100, 300, 1000,
                                          3000, 10000, 30000)
                              if k <= n_components]
                    # Cumulative share of sum_{c,u} H[c,u]^2 held by the top-k.
                    ranked_power = squared[order].sum(axis=1)
                    power_share = np.cumsum(ranked_power) / ranked_power.sum()

                    results["removal"][cell] = {
                        "z_baseline": results["meta"][key]["z_baseline"][
                            0 if target == SENTENCE_FREQ else 1
                        ],
                        "shape": curve_summary(z_removed),
                        "z_at_k": {str(k): float(z_removed[k - 1])
                                   for k in k_grid},
                        "z_at_percent": {
                            f"{pct}%": float(
                                z_removed[max(int(n_components * pct / 100), 1) - 1]
                            )
                            for pct in (1, 5, 10, 25, 50, 75, 90)
                        },
                        "power_share_at_k": {str(k): float(power_share[k - 1])
                                             for k in k_grid},
                        "locations": component_locations(order, n_dims),
                    }
                    results["retention"][cell] = {
                        "shape": curve_summary(z_retained),
                        "k_reaches_threshold": first_at_or_above(z_retained),
                        "z_at_k": {str(k): float(z_retained[k - 1])
                                   for k in k_grid},
                    }
                    if ranking == "magnitude":
                        top = int(order[0])
                        results["retention"][cell]["top_component"] = {
                            "layer": top // n_dims,
                            "dim": top % n_dims,
                            "solo_z": float(z_retained[0]),
                            "power_share": float(power_share[0]),
                        }
                        rand_rem, rand_ret = random_subset_control(
                            squared, target_index, noise_mask, k_grid
                        )
                        results["removal"][cell]["random_control"] = rand_rem
                        results["retention"][cell]["random_control"] = rand_ret
                        results["removal"][cell]["interference"] = (
                            interference_diagnostic(squared, order, target)
                        )

                    shape = results["removal"][cell]["shape"]
                    ret_shape = results["retention"][cell]["shape"]
                    print(f"    @{target_label:>4s} {ranking:12s} "
                          f"removal min z={shape['z_min']:7.2f} at k="
                          f"{shape['k_at_z_min']:<6d} "
                          f"final={shape['z_final']:7.2f} "
                          f"above-thr {100 * shape['fraction_k_above_threshold']:5.1f}% "
                          f"| retention z(k=1)={z_retained[0]:6.2f} "
                          f"min={ret_shape['z_min']:6.2f}")
                    del scores, order, z_removed, z_retained

            del activations, components, squared

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")
    make_figures(results, curves, act_dir, fig_dir)
    return results, curves


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def make_figures(results, curves, act_dir, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "magnitude": "#0072B2",
        "rawpower": "#E69F00",
        "sqpower": "#009E73",
        "aligned": "#D55E00",
    }
    panels = [
        ("gpt2-zh", "phrase+sentence", "0.25"),
        ("gpt2-zh", "phrase+sentence", "0.5"),
        ("gpt2-zh", "phrase", "0.5"),
        ("gpt2-zh", "reversed", "0.5"),
        ("rwkv7", "phrase+sentence", "0.25"),
        ("rwkv7", "phrase+sentence", "0.5"),
        ("rwkv7", "phrase", "0.5"),
        ("rwkv7", "reversed", "0.5"),
    ]

    for direction, index, filename, ylabel in (
        ("removal", 0, "macroscopic-ablation-removal.png",
         "z after removing the top-$k$"),
        ("retention", 1, "macroscopic-ablation-retention.png",
         "z keeping only the top-$k$"),
    ):
        fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=False)
        for ax, (model, corpus, target_label) in zip(axes.ravel(), panels):
            key = f"{model}/{corpus}"
            n_components = results["meta"][key]["n_components"]
            baseline = results["meta"][key]["z_baseline"][
                0 if target_label == "0.25" else 1
            ]
            for ranking in RANKINGS:
                cell = f"{key}@{target_label}/{ranking}"
                if cell not in curves:
                    continue
                curve = curves[cell][index]
                ax.plot(np.arange(1, len(curve) + 1), curve, lw=1.2,
                        color=colors[ranking], label=RANKING_LABEL[ranking])
            control_cell = f"{key}@{target_label}/magnitude"
            control = results[direction].get(control_cell, {}).get("random_control")
            if control:
                ks = np.array([int(k) for k in control])
                means = np.array([control[k][0] for k in control])
                sds = np.array([control[k][1] for k in control])
                ax.plot(ks, means, color="0.45", lw=1.0, ls="-.",
                        label="random subset")
                ax.fill_between(ks, means - sds, means + sds,
                                color="0.6", alpha=0.25, lw=0)
            ax.axhline(Z_THRESHOLD, color="0.35", ls="--", lw=0.9)
            ax.axhline(baseline, color="0.6", ls=":", lw=0.9)
            ax.set_xscale("log")
            ax.set_xlim(1, n_components)
            ax.set_title(f"{MODEL_LABEL[model]} / {corpus} @ {target_label}"
                         f"  (baseline z={baseline:.1f})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("$k$ components", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
        axes[0, 0].legend(fontsize=7, loc="best")
        fig.suptitle(
            f"Component ablation of $x(u)=\\|H_u\\|_F$ -- {direction}; "
            f"dashed $z=3$, dotted = unablated baseline", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=200)
        plt.close(fig)
        print(f"wrote {fig_dir / filename}")

    # Layer histogram of the top-100 components.
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5))
    for ax, (model, corpus) in zip(
        axes.ravel(),
        [(m, c) for m in MODELS for c in ABLATION_CORPORA],
    ):
        key = f"{model}/{corpus}"
        n_layers = results["meta"][key]["n_layers"]
        target_label = "0.5"
        width = 0.8 / len(RANKINGS)
        for i, ranking in enumerate(RANKINGS):
            cell = f"{key}@{target_label}/{ranking}"
            if cell not in results["removal"]:
                continue
            hist = np.array(results["removal"][cell]["locations"]["layer_histogram"])
            padded = np.zeros(n_layers)
            padded[:len(hist)] = hist
            ax.bar(np.arange(n_layers) + i * width - 0.4, padded, width=width,
                   color=colors[ranking], label=RANKING_LABEL[ranking])
        ax.set_title(f"{MODEL_LABEL[model]} / {corpus} @ 0.5", fontsize=8)
        ax.set_xlabel("layer", fontsize=8)
        ax.set_ylabel("count in top-100", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Layer of the 100 top-ranked components", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "macroscopic-ablation-layers.png", dpi=200)
    plt.close(fig)
    print(f"wrote {fig_dir / 'macroscopic-ablation-layers.png'}")

    # Position profile of x(u).
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for ax, (model, corpus) in zip(
        axes.ravel(),
        [(m, c) for m in MODELS for c in ABLATION_CORPORA],
    ):
        activations = np.load(act_dir / model / f"{corpus}.npy")
        signal = global_signal(activations)
        ax.plot(signal, lw=0.9, color="#333333")
        ax.plot(0, signal[0], "o", ms=4, color="#D55E00")
        info = results["position"][f"{model}/{corpus}"]
        ax.set_title(
            f"{MODEL_LABEL[model]} / {corpus}\n"
            f"x(0)/median={info['first_over_median']:.2f}, "
            f"max share={100 * info['max_power_share']:.2f}%", fontsize=8)
        ax.set_xlabel("position $u$", fontsize=8)
        ax.set_ylabel("$x(u)$", fontsize=8)
        ax.tick_params(labelsize=7)
        del activations
    fig.suptitle("Position profile of the aggregate signal "
                 "(orange = first position)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "macroscopic-ablation-position.png", dpi=200)
    plt.close(fig)
    print(f"wrote {fig_dir / 'macroscopic-ablation-position.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activations", type=Path, required=True,
                        help="directory holding <model>/<corpus>.npy")
    parser.add_argument("--out", type=Path, default=FIG_DIR,
                        help="figure directory (default: analysis/figures)")
    args = parser.parse_args()
    run(args.activations, args.out)
