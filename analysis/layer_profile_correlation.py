"""Per-layer z profiles at 0.5 and 0.25 unit^-1: their correlation and z = 3 crossings.

Signal and z-score are imported from figures.py; the run exits unless the aggregate
z-scores match PUBLISHED. Writes layer_profile_correlation_results.json.

Run:
    uv run --with numpy --with scipy -- python analysis/layer_profile_correlation.py \
        --activations <OUTPUT_DIR>/activations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures import (  # noqa: E402
    PHRASE_FREQ,
    SENTENCE_FREQ,
    Z_THRESHOLD,
    global_signal,
    layerwise_z_scores,
    target_z_scores,
)

OUT_JSON = Path(__file__).resolve().parent / "layer_profile_correlation_results.json"

MODELS = ("gpt2-zh", "rwkv7")
MODEL_LABEL = {"gpt2-zh": "GPT-2", "rwkv7": "RWKV-7"}
CORPORA = ("phrase+sentence", "phrase", "random")

# Aggregate z-scores (0.25, 0.5) quoted in the manuscript; checked by validate().
PUBLISHED = {
    ("gpt2-zh", "phrase+sentence"): (9.0, 11.8),
    ("gpt2-zh", "phrase"): (None, 11.1),
    ("gpt2-zh", "random"): (-0.3, -1.4),
    ("rwkv7", "phrase+sentence"): (14.2, 56.7),
    ("rwkv7", "phrase"): (None, 44.0),
}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def partial_corr_controlling_layer(a, b):
    """Pearson (r, p) of a and b after regressing a linear layer trend out of both."""
    layer = np.arange(len(a), dtype=np.float64)
    design = np.column_stack([np.ones_like(layer), layer])
    resid = []
    for series in (np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)):
        coef, *_ = np.linalg.lstsq(design, series, rcond=None)
        resid.append(series - design @ coef)
    n = len(layer)
    if n <= 3:
        return None, None
    r = float(np.corrcoef(resid[0], resid[1])[0, 1])
    # df = n - 2 - (number of controlled variables) = n - 3
    df = n - 3
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * np.sqrt(df / (1.0 - r ** 2))
    return r, float(2.0 * stats.t.sf(abs(t), df))


def crossing(z, threshold=Z_THRESHOLD):
    """Where a profile crosses the threshold, and where it peaks."""
    z = np.asarray(z, dtype=np.float64)
    above = z > threshold
    first = int(np.argmax(above)) if above.any() else None
    return {
        "n_layers": int(z.size),
        "n_above": int(above.sum()),
        "first_above": first,
        "monotone_above_after_first": bool(above[first:].all()) if first is not None else None,
        "last_above": int(np.max(np.flatnonzero(above))) if above.any() else None,
        "argmax_layer": int(np.argmax(z)),
        "max": float(z.max()),
        "min": float(z.min()),
        "dips_below_after_first": (
            [int(i) for i in np.flatnonzero(~above[first:]) + first]
            if first is not None else []
        ),
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
    reproduced = {}
    for model in MODELS:
        for corpus in CORPORA:
            path = act_dir / model / f"{corpus}.npy"
            if not path.exists():
                raise SystemExit(f"missing activation cache: {path}")
            activations = np.load(path)
            z_sentence, z_phrase = target_z_scores(global_signal(activations))
            reproduced[f"{model}/{corpus}"] = {
                "z_sentence": z_sentence,
                "z_phrase": z_phrase,
                "published": PUBLISHED.get((model, corpus), (None, None)),
            }
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
    return reproduced


def run(act_dir):
    reproduced = validate(act_dir)
    results = {
        "meta": {
            "sentence_freq": SENTENCE_FREQ,
            "phrase_freq": PHRASE_FREQ,
            "z_threshold": Z_THRESHOLD,
            "layer_index": "0-based, embedding layer dropped, as plotted in Fig. 3",
            "reused_from_figures_py": [
                "global_signal", "layer_signals", "spectrum", "z_at",
                "target_z_scores", "layerwise_z_scores",
            ],
        },
        "aggregate": reproduced,
        "layerwise": {},
        "correlation": {},
        "crossings": {},
    }

    print("=" * 78)
    print("STEP 2  layer-wise profiles, their correlation and their crossings")
    print("=" * 78)

    for model in MODELS:
        for corpus in CORPORA:
            key = f"{model}/{corpus}"
            activations = np.load(act_dir / model / f"{corpus}.npy")
            z_sentence, z_phrase = layerwise_z_scores(activations)
            del activations

            n = int(z_sentence.size)
            pearson = stats.pearsonr(z_phrase, z_sentence)
            spearman = stats.spearmanr(z_phrase, z_sentence)
            partial_r, partial_p = partial_corr_controlling_layer(z_phrase, z_sentence)

            results["layerwise"][key] = {
                "n_layers": n,
                "z_sentence": [float(v) for v in z_sentence],
                "z_phrase": [float(v) for v in z_phrase],
            }
            results["correlation"][key] = {
                "n": n,
                "pearson_r": float(pearson[0]),
                "pearson_p": float(pearson[1]),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "partial_pearson_r_given_layer": partial_r,
                "partial_pearson_p_given_layer": partial_p,
                "pearson_r_phrase_vs_layer": float(
                    stats.pearsonr(np.arange(n), z_phrase)[0]),
                "pearson_r_sentence_vs_layer": float(
                    stats.pearsonr(np.arange(n), z_sentence)[0]),
            }
            results["crossings"][key] = {
                "sentence": crossing(z_sentence),
                "phrase": crossing(z_phrase),
            }

            corr = results["correlation"][key]
            print(f"\n{MODEL_LABEL[model]} / {corpus}   n = {n} layers")
            print(f"  Pearson  r = {corr['pearson_r']:+.3f}  p = {corr['pearson_p']:.3g}")
            print(f"  Spearman rho = {corr['spearman_rho']:+.3f}  "
                  f"p = {corr['spearman_p']:.3g}")
            print(f"  partial (layer controlled) r = {partial_r:+.3f}  p = {partial_p:.3g}")
            print(f"  r vs depth: phrase {corr['pearson_r_phrase_vs_layer']:+.3f}, "
                  f"sentence {corr['pearson_r_sentence_vs_layer']:+.3f}")
            for level in ("phrase", "sentence"):
                c = results["crossings"][key][level]
                print(f"  {level:8s} first z>3 at layer {c['first_above']}, "
                      f"{c['n_above']}/{c['n_layers']} layers above, "
                      f"stays above: {c['monotone_above_after_first']}, "
                      f"max {c['max']:.2f} at layer {c['argmax_layer']}")

    print("\n" + "=" * 78)
    print("STEP 3  per-layer tables")
    print("=" * 78)
    for model in MODELS:
        for corpus in CORPORA:
            key = f"{model}/{corpus}"
            entry = results["layerwise"][key]
            print(f"\n{MODEL_LABEL[model]} / {corpus}")
            print(f"{'layer':>5s} {'z(0.5) phrase':>14s} {'z(0.25) sentence':>17s}")
            for i, (zp, zs) in enumerate(zip(entry["z_phrase"], entry["z_sentence"])):
                flag_p = "*" if zp > Z_THRESHOLD else " "
                flag_s = "*" if zs > Z_THRESHOLD else " "
                print(f"{i:5d} {zp:13.2f}{flag_p} {zs:16.2f}{flag_s}")

    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activations", type=Path, required=True,
                        help="directory holding <model>/<corpus>.npy")
    run(parser.parse_args().activations)
