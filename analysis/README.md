# Supporting analyses

One-off scripts that read the cached activations (or, for `tokens_per_unit.py`, the corpora and tokenizers) and write figures and a JSON of results. Nothing in the figure pipeline imports them.

| Script | What it computes |
|---|---|
| `norm_ablation.py` | z at 0.25 and 0.5 unit⁻¹ after removing or keeping the top-k components of the aggregate signal |
| `nyquist_diagnostics.py` | component pooling and phase at 0.5 unit⁻¹ (Nyquist) and 0.25 unit⁻¹ |
| `layer_profile_correlation.py` | per-layer z profiles, their correlation and their z = 3 crossings |
| `tokens_per_unit.py` | tokens per unit and units per token for every corpus under both tokenizers |

```
uv run --with numpy --with matplotlib -- python analysis/norm_ablation.py --activations <OUTPUT_DIR>/activations
uv run --with numpy --with matplotlib -- python analysis/nyquist_diagnostics.py --activations <OUTPUT_DIR>/activations
uv run --with numpy --with scipy -- python analysis/layer_profile_correlation.py --activations <OUTPUT_DIR>/activations
uv run --with transformers --with torch -- python analysis/tokens_per_unit.py --corpus-dir <CORPUS_DIR>
```

Figures go to `analysis/figures/` unless `--out` names another directory; the JSON files are written next to the scripts. `layer_profile_correlation_results.json` and `tokens_per_unit_results.json` are committed; the other outputs are not.
