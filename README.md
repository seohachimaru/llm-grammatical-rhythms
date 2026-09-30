# llm-grammatical-rhythms

Code for the Brief Report by Seo Hachimaru and Fumito Mori, which applies the frequency-tagging paradigm of Ding et al. (2016) to the hidden states of language models.

- `notebook.ipynb` computes the target-frequency z-scores and the structured-versus-reversed tests, and caches the hidden states to Drive.
- `figures.py` draws the data figures from that cache.
- `analysis/` holds supporting analyses; see `analysis/README.md`.

| Model | Hugging Face |
|---|---|
| GPT-2 (Chinese) | `uer/gpt2-large-chinese-cluecorpussmall` |
| RWKV-7 | `fla-hub/rwkv7-1.5B-world` |

## Running

1. Obtain the two corpora taken from Ding et al. (2016): `data/README.md`.
2. Put them and `data/Chinese-random.txt` in a Drive folder and point `CORPUS_DIR` at it.
3. Run `notebook.ipynb` top to bottom in Google Colab on a GPU runtime. The last cell compares the z-scores with those of the published run.

Outputs, under `OUTPUT_DIR`:

| Path | Contents |
|---|---|
| `activations/<model>/<corpus>.npy` | unit-aligned hidden states, `[layers, dim, positions]` |
| `target_z_scores.json` | target-frequency z-scores |
| `structured_vs_reversed.json` | Mann–Whitney comparisons |

GPU and library versions of the published run: `ENVIRONMENT.md`.

## Figures

```
python assets/build_icons.py --out build/icons
python figures.py --activations <OUTPUT_DIR>/activations --assets build/icons --out <dir>
```

Writes `global_2x5.pdf`, `layerwise_2x5.pdf` and `z_scores.json`. `--corpora` selects and orders the rows, `--style base` drops the decoration, and `--panels` writes one file per panel.

## Provenance

The code was written with Claude (Anthropic), mainly Claude Opus 4.5 to 5.5 and Claude Fable 5.0 to 5.1. The corpus-structure icons are the authors' own artwork.

## License

MIT, see `LICENSE`. The corpora, including `data/Chinese-random.txt`, are not covered by it.
