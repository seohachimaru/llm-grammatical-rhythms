# Environment

Hidden-state values depend on the GPU and library versions, so a rerun reproduces the z-scores within the tolerance of the notebook's last cell rather than exactly.

Recorded in the Google Colab sessions that extracted the published activations of GPT-2 (2026-07-22) and of RWKV-7 on the phrase-plus-sentence, reversed-phrases and English corpora (2026-07-26):

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, driver 580.82.07 |
| `torch` | 2.11.0+cu128 |
| `triton` | 3.6.0 |
| `transformers` | 5.13.1 |
| `tokenizers` | 0.22.2 |
| `rwkv-fla` | 0.7.202508221413 |
| `numpy` | 2.0.2 |
| `scipy` | 1.16.3 |
