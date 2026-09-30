# Corpora

The notebook reads three files from `CORPUS_DIR`.

| File | Contents | Source |
|---|---|---|
| `Chinese-sentences.txt` | 50 four-character Chinese sentences | Ding et al. (2016), Supplementary Table S1; not redistributed here |
| `English-sentences.txt` | 60 four-word English sentences | Ding et al. (2016), Supplementary Table S1; not redistributed here |
| `Chinese-random.txt` | 200 Chinese characters | deposited in this folder |

`Chinese-random.txt` holds the 200 characters of the 50 four-character verb phrases (a two-character verb followed by a two-character noun) in Ding et al.'s Supplementary Table S1, in random order. It is not Ding et al.'s random condition, which replaces each syllable of a sentence with the syllable at the same position in a randomly chosen sentence. Copy it into `CORPUS_DIR` next to the other two files.

The phrase-only and reversed-phrases corpora are derived from `Chinese-sentences.txt` in the notebook; the reversal follows Lo et al. (2022).

Whitespace is removed from the Chinese files and used as the word separator in the English file. The notebook prints the unit count of every corpus: 200, 100, 200, 200 and 240.

> Ding N, Melloni L, Zhang H, Tian X, Poeppel D (2016). Cortical tracking of hierarchical linguistic structures in connected speech. *Nature Neuroscience* 19:158–164. doi:10.1038/nn.4186

> Lo CW, Tung TY, Ke AH, Brennan JR (2022). Hierarchy, not lexical regularity, modulates low-frequency neural synchrony during language comprehension. *Neurobiology of Language* 3:538–555.
