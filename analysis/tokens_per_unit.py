"""Tokens per unit and units per token for every corpus under both tokenizers.

Also counts RWKV vocabulary entries with two or more CJK characters.
Writes tokens_per_unit_results.json.

The RWKV tokenizer is loaded from the cached checkpoint directly, since
`AutoTokenizer` on this checkpoint imports the model package (`fla`).

Run:
    uv run --with transformers --with torch -- python analysis/tokens_per_unit.py \
        --corpus-dir <CORPUS_DIR>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer  # noqa: E402

HF_CACHE = Path("~/.cache/huggingface/hub").expanduser()
OUT = Path(__file__).with_name("tokens_per_unit_results.json")

SENTENCE_LEN = 4
CJK = re.compile(r"[一-鿿]")

MODELS = {
    "gpt2-zh": {"hf": "uer/gpt2-large-chinese-cluecorpussmall", "tokenizer": {}},
    "rwkv7": {"hf": "fla-hub/rwkv7-1.5B-world", "tokenizer": {"add_bos_token": True}},
}


# --- corpora, as the notebook builds them -----------------------------------

def read_units(path: Path, language: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if language == "en":
        return text.split()
    return list(re.sub(r"\s+", "", text))


def phrase_only(units: list[str]) -> list[str]:
    return [u for i in range(0, len(units), SENTENCE_LEN)
            for u in units[i:i + SENTENCE_LEN // 2]]


def reversed_phrases(units: list[str]) -> list[str]:
    out = []
    for i in range(0, len(units), SENTENCE_LEN):
        first, second, third, fourth = units[i:i + SENTENCE_LEN]
        out += [second, first, fourth, third]
    return out


def build_corpora(corpus_dir: Path) -> dict[str, tuple[str, list[str]]]:
    zh = read_units(corpus_dir / "Chinese-sentences.txt", "zh")
    return {
        "random": ("zh", read_units(corpus_dir / "Chinese-random.txt", "zh")),
        "phrase": ("zh", phrase_only(zh)),
        "phrase+sentence": ("zh", zh),
        "reversed": ("zh", reversed_phrases(zh)),
        "english": ("en", read_units(corpus_dir / "English-sentences.txt", "en")),
    }


# --- alignment, as the notebook applies it ----------------------------------

def manual_offsets(tokenizer, input_ids, text):
    special = set(tokenizer.all_special_ids)
    spans, pos = [], 0
    for token_id in input_ids:
        if token_id in special:
            spans.append((0, 0))
            continue
        piece = tokenizer.decode([token_id])
        start = text.find(piece, pos)
        if start < 0:
            start = pos
        spans.append((start, start + len(piece)))
        pos = start + len(piece)
    return spans


def unit_token_indices(units, text, spans, language):
    indices, pos = [], 0
    for unit in units:
        found = text.find(unit, pos)
        start = found if found >= 0 else pos
        end = start + len(unit)
        indices.append([i for i, (s, e) in enumerate(spans) if start < e and s < end])
        pos = end
        if language == "en":
            while pos < len(text) and text[pos].isspace():
                pos += 1
    return indices


# --- tokenizers ---------------------------------------------------------------

def snapshot_dir(hf_name: str) -> Path:
    snaps = HF_CACHE / f"models--{hf_name.replace('/', '--')}" / "snapshots"
    return next(snaps.iterdir())


def load_tokenizer(key: str):
    spec = MODELS[key]
    if key == "rwkv7":
        snap = snapshot_dir(spec["hf"])
        module_spec = importlib.util.spec_from_file_location(
            "hf_rwkv_tokenizer", snap / "hf_rwkv_tokenizer.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module.RwkvTokenizer(str(snap / "rwkv_vocab_v20230424.txt"),
                                    **spec["tokenizer"])
    return AutoTokenizer.from_pretrained(spec["hf"], trust_remote_code=True,
                                         **spec["tokenizer"])


def rwkv_vocab_cjk_entries() -> dict:
    """How many RWKV vocabulary entries contain two or more CJK characters."""
    vocab = snapshot_dir(MODELS["rwkv7"]["hf"]) / "rwkv_vocab_v20230424.txt"
    n_entries = n_multi = 0
    for line in vocab.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        n_entries += 1
        piece = eval(line[line.index(" "):line.rindex(" ")])  # the file's own format
        raw = piece.encode("utf-8") if isinstance(piece, str) else piece
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(CJK.findall(text)) >= 2:
            n_multi += 1
    return {"vocab_entries": n_entries, "entries_with_2plus_cjk": n_multi}


# --- main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, required=True,
                        help="directory holding the three corpus files (data/README.md)")
    corpora = build_corpora(parser.parse_args().corpus_dir)
    results = {"alignment": {}, "rwkv_vocab": rwkv_vocab_cjk_entries()}
    for key in MODELS:
        tokenizer = load_tokenizer(key)
        for name, (language, units) in corpora.items():
            text = " ".join(units) if language == "en" else "".join(units)
            encoded = tokenizer(text, add_special_tokens=True,
                                return_offsets_mapping=tokenizer.is_fast)
            ids = encoded["input_ids"]
            spans = (encoded["offset_mapping"] if tokenizer.is_fast
                     else manual_offsets(tokenizer, ids, text))
            idx = unit_token_indices(units, text, spans, language)
            units_of_token: Counter = Counter(t for i in idx for t in i)
            entry = {
                "units": len(units),
                "content_tokens": sum(1 for s in spans if s != (0, 0)),
                "tokens_per_unit": dict(sorted(Counter(len(i) for i in idx).items())),
                "units_per_token": dict(sorted(Counter(units_of_token.values()).items())),
            }
            results["alignment"][f"{key}/{name}"] = entry
            print(f"{key:8s} {name:16s} units={entry['units']:4d} "
                  f"tokens={entry['content_tokens']:4d} "
                  f"tokens/unit={entry['tokens_per_unit']} "
                  f"units/token={entry['units_per_token']}")
    print("RWKV vocab:", results["rwkv_vocab"])
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
