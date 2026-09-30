"""Build the icon set figures.py embeds from the artwork in icons/.

Each drawing is recolored in its row's ink, and span arrows are drawn under the
structured ones. Run this first, then point `figures.py --assets` at its output:

    python assets/build_icons.py --out build/icons
    python figures.py --activations <dir> --assets build/icons --out <dir>
"""

import argparse
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent
sys.path.insert(0, str(ASSETS))
sys.path.insert(0, str(ASSETS.parent))

from annotate_icon import DEFAULT_TINT, annotate  # noqa: E402
from figures import CHINESE_INK, ENGLISH_INK, PHRASE_COLOR, SENTENCE_COLOR  # noqa: E402
from recolor_icon import recolor  # noqa: E402

# The ink the artwork is drawn in; only the raster icon needs it named.
SOURCE_INK = "#0500D7"

# An arrow is (leaves spanned, color).
PHRASE_ARROW = (2, PHRASE_COLOR)
SENTENCE_ARROW = (4, SENTENCE_COLOR)

# source artwork, icon to write, ink, arrows, ink to replace (raster only)
ICONS = [
    ("random.pdf", "random.pdf", CHINESE_INK, [], None),
    ("phrase.pdf", "phrase.pdf", CHINESE_INK, [PHRASE_ARROW], None),
    ("phrase-sentence.pdf", "phrase-sentence.pdf", CHINESE_INK,
     [SENTENCE_ARROW, PHRASE_ARROW], None),
    ("reversed.pdf", "reversed.pdf", CHINESE_INK, [], SOURCE_INK),
    ("phrase-sentence.pdf", "phrase-sentence-en.pdf", ENGLISH_INK,
     [SENTENCE_ARROW, PHRASE_ARROW], None),
]


def build(out, tint_amount=DEFAULT_TINT):
    out.mkdir(parents=True, exist_ok=True)
    for source, name, ink, arrows, only in ICONS:
        destination = out / name
        # pymupdf cannot save non-incrementally over the file it has open, so
        # the recolored artwork goes to a temporary file first.
        plain = destination.with_suffix(".plain.pdf") if arrows else destination
        recolor(ASSETS / "icons" / source, plain, ink, only)
        if arrows:
            annotate(plain, destination, arrows, tint_amount=tint_amount)
            plain.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/icons",
                        help="directory to write the icon set to "
                             "(default: build/icons)")
    parser.add_argument("--tint", type=float, default=DEFAULT_TINT,
                        help="share of the arrow hue kept against white, 0-1 "
                             f"(default {DEFAULT_TINT})")
    args = parser.parse_args()
    build(Path(args.out), args.tint)


if __name__ == "__main__":
    main()
