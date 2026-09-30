"""Draw span arrows under a structure icon, one per constituent.

Leaves are the filled circles of the artwork and levels its horizontal rules.
Each --arrow gives the number of leaves one arrow spans, outermost level
first, and its color.

    python assets/annotate_icon.py assets/icons/phrase-sentence.pdf out.pdf \
        --arrow 4:#E27100 --arrow 2:#36883A
"""

import argparse
from pathlib import Path

# Share of the arrow's color kept against white (1 = full color).
DEFAULT_TINT = 0.5

# Arrow proportions, relative to the icon's page width.
SHAFT = 0.0405  # half-thickness of the shaft
HEAD_HEIGHT = 2.4  # half-height of the head, as a multiple of SHAFT
HEAD_LENGTH = 1.15  # length of the head, as a multiple of the shaft's thickness


def tint(hex_color, amount=DEFAULT_TINT):
    """`#RRGGBB` mixed with white, as PDF's three 0-1 operands."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected #RRGGBB, got {hex_color!r}")
    channels = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(1.0 - (1.0 - c) * amount for c in channels)


def read_artwork(page):
    """Leaf x-extents and level heights, in PDF user space (y up).

    PyMuPDF reports drawings with y down from the top, so y is flipped here.
    """
    height = page.rect.height
    leaves, levels = [], set()
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if drawing["type"] == "f":  # the page's white ground
            continue
        if all(item[0] == "c" for item in drawing["items"]):
            leaves.append((rect.x0, rect.x1))
        elif abs(rect.y0 - rect.y1) < 1e-6:  # a horizontal rule: one level
            levels.add(round(height - rect.y0, 4))
    if not leaves:
        raise SystemExit("no filled circles found: this icon has no leaves")
    # The outermost level sits highest on the page.
    return sorted(leaves), sorted(levels, reverse=True)


def arrow_path(x0, x1, y, shaft):
    """A right-pointing arrow as one closed polygon, in PDF operators."""
    head_h = HEAD_HEIGHT * shaft
    head_l = HEAD_LENGTH * 2 * shaft
    neck = x1 - head_l
    points = [(x0, y - shaft), (neck, y - shaft), (neck, y - head_h), (x1, y),
              (neck, y + head_h), (neck, y + shaft), (x0, y + shaft)]
    moves = [f"{px:.4f} {py:.4f} " + ("m" if i == 0 else "l")
             for i, (px, py) in enumerate(points)]
    return "\n".join(moves) + "\nh\nf\n"


def annotate(source, destination, arrows, tint_amount=DEFAULT_TINT,
             pad_ratio=0.24):
    import pymupdf

    document = pymupdf.open(source)
    page = document[0]
    leaves, levels = read_artwork(page)
    if len(arrows) > len(levels):
        raise SystemExit(f"{len(arrows)} arrow levels asked of an icon with "
                         f"{len(levels)}")

    shaft = SHAFT * page.rect.width
    pad = pad_ratio * min(x1 - x0 for x0, x1 in leaves)
    limit_low, limit_high = 0.0, page.rect.width
    operators = ["q\n/DeviceRGB cs\n"]
    for (span, color), y in zip(arrows, levels):
        if len(leaves) % span:
            raise SystemExit(f"{len(leaves)} leaves do not divide into {span}")
        red, green, blue = tint(color, tint_amount)
        operators.append(f"{red:.6f} {green:.6f} {blue:.6f} rg\n")
        for start in range(0, len(leaves), span):
            group = leaves[start:start + span]
            x0 = max(limit_low, group[0][0] - pad)
            x1 = min(limit_high, group[-1][1] + pad)
            operators.append(arrow_path(x0, x1, y, shaft))
    operators.append("Q\n")

    # Insert after the white ground (the first fill) and before the artwork.
    page.clean_contents()
    xref = page.get_contents()[0]
    stream = page.read_contents()
    cut = stream.find(b" f ")
    if cut < 0:
        raise SystemExit("could not find the ground this icon is painted on")
    cut += len(b" f ")
    document.update_stream(
        xref, stream[:cut] + "".join(operators).encode() + stream[cut:]
    )
    document.save(destination)
    document.close()
    print(f"wrote {destination} ({sum(len(leaves) // s for s, _ in arrows)} "
          f"arrows over {len(leaves)} leaves)")


def parse_arrow(text):
    span, _, color = text.partition(":")
    if not color:
        raise argparse.ArgumentTypeError(f"expected SPAN:#RRGGBB, got {text!r}")
    return int(span), color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="icon PDF to read")
    parser.add_argument("destination", help="icon PDF to write")
    parser.add_argument("--arrow", type=parse_arrow, action="append",
                        required=True, metavar="SPAN:#RRGGBB",
                        help="leaves per arrow and its hue, outermost first")
    parser.add_argument("--tint", type=float, default=DEFAULT_TINT,
                        help="share of the hue kept against white, 0-1 "
                             f"(default {DEFAULT_TINT})")
    args = parser.parse_args()
    if not 0.0 < args.tint <= 1.0:
        raise SystemExit(f"--tint must be in (0, 1], got {args.tint}")
    annotate(Path(args.source), Path(args.destination), args.arrow, args.tint)


if __name__ == "__main__":
    main()
