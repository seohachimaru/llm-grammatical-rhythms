"""Redraw one icon PDF in a different ink by rewriting its color operators.

    python assets/recolor_icon.py assets/icons/phrase-sentence.pdf \
        out.pdf --to "#4C7DFF"
"""

import argparse
import re
from pathlib import Path

# Operands of `rg` (fill) and `RG` (stroke).
COLOR_OPERATOR = re.compile(
    rb"(?P<r>[\d.]+)\s+(?P<g>[\d.]+)\s+(?P<b>[\d.]+)\s+(?P<op>rg|RG)"
)


def to_pdf_rgb(hex_color):
    """`#RRGGBB` as the three 0-1 operands a PDF color operator takes."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected #RRGGBB, got {hex_color!r}")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def recolor_raster(document, page, target_hex, only_hex, tolerance=42):
    """Replace ink `only_hex` with `target_hex` in every image on the page.

    Pixels are remapped by their position along the ink-to-paper line, so
    antialiased edges follow; pixels off that line are left alone. Returns the
    number of pixels changed.
    """
    import numpy as np
    from PIL import Image
    import io

    if only_hex is None:
        raise SystemExit("a raster needs --only to say which ink to replace")
    source_rgb = np.array(to_pdf_rgb(only_hex)) * 255.0
    target_rgb = np.array(to_pdf_rgb(target_hex)) * 255.0
    paper = np.array([255.0, 255.0, 255.0])
    replaced = 0

    for xref, *_ in page.get_images():
        raw = document.extract_image(xref)
        image = Image.open(io.BytesIO(raw["image"])).convert("RGB")
        pixels = np.asarray(image).astype(np.float64)

        # Project each pixel onto the ink-to-paper line; the position is the
        # ink's share.
        span = source_rgb - paper
        share = ((pixels - paper) @ span) / float(span @ span)
        share = np.clip(share, 0.0, 1.0)
        distance = np.linalg.norm(
            pixels - (paper + share[..., None] * span), axis=-1
        )
        # Bare paper also sits on the line, at zero share; exclude it.
        hit = (distance < tolerance) & (share > 0.02)
        if not hit.any():
            continue

        rebuilt = paper + share[..., None] * (target_rgb - paper)
        pixels[hit] = rebuilt[hit]
        replaced += int(hit.sum())

        buffer = io.BytesIO()
        Image.fromarray(pixels.round().clip(0, 255).astype(np.uint8)).save(
            buffer, format="PNG"
        )
        page.replace_image(xref, stream=buffer.getvalue())
    return replaced


def recolor(source, destination, target_hex, only_hex=None):
    """Copy `source` to `destination` with its ink replaced.

    With `only_hex`, just that one ink is replaced and everything else is left
    alone; without it, every color operator in the stream is rewritten.
    """
    import pymupdf

    red, green, blue = to_pdf_rgb(target_hex)
    replacement = f"{red:.10f} {green:.10f} {blue:.10f}".encode()
    match_rgb = to_pdf_rgb(only_hex) if only_hex else None
    replaced = 0

    document = pymupdf.open(source)
    page = document[0]

    def substitute(match):
        nonlocal replaced
        if match_rgb is not None:
            current = tuple(float(match.group(c)) for c in ("r", "g", "b"))
            if any(abs(a - b) > 1e-6 for a, b in zip(current, match_rgb)):
                return match.group(0)
        replaced += 1
        return replacement + b" " + match.group("op")

    # Merge the page's content streams into one so all of it is rewritten.
    page.clean_contents()
    xref = page.get_contents()[0]
    rewritten = COLOR_OPERATOR.sub(substitute, page.read_contents())
    document.update_stream(xref, rewritten)

    pixels = recolor_raster(document, page, target_hex, only_hex) \
        if page.get_images() else 0
    if not replaced and not pixels:
        raise SystemExit(f"nothing to recolor in {source}")

    document.save(destination)
    document.close()
    drawn = [f"{replaced} color operators"] if replaced else []
    drawn += [f"{pixels} pixels"] if pixels else []
    print(f"wrote {destination} ({' and '.join(drawn)} -> {target_hex})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="icon PDF to read")
    parser.add_argument("destination", help="icon PDF to write")
    parser.add_argument("--to", required=True, help="target ink, #RRGGBB")
    parser.add_argument("--only", help="replace just this ink, #RRGGBB")
    args = parser.parse_args()
    recolor(Path(args.source), Path(args.destination), args.to, args.only)


if __name__ == "__main__":
    main()
