"""Draw the spectrum and layer-wise figures from the cached activation arrays.

    python figures.py --activations <dir> --out <dir>              # reference
    python figures.py --activations <dir> --out <dir> --style base # undecorated

Frequencies are per unit, i.e. per position in the activation arrays.
"""

import argparse
import json
from pathlib import Path

import numpy as np

SENTENCE_FREQ = 0.25  # unit^-1
PHRASE_FREQ = 0.50  # unit^-1
Z_THRESHOLD = 3.0

# One ink per stimulus language, used by the row icons and the column headers.
CHINESE_INK = "#191970"
ENGLISH_INK = "#4C7DFF"
SUBTITLE_COLOR = "0.35"

MODELS = {
    "gpt2-zh": {
        "label": "GPT-2",
        "languages": (("Chinese", CHINESE_INK), (" only", SUBTITLE_COLOR)),
    },
    "rwkv7": {
        "label": "RWKV-7",
        "languages": (("Chinese", CHINESE_INK), (", ", SUBTITLE_COLOR),
                      ("English", ENGLISH_INK), (", etc.", SUBTITLE_COLOR)),
    },
}

# Row order, and the icon for each corpus (built by assets/build_icons.py).
CORPORA = {
    "random": {"label": "Random", "icon": "random.pdf"},
    "phrase": {"label": "Phrase only", "icon": "phrase.pdf"},
    "phrase+sentence": {"label": "Phrase + sentence", "icon": "phrase-sentence.pdf"},
    "reversed": {"label": "Reversed phrases", "icon": "reversed.pdf"},
    "english": {"label": "Phrase + sentence", "icon": "phrase-sentence-en.pdf"},
}

# The control row is set off by a gap and a rule, and each block of rows is
# labelled with its stimulus language.
CONTROL_ROW = "english"
GROUP_GAP = 0.30  # extra space above the control row, inches
GROUP_LABEL_PAD = 0.02  # space between an icon and the name over it, inches
GROUP_LABEL_INDENT = 0.07  # how far left of its icon a name starts, inches
GROUP_LABEL_FONTSIZE = 8
GROUP_RULE_CLEARANCE = 0.045  # space the rule keeps off what is under it, inches
GROUP_RULE = {"color": "0.45", "linestyle": (0, (3.5, 2.5)), "lw": 0.6}
CHINESE_GROUP = ("Chinese", CHINESE_INK)
ENGLISH_GROUP = ("English", ENGLISH_INK)

# One hue per linguistic level, shared by both figures.
SENTENCE_COLOR = "#E27100"
PHRASE_COLOR = "#36883A"
SPECTRUM_FILL = "#DCDCD6"

# Legend glyph color; the marks in the panels take the hue of their frequency.
MARKER_COLOR = "#222222"

SENTENCE_LEGEND = "Sentence rate"
PHRASE_LEGEND = "Phrase rate"
# Thin spaces by hand: the custom mathtext fontset drops relation spacing.
Z_MARKER_LEGEND = r"$z\,{>}\,3$"
THRESHOLD_LEGEND = r"Threshold ($z\,{=}\,3$)"
FREQUENCY_LABEL = r"Frequency (unit$^{-1}$)"
LAYER_LABEL = r"Layer $\ell$"
ZSCORE_LABEL = r"$z$-score"
TARGETS = ((SENTENCE_FREQ, SENTENCE_COLOR, "0.25"), (PHRASE_FREQ, PHRASE_COLOR, "0.5"))
# x ticks as (position, color, label); a None color leaves the label plain.
XTICKS = ((0.0, None, "0"),) + TARGETS

FONT_STACK = ("Helvetica", "Arial", "DejaVu Sans")

PRINT_WIDTH = 3.5  # PNAS single column, inches
TITLE_FONTSIZE = 8
SUBTITLE_FONTSIZE = 6
LABEL_FONTSIZE = 7
TICK_FONTSIZE = 6
LEGEND_FONTSIZE = 6
ROW_LABEL_FONTSIZE = 6

ROW_HEIGHT = 0.39  # inches of panel per corpus row
ROW_GAP = 0.164  # inches between two panels
# Bands above and below the grid, inches.
HEADER_BAND = 0.855  # column headers and the legend
HEADER_BAND_BARE = 0.435  # the same band with neither drawn
FOOTER_BAND = 0.394  # x tick labels and the axis title
LEGEND_X = 0.575
ICON_RATIO = 0.85
# Icon baseline, as a fraction of the cell height from its bottom.
ICON_AREA_BOTTOM = 0.20
ICON_BLEED = 0.45  # fraction of the gap above a cell an icon may grow into

BAND_HALFWIDTH = 0.015
BAND_ALPHA = 0.16
MARKER_GLYPH = {"marker": (6, 2, 0), "ms": 7.0, "mew": 1.2}

# Size of one undecorated panel emitted on its own, inches.
PANEL_SIZE = (1.45, 0.95)

# The bare concept-figure panel: size in inches, then line widths and band.
PRESHRINK_PANEL = (0.8704, 0.6798)
BARE_CURVE_WIDTH = 0.6
BARE_SPINE_WIDTH = 0.4
BARE_BAND_HALFWIDTH = 0.022


# --- signals -----------------------------------------------------------------

def global_signal(activations):
    """Frobenius norm over layers and dims, one value per position."""
    return np.sqrt(np.sum(activations.astype(np.float64) ** 2, axis=(0, 1)))


def layer_signals(activations):
    """Per-layer L2 norm, shape [layers, positions]."""
    return np.linalg.norm(activations.astype(np.float64), axis=1)


def spectrum(signal):
    """Magnitude spectrum of a signal sampled once per position."""
    return np.fft.rfftfreq(len(signal), d=1.0), np.abs(np.fft.rfft(signal))


def z_at(freqs, mags, target, exclude=()):
    """Target bin against the mean and s.d. of the other bins, DC excluded."""
    index = int(np.argmin(np.abs(freqs - target)))
    drop = {0, index, *(int(np.argmin(np.abs(freqs - f))) for f in exclude)}
    noise = np.array([mags[i] for i in range(1, len(mags)) if i not in drop])
    if noise.size == 0 or noise.std() == 0:
        return 0.0
    return float((mags[index] - noise.mean()) / noise.std())


def target_z_scores(signal):
    """(z at the sentential rate, z at the phrasal rate)."""
    freqs, mags = spectrum(signal)
    return (
        z_at(freqs, mags, SENTENCE_FREQ, exclude=(PHRASE_FREQ,)),
        z_at(freqs, mags, PHRASE_FREQ, exclude=(SENTENCE_FREQ,)),
    )


def layerwise_z_scores(activations):
    """Per-layer (z_sentential, z_phrasal), each shape [layers]."""
    scores = np.array([target_z_scores(s) for s in layer_signals(activations)])
    return scores[:, 0], scores[:, 1]


def load_activations(activations_dir):
    """Arrays found, keyed by (model, corpus), and a list of the missing ones."""
    activations_dir = Path(activations_dir)
    found, missing = {}, []
    for model_key in MODELS:
        for corpus_key in CORPORA:
            path = activations_dir / model_key / f"{corpus_key}.npy"
            if path.exists():
                found[model_key, corpus_key] = np.load(path)
            else:
                missing.append(f"{model_key}/{corpus_key}")
    return found, missing


# --- style -------------------------------------------------------------------

def sans_family():
    """The first font of FONT_STACK this machine actually carries."""
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in FONT_STACK if name in installed),
                FONT_STACK[-1])


def styled_pyplot():
    """`matplotlib.pyplot` with the figure fonts set and fonts embedded as Type 42."""
    import matplotlib
    import matplotlib.pyplot as plt

    family = sans_family()
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_STACK),
        # Set mathtext in the text font; the default fontset uses DejaVu.
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
    })
    return plt


# --- shared layout -----------------------------------------------------------

def control_break(corpora):
    """Row index where the control row starts; None if it is absent or first."""
    if CONTROL_ROW not in corpora:
        return None
    return corpora.index(CONTROL_ROW) or None


def grid_layout(n_rows, n_columns, with_icons=True, gap_before=None):
    """Geometry for the panel grid, with an icon column when decorated."""
    header = HEADER_BAND if with_icons else HEADER_BAND_BARE
    gap = GROUP_GAP if gap_before else 0.0
    grid = ROW_HEIGHT * n_rows + ROW_GAP * (n_rows - 1)
    fig_height = grid + gap + header + FOOTER_BAND
    grid_top = 1.0 - header / fig_height
    width_ratios = ([ICON_RATIO] if with_icons else []) + [1.0] * n_columns
    return {
        "figsize": (PRINT_WIDTH, fig_height),
        "gridspec_kw": {
            "width_ratios": width_ratios,
            "wspace": 0.5,
            # hspace is a fraction of the row height.
            "hspace": ROW_GAP / ROW_HEIGHT,
        },
        "subplots_adjust": {
            "left": 0.02 if with_icons else 0.10,
            "right": 0.965,
            "top": grid_top,
            "bottom": FOOTER_BAND / fig_height,
        },
        "legend_y": grid_top + 0.34 / fig_height,
    }


def open_grid(n_rows, n_columns, with_icons=True, gap_before=None):
    plt = styled_pyplot()

    layout = grid_layout(n_rows, n_columns, with_icons, gap_before)
    fig, axes = plt.subplots(
        n_rows,
        n_columns + (1 if with_icons else 0),
        squeeze=False,
        figsize=layout["figsize"],
        gridspec_kw=layout["gridspec_kw"],
    )
    fig.subplots_adjust(**layout["subplots_adjust"])
    if gap_before:
        apply_group_gap(axes, layout, gap_before)
    return fig, axes, layout


def apply_group_gap(axes, layout, gap_before):
    """Place rows by hand, pushing rows from `gap_before` down by GROUP_GAP."""
    fig_height = layout["figsize"][1]
    adjust = layout["subplots_adjust"]
    pitch = ROW_HEIGHT + ROW_GAP

    for i, row in enumerate(axes):
        top = adjust["top"] * fig_height - i * pitch
        if i >= gap_before:
            top -= GROUP_GAP
        for ax in row:
            box = ax.get_position()
            ax.set_position([box.x0, (top - ROW_HEIGHT) / fig_height,
                             box.width, ROW_HEIGHT / fig_height])


def group_labels(corpora):
    """(language, ink) for each block of rows, keyed by the block's first row."""
    gap_before = control_break(corpora)
    if gap_before is None:
        leads = bool(corpora) and corpora[0] == CONTROL_ROW
        return {0: ENGLISH_GROUP if leads else CHINESE_GROUP}
    return {0: CHINESE_GROUP, gap_before: ENGLISH_GROUP}


def label_groups(fig, corpora, frames):
    """Name each block's language at the top left of its first icon."""
    page_w = fig.get_figwidth() * 72.0
    page_h = fig.get_figheight() * 72.0
    texts = {}
    for row, (name, ink) in group_labels(corpora).items():
        x0, _, _, top = frames[row]
        x0 -= GROUP_LABEL_INDENT * 72.0
        texts[row] = fig.text(
            x0 / page_w, (top + GROUP_LABEL_PAD * 72.0) / page_h, name,
            color=ink, fontsize=GROUP_LABEL_FONTSIZE, ha="left", va="bottom",
        )
    return texts


def draw_group_rule(fig, frame, label=None):
    """Full-width rule just above the control row's icon, or `label` if higher."""
    from matplotlib.lines import Line2D

    page_h = fig.get_figheight() * 72.0
    ceiling = frame[3]
    if label is not None:
        fig.canvas.draw()
        extent = label.get_window_extent(fig.canvas.get_renderer())
        ceiling = max(ceiling, extent.y1 / fig.dpi * 72.0)
    y = (ceiling + GROUP_RULE_CLEARANCE * 72.0) / page_h
    fig.add_artist(Line2D([0.0, 1.0], [y, y], transform=fig.transFigure,
                          **GROUP_RULE))


def label_row(ax, corpus_key):
    ax.axis("off")
    label = CORPORA[corpus_key]["label"]
    if label:
        ax.text(
            0.5,
            ICON_AREA_BOTTOM - 0.06,
            label,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=ROW_LABEL_FONTSIZE,
        )
    return ax, CORPORA[corpus_key]["icon"]


def column_header(ax, model_key, pad):
    """Model name as the title, with a subtitle line of colored text runs.

    One Text cannot mix colors, so the subtitle is one TextArea per run.
    """
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    from matplotlib.transforms import ScaledTranslation

    runs = [
        TextArea(text, textprops={"color": color, "fontsize": SUBTITLE_FONTSIZE})
        for text, color in MODELS[model_key]["languages"]
    ]
    lifted = ax.transAxes + ScaledTranslation(
        0, pad / 72.0, ax.figure.dpi_scale_trans
    )
    ax.add_artist(AnchoredOffsetbox(
        loc="lower center", pad=0.0, borderpad=0.0, frameon=False,
        child=HPacker(children=runs, pad=0, sep=0, align="baseline"),
        bbox_to_anchor=(0.5, 1.0), bbox_transform=lifted,
    ))
    ax.set_title(
        MODELS[model_key]["label"],
        fontsize=TITLE_FONTSIZE,
        pad=pad + SUBTITLE_FONTSIZE + 2,
    )


def icon_frames(icon_axes, assets_dir):
    """Icon rectangles (x0, y0, x1, y1) in points, y measured up.

    All icons share one width and sit on the same baseline in their cells.
    The width is fitted to the tallest icon in CORPORA, not only in the rows
    drawn, so a subset figure keeps the same icon size.
    """
    import pymupdf

    fig = icon_axes[0][0].figure
    page_w = fig.get_figwidth() * 72.0
    page_h = fig.get_figheight() * 72.0
    pad = 0.04

    def aspect(icon_file):
        source = pymupdf.open(Path(assets_dir) / icon_file)
        ratio = source[0].rect.width / source[0].rect.height
        source.close()
        return ratio

    boxes = [ax.get_position() for ax, _ in icon_axes]
    aspects = [aspect(icon_file) for _, icon_file in icon_axes]
    catalogue = [entry["icon"] for entry in CORPORA.values()]
    tallest = min(aspect(icon_file) for icon_file in catalogue
                  if (Path(assets_dir) / icon_file).exists())

    cell_w = min(box.x1 - box.x0 for box in boxes) * page_w
    cell_h = min(box.y1 - box.y0 for box in boxes) * page_h
    gaps = [boxes[i].y0 - boxes[i + 1].y1 for i in range(len(boxes) - 1)]
    headroom = cell_h * (1 - ICON_AREA_BOTTOM)
    if gaps:
        headroom += ICON_BLEED * min(gaps) * page_h
    width = min(cell_w * (1 - 2 * pad), headroom * tallest)

    frames = []
    for box, aspect in zip(boxes, aspects):
        x0 = box.x0 * page_w + ((box.x1 - box.x0) * page_w - width) / 2
        baseline = (box.y0 + ICON_AREA_BOTTOM * (box.y1 - box.y0)) * page_h
        frames.append((x0, baseline, x0 + width, baseline + width / aspect))
    return frames


def place_vector_icons(pdf_path, fig, icon_axes, assets_dir, frames):
    """Embed the icon PDFs into their cells, keeping the figure fully vector.

    Call after savefig, which must run without bbox_inches="tight" so the page
    equals figsize*72pt and axis positions map linearly to page coordinates.
    """
    import pymupdf

    page_h = fig.get_figheight() * 72.0
    document = pymupdf.open(pdf_path)
    page = document[0]

    for (x0, y0, x1, y1), (_, icon_file) in zip(frames, icon_axes):
        source = pymupdf.open(Path(assets_dir) / icon_file)
        page.show_pdf_page(
            pymupdf.Rect(x0, page_h - y1, x1, page_h - y0), source, 0
        )
        source.close()
    document.save(pdf_path, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
    document.close()


def bare_axes(ax):
    """Strip a panel to the data and a plain frame."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", length=2.5, width=0.6, labelsize=TICK_FONTSIZE)


# --- Fig. 2: aggregate spectra -----------------------------------------------

def draw_spectrum(ax, freqs, mags, z_scores, ylim_top, decorated=True,
                  show_xticklabels=True, show_exponent=True):
    """One spectrum panel."""
    from matplotlib.ticker import MaxNLocator

    if decorated:
        for freq, color, _ in TARGETS:
            ax.axvspan(freq - BAND_HALFWIDTH, freq + BAND_HALFWIDTH,
                       color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.fill_between(freqs[1:], 0, mags[1:], color=SPECTRUM_FILL, lw=0, zorder=3)
    ax.plot(freqs[1:], mags[1:], color="black", lw=0.9, zorder=4)
    if decorated:
        for (freq, color, _), z in zip(TARGETS, z_scores):
            if z > Z_THRESHOLD:
                ax.plot(freq, 1.0, transform=ax.get_xaxis_transform(),
                        color=color, mec=color, mfc=color,
                        clip_on=False, zorder=6, **MARKER_GLYPH)

    # The curve ends at the Nyquist limit (0.5); pad by the band half-width so
    # the phrasal band there is not clipped.
    ax.set_xlim(0, PHRASE_FREQ + BAND_HALFWIDTH)
    ax.set_ylim(0, ylim_top)
    ax.set_xticks([freq for freq, _, _ in XTICKS])
    if show_xticklabels:
        ax.set_xticklabels([text for _, _, text in XTICKS], fontsize=TICK_FONTSIZE)
        if decorated:
            for label, (_, color, _) in zip(ax.get_xticklabels(), XTICKS):
                if color is None:
                    continue
                label.set_color(color)
                label.set_fontweight("bold")
    else:
        ax.set_xticklabels([])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(TICK_FONTSIZE - 0.5)
    ax.yaxis.get_offset_text().set_visible(show_exponent)
    bare_axes(ax)


def make_global_figure(path, activations, z_scores, assets_dir, decorated=True,
                       corpora=None):
    plt = styled_pyplot()
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    corpora = tuple(corpora or CORPORA)
    spectra = {key: spectrum(global_signal(a)) for key, a in activations.items()}
    # y limits from the rows drawn only.
    ylim_top = {
        model_key: 1.2 * max(
            spectra[model_key, corpus_key][1][1:].max()
            for corpus_key in corpora if (model_key, corpus_key) in spectra
        )
        for model_key in MODELS
    }

    gap_before = control_break(corpora)
    fig, axes, layout = open_grid(len(corpora), len(MODELS),
                                  with_icons=decorated, gap_before=gap_before)
    icon_axes = []
    offset = 1 if decorated else 0
    for i, corpus_key in enumerate(corpora):
        if decorated:
            icon_axes.append(label_row(axes[i][0], corpus_key))
        for j, model_key in enumerate(MODELS):
            ax = axes[i][j + offset]
            if (model_key, corpus_key) not in spectra:
                ax.axis("off")
                continue
            last_row = i == len(corpora) - 1
            freqs, mags = spectra[model_key, corpus_key]
            draw_spectrum(ax, freqs, mags, z_scores[model_key, corpus_key],
                          ylim_top[model_key], decorated,
                          show_xticklabels=last_row or not decorated,
                          show_exponent=i == 0 or not decorated)
            if decorated:
                if i == 0:
                    column_header(ax, model_key, pad=12)
                if last_row:
                    ax.set_xlabel(FREQUENCY_LABEL, fontsize=LABEL_FONTSIZE)
                if j == 0 and i == len(corpora) // 2:
                    ax.set_ylabel("Magnitude", fontsize=LABEL_FONTSIZE)

    frames = icon_frames(icon_axes, assets_dir) if icon_axes else []
    if frames:
        labels = label_groups(fig, corpora, frames)
        if gap_before:
            draw_group_rule(fig, frames[gap_before], labels.get(gap_before))

    if decorated:
        handles = [
            Patch(facecolor=to_rgba(color, BAND_ALPHA), lw=0, label=label)
            for color, label in ((SENTENCE_COLOR, SENTENCE_LEGEND),
                                 (PHRASE_COLOR, PHRASE_LEGEND))
        ]
        handles.append(
            Line2D([], [], color="none", mec=MARKER_COLOR, mfc=MARKER_COLOR,
                   label=Z_MARKER_LEGEND, **MARKER_GLYPH)
        )
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(LEGEND_X, layout["legend_y"]), ncol=2,
                   frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=1.6,
                   columnspacing=1.4, labelspacing=0.5)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if frames:
        place_vector_icons(path, fig, icon_axes, assets_dir, frames)
    plt.close(fig)
    print(f"wrote {path}")


# --- Fig. 1: the concept figure's bare panel ---------------------------------

def make_bare_spectrum(path, activations, model_key, corpus_key,
                       size=PRESHRINK_PANEL):
    """One spectrum with the two bands only, scaled to its own peak."""
    plt = styled_pyplot()

    freqs, mags = spectrum(global_signal(activations[model_key, corpus_key]))
    fig = plt.figure(figsize=size)
    # Inset slightly so the spines are not clipped at the page edge.
    ax = fig.add_axes((0.015, 0.015, 0.97, 0.97))

    for freq, color, _ in TARGETS:
        ax.axvspan(freq - BARE_BAND_HALFWIDTH, freq + BARE_BAND_HALFWIDTH,
                   color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.plot(freqs[1:], mags[1:], color="black", lw=BARE_CURVE_WIDTH, zorder=4)

    # As in the grid: pad past the Nyquist limit by the band half-width.
    ax.set_xlim(0, PHRASE_FREQ + BARE_BAND_HALFWIDTH)
    ax.set_ylim(0, 1.1 * mags[1:].max())
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(BARE_SPINE_WIDTH)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# --- Fig. 3: layer-wise responses --------------------------------------------

def draw_layerwise(ax, z_sentence, z_phrase, ylim, decorated=True):
    from matplotlib.ticker import MaxNLocator

    layers = np.arange(len(z_sentence))
    style = (
        {"color": SENTENCE_COLOR, "marker": "o"} if decorated
        else {"color": "black", "marker": "o"}
    )
    style_p = (
        {"color": PHRASE_COLOR, "marker": "x"} if decorated
        else {"color": "0.45", "marker": "x"}
    )
    line_s, = ax.plot(layers, z_sentence, markersize=1.5, lw=0.8,
                      label=SENTENCE_LEGEND, **style)
    line_p, = ax.plot(layers, z_phrase, markersize=2, lw=0.8,
                      label=PHRASE_LEGEND, **style_p)
    threshold = ax.axhline(y=Z_THRESHOLD, color="0.35", linestyle="--", lw=0.8,
                           label=THRESHOLD_LEGEND)

    ax.set_ylim(ylim)
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if decorated:
        ax.grid(True, alpha=0.3)
    bare_axes(ax)
    return line_s, line_p, threshold


def layerwise_ylims(layerwise, corpora):
    return {
        model_key: (
            -2,
            max(4.0, 1.08 * max(
                max(z_s.max(), z_p.max())
                for corpus_key in corpora
                if (model_key, corpus_key) in layerwise
                for z_s, z_p in [layerwise[model_key, corpus_key]]
            )),
        )
        for model_key in MODELS
    }


def make_layerwise_figure(path, activations, assets_dir, decorated=True,
                          corpora=None):
    plt = styled_pyplot()

    corpora = tuple(corpora or CORPORA)
    layerwise = {key: layerwise_z_scores(a) for key, a in activations.items()}
    # y limits from the rows drawn only.
    ylims = layerwise_ylims(layerwise, corpora)

    gap_before = control_break(corpora)
    fig, axes, layout = open_grid(len(corpora), len(MODELS),
                                  with_icons=decorated, gap_before=gap_before)
    icon_axes = []
    offset = 1 if decorated else 0
    handles = None
    for i, corpus_key in enumerate(corpora):
        if decorated:
            icon_axes.append(label_row(axes[i][0], corpus_key))
        for j, model_key in enumerate(MODELS):
            ax = axes[i][j + offset]
            if (model_key, corpus_key) not in layerwise:
                ax.axis("off")
                continue
            z_sentence, z_phrase = layerwise[model_key, corpus_key]
            handles = draw_layerwise(ax, z_sentence, z_phrase,
                                     ylims[model_key], decorated)
            if decorated:
                if i == 0:
                    column_header(ax, model_key, pad=3)
                if i == len(corpora) - 1:
                    ax.set_xlabel(LAYER_LABEL, fontsize=LABEL_FONTSIZE)
                if j == 0 and i == len(corpora) // 2:
                    ax.set_ylabel(ZSCORE_LABEL, fontsize=LABEL_FONTSIZE)

    frames = icon_frames(icon_axes, assets_dir) if icon_axes else []
    if frames:
        labels = label_groups(fig, corpora, frames)
        if gap_before:
            draw_group_rule(fig, frames[gap_before], labels.get(gap_before))

    if decorated and handles:
        fig.legend(handles=list(handles), loc="lower center",
                   bbox_to_anchor=(LEGEND_X, layout["legend_y"]), ncol=2,
                   frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=1.6,
                   columnspacing=1.4, labelspacing=0.5)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if frames:
        place_vector_icons(path, fig, icon_axes, assets_dir, frames)
    plt.close(fig)
    print(f"wrote {path}")


def make_layerwise_panels(directory, activations):
    """One undecorated file per model x corpus, before any tiling."""
    plt = styled_pyplot()

    layerwise = {key: layerwise_z_scores(a) for key, a in activations.items()}
    ylims = layerwise_ylims(layerwise, tuple(CORPORA))

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for (model_key, corpus_key), (z_sentence, z_phrase) in layerwise.items():
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        draw_layerwise(ax, z_sentence, z_phrase, ylims[model_key], decorated=False)
        slug = corpus_key.replace("+", "-")
        path = directory / f"layerwise_{model_key}_{slug}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path.name)
    print(f"wrote {len(written)} panels to {directory}")
    return sorted(written)


def make_spectrum_panels(directory, activations, z_scores):
    """One undecorated spectrum per model x corpus, before any tiling."""
    plt = styled_pyplot()

    spectra = {key: spectrum(global_signal(a)) for key, a in activations.items()}
    ylim_top = {
        model_key: 1.2 * max(
            spectra[model_key, corpus_key][1][1:].max()
            for corpus_key in CORPORA if (model_key, corpus_key) in spectra
        )
        for model_key in MODELS
    }

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for (model_key, corpus_key), (freqs, mags) in spectra.items():
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        draw_spectrum(ax, freqs, mags, z_scores[model_key, corpus_key],
                      ylim_top[model_key], decorated=False)
        slug = corpus_key.replace("+", "-")
        path = directory / f"spectrum_{model_key}_{slug}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path.name)
    print(f"wrote {len(written)} panels to {directory}")
    return sorted(written)


# --- entry point -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True,
                        help="directory holding <model>/<corpus>.npy")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--assets", help="directory holding the icon PDFs "
                                         "(required unless --style base)")
    parser.add_argument("--style", choices=("reference", "base", "both"),
                        default="reference")
    parser.add_argument("--panels", action="store_true",
                        help="also emit one undecorated file per panel")
    parser.add_argument("--bare-spectrum", metavar="MODEL:CORPUS",
                        help="also emit one spectrum carrying the two bands "
                             "and nothing else, for the concept figure")
    parser.add_argument("--corpora",
                        help="comma-separated subset of "
                             f"{','.join(CORPORA)}, in the order to draw them "
                             "(default: all of them)")
    args = parser.parse_args()

    corpora = tuple(CORPORA)
    if args.corpora:
        corpora = tuple(key.strip() for key in args.corpora.split(","))
        unknown = [key for key in corpora if key not in CORPORA]
        if unknown:
            raise SystemExit(f"unknown corpus: {', '.join(unknown)}")

    activations, missing = load_activations(args.activations)
    if not activations:
        raise SystemExit(f"no activation arrays under {args.activations}")
    if missing:
        print(f"WARNING: {len(missing)} missing, figures are partial: "
              f"{', '.join(missing)}")

    z_scores = {key: target_z_scores(global_signal(a))
                for key, a in activations.items()}
    out = Path(args.out)
    # The grid size in the filename keeps a subset from overwriting the full figure.
    grid = f"{len(MODELS)}x{len(corpora)}"

    if args.style in ("reference", "both"):
        if not args.assets:
            raise SystemExit("--assets is required for the reference style")
        make_global_figure(out / f"global_{grid}.pdf", activations, z_scores,
                           args.assets, decorated=True, corpora=corpora)
        make_layerwise_figure(out / f"layerwise_{grid}.pdf", activations,
                              args.assets, decorated=True, corpora=corpora)
    if args.style in ("base", "both"):
        make_global_figure(out / f"global_{grid}_base.pdf", activations,
                           z_scores, None, decorated=False, corpora=corpora)
        make_layerwise_figure(out / f"layerwise_{grid}_base.pdf", activations,
                              None, decorated=False, corpora=corpora)
    if args.bare_spectrum:
        model_key, _, corpus_key = args.bare_spectrum.partition(":")
        if (model_key, corpus_key) not in activations:
            raise SystemExit(f"no activations for {args.bare_spectrum!r}; "
                             f"expected MODEL:CORPUS")
        slug = corpus_key.replace("+", "-")
        make_bare_spectrum(out / f"bare_{model_key}_{slug}.pdf",
                           activations, model_key, corpus_key)
    if args.panels:
        make_spectrum_panels(out / "panels", activations, z_scores)
        make_layerwise_panels(out / "panels", activations)

    (out / "z_scores.json").write_text(json.dumps(
        [{"model": m, "corpus": c, "z_sentential": round(z[0], 3),
          "z_phrasal": round(z[1], 3)}
         for (m, c), z in z_scores.items()], indent=2))
    print(f"wrote {out / 'z_scores.json'}")


if __name__ == "__main__":
    main()
