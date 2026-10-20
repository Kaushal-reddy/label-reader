"""Stage 8: analysis. Turn the Stage 7 measurements into printed tables,
confusion matrices and figures.

Run with:  python analyze.py   (after evaluate.py; from the repo folder, with the venv active)

evaluate.py does the slow part (every stage on every photo, timed 3 times).
This script only reads its output/evaluation/per_photo.csv, so it runs in a
couple of seconds and can be re-run freely. If barcode_reader.py has been run,
its variant matrix is plotted too.

Every figure has a table with the same numbers (printed, saved as CSV, and in
report.md), so nothing can only be read off a picture. With 12 photos one
photo moves a rate by 8 points, so rates come with 95% Wilson intervals: the
range the true rate could plausibly be in, given this few photos.

Outputs in output/analysis/:
  *.png       the figures
  *.csv       the table behind each figure
  report.md   every table and figure on one page
"""

import math
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")   # draw to files only: needs no window system (works over SSH, in CI)
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb  # noqa: E402

EVAL_CSV = os.path.join("output", "evaluation", "per_photo.csv")
VARIANTS_CSV = os.path.join("output", "barcode", "variant_matrix.csv")
OUT_DIR = os.path.join("output", "analysis")

# ---------- look ----------

# A validated colour-blind-safe palette. Series colours are used in this fixed
# order; green/red are kept for "correct"/"wrong" only, always with a ✓/✗ or a
# word next to them so colour is never the only cue.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
# One hue, light -> dark, for "how many / how much" in heatmaps.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#eef4fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    # DejaVu Sans ships with matplotlib, so figures look the same on every OS,
    # and it has the ✓ ✗ ∅ glyphs (Helvetica Neue, Segoe UI and Arial lack some).
    "font.family": "DejaVu Sans",
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def tint(color: str, amount: float) -> tuple[float, float, float]:
    """color mixed with the background: amount 1 = full colour, 0 = background."""
    c, s = np.array(to_rgb(color)), np.array(to_rgb(SURFACE))
    return tuple(s + amount * (c - s))


PASS_FAIL = ListedColormap([tint(CRITICAL, 0.28), tint(GOOD, 0.28)])   # 0 = fail, 1 = pass


# ---------- numbers ----------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n.

    The usual p +- 1.96*sqrt(p(1-p)/n) breaks down for small n and for rates
    near 0% or 100% (it would claim 6/6 = 100% +- 0). Wilson's doesn't.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)" if n else "n/a"


def align(true: str, read: str) -> list[tuple[str, str]]:
    """Line two strings up character by character using the fewest edits.

    Same edit distance as ocr_reader.levenshtein, but it keeps the whole table
    so it can walk back through it and say WHICH edits: returns
    (true char, read char) pairs, with "" where OCR missed a character (read
    side) or added one (true side).
    """
    n, m = len(true), len(read)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0], d[0, :] = range(n + 1), range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (true[i - 1] != read[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (true[i - 1] != read[j - 1]):
            pairs.append((true[i - 1], read[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            pairs.append((true[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", read[j - 1]))
            j -= 1
    return pairs[::-1]


# ---------- loading ----------

def load() -> pd.DataFrame:
    if not os.path.exists(EVAL_CSV):
        sys.exit(f"{EVAL_CSV} not found. Run evaluate.py first; this script analyses its results.")
    # keep_default_na=False: an empty cell (e.g. no serial found) stays "", not NaN.
    df = pd.read_csv(EVAL_CSV, keep_default_na=False)
    if "s5_combined_closest" not in df.columns:
        sys.exit(f"{EVAL_CSV} is from an older evaluate.py. Re-run evaluate.py, then this script.")

    # Photo names follow <device>_<number>[_<condition>], e.g. Laptop_2_Glare.jpeg.
    df["photo"] = df.filename.str.replace(r"\.[^.]+$", "", regex=True)
    parts = df.photo.str.split("_")
    df["device"] = parts.str[0]
    df["condition"] = (parts.str[2:].str.join(" ").str.lower()
                       .replace({"": "straight-on", "sideangle": "side angle"}))

    # Derived pass/fail columns, so every measure below is just a True/False column.
    df["s4_crop_found"] = df.s4_n_crops > 0
    df["s5_combined_near"] = df.s5_combined_wrong <= 1
    df["s6_accepted"] = df.s6_status == "accepted"
    df["s6_accepted_correct"] = df.s6_accepted & df.s6_correct
    return df


# (column, label) for every pass/fail measure, in pipeline order.
MEASURES = [
    ("s2_barcode_ok",          "Barcode, raw photo",            "S2"),
    ("s2_ocr_ok",              "OCR, raw photo",                "S2"),
    ("s3_barcode_ok",          "Barcode, retries + type filter", "S3"),
    ("s4_crop_found",          "Label crop found",              "S4"),
    ("s4_barcode_on_crops_ok", "Barcode on the crops",          "S4"),
    ("s5_whole_ok",            "OCR whole crop, exact",         "S5"),
    ("s5_lines_ok",            "OCR line by line, exact",       "S5"),
    ("s5_combined_ok",         "OCR combined, exact",           "S5"),
    ("s5_combined_near",       "OCR combined, <=1 char wrong",  "S5"),
    ("s6_accepted_correct",    "Accepted and correct",          "S6"),
    ("s6_correct",             "Best guess correct (incl. review)", "S6"),
]


# ---------- drawing helpers ----------

class Report:
    """Collects tables and figures; prints tables as it goes; writes report.md at the end."""

    def __init__(self, n: int) -> None:
        self.md = [f"# Analysis\n\nFrom `{EVAL_CSV}` ({n} photos). Re-create with `python analyze.py`.\n"]

    def section(self, title: str, note: str = "") -> None:
        print(f"\n=== {title} ===")
        if note:
            print(note)
        self.md += [f"\n## {title}\n", note + "\n" if note else ""]

    def table(self, df: pd.DataFrame, name: str, index: bool = False, show: bool = True) -> None:
        if show:
            print(df.to_string(index=index))
        df.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=index)
        t = df.reset_index() if index else df
        self.md.append("| " + " | ".join(map(str, t.columns)) + " |")
        self.md.append("|" + "---|" * len(t.columns))
        self.md += ["| " + " | ".join(map(str, row)) + " |" for row in t.itertuples(index=False)]
        self.md.append("")

    def text(self, line: str) -> None:
        print(line)
        self.md.append(line + "  ")

    def figure(self, fig: plt.Figure, name: str) -> None:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[figure] {os.path.join(OUT_DIR, name)}.png")
        self.md.append(f"\n![{name}]({name}.png)\n")

    def save(self) -> None:
        with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.md) + "\n")


def heatmap(ax: plt.Axes, values: np.ndarray, rows: list[str], cols: list[str],
            labels: list[list[str]], cmap=BLUES, vmin: float | None = None,
            vmax: float | None = None, dark_text_above: float = 0.6) -> None:
    """Coloured grid with a text label in every cell and thin background-coloured gaps."""
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            on_dark = cmap is not PASS_FAIL and im.norm(values[i, j]) > dark_text_above
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9,
                    color="white" if on_dark else INK)


def pass_fail_grid(ax: plt.Axes, ok: np.ndarray, rows: list[str], cols: list[str]) -> None:
    """Green ✓ / red ✗ grid. The glyph carries the meaning; the tint only helps scanning."""
    labels = [["✓" if v else "✗" for v in row] for row in ok]
    heatmap(ax, ok.astype(float), rows, cols, labels, cmap=PASS_FAIL, vmin=0, vmax=1)


def legend_above(ax: plt.Axes, ncol: int) -> None:
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, fontsize=8,
              handlelength=1.2, borderaxespad=0.3)


# ---------- the analyses ----------

def stage_accuracy(df: pd.DataFrame, rep: Report) -> None:
    n = len(df)
    rep.section("Accuracy per stage", "Share of photos where the serial was read correctly at each stage.")
    rows = []
    for col, label, stage in MEASURES:
        k = int(df[col].sum())
        lo, hi = wilson(k, n)
        rows.append({"stage": stage, "measure": label, "correct": k, "of": n,
                     "rate_%": round(100 * k / n), "ci95_low_%": round(100 * lo), "ci95_high_%": round(100 * hi)})
    t = pd.DataFrame(rows)
    rep.table(t, "stage_accuracy")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(t))[::-1]
    rates = t["rate_%"] / 100
    err = [rates - t["ci95_low_%"] / 100, t["ci95_high_%"] / 100 - rates]
    ax.barh(y, rates, height=0.55, color=SERIES[0])
    ax.errorbar(rates, y, xerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    for yi, hi, k in zip(y, t["ci95_high_%"] / 100, t.correct):
        ax.text(hi + 0.02, yi, f"{k}/{n}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [f"{r.stage}  {r.measure}" for r in t.itertuples()])
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{p:.0%}" for p in np.linspace(0, 1, 6)])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Accuracy per stage  (n = {n} photos)")
    ax.text(0, -0.13, "Whiskers: 95% Wilson interval. Labels show photos correct; "
            "whiskers end where they do, not at the label.", transform=ax.transAxes,
            fontsize=7.5, color=MUTED)
    rep.figure(fig, "stage_accuracy")


def photo_outcomes(df: pd.DataFrame, rep: Report) -> None:
    rep.section("Per-photo outcomes", "✓ (figure) / '#' (printed) = serial read correctly at that "
                "stage. Read across a row to see where a photo is lost.")
    cols = [c for c, _, _ in MEASURES]
    names = [f"{s} {label}" for _, label, s in MEASURES]
    t = df[["photo"] + cols].copy()
    t.columns = ["photo"] + names
    rep.table(t.replace({True: "yes", False: "no"}), "photo_outcomes", show=False)
    # Too wide for a terminal with full names, so print it like barcode_reader.py does.
    print(f"{'':20}" + " ".join(f"{i:>2}" for i in range(len(cols))))
    for r in df.itertuples():
        print(f"{r.photo:20}" + " ".join(f"{'#' if getattr(r, c) else '.':>2}" for c in cols))
    print("'#' = correct.  Columns: " + ", ".join(f"{i}={name}" for i, name in enumerate(names)))

    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(df) + 2.6))
    pass_fail_grid(ax, df[cols].to_numpy(), list(df.photo), names)
    ax.set_xticklabels(names, rotation=40, ha="left", rotation_mode="anchor", fontsize=8)
    ax.set_title("Per-photo outcome at each stage", pad=10)
    rep.figure(fig, "photo_outcomes")


def decision_confusion(df: pd.DataFrame, rep: Report) -> None:
    """2x2: what the pipeline decided vs whether its serial was actually right."""
    acc, ok = df.s6_accepted, df.s6_correct
    tp, fp = int((acc & ok).sum()), int((acc & ~ok).sum())
    fn, tn = int((~acc & ok).sum()), int((~acc & ~ok).sum())
    n = len(df)
    rep.section("Final decision confusion matrix",
                "Positive = the pipeline's serial (or best guess) is correct. "
                "Accepting is the pipeline saying 'this is right'.")
    t = pd.DataFrame({"best guess correct": [tp, fn], "best guess wrong / none": [fp, tn]},
                     index=["accepted", "needs review"])
    rep.table(t, "decision_confusion", index=True)
    rep.text(f"Precision (accepted that are right):         {rate(tp, tp + fp)}")
    rep.text(f"Recall / coverage (right guesses accepted):  {rate(tp, tp + fn)}")
    rep.text(f"Wrong guesses caught (sent to review):       {rate(tn, tn + fp)}")
    rep.text(f"Automation rate (accepted, no human):        {rate(tp + fp, n)}")
    rep.text(f"Best guess correct, any decision:            {rate(tp + fn, n)}")
    if tp + fp and tp + fn:
        p, r = tp / (tp + fp), tp / (tp + fn)
        rep.text(f"F1:                                          {2 * p * r / (p + r):.2f}" if p + r else "F1: 0")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    vals = np.array([[tp, fp], [fn, tn]])
    tags = [["correct, accepted", "WRONG, accepted"], ["correct, sent to review", "wrong, sent to review"]]
    labels = [[f"{vals[i, j]}  ({100 * vals[i, j] / n:.0f}%)\n{tags[i][j]}" for j in range(2)] for i in range(2)]
    heatmap(ax, vals, ["Accepted", "Needs review"], ["Best guess correct", "Best guess wrong / none"],
            labels, vmin=0, vmax=n)
    ax.set_title("Decision vs truth", pad=24)
    rep.figure(fig, "decision_confusion")


def source_agreement(df: pd.DataFrame, rep: Report) -> None:
    """Do barcode and OCR fail on the same photos? If not, combining them pays off."""
    rep.section("Barcode vs OCR", "Stage 3 barcode against Stage 5 OCR (combined): "
                "where they fail on different photos, one rescues the other.")
    no_text = df.s5_combined_closest == ""
    ocr_bucket = np.select([df.s5_combined_wrong == 0, df.s5_combined_wrong <= 2, ~no_text],
                           ["exact", "1-2 chars off", "3+ chars off"], "no text read")
    cols = ["exact", "1-2 chars off", "3+ chars off", "no text read"]
    t = (pd.crosstab(np.where(df.s3_barcode_ok, "barcode read serial", "barcode failed"), ocr_bucket)
         .reindex(index=["barcode read serial", "barcode failed"], columns=cols, fill_value=0))
    t.index.name, t.columns.name = "barcode \\ OCR", None
    rep.table(t, "source_agreement", index=True)
    both = int((df.s3_barcode_ok & df.s5_combined_ok).sum())
    either = int((df.s3_barcode_ok | df.s5_combined_ok).sum())
    rep.text(f"Both exact: {both}/{len(df)}.  At least one exact: {either}/{len(df)}.  "
             f"Barcode alone: {int(df.s3_barcode_ok.sum())}, OCR alone: {int(df.s5_combined_ok.sum())}.")

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    vals = t.to_numpy()
    heatmap(ax, vals, list(t.index), cols, [[str(v) for v in row] for row in vals], vmin=0, vmax=vals.max())
    ax.set_title("Barcode vs OCR on the same photo", pad=24)
    rep.figure(fig, "source_agreement")


def char_confusion(df: pd.DataFrame, rep: Report) -> None:
    """Which characters does OCR misread, and as what?"""
    rep.section("OCR character confusions",
                "Each true serial lined up against the closest stretch of OCR text (Stage 5, combined). "
                "∅ = missing (OCR skipped a character) or extra (OCR added one). "
                "Photos where OCR read no text at all are left out.")
    pairs, where = Counter(), {}
    used = df[df.s5_combined_closest != ""]
    for r in used.itertuples():
        for tc, rc in align(r.true_serial.upper(), r.s5_combined_closest):
            pairs[(tc, rc)] += 1
            if tc != rc:
                where.setdefault((tc, rc), []).append(r.photo)
    errors = {k: v for k, v in pairs.items() if k[0] != k[1]}
    total = sum(pairs.values())
    rep.text(f"{len(used)} photos, {total} aligned characters, {sum(errors.values())} errors "
             f"({100 * sum(errors.values()) / max(total, 1):.1f}%).")
    if not errors:
        return
    t = pd.DataFrame([{"true": tc or "∅", "read as": rc or "∅", "times": c, "photos": ", ".join(where[(tc, rc)])}
                      for (tc, rc), c in sorted(errors.items(), key=lambda kv: -kv[1])])
    rep.table(t, "char_confusions")

    # Matrix over just the characters involved in an error. Rows = true chars
    # that were misread, columns = what OCR read; the diagonal counts correct
    # reads of that char, so each row shows how often it goes wrong.
    true_side = sorted({tc for tc, _ in errors if tc})
    read_only = sorted({rc for _, rc in errors if rc} - set(true_side))
    rows = true_side + (["∅"] if any(tc == "" for tc, _ in errors) else [])
    cols = true_side + read_only + (["∅"] if any(rc == "" for _, rc in errors) else [])
    key = {c: c for c in true_side + read_only} | {"∅": ""}
    counts = np.array([[pairs.get((key[a], key[b]), 0) for b in cols] for a in rows])
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)   # each row sums to 1
    labels = [[str(v) if v else "" for v in row] for row in counts]
    fig, ax = plt.subplots(figsize=(0.5 * len(cols) + 1.8, 0.5 * len(rows) + 1.6))
    heatmap(ax, share, rows, cols, labels, vmin=0, vmax=1)
    ax.set_xlabel("read by OCR as")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("true character")
    ax.set_title("OCR character confusion matrix", pad=34)
    ax.text(0, -0.08, "Cell = number of times; colour = share of that row "
            "(how often the true character was read that way).",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
    rep.figure(fig, "char_confusion")


def ocr_methods(df: pd.DataFrame, rep: Report) -> None:
    rep.section("OCR methods: characters wrong per photo",
                "0 = exact. A missing or junk read counts as the whole serial wrong.")
    methods = [("s5_whole_wrong", "whole crop"), ("s5_lines_wrong", "line by line"),
               ("s5_combined_wrong", "combined")]
    t = df[["photo"] + [c for c, _ in methods]].copy()
    t.columns = ["photo"] + [m for _, m in methods]
    rep.table(t, "ocr_chars_wrong")

    vals = t[[m for _, m in methods]].to_numpy()
    fig, ax = plt.subplots(figsize=(4.6, 0.36 * len(df) + 1.4))
    heatmap(ax, vals, list(t.photo), [m for _, m in methods], [[str(v) for v in row] for row in vals],
            vmin=0, vmax=max(vals.max(), 1), dark_text_above=0.5)
    ax.set_title("OCR characters wrong (0 = exact)", pad=24)
    rep.figure(fig, "ocr_chars_wrong")

    # How many photos would count as read if we tolerated k wrong characters?
    ks = list(range(6))
    curve = pd.DataFrame({"chars_allowed": ks} |
                         {m: [int((df[c] <= k).sum()) for k in ks] for c, m in methods})
    rep.section("OCR tolerance curve", "Photos whose serial OCR got within k characters.")
    rep.table(curve, "ocr_tolerance")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for color, (_, m) in zip(SERIES, methods):
        # Lines often coincide, so no end labels (they'd collide); the legend names them.
        ax.plot(ks, curve[m], color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=m)
    ax.set_xticks(ks)
    ax.set_ylim(0, len(df) + 0.5)
    ax.set_xlabel("characters wrong allowed")
    ax.set_ylabel(f"photos (of {len(df)})")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    legend_above(ax, 3)
    ax.set_title("OCR: photos read within k characters", pad=22)
    rep.figure(fig, "ocr_tolerance")


def acceptance_policies(df: pd.DataFrame, rep: Report) -> None:
    """What if the acceptance rule were looser? Same reads, different rule."""
    rep.section("Acceptance policies",
                "Each policy accepts what the one above it does, plus one more kind of evidence. "
                "Uses the serial validate.py chose, so only the rule changes.")

    def evidence(r) -> int:
        if r.s6_status == "accepted":
            return 0                                   # barcode and OCR agree
        if r.s6_source == "barcode":
            return 1                                   # barcode alone (it has a checksum)
        if r.s6_source == "ocr" and r.s6_serial in r.s6_ocr_labelled.split():
            return 2                                   # OCR, written after an SN key
        if r.s6_source == "ocr":
            return 3                                   # any serial-shaped OCR text
        return 4                                       # nothing found
    level = np.array([evidence(r) for r in df.itertuples()])
    names = ["barcode and OCR agree (current)", "+ barcode alone", "+ OCR after an SN key",
             "+ any serial-shaped OCR"]
    rows = []
    for k, name in enumerate(names):
        take = level <= k
        good, bad = int((take & df.s6_correct).sum()), int((take & ~df.s6_correct).sum())
        rows.append({"policy": name, "accepted": good + bad, "correct": good, "wrong": bad,
                     "precision_%": round(100 * good / (good + bad)) if good + bad else None,
                     "to_review": len(df) - good - bad})
    t = pd.DataFrame(rows)
    rep.table(t, "acceptance_policies")

    fig, ax = plt.subplots(figsize=(7, 2.8))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.correct, height=0.55, color=GOOD, label="accepted, correct ✓", edgecolor=SURFACE, linewidth=1.5)
    ax.barh(y, t.wrong, left=t.correct, height=0.55, color=CRITICAL, label="accepted, WRONG ✗",
            edgecolor=SURFACE, linewidth=1.5)
    for yi, r in zip(y, t.itertuples()):
        ax.text(r.accepted + 0.15, yi, f"{r.accepted} accepted, {r.wrong} wrong", va="center",
                fontsize=8, color=INK_2)
    ax.set_yticks(y, t.policy)
    ax.set_xlim(0, len(df) + 3.5)
    ax.set_xticks(range(0, len(df) + 1, 2))
    ax.set_xlabel(f"photos (of {len(df)})")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    legend_above(ax, 2)
    ax.set_title("Looser acceptance rules: more automation vs wrong serials saved", pad=22)
    rep.figure(fig, "acceptance_policies")


def by_condition(df: pd.DataFrame, rep: Report) -> None:
    rep.section("By capture condition and device",
                "Conditions come from the file names (<device>_<n>_<condition>). Groups are tiny: "
                "read these as hints, not rates.")
    measures = [("s3_barcode_ok", "barcode"), ("s5_combined_ok", "OCR exact"), ("s6_accepted", "accepted")]
    for col_name, fname in (("condition", "by_condition"), ("device", "by_device")):
        g = df.groupby(col_name)
        t = pd.DataFrame({"photos": g.size()} | {m: g[c].sum().astype(int) for c, m in measures})
        rep.table(t, fname, index=True)
        if col_name != "condition":
            continue
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x = np.arange(len(t))
        w = 0.26
        for i, (color, (_, m)) in enumerate(zip(SERIES, measures)):
            ax.bar(x + (i - 1) * w, t[m] / t.photos, width=w - 0.03, color=color, label=m)
            # k/n on every bar: groups are tiny, and a 0 bar is otherwise invisible.
            for xi, k, total in zip(x, t[m], t.photos):
                ax.text(xi + (i - 1) * w, k / total + 0.02, f"{k}/{total}", ha="center",
                        fontsize=7, color=INK_2)
        ax.set_xticks(x, [f"{c}\n(n={k})" for c, k in zip(t.index, t.photos)])
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.linspace(0, 1, 5), [f"{p:.0%}" for p in np.linspace(0, 1, 5)])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        legend_above(ax, 3)
        ax.set_title("Success rate by capture condition", pad=22)
        rep.figure(fig, "by_condition")


def timing(df: pd.DataFrame, rep: Report) -> None:
    steps = [("t_label_find", "label finding"), ("t_barcode", "barcode"), ("t_ocr", "OCR")]
    rep.section("Time per photo (seconds, median of evaluate.py's runs)",
                "Label finding (~0.01 s) is too thin to see in the chart; decision time "
                "(< 0.01 s) is left out.")
    keys = [k for k, _ in steps] + ["t_decide", "t_total"]
    t = pd.DataFrame({"mean": df[keys].mean(), "median": df[keys].median(),
                      "p90": df[keys].quantile(0.9), "max": df[keys].max()}).round(2)
    t["share_of_total_%"] = (100 * df[keys].sum() / df.t_total.sum()).round(0)
    t.index = [k.replace("t_", "") for k in keys]
    rep.table(t, "timing_summary", index=True)

    d = df.sort_values("t_total")
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(d) + 1.6))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for color, (k, label) in zip(SERIES, steps):
        ax.barh(y, d[k], left=left, height=0.6, color=color, label=label, edgecolor=SURFACE, linewidth=1.5)
        left += d[k].to_numpy()
    for yi, total in zip(y, d.t_total):
        ax.text(total + 0.05, yi, f"{total:.1f}s", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, d.photo)
    ax.set_xlim(0, d.t_total.max() * 1.15)
    ax.set_xlabel("seconds")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    legend_above(ax, 3)
    ax.set_title("Where the time goes, per photo", pad=22)
    rep.figure(fig, "timing")


def failures(df: pd.DataFrame, rep: Report) -> None:
    rep.section("Photos sent to review, and where they were lost")
    rev = df[~df.s6_accepted]
    t = pd.DataFrame({
        "photo": rev.photo,
        "label crop": np.where(rev.s4_crop_found, "found", "NOT found"),
        "barcode": np.where(rev.s3_barcode_ok, "read", "failed"),
        "OCR chars wrong": rev.s5_combined_wrong,
        "best guess": np.where(rev.s6_correct, "correct", "wrong / none"),
        "reason": rev.s6_reason,
    })
    rep.table(t, "review_photos")


def barcode_variants(rep: Report) -> None:
    if not os.path.exists(VARIANTS_CSV):
        rep.section("Barcode variants")
        rep.text(f"Skipped: {VARIANTS_CSV} not found (run barcode_reader.py to add this figure).")
        return
    m = pd.read_csv(VARIANTS_CSV)
    variants = [c for c in m.columns if c != "filename"]
    rep.section("Barcode variants", "Which image variant let ZBar read the serial (barcode_reader.py).")
    totals = pd.DataFrame({"variant": variants, "photos_read": [int(m[v].sum()) for v in variants],
                           "only_this_variant": [int((m[v] & (m[variants].sum(axis=1) == 1)).sum())
                                                 for v in variants]})
    rep.table(totals, "barcode_variants")
    rows = list(m.filename.str.replace(r"\.[^.]+$", "", regex=True))
    cols = [f"{v}\n{int(m[v].sum())}/{len(m)}" for v in variants]
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(m) + 1.8))
    pass_fail_grid(ax, m[variants].to_numpy(), rows, cols)
    ax.set_xticklabels(cols, rotation=40, ha="left", rotation_mode="anchor", fontsize=8)
    ax.set_title("Barcode read per image variant", pad=10)
    rep.figure(fig, "barcode_variants")


def main() -> None:
    df = load()
    os.makedirs(OUT_DIR, exist_ok=True)
    rep = Report(len(df))
    stage_accuracy(df, rep)
    photo_outcomes(df, rep)
    decision_confusion(df, rep)
    acceptance_policies(df, rep)
    source_agreement(df, rep)
    ocr_methods(df, rep)
    char_confusion(df, rep)
    barcode_variants(rep)
    by_condition(df, rep)
    timing(df, rep)
    failures(df, rep)
    rep.save()
    print(f"\nAll tables and figures: {os.path.join(OUT_DIR, 'report.md')}")


if __name__ == "__main__":
    main()
