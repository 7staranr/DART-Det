"""DART framework schematic: diagnose -> decompose (three-way partition) -> repair.
A clean single-row pipeline figure for the principle section."""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(ROOT, "paper", "figures", "fig_dart.pdf")

fig, ax = plt.subplots(figsize=(11, 3.1))
ax.set_xlim(0, 11)
ax.set_ylim(0, 3.1)
ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                                fc=fc, ec="#333333", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.6, color="#333333"))


# Stage 1: input + ranked head
box(0.15, 1.1, 2.0, 0.95, "Dense scene\n+ NMS-free\ntop-$K$ head", "#dbe9f6")
arrow(2.2, 1.57, 2.9, 1.57)
# Stage 2: RTIL diagnose
box(2.95, 1.1, 2.05, 0.95, "RTIL\nrank-truncation\n= info loss", "#fde7c9")
arrow(5.05, 1.57, 5.75, 1.57)

# Stage 3: three-way partition (stacked bar)
ax.text(7.0, 2.92, "Three-way slot partition", ha="center", fontsize=10.0)
parts = [("positive (kept TP)", "#bfe3c0", 0.80),
         ("boundary: rank-displaced,\nrecoverable (51-56%)", "#ffe08a", 0.95),
         ("negative: head\nresidual (10-14%)", "#f4b8b8", 0.70)]
y0 = 0.35
for lbl, c, h in parts:
    ax.add_patch(FancyBboxPatch((5.8, y0), 2.4, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc=c, ec="#555555", lw=1.0))
    ax.text(7.0, y0 + h / 2, lbl, ha="center", va="center", fontsize=8.3)
    y0 += h
arrow(8.25, 1.57, 8.95, 1.57)

# Stage 4: DABA repair
box(9.0, 1.1, 1.85, 0.95, "DABA\ndensity$\\to K$\nrepair", "#d8d0ec")

ax.text(5.5, 0.12, r"diagnose $\;\longrightarrow\;$ decompose $\;\longrightarrow\;$ repair",
        ha="center", fontsize=10.5, style="italic", color="#444444")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
