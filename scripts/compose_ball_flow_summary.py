"""Compose the current ball-flow still-image artifacts into one inspection sheet."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viz-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ("raw_crossing_fan.png", "raw_gallery.png",
             "raw_curves.png", "mechanism_multiview.png")
    labels = ("(a) Raw crossing fan", "(b) Raw side-view gallery",
              "(c) Raw temperature-1 metrics", "(d) Expansion mechanism")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    for axis, name, label in zip(axes.flat, names, labels):
        axis.imshow(plt.imread(args.viz_dir / name))
        axis.set_title(label, fontsize=18, weight="bold", pad=22)
        axis.axis("off")
    fig.tight_layout(pad=2.0, h_pad=2.4)
    fig.savefig(args.output, dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
