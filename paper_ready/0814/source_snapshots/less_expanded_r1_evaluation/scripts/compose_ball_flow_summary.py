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
    # --gallery-view decides which gallery exists; fall back to whichever
    # one was rendered instead of assuming the side view is present.
    gallery, gallery_label = (
        ("raw_gallery.png", "(b) Raw side-view gallery")
        if (args.viz_dir / "raw_gallery.png").is_file()
        else ("raw_gallery_headon.png", "(b) Raw head-on gallery")
    )
    names = ("raw_crossing_fan.png", gallery,
             "raw_curves.png", "mechanism_multiview.png")
    labels = ("(a) Raw crossing fan", gallery_label,
              "(c) Raw temperature-1 metrics", "(d) Expansion mechanism")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    for axis, name, label in zip(axes.flat, names, labels):
        path = args.viz_dir / name
        if not path.is_file():
            axis.axis("off")
            continue
        axis.imshow(plt.imread(path))
        axis.set_title(label, fontsize=18, weight="bold", pad=22)
        axis.axis("off")
    fig.tight_layout(pad=2.0, h_pad=2.4)
    fig.savefig(args.output, dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
