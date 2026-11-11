"""Bar charts of actual vs. approximated WER/CER from `results summarize` (Figs. 2, 4, 5)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ACTUAL_COLOR = "#0072B2"
APPROX_COLOR = "#D55E00"


def plot_actual_vs_approximated(
    summary: pd.DataFrame, metric: str, panels: list[str], panel_by: str, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_by = "dataset" if panel_by == "model" else "model"
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(max(5, 0.45 * summary[x_by].nunique() + 2), 3.5 * len(panels)),
        squeeze=False,
    )
    for ax, panel in zip(axes[:, 0], panels, strict=True):
        rows = summary[summary[panel_by] == panel].sort_values(x_by)
        if rows.empty:
            raise ValueError(f"No rows for {panel_by}={panel} in the summary")
        x = np.arange(len(rows))
        width = 0.35
        ax.bar(
            x - width / 2,
            rows[f"{metric}_mean"],
            width=width,
            color=ACTUAL_COLOR,
            hatch="///",
            edgecolor="black",
            zorder=3,
            label=f"Actual {metric}",
        )
        ax.bar(
            x + width / 2,
            rows[f"a{metric}_mean"],
            width=width,
            color=APPROX_COLOR,
            hatch="...",
            edgecolor="black",
            zorder=3,
            label=f"Approximated {metric}",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(rows[x_by], fontsize=11, rotation=60, ha="right")
        ax.set_ylabel(f"{metric} (%)", fontsize=12)
        ax.set_title(panel, fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=0)
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("runs/results/summary.csv"))
    parser.add_argument("--metric", choices=["WER", "CER"], default="WER")
    parser.add_argument("--panel-by", choices=["model", "dataset"], default="model")
    parser.add_argument("--panels", nargs="+", required=True, help="Models or datasets to plot")
    parser.add_argument(
        "--include", nargs="+", default=None, help="Restrict the x axis to these names"
    )
    parser.add_argument("--output", type=Path, default=Path("runs/results/actual_vs_approx.pdf"))
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    if args.include:
        x_by = "dataset" if args.panel_by == "model" else "model"
        summary = summary[summary[x_by].isin(args.include)]
    plot_actual_vs_approximated(summary, args.metric, args.panels, args.panel_by, args.output)


if __name__ == "__main__":
    main()
