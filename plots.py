"""Matplotlib plots for the compression benchmark.

Three plots are produced:

1. compression_ratio.png — grouped bar chart of compression ratio (%) by
   algorithm and dataset. Lower is better.
2. ratio_vs_time.png — scatter of bits-per-base versus total encode+decode
   time, one point per (algorithm, dataset). The Pareto front in the
   bottom-left dominates.
3. ablation_contributions.png — average per-flag marginal contribution to
   compressed size for VCSD+ features.

All plots use ``matplotlib.use("Agg")`` so they render without a display.
"""
from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results"
PLOTS_DIR = RESULTS_DIR / "plots"


def _read_metrics_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_ablation_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_compression_ratio(metrics: Sequence[dict], output_path: Path) -> None:
    """Grouped bar chart: compression ratio per (algorithm, dataset)."""
    by_algorithm: Dict[str, Dict[str, float]] = defaultdict(dict)
    datasets: list[str] = []
    for row in metrics:
        ratio = float(row["compression_ratio_%"])
        by_algorithm[row["algorithm"]][row["dataset"]] = ratio
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    algorithms = sorted(by_algorithm.keys())
    n_groups = len(datasets)
    n_algos = len(algorithms)
    bar_width = 0.8 / max(n_algos, 1)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.4 * n_groups), 5.0))
    for i, algo in enumerate(algorithms):
        values = [by_algorithm[algo].get(dataset, 0.0) for dataset in datasets]
        positions = [j + i * bar_width for j in range(n_groups)]
        ax.bar(positions, values, width=bar_width, label=algo)

    ax.set_xticks([j + bar_width * (n_algos - 1) / 2 for j in range(n_groups)])
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Compression ratio (%, lower is better)")
    ax.set_title("Compression ratio across codecs and datasets")
    ax.axhline(y=25.0, color="gray", linewidth=0.8, linestyle="--",
               label="2-bit-pack floor (25%)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_ratio_vs_time(metrics: Sequence[dict], output_path: Path) -> None:
    """Scatter: bits-per-base vs total encode+decode time, colored by algorithm."""
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    by_algorithm: Dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in metrics:
        try:
            bpb = float(row["bits_per_base"])
            total_time = float(row["encode_time"]) + float(row["decode_time"])
        except (KeyError, ValueError):
            continue
        by_algorithm[row["algorithm"]].append((total_time, bpb))

    markers = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h")
    for marker, (algo, points) in zip(
        itertools.cycle(markers),
        sorted(by_algorithm.items()),
    ):
        xs, ys = zip(*points) if points else ([], [])
        ax.scatter(xs, ys, label=algo, s=60, alpha=0.8, marker=marker)

    ax.set_xscale("log")
    ax.set_xlabel("Total encode + decode time (seconds, log scale)")
    ax.set_ylabel("Bits per base (lower is better)")
    ax.axhline(y=2.0, color="gray", linewidth=0.8, linestyle="--",
               label="i.i.d. floor (2.0 bpb)")
    ax.set_title("Compression efficiency vs runtime")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_ablation_marginals(rows: Sequence[dict], output_path: Path) -> None:
    """Bar chart of average marginal contribution per VCSD+ flag, per dataset."""
    if not rows:
        return
    flags = ("sequence_ordering", "use_approximate", "use_reverse_complement", "use_token_entropy")
    by_dataset: Dict[str, Dict[str, float]] = defaultdict(dict)
    for dataset_name in {row["dataset"] for row in rows}:
        dataset_rows = [r for r in rows if r["dataset"] == dataset_name]
        for flag in flags:
            deltas = []
            for off_row in dataset_rows:
                if off_row.get(flag, "False") == "False":
                    other_flags = {f: off_row[f] for f in flags}
                    other_flags[flag] = "True"
                    on_match = next(
                        (r for r in dataset_rows
                         if all(r[f] == other_flags[f] for f in flags)),
                        None,
                    )
                    if on_match:
                        try:
                            off_size = int(off_row["compressed_bytes"])
                            on_size = int(on_match["compressed_bytes"])
                            if off_size > 0:
                                deltas.append((on_size - off_size) / off_size * 100.0)
                        except (ValueError, KeyError):
                            continue
            if deltas:
                by_dataset[dataset_name][flag] = sum(deltas) / len(deltas)

    if not by_dataset:
        return
    datasets = sorted(by_dataset.keys())
    n_groups = len(datasets)
    bar_width = 0.8 / len(flags)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.4 * n_groups), 5.0))
    for i, flag in enumerate(flags):
        values = [by_dataset[d].get(flag, 0.0) for d in datasets]
        positions = [j + i * bar_width for j in range(n_groups)]
        ax.bar(positions, values, width=bar_width, label=flag)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks([j + bar_width * (len(flags) - 1) / 2 for j in range(n_groups)])
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Mean marginal Δ size (%)\nnegative = improvement")
    ax.set_title("VCSD+ feature ablation: marginal contribution per flag")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    metrics_path = RESULTS_DIR / "performance_metrics.csv"
    if not metrics_path.exists():
        print(f"{metrics_path} missing; run comparison.py --full-pipeline first.")
        return
    metrics = _read_metrics_csv(metrics_path)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_compression_ratio(metrics, PLOTS_DIR / "compression_ratio.png")
    plot_ratio_vs_time(metrics, PLOTS_DIR / "ratio_vs_time.png")
    ablation_rows = _read_ablation_csv(RESULTS_DIR / "ablation_results.csv")
    if ablation_rows:
        plot_ablation_marginals(ablation_rows, PLOTS_DIR / "ablation_contributions.png")
    print(f"Wrote plots under {PLOTS_DIR}")


if __name__ == "__main__":
    main()
