#!/usr/bin/env python3
"""Plot PER vs noise level from aggregated metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="PER vs Noise")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Path(args.metrics).open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    clean_per = metrics["clean"]["per"]
    by_snr = metrics["by_snr"]
    snr_values = sorted((int(k), v["per"]) for k, v in by_snr.items())

    labels = ["clean"] + [str(snr) for snr, _ in sorted(snr_values, reverse=True)]
    values = [clean_per] + [per for _, per in sorted(snr_values, reverse=True)]

    x = list(range(len(labels)))

    plt.figure(figsize=(9, 5.5))
    plt.plot(x, values, marker="o", linewidth=2.2, color="#1f77b4")
    for xi, yi in zip(x, values):
        plt.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, labels)
    plt.ylim(0.0, min(1.0, max(values) + 0.08))
    plt.xlabel("Condition (clean then SNR dB)")
    plt.ylabel("PER")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)


if __name__ == "__main__":
    main()
