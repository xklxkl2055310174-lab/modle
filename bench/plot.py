import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="仅从真实 CSV 生成图表")
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = Path(args.directory)
    summary = pd.read_csv(directory / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(summary["case"], summary.success_qps, color="#2563eb")
    axes[0].set_ylabel("Successful requests / second")
    axes[0].tick_params(axis="x", rotation=80)
    axes[1].scatter(summary.success_qps, summary.accepted_p99_ms, color="#059669")
    for _, row in summary.iterrows():
        axes[1].annotate(row["case"], (row.success_qps, row.accepted_p99_ms), fontsize=7)
    axes[1].set(xlabel="Successful requests / second", ylabel="Successful P99 (ms)")
    fig.suptitle("Measured throughput / tail latency (see environment.json)")
    fig.tight_layout()
    fig.savefig(directory / "throughput-latency.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    for resources in directory.glob("*/resources.csv"):
        data = pd.read_csv(resources)
        if data.empty:
            continue
        elapsed = data.timestamp - data.timestamp.iloc[0]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(elapsed, data.rss_bytes / 1024**2, label="RSS MiB")
        ax.set(xlabel="Elapsed seconds", ylabel="Process-tree RSS (MiB)")
        twin = ax.twinx()
        twin.plot(elapsed, data.cpu_percent, color="#dc2626", label="CPU %")
        twin.set_ylabel("CPU % (100 = one core)")
        fig.tight_layout()
        fig.savefig(resources.with_suffix(".png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
