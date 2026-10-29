import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="模型基准 CSV 绘图")
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = Path(args.directory)
    data = pd.read_csv(directory / "model.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(data.batch_size, data.sample_ms, "o-", color="#2563eb")
    axes[0].set(
        xlabel="Batch size", ylabel="Milliseconds per sample", xscale="log", title="Latency"
    )
    axes[1].plot(data.batch_size, data.samples_per_second, "o-", color="#059669")
    axes[1].set(xlabel="Batch size", ylabel="Samples / second", xscale="log", title="Throughput")
    for axis in axes:
        axis.set_xticks(data.batch_size, data.batch_size)
        axis.grid(alpha=0.15)
    fig.suptitle(f"{data.model.iloc[0]} on {data.device.iloc[0]} — measured model execution")
    fig.tight_layout()
    fig.savefig(directory / "model.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
