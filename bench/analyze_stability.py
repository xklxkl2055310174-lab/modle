import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bench.common import write_json


def main():
    parser = argparse.ArgumentParser(description="稳定性趋势证据；短时运行不能证明长期稳定")
    parser.add_argument("point", help="含 resources.csv/requests.csv 的单个测量目录")
    args = parser.parse_args()
    point = Path(args.point)
    resources = pd.read_csv(point / "resources.csv")
    requests = pd.read_csv(point / "requests.csv")
    if len(resources) < 3 or requests.empty:
        parser.error("样本不足")
    elapsed = resources.timestamp - resources.timestamp.iloc[0]
    slope = np.polyfit(elapsed, resources.rss_bytes / 1024**2, 1)[0]
    requests["minute"] = ((requests.timestamp - requests.timestamp.min()) // 60).astype(int)
    errors = requests.groupby("minute").success.agg(["count", "mean"])
    errors["failure_rate"] = 1 - errors["mean"]
    errors.to_csv(point / "errors-per-minute.csv")
    write_json(
        point / "stability.json",
        {
            "sampled_seconds": float(elapsed.iloc[-1]),
            "rss_trend_mib_per_minute": float(slope * 60),
            "rss_first_mib": float(resources.rss_bytes.iloc[0] / 1024**2),
            "rss_last_mib": float(resources.rss_bytes.iloc[-1] / 1024**2),
            "at_least_30_minutes": bool(elapsed.iloc[-1] >= 1800),
            "interpretation": (
                "结合每分钟错误率、内存曲线与分配器缓存解释；线性斜率不直接证明泄漏。"
            ),
        },
    )


if __name__ == "__main__":
    main()
