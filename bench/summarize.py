import json
from pathlib import Path

import pandas as pd


def summarize(directory, duration):
    directory = Path(directory)
    data = pd.read_csv(directory / "requests.csv")
    resources = pd.read_csv(directory / "resources.csv")
    window = directory / "requests.window.json"
    if window.exists():
        duration = json.loads(window.read_text(encoding="utf-8"))["elapsed_s"]
    successful = data.loc[data.success == 1, "latency_ms"]
    return {
        "requests": len(data),
        "measured_seconds": duration,
        "qps": len(data) / duration,
        "success_qps": len(successful) / duration,
        "failure_rate": 1 - len(successful) / len(data) if len(data) else 0,
        "p50_ms": data.latency_ms.quantile(0.5),
        "p95_ms": data.latency_ms.quantile(0.95),
        "p99_ms": data.latency_ms.quantile(0.99),
        "accepted_p99_ms": successful.quantile(0.99),
        "cpu_mean_percent": resources.cpu_percent.iloc[1:].mean(),
        "rss_peak_bytes": resources.rss_bytes.max(),
    }
