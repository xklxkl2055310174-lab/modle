import json

import pandas as pd

from bench.summarize import summarize


def test_summary_includes_failures_and_real_measurement_window(tmp_path):
    pd.DataFrame({"success": [1, 0, 1, 0], "latency_ms": [100, 1, 200, 2]}).to_csv(
        tmp_path / "requests.csv",
        index=False,
    )
    pd.DataFrame({"cpu_percent": [0, 100, 200], "rss_bytes": [10, 20, 30]}).to_csv(
        tmp_path / "resources.csv",
        index=False,
    )
    (tmp_path / "requests.window.json").write_text(json.dumps({"elapsed_s": 2}))
    result = summarize(tmp_path, 1)
    assert result["qps"] == 2
    assert result["success_qps"] == 1
    assert result["failure_rate"] == 0.5
    assert result["cpu_mean_percent"] == 150
    assert result["accepted_p99_ms"] > result["p99_ms"]
