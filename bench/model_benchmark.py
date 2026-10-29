import argparse
import statistics
import time
from pathlib import Path

import psutil

from app.config import Settings
from app.models import create_runner
from bench.common import environment, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(
        description="模型执行基准：预处理不计入，含 H2D/D2H 和 softmax"
    )
    parser.add_argument("--image", default="reports/sample.png")
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="reports/runs/model")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("iterations/warmup 必须为正整数")
    settings = Settings()
    runner = create_runner(settings)
    sample = runner.preprocess(Path(args.image).read_bytes())
    output = Path(args.output)
    write_json(
        output / "environment.json",
        {**environment(settings), "arguments": vars(args), "resolved_device": runner.device},
    )
    rows, raw = [], []
    for size in [int(x) for x in args.batches.split(",")]:
        if size < 1:
            parser.error("批大小必须为正整数")
        if runner.device == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            runner.infer_batch([sample] * size)
        durations = []
        for iteration in range(args.iterations):
            start = time.perf_counter()
            runner.infer_batch([sample] * size)
            duration = (time.perf_counter() - start) * 1000
            durations.append(duration)
            raw.append({"batch_size": size, "iteration": iteration, "latency_ms": duration})
        latency = statistics.mean(durations)
        rows.append(
            {
                "model": settings.model_name,
                "device": runner.device,
                "batch_size": size,
                "batch_ms": latency,
                "sample_ms": latency / size,
                "samples_per_second": size * 1000 / latency,
                "rss_bytes": psutil.Process().memory_info().rss,
                "gpu_peak_bytes": torch.cuda.max_memory_allocated()
                if runner.device == "cuda"
                else "",
            }
        )
        # 每个点落盘，后续 OOM 不会抹掉已经真实测得的数据。
        write_csv(output / "model.csv", rows)
        write_csv(output / "raw.csv", raw)
        print(rows[-1])


if __name__ == "__main__":
    main()
