import argparse
from pathlib import Path

import torch

from app.config import Settings
from app.models import create_runner
from bench.common import environment, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description="Top 15 算子与 Chrome trace")
    parser.add_argument("--image", default="reports/sample.png")
    parser.add_argument("--batches", default="1,16")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default="reports/runs/profile")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations 必须为正整数")
    settings = Settings()
    if settings.model_name != "resnet18":
        parser.error("算子分析仅适用于真实 ResNet18")
    runner = create_runner(settings)
    output = Path(args.output)
    write_json(output / "environment.json", environment(settings))
    sample = runner.preprocess(Path(args.image).read_bytes())
    activities = [torch.profiler.ProfilerActivity.CPU]
    if runner.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    for size in map(int, args.batches.split(",")):
        for _ in range(3):
            runner.infer_batch([sample] * size)
        with torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True
        ) as profile:
            for _ in range(args.iterations):
                runner.infer_batch([sample] * size)
        profile.export_chrome_trace(str(output / f"batch-{size}.json"))
        events = profile.key_averages()
        total_cpu = sum(event.self_cpu_time_total for event in events)
        total_device = sum(getattr(event, "self_device_time_total", 0) for event in events)
        for device, total, attr in [
            ("cpu", total_cpu, "self_cpu_time_total"),
            ("device", total_device, "self_device_time_total"),
        ]:
            if total <= 0:
                continue
            rows = [
                {
                    "operator": event.key,
                    "calls": event.count,
                    "self_time_us": getattr(event, attr, 0),
                    "share_percent": getattr(event, attr, 0) / total * 100,
                }
                for event in sorted(events, key=lambda x: getattr(x, attr, 0), reverse=True)[:15]
            ]
            write_csv(output / f"batch-{size}-{device}.csv", rows)


if __name__ == "__main__":
    main()
