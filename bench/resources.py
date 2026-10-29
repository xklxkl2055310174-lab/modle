import argparse
import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil


class ResourceSampler:
    """后台采样线程：跟踪父进程及 spawn 子进程，RSS 之和可能重复计算共享页。"""

    def __init__(self, pid, output, interval=1):
        self.pid, self.output, self.interval = pid, Path(output), interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.error = None

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.error:
            raise self.error

    def run(self):
        try:
            self._sample()
        except Exception as exc:
            self.error = exc

    def _sample(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        processes = {}
        with self.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "timestamp",
                    "cpu_percent",
                    "rss_bytes",
                    "gpu_memory_bytes",
                    "gpu_util_percent",
                ],
            )
            writer.writeheader()
            while not self.stop_event.is_set():
                try:
                    parent = psutil.Process(self.pid)
                    current = {
                        p.pid: processes.get(p.pid, p)
                        for p in [parent, *parent.children(recursive=True)]
                    }
                except psutil.NoSuchProcess:
                    break
                cpu, memory = 0.0, 0
                for process in current.values():
                    try:
                        cpu += process.cpu_percent()
                        memory += process.memory_info().rss
                    except psutil.Error:
                        pass
                processes = current
                gpu_memory, gpu_util = "", ""
                executable = shutil.which("nvidia-smi")
                if executable:
                    try:
                        apps = subprocess.check_output(
                            [
                                executable,
                                "--query-compute-apps=pid,used_memory",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=2,
                        )
                        gpu_memory = sum(
                            float(line.split(",")[1]) * 1024**2
                            for line in apps.splitlines()
                            if int(line.split(",")[0]) in current
                        )
                        usage = subprocess.check_output(
                            [
                                executable,
                                "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=2,
                        )
                        gpu_util = max(map(float, usage.splitlines()), default=0)
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
                writer.writerow(
                    {
                        "timestamp": time.time(),
                        "cpu_percent": cpu,
                        "rss_bytes": memory,
                        "gpu_memory_bytes": gpu_memory,
                        "gpu_util_percent": gpu_util,
                    }
                )
                stream.flush()
                self.stop_event.wait(self.interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--output", default="reports/runs/resources.csv")
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("seconds/interval 必须为正数")
    sampler = ResourceSampler(args.pid, args.output, args.interval).start()
    try:
        time.sleep(args.seconds)
    finally:
        sampler.stop()


if __name__ == "__main__":
    main()
