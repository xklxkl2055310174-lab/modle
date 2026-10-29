import csv
import threading
import time

import httpx


class HealthProbe:
    """独立线程探测事件循环响应，提供实验 D 的外部证据。"""

    def __init__(self, url, output):
        self.url, self.output = url, output
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=4)

    def run(self):
        with self.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["timestamp", "latency_ms", "status"])
            writer.writeheader()
            with httpx.Client(timeout=2, trust_env=False) as client:
                while not self.stop_event.is_set():
                    started = time.perf_counter()
                    try:
                        status = client.get(self.url).status_code
                    except httpx.HTTPError:
                        status = 0
                    writer.writerow(
                        {
                            "timestamp": time.time(),
                            "status": status,
                            "latency_ms": (time.perf_counter() - started) * 1000,
                        }
                    )
                    stream.flush()
                    self.stop_event.wait(0.2)
