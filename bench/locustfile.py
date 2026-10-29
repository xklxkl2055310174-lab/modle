import csv
import json
import os
import time
from pathlib import Path

from locust import HttpUser, between, events, task

_stream = None
_writer = None
_started = None


@events.test_start.add_listener
def start_capture(environment, **kwargs):
    global _stream, _writer, _started
    _started = time.time()
    raw = os.getenv("BENCH_RAW_CSV")
    if raw:
        Path(raw).parent.mkdir(parents=True, exist_ok=True)
        _stream = open(raw, "w", newline="", encoding="utf-8")
        _writer = csv.DictWriter(
            _stream,
            fieldnames=[
                "timestamp",
                "name",
                "status",
                "latency_ms",
                "success",
                "batch_size",
            ],
        )
        _writer.writeheader()


@events.request.add_listener
def record_request(name, response_time, response=None, exception=None, **kwargs):
    if _writer:
        batch = ""
        if response is not None and response.status_code == 200:
            try:
                batch = response.json().get("batch_size", "")
            except ValueError:
                pass
        _writer.writerow(
            {
                "timestamp": time.time(),
                "name": name,
                "status": response.status_code if response is not None else 0,
                "latency_ms": response_time,
                "success": int(exception is None),
                "batch_size": batch,
            }
        )


@events.test_stop.add_listener
def stop_capture(environment, **kwargs):
    global _stream, _writer
    if _stream:
        Path(_stream.name).with_suffix(".window.json").write_text(
            json.dumps(
                {"started": _started, "ended": time.time(), "elapsed_s": time.time() - _started}
            ),
            encoding="utf-8",
        )
        _stream.close()
    _stream = _writer = None


class PredictionUser(HttpUser):
    wait_time = between(0, 0)

    def on_start(self):
        self.image = Path(os.getenv("BENCH_IMAGE", "reports/sample.png")).read_bytes()
        self.model = os.getenv("BENCH_MODEL", "resnet18")
        self.timeout = float(os.getenv("BENCH_TIMEOUT_S", "10"))

    @task
    def predict(self):
        # 不调用 SDK：关闭自动重试，测量真实服务失败率和延迟。
        with self.client.post(
            f"/v1/models/{self.model}/predict",
            files={"file": ("sample.png", self.image, "image/png")},
            data={"top_k": "5"},
            name="predict",
            timeout=self.timeout,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    body = response.json()
                    if (
                        len(body["predictions"]) != 5
                        or body["batch_size"] < 1
                        or not body["request_id"]
                        or "total_ms" not in body["timing"]
                    ):
                        response.failure("预测响应格式错误")
                except (ValueError, KeyError, TypeError):
                    response.failure("预测响应格式错误")
