import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from app.api.errors import Failure
from app.models.base import ModelRunner

_model_lock = threading.Lock()


@dataclass
class Work:
    data: bytes
    top_k: int
    submitted: float
    deadline: float
    prepared: Any = None
    preprocess_ms: float = 0


@dataclass
class Result:
    predictions: list[dict]
    batch_size: int
    queue_ms: float
    preprocess_ms: float
    inference_ms: float


def is_oom(exc: Exception) -> bool:
    # DummyRunner 可用同名异常离线模拟 CUDA OOM，无需导入 torch。
    return type(exc).__name__ == "OutOfMemoryError"


def execute_batch(runner: ModelRunner, items: list[Work]) -> list[Result | Failure]:
    """工作线程/子进程执行；逐条解码隔离坏样本；批失败仅重试一次单条。"""
    outputs: list[Result | Failure] = [Failure(504, "timeout", "请求已过期") for _ in items]
    valid = []
    for index, item in enumerate(items):
        start = time.monotonic()
        if start >= item.deadline:
            continue
        try:
            value = item.prepared if item.prepared is not None else runner.preprocess(item.data)
            prep_ms = item.preprocess_ms + (time.monotonic() - start) * 1000
            valid.append((index, value, prep_ms, (start - item.submitted) * 1000))
        except ValueError as exc:
            outputs[index] = Failure(400, "invalid_image", str(exc))
        except Exception:
            outputs[index] = Failure(500, "preprocess_failed", "图片预处理失败")

    def infer(group, may_retry=True, elapsed_ms=0.0):
        group = [x for x in group if time.monotonic() < items[x[0]].deadline]
        if not group:
            return
        started = time.monotonic()
        try:
            guard = _model_lock if runner.settings.exp_global_lock else nullcontext()
            with guard:
                predictions = runner.infer_batch([x[1] for x in group])
            duration = elapsed_ms + (time.monotonic() - started) * 1000
            if len(predictions) != len(group):
                raise RuntimeError("模型输出数量错误")
            for (index, _, prep, queue), prediction in zip(group, predictions, strict=True):
                outputs[index] = Result(
                    runner.postprocess(prediction, items[index].top_k),
                    len(group),
                    queue,
                    prep,
                    duration,
                )
        except Exception as exc:
            elapsed_ms += (time.monotonic() - started) * 1000
            if is_oom(exc) and len(group) > 1:
                runner.clear_cache()
                half = len(group) // 2
                infer(group[:half], False, elapsed_ms)
                infer(group[half:], False, elapsed_ms)
            elif may_retry:
                for sample in group:
                    infer([sample], False, elapsed_ms)
            else:
                for index, *_ in group:
                    outputs[index] = Failure(500, "inference_failed", "模型推理失败")

    infer(valid)
    return outputs
