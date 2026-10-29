import asyncio
import threading
import time

import pytest

from app.api.errors import Failure, ServiceError
from app.models.dummy import DummyRunner
from app.observability.metrics import Metrics
from app.serving.batcher import BatchService
from app.serving.execution import Work, execute_batch


class OutOfMemoryError(RuntimeError):
    pass


class ControlledRunner(DummyRunner):
    def __init__(self, settings):
        super().__init__(settings)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def warmup(self):
        return None

    def infer_batch(self, inputs):
        self.calls += 1
        self.entered.set()
        self.release.wait(2)
        return super().infer_batch(inputs)


async def test_queue_full_expiry_cancellation_and_recovery(settings, image_bytes):
    settings.max_batch_size, settings.queue_maxsize = 1, 1
    runner = ControlledRunner(settings)
    service = BatchService(settings, Metrics(), runner)
    await service.start()
    first = asyncio.create_task(service.predict(image_bytes, 2, time.monotonic() + 3))
    try:
        assert await asyncio.to_thread(runner.entered.wait, 1)
        second = asyncio.create_task(service.predict(image_bytes, 2, time.monotonic() + 0.03))
        await asyncio.sleep(0)
        with pytest.raises(ServiceError) as caught:
            await service.predict(image_bytes, 2, time.monotonic() + 3)
        assert caught.value.failure.code == "queue_full"
        with pytest.raises(ServiceError) as caught:
            await second
        assert caught.value.failure.status == 504
        runner.release.set()
        assert (await first).batch_size == 1
        await service.queue.join()
        assert runner.calls == 1  # 已过期请求没有执行模型。
        task = asyncio.create_task(service.predict(image_bytes, 1, time.monotonic() + 2))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await service.queue.join()
        result = await service.predict(image_bytes, 1, time.monotonic() + 2)
        assert len(result.predictions) == 1
    finally:
        runner.release.set()
        await service.stop()
    assert not service.pending


async def test_shutdown_deadline_fails_remaining(settings, image_bytes):
    settings.shutdown_timeout_s, settings.max_batch_size = 0.02, 1
    runner = ControlledRunner(settings)
    service = BatchService(settings, Metrics(), runner)
    await service.start()
    task = asyncio.create_task(service.predict(image_bytes, 1, time.monotonic() + 5))
    assert await asyncio.to_thread(runner.entered.wait, 1)
    try:
        await asyncio.wait_for(service.stop(), 0.5)
        with pytest.raises(ServiceError) as caught:
            await task
        assert caught.value.failure.status == 503
        with pytest.raises(ServiceError):
            await service.predict(image_bytes, 1, time.monotonic() + 5)
    finally:
        runner.release.set()


def work(data):
    return Work(data, 2, time.monotonic(), time.monotonic() + 10)


def test_bad_sample_isolation(settings, image_bytes):
    results = execute_batch(DummyRunner(settings), [work(image_bytes), work(b"bad")])
    assert results[0].batch_size == 1
    assert isinstance(results[1], Failure) and results[1].status == 400


def test_oom_halves_actual_batch_sizes(settings, image_bytes):
    class OOMRunner(DummyRunner):
        def infer_batch(self, inputs):
            if len(inputs) > 2:
                raise OutOfMemoryError("test")
            return super().infer_batch(inputs)

    results = execute_batch(OOMRunner(settings), [work(image_bytes) for _ in range(5)])
    assert [x.batch_size for x in results] == [2, 2, 1, 2, 2]


def test_generic_failure_retries_each_item_once(settings, image_bytes):
    class RetryRunner(DummyRunner):
        calls = 0

        def infer_batch(self, inputs):
            self.calls += 1
            if len(inputs) > 1 or inputs[0][0] == 0:
                raise RuntimeError("bad sample")
            return super().infer_batch(inputs)

    runner = RetryRunner(settings)
    good, bad = work(image_bytes), work(image_bytes)
    good.prepared, bad.prepared = [10, 10, 10], [0, 0, 0]
    results = execute_batch(runner, [good, bad])
    assert results[0].batch_size == 1
    assert results[1].status == 500
    assert runner.calls == 3


def test_expired_before_preprocess(settings):
    item = work(b"bad")
    item.deadline = time.monotonic() - 1
    assert execute_batch(DummyRunner(settings), [item])[0].status == 504


def test_warmup_respects_small_pixel_limit(settings):
    settings.max_image_pixels = 1
    DummyRunner(settings).warmup()
