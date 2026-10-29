import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.api.errors import Failure, ServiceError
from app.config import Settings
from app.models import create_runner
from app.serving.execution import Result, Work, execute_batch


@dataclass
class Pending:
    work: Work
    future: asyncio.Future


class BatchService:
    """事件循环只调度有界队列；解码、模型推理均在专用工作线程执行。"""

    worker_count = 1

    def __init__(self, settings: Settings, metrics, runner=None):
        self.settings = settings
        self.metrics = metrics
        self.runner = runner
        self.max_batch = settings.max_batch_size
        self.queue = asyncio.Queue(0 if settings.exp_unbounded_queue else settings.queue_maxsize)
        self.pending: dict[asyncio.Future, Pending] = {}
        self.tasks = []
        self.accepting = False
        self.executor = None
        self.device = "unknown"
        self.num_classes = 1000

    def initialize_runner(self):
        if self.runner is None:
            self.runner = create_runner(self.settings)
        self.runner.warmup()
        return self.runner.device, self.runner.num_classes

    async def initialize_executor(self):
        self.executor = ThreadPoolExecutor(
            max_workers=self.worker_count, thread_name_prefix="infer"
        )
        self.device, self.num_classes = await asyncio.get_running_loop().run_in_executor(
            self.executor,
            self.initialize_runner,
        )

    async def start(self):
        try:
            await asyncio.wait_for(self.initialize_executor(), self.settings.startup_timeout_s)
        except BaseException:
            if self.executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
            raise
        self.accepting = True
        self.tasks = [asyncio.create_task(self.worker()) for _ in range(self.worker_count)]

    async def predict(self, data: bytes, top_k: int, deadline: float) -> Result:
        if not self.accepting:
            raise ServiceError(503, "not_ready", "服务未就绪或正在停机")
        if time.monotonic() >= deadline:
            raise ServiceError(504, "timeout", "请求已过期")
        future = asyncio.get_running_loop().create_future()
        item = Pending(Work(data, top_k, time.monotonic(), deadline), future)
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise ServiceError(503, "queue_full", "请求队列已满") from exc
        self.pending[future] = item
        self.metrics.queue_depth.set(self.queue.qsize())
        try:
            return await asyncio.wait_for(future, max(0, deadline - time.monotonic()))
        except TimeoutError as exc:
            raise ServiceError(504, "timeout", "等待推理结果超时") from exc
        finally:
            self.pending.pop(future, None)

    async def execute(self, work):
        if self.settings.exp_inline_preprocess:
            # 实验 D：故意阻塞事件循环，仅用于定位预处理开销。
            for item in work:
                started = time.monotonic()
                try:
                    item.prepared = self.runner.preprocess(item.data)
                    item.preprocess_ms = (time.monotonic() - started) * 1000
                except ValueError:
                    pass  # 交给工作线程统一返回输入错误。
        return await asyncio.get_running_loop().run_in_executor(
            self.executor,
            execute_batch,
            self.runner,
            work,
        )

    async def worker(self):
        while True:
            group = []
            try:
                first = await self.queue.get()
                group.append(first)
                until = min(
                    first.work.deadline, time.monotonic() + self.settings.max_wait_ms / 1000
                )
                while len(group) < self.max_batch and not first.future.done():
                    remaining = until - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        try:
                            item = await asyncio.wait_for(self.queue.get(), remaining)
                        except TimeoutError:
                            break
                    group.append(item)
                    until = min(until, item.work.deadline)
                self.metrics.queue_depth.set(self.queue.qsize())
                active = [x for x in group if not x.future.done()]
                if not active:
                    continue
                results = await self.execute([x.work for x in active])
                # 直方图按一次原始调度记录；真实子批大小另在响应内返回。
                self.metrics.batch_size.observe(len(active))
                for item, result in zip(active, results, strict=True):
                    if item.future.done():
                        continue
                    if isinstance(result, Failure):
                        item.future.set_exception(
                            ServiceError(
                                result.status,
                                result.code,
                                result.message,
                            )
                        )
                    elif time.monotonic() >= item.work.deadline:
                        item.future.set_exception(ServiceError(504, "timeout", "推理超过截止时间"))
                    else:
                        self.metrics.queue_seconds.observe(result.queue_ms / 1000)
                        self.metrics.inference_seconds.observe(result.inference_ms / 1000)
                        item.future.set_result(result)
            except asyncio.CancelledError:
                raise
            except Exception:
                for item in group:
                    if not item.future.done():
                        item.future.set_exception(ServiceError(500, "inference_failed", "推理失败"))
            finally:
                for _ in group:
                    self.queue.task_done()

    def begin_shutdown(self):
        self.accepting = False

    async def stop(self):
        self.begin_shutdown()
        try:
            await asyncio.wait_for(self.queue.join(), self.settings.shutdown_timeout_s)
        except TimeoutError:
            pass
        for item in list(self.pending.values()):
            if not item.future.done():
                item.future.set_exception(ServiceError(503, "shutting_down", "服务停机超时"))
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        self.metrics.queue_depth.set(0)
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
