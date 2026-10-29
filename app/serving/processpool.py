import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from app.config import Settings
from app.models import create_runner
from app.serving.batcher import BatchService
from app.serving.execution import execute_batch

_runner = None
_startup_barrier = None


def initialize_process(config, barrier):
    """spawn 子进程初始化器：每个进程独立加载、设置 torch 线程并预热。"""
    global _runner, _startup_barrier
    _runner = create_runner(Settings(**config))
    _runner.warmup()
    _startup_barrier = barrier


def process_ready():
    # 屏障确保所有进程都加载并预热后，父进程才能报告 ready。
    _startup_barrier.wait(timeout=_runner.settings.startup_timeout_s)
    return os.getpid(), _runner.device, _runner.num_classes


def process_execute(items):
    return execute_batch(_runner, items)


class ProcessPoolService(BatchService):
    def __init__(self, settings, metrics, runner=None):
        if runner is not None:
            raise ValueError("processpool 通过 MODEL_NAME 选择模型，不传递父进程模型对象")
        super().__init__(settings, metrics)
        self.worker_count = settings.process_workers
        self.max_batch = 1

    async def initialize_executor(self):
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(self.worker_count)
        self.executor = ProcessPoolExecutor(
            max_workers=self.worker_count,
            mp_context=context,
            initializer=initialize_process,
            initargs=(self.settings.model_dump(), barrier),
        )
        loop = asyncio.get_running_loop()
        infos = await asyncio.gather(
            *[loop.run_in_executor(self.executor, process_ready) for _ in range(self.worker_count)]
        )
        self.device, self.num_classes = infos[0][1:]
        self.worker_pids = [x[0] for x in infos]

    async def execute(self, work):
        # 只传原始图片字节和标量元数据；不在进程间传输大张量。
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, process_execute, work
        )
