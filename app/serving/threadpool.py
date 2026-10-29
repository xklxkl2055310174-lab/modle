from app.serving.batcher import BatchService


class ThreadPoolService(BatchService):
    def __init__(self, settings, metrics, runner=None):
        super().__init__(settings, metrics, runner)
        self.worker_count = settings.threadpool_size
        self.max_batch = 1
