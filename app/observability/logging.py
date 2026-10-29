import json
import logging
import logging.handlers
import os
import queue
import sys
from datetime import UTC, datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False)


class BoundedQueueHandler(logging.handlers.QueueHandler):
    dropped = 0

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1


class DrainListener(logging.handlers.QueueListener):
    def enqueue_sentinel(self):
        self.queue.put(self._sentinel)


class SyncHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()
        os.fsync(self.stream.fileno())


class AccessLogger:
    def __init__(self, settings):
        self.logger = logging.Logger(f"inference.{id(self)}", settings.log_level)
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(JsonFormatter())
        self.queue_handler = BoundedQueueHandler(queue.Queue(settings.log_queue_size))
        self.logger.addHandler(self.queue_handler)
        self.listener = DrainListener(self.queue_handler.queue, stream)
        self.listener.start()
        self.sync = None
        if settings.exp_sync_log:
            Path(settings.exp_log_path).parent.mkdir(parents=True, exist_ok=True)
            self.sync = SyncHandler(settings.exp_log_path, encoding="utf-8")
            self.sync.setFormatter(JsonFormatter())
            self.logger.addHandler(self.sync)

    def access(self, **fields):
        self.logger.info("request", extra={"fields": fields})

    def close(self):
        self.listener.stop()
        if self.sync:
            self.sync.close()
