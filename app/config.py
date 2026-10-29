import math
import os
from pathlib import Path
from typing import Literal

import psutil
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def available_cpus(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """启动线程调用：同时考虑 CPU affinity、cgroup v2/v1 与父级配额。"""
    count = float(os.cpu_count() or 1)
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    else:
        try:
            count = min(count, len(psutil.Process().cpu_affinity()))
        except (AttributeError, psutil.Error):
            pass
    roots = [cgroup_root]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            base = cgroup_root if not controllers else cgroup_root / "cpu"
            current = base / relative.lstrip("/")
            if current.is_relative_to(base):
                roots.extend([current, *[p for p in current.parents if p.is_relative_to(base)]])
    except (OSError, ValueError):
        pass
    for root in roots:
        try:
            quota, period = (root / "cpu.max").read_text().split()
            if quota != "max":
                count = min(count, int(quota) / int(period))
        except (OSError, ValueError, ZeroDivisionError):
            pass
        for location in (root, root / "cpu"):
            try:
                quota = int((location / "cpu.cfs_quota_us").read_text())
                period = int((location / "cpu.cfs_period_us").read_text())
                if quota > 0:
                    count = min(count, quota / period)
            except (OSError, ValueError, ZeroDivisionError):
                pass
    return max(1, math.floor(count))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    model_name: Literal["resnet18", "dummy"] = "resnet18"
    weights_path: str = "weights/resnet18-f37072fd.pth"
    serve_mode: Literal["threadpool", "processpool", "async_batch"] = "async_batch"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    torch_threads: str = "auto"
    threadpool_size: int = Field(4, ge=1, le=128)
    process_workers: int = Field(2, ge=1, le=64)
    max_batch_size: int = Field(16, ge=1, le=64)
    max_wait_ms: float = Field(5, ge=0, le=1000)
    queue_maxsize: int = Field(128, ge=1)
    request_timeout_ms: int = Field(5000, gt=0)
    shutdown_timeout_s: float = Field(15, gt=0)
    startup_timeout_s: float = Field(180, gt=0)
    max_upload_bytes: int = Field(5 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(20_000_000, gt=0)
    rate_limit_qps: float = Field(100, ge=0)
    rate_limit_burst: int = Field(200, ge=1)
    key_rate_limit_qps: float = Field(20, ge=0)
    key_rate_limit_burst: int = Field(40, ge=1)
    rate_limit_max_keys: int = Field(10000, ge=1)
    rate_limit_key_ttl_s: float = Field(300, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_queue_size: int = Field(10000, ge=1)
    warmup_batches: str = "1,2,4,8,16,32,64"
    warmup_runs: int = Field(2, ge=1)
    resource_interval_s: float = Field(1, gt=0)
    exp_sync_log: bool = False
    exp_log_path: str = "reports/sync-access.log"
    exp_global_lock: bool = False
    exp_inline_preprocess: bool = False
    exp_unbounded_queue: bool = False
    dummy_delay_ms: float = Field(0, ge=0)
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)

    @model_validator(mode="after")
    def validate_parameters(self):
        if self.torch_threads != "auto":
            try:
                if int(self.torch_threads) < 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("TORCH_THREADS 必须为 auto、0 或正整数") from exc
        try:
            if not all(1 <= int(x) <= 64 for x in self.warmup_batches.split(",")):
                raise ValueError
        except ValueError as exc:
            raise ValueError("WARMUP_BATCHES 必须为 1..64 的逗号分隔整数") from exc
        return self

    def thread_count(self) -> int:
        if self.torch_threads != "auto":
            return int(self.torch_threads)
        workers = self.process_workers if self.serve_mode == "processpool" else 1
        return max(1, available_cpus() // workers)

    def batch_sizes(self) -> list[int]:
        if self.serve_mode != "async_batch":
            return [1]
        return sorted(
            {self.max_batch_size, 1}
            | {int(x) for x in self.warmup_batches.split(",") if int(x) <= self.max_batch_size}
        )
