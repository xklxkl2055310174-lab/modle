import os
import shutil
import subprocess

import psutil
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        common = {"registry": self.registry}
        buckets = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
        self.requests = Counter(
            "inference_requests_total", "请求总数", ["route", "status"], **common
        )
        self.request_seconds = Histogram(
            "inference_request_seconds",
            "端到端耗时",
            ["route"],
            buckets=buckets,
            **common,
        )
        self.queue_seconds = Histogram(
            "inference_queue_seconds", "排队耗时", buckets=buckets, **common
        )
        self.inference_seconds = Histogram(
            "inference_execution_seconds",
            "每个成功请求所在批次的推理耗时",
            buckets=buckets,
            **common,
        )
        self.batch_size = Histogram(
            "inference_batch_size",
            "调度批大小",
            buckets=(1, 2, 4, 8, 16, 32, 64),
            **common,
        )
        self.queue_depth = Gauge("inference_queue_depth", "待调度队列深度", **common)
        self.inflight = Gauge("inference_inflight", "HTTP 在途请求", **common)
        self.rejections = Counter("inference_rejections_total", "拒绝数", ["reason"], **common)
        self.cpu = Gauge("inference_cpu_percent", "主进程及子进程 CPU，100% 为一核", **common)
        self.memory = Gauge("inference_memory_bytes", "主进程及子进程 RSS 之和", **common)
        self.gpu_memory = Gauge("inference_gpu_memory_bytes", "服务进程树 GPU 显存", **common)
        self.gpu_available = Gauge("inference_gpu_sample_available", "GPU 采样是否有效", **common)
        self.processes = {}

    def sample_resources(self):
        """默认工作线程采样进程树及可选 nvidia-smi，不阻塞事件循环。"""
        parent = psutil.Process(os.getpid())
        current = {p.pid: self.processes.get(p.pid, p) for p in [parent, *parent.children(True)]}
        cpu, memory = 0.0, 0
        for process in current.values():
            try:
                cpu += process.cpu_percent()
                memory += process.memory_info().rss
            except psutil.Error:
                pass
        self.processes = current
        self.cpu.set(cpu)
        self.memory.set(memory)
        executable = shutil.which("nvidia-smi")
        self.gpu_available.set(0)
        if executable:
            try:
                result = subprocess.run(
                    [
                        executable,
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                total = sum(
                    float(line.split(",")[1]) * 1024**2
                    for line in result.stdout.splitlines()
                    if int(line.split(",")[0]) in current
                )
                self.gpu_memory.set(total)
                self.gpu_available.set(1)
            except (ValueError, OSError, subprocess.SubprocessError):
                pass
