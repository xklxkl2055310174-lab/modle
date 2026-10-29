import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

from app.config import Settings
from bench.common import environment, write_csv, write_json
from bench.health_probe import HealthProbe
from bench.resources import ResourceSampler
from bench.summarize import summarize


def cases(scenario):
    if scenario == "modes":
        for mode in ("threadpool", "processpool", "async_batch"):
            for concurrency in (1, 8, 32, 64, 128):
                yield f"{mode}-c{concurrency}", concurrency, {"SERVE_MODE": mode}
    elif scenario == "batch":
        for batch in (1, 4, 8, 16, 32, 64):
            for wait in (0, 1, 5, 10, 20):
                yield (
                    f"batch{batch}-wait{wait}",
                    64,
                    {
                        "SERVE_MODE": "async_batch",
                        "MAX_BATCH_SIZE": str(batch),
                        "MAX_WAIT_MS": str(wait),
                    },
                )
    elif scenario == "overload":
        for limited in (False, True):
            for bounded in (False, True):
                yield (
                    f"limit{int(limited)}-bound{int(bounded)}",
                    128,
                    {
                        "RATE_LIMIT_QPS": "20" if limited else "0",
                        "RATE_LIMIT_BURST": "40",
                        "QUEUE_MAXSIZE": "16",
                        "EXP_UNBOUNDED_QUEUE": str(not bounded).lower(),
                        "SERVE_MODE": "async_batch",
                    },
                )
    elif scenario == "stability":
        yield "stability", 32, {"SERVE_MODE": "async_batch"}
    elif scenario == "bottlenecks":
        for name, mode, switch, before, after in [
            ("A", "processpool", "TORCH_THREADS", "0", "auto"),
            ("B", "async_batch", "EXP_SYNC_LOG", "1", "0"),
            ("C", "threadpool", "EXP_GLOBAL_LOCK", "1", "0"),
            ("D", "async_batch", "EXP_INLINE_PREPROCESS", "1", "0"),
        ]:
            for label, value in (("before", before), ("after", after)):
                yield f"{name}-{label}", 32, {"SERVE_MODE": mode, switch: value}
    elif scenario == "smoke":
        for mode in ("threadpool", "processpool", "async_batch"):
            yield mode, 4, {"SERVE_MODE": mode}


def set_affinity(pid, cpus):
    if cpus:
        psutil.Process(pid).cpu_affinity([int(x) for x in cpus.split(",")])


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()
    try:
        process.wait(timeout=35)
    except subprocess.TimeoutExpired:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.Error:
                pass
        process.kill()
        process.wait(timeout=10)


def run_locust(args, env, users, seconds, prefix, cpus):
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "bench/locustfile.py",
        "--headless",
        "--host",
        f"http://127.0.0.1:{args.port}",
        "-u",
        str(users),
        "-r",
        str(users * 10),
        "--run-time",
        f"{seconds}s",
        "--stop-timeout",
        "10",
        "--csv",
        str(prefix),
        "--csv-full-history",
        "--only-summary",
        "--exit-code-on-error",
        "0",
    ]
    with prefix.with_suffix(".log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        set_affinity(process.pid, cpus)
        try:
            result = process.wait(timeout=seconds + 90)
        except BaseException:
            stop_process(process)
            raise
        if result:
            raise RuntimeError(f"Locust 启动/运行失败，检查 {prefix}.log")


def main():
    parser = argparse.ArgumentParser(description="本机单服务矩阵；每点重启，预热后独立测量")
    parser.add_argument(
        "--scenario",
        choices=["modes", "batch", "overload", "stability", "bottlenecks", "smoke"],
        default="modes",
    )
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--image", default="reports/sample.png")
    parser.add_argument("--output", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-cpus", default="")
    parser.add_argument("--client-cpus", default="")
    parser.add_argument("--limit", type=int, default=0, help="只运行前 N 点，0 为全部")
    args = parser.parse_args()
    duration = (
        args.duration
        if args.duration is not None
        else (1800 if args.scenario == "stability" else 60)
    )
    if duration <= 0 or args.warmup <= 0:
        parser.error("duration/warmup 必须为正数")
    if set(args.server_cpus.split(",")) & set(args.client_cpus.split(",")) - {""}:
        parser.error("服务和压测 CPU 集合不能重叠")
    if not Path(args.image).is_file():
        parser.error("图片不存在，请先运行 python -m scripts.make_sample")
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error("压测端口已被占用；请选择空闲端口，避免误测其他服务")
    output = Path(args.output or f"reports/runs/{args.scenario}-{time.time_ns()}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "environment.json",
        {
            **environment(Settings()),
            "arguments": vars(args),
            "duration_s": duration,
            "cpu_isolated": bool(args.server_cpus and args.client_cpus),
        },
    )
    summaries = []
    for index, (name, users, overrides) in enumerate(cases(args.scenario)):
        if args.limit and index >= args.limit:
            break
        point = output / name
        point.mkdir()
        env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "RATE_LIMIT_QPS": "0",
            "KEY_RATE_LIMIT_QPS": "0",
            "EXP_SYNC_LOG": "0",
            "EXP_GLOBAL_LOCK": "0",
            "EXP_INLINE_PREPROCESS": "0",
            "EXP_UNBOUNDED_QUEUE": "0",
            "PORT": str(args.port),
            "HOST": "127.0.0.1",
            "BENCH_IMAGE": str(Path(args.image).resolve()),
            **overrides,
        }
        env["BENCH_MODEL"] = env.get("MODEL_NAME", Settings().model_name)
        write_json(
            point / "config.json",
            {"users": users, "overrides": overrides, "model": env["BENCH_MODEL"]},
        )
        with (point / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "app"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            try:
                set_affinity(process.pid, args.server_cpus)
                deadline = time.monotonic() + 240
                with httpx.Client(timeout=2, trust_env=False) as client:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"服务启动失败：{point / 'server.log'}")
                        try:
                            if (
                                client.get(f"http://127.0.0.1:{args.port}/readyz").status_code
                                == 200
                            ):
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError("等待服务 ready 超时")
                        time.sleep(0.2)
                    write_json(
                        point / "model.json",
                        client.get(f"http://127.0.0.1:{args.port}/v1/models").json(),
                    )
                run_locust(args, env, users, args.warmup, point / "warmup", args.client_cpus)
                measured_env = {**env, "BENCH_RAW_CSV": str((point / "requests.csv").resolve())}
                sampler = ResourceSampler(process.pid, point / "resources.csv").start()
                health = HealthProbe(
                    f"http://127.0.0.1:{args.port}/healthz", point / "health.csv"
                ).start()
                try:
                    run_locust(
                        args, measured_env, users, duration, point / "locust", args.client_cpus
                    )
                finally:
                    health.stop()
                    sampler.stop()
                summary = {
                    "case": name,
                    "concurrency": users,
                    "duration_s": duration,
                    **summarize(point, duration),
                }
                summaries.append(summary)
                write_csv(output / "summary.csv", summaries)
                print(summary, flush=True)
            finally:
                stop_process(process)


if __name__ == "__main__":
    main()
