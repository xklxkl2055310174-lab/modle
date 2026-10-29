import argparse
import csv
import json
import os
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description="从原始 CSV 自动生成报告，禁止手工编造测量值")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--http", required=True)
    parser.add_argument("--output", default="docs/report.md")
    args = parser.parse_args()
    model, profile, http, output = map(Path, [args.model, args.profile, args.http, args.output])
    env = json.loads((http / "environment.json").read_text(encoding="utf-8"))
    model_env = json.loads((model / "environment.json").read_text(encoding="utf-8"))
    measurements = rows(http / "summary.csv")
    if env["arguments"]["scenario"] != "smoke" or model_env["resolved_device"] != "cpu":
        parser.error("本报告模板仅用于 CPU smoke；正式实验请按压测指南撰写对应报告")

    def link(path):
        return os.path.relpath(path, output.parent).replace(os.sep, "/")

    lines = [
        "# 压测报告（真实 CPU 冒烟）",
        "",
        "本报告由 `python -m bench.report` 从 CSV 自动生成。数据用于证明实现与测量链路可运行，",
        "不能支持长期稳定性、生产容量或优化百分比结论。完整矩阵和 30 分钟实验尚待运行。",
        "",
        "## 测试环境与边界",
        "",
        f"- 时间：{env['time_utc']}（UTC）；OS：{env['platform']}；Python {env['python']}。",
        f"- 逻辑 CPU：{env['logical_cpus']}；物理内存：{env['memory_bytes'] / 1024**3:.2f} GiB。",
        f"- PyTorch {env['packages']['torch']} / torchvision {env['packages']['torchvision']}；"
        f"FastAPI {env['packages']['fastapi']} / Locust {env['packages']['locust']}。",
        f"- HTTP 代码版本：`{env['commit']}`；服务核 {env['arguments']['server_cpus']}，"
        f"客户端核 {env['arguments']['client_cpus']}；"
        f"TORCH_THREADS={env['settings']['torch_threads']}。",
        f"- 输入：`{env['arguments']['image']}`。本次合成图不能用于准确率评估。机器运行桌面应用，"
        "CPU affinity 不代表物理核独占，也未固定频率。",
        f"- HTTP 每模式预热 {env['arguments']['warmup']} 秒、目标测量 {env['duration_s']} 秒，"
        f"并发 4；限流关闭，批上限 {env['settings']['max_batch_size']}。"
        "QPS 分母使用实际窗口（含排空）。",
        f"- 模型基准每批预热 {model_env['arguments']['warmup']} 次、测量 "
        f"{model_env['arguments']['iterations']} 次；未隔离 CPU，不能直接与 HTTP 模式对照。",
        "- 本报告仅验证 CPU 运行；CUDA 运行时、GPU 推理和设备利用率须另行验证。",
        f"- [HTTP 完整环境]({link(http / 'environment.json')})；"
        f"[模型完整环境]({link(model / 'environment.json')})。",
        "",
        "## 三模式 HTTP 冒烟",
        "",
        "| 模式 | 请求数 | 实际窗口 s | 成功 QPS | P50 ms | P95 ms | P99 ms |"
        " 失败率 | 峰值 RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in measurements:
        lines.append(
            f"| {row['case']} | {row['requests']} | {float(row['measured_seconds']):.2f} | "
            f"{float(row['success_qps']):.2f} | {float(row['p50_ms']):.2f} | "
            f"{float(row['p95_ms']):.2f} | {float(row['p99_ms']):.2f} | "
            f"{float(row['failure_rate']):.2%} | {float(row['rss_peak_bytes']) / 1024**2:.2f} |"
        )
    total = sum(int(row["requests"]) for row in measurements)
    winner = max(measurements, key=lambda row: float(row["success_qps"]))["case"]
    lines += [
        "",
        f"共 {total} 次测量请求。本次最高成功吞吐为 **{winner}**；"
        "动态批处理没有在这组低并发 CPU 冒烟中胜出。不能据此给三种模式做通用排名。",
        "进程池的模型副本增加 RSS；短时 P99 受冷路径、调度和样本数量影响，需长时间、多次复测。",
        "",
        f"![吞吐与尾延迟]({link(http / 'throughput-latency.png')})",
        "",
        f"[汇总 CSV]({link(http / 'summary.csv')})；每模式目录保留 requests.csv、"
        "resources.csv、health.csv、Locust CSV 与实际时间窗口。",
        "",
        "## 模型基准",
        "",
        "只计 infer_batch（stack、设备传输、模型和 softmax），不含解码；均值不是 P99。",
        "",
        "| 批大小 | 单批 ms | 单样本 ms | samples/s |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows(model / "model.csv"):
        lines.append(
            f"| {row['batch_size']} | {float(row['batch_ms']):.2f} | "
            f"{float(row['sample_ms']):.2f} | {float(row['samples_per_second']):.2f} |"
        )
    lines += [
        "",
        "批大小增大没有带来单调吞吐提升。这里只能观察到当前配置下的固定开销与批量计算取舍，"
        "不能证明瓶颈原因。",
        "",
        f"![模型基准]({link(model / 'model.png')})",
        "",
        f"[逐次迭代数据]({link(model / 'raw.csv')})。",
        "",
        "## 算子分布",
        "",
    ]
    for batch in (1, 16):
        path = profile / f"batch-{batch}-cpu.csv"
        top = rows(path)[0]
        lines.append(
            f"- batch {batch}：最高 CPU self-time 算子 `{top['operator']}`，"
            f"占 {float(top['share_percent']):.2f}%；[Top 15]({link(path)})，"
            f"[Chrome trace]({link(profile / f'batch-{batch}.json')})。"
        )
    lines += [
        "",
        "算子 self-time 占比用于定位，不等同于端到端耗时占比。采样本身存在开销。",
        "",
        "## 尚待验证",
        "",
        "- 三模式 × 并发 1/8/32/64/128 的完整矩阵和重复运行。",
        "- 批大小 × 等待窗口、过载四组合、A/B/C/D 瓶颈前后对照。",
        "- 30 分钟稳定性；当前短测不能宣称内存和错误率长期稳定。",
        "- Docker 镜像/监控栈、JMeter 交叉验证、Linux py-spy/pidstat/iostat 证据。",
        "- 真实 GPU 推理及 OOM；离线模拟仅验证拆批控制流。",
        "",
        "复现实验命令见 [压测指南](bench-guide.md)。每个结论应同时报告成功 QPS、成功 P99、"
        "失败率、CPU/RSS 以及原始数据路径，保留不符合预期的结果。",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
