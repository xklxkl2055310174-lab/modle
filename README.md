# 小模型推理服务与并发压测

Python 3.11 / PyTorch / FastAPI 学习项目。把 ImageNet 预训练 ResNet18 封装为图像分类服务，比较线程池、spawn 进程池和异步动态批处理的吞吐、尾延迟与资源开销。

包含有界队列、截止时间、批内故障隔离、CUDA OOM 拆批、客户端退避重试、双层令牌桶、JSON 日志、Prometheus/Grafana、Locust/JMeter、算子 profiling 和离线测试。性能与验证范围见 [报告](docs/report.md)，开发记录见 [PROGRESS](docs/PROGRESS.md)。

## 本机启动

从仓库根目录执行，Python 必须为 3.11：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt -r requirements-bench.txt
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m scripts.download_weights
python -m scripts.make_sample
```

复制 `.env.example` 为 `.env`（Linux `cp .env.example .env`，PowerShell `Copy-Item .env.example .env`），然后：

```bash
python -m app
# 另一个终端：Windows 用 curl.exe，避免 PowerShell 的 curl 别名
curl -F "file=@reports/sample.png" -F "top_k=5" http://127.0.0.1:8000/v1/models/resnet18/predict
```

打开 [交互接口](http://127.0.0.1:8000/docs)、[就绪状态](http://127.0.0.1:8000/readyz)、[指标](http://127.0.0.1:8000/metrics)。服务只从本地加载权重，启动不会自动下载。合成图仅用于功能与性能验证。

**无需 torch/权重的离线演示**：安装 `requirements-dev.txt`，在 `.env` 中设 `MODEL_NAME=dummy`，运行 `python -m app`，请求路径改为 `/v1/models/dummy/predict`。DummyRunner 不具备真实分类能力。

```bash
python -m ruff check .
python -m pytest -q
```

测试不访问网络，不下载权重；Windows 测试环境需允许 asyncio 使用本机 socketpair。

## Docker 与监控

先准备 `.env`、权重，在 `.env` 设置 `GRAFANA_ADMIN_PASSWORD`，然后：

```bash
docker compose --env-file .env -f deploy/docker-compose.yml up --build -d
# NVIDIA GPU（需宿主驱动和 NVIDIA Container Toolkit）：
docker compose --env-file .env -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml up --build -d
```

[Grafana](http://127.0.0.1:3000) 用户为 `admin`，密码为自己设置的值；[Prometheus](http://127.0.0.1:9090)。GPU 镜像使用 torch 2.6 / torchvision 0.21 cu124，CPU 镜像使用本次实测的 torch 2.15 / torchvision 0.29；跨版本数据不可直接归因于设备差异。镜像与 GPU 验证状态详见进度记录。

## 切换服务模式与压测

在 `.env` 中设置 `SERVE_MODE=threadpool`、`processpool` 或 `async_batch`（默认）后重启服务。所有模式共享相同 API、限流、队列上限、日志及指标。

```bash
# 先停止手动启动的服务；矩阵脚本自行启动/停止服务
python -m bench.run_matrix --scenario smoke --warmup 3 --duration 10
python -m bench.run_matrix --scenario modes --server-cpus 0,1 --client-cpus 2,3
python -m bench.plot reports/runs/<本次目录>
```

完整操作、CPU 隔离、公平性、30 分钟稳定性和瓶颈实验见 [压测指南](docs/bench-guide.md)。矩阵默认关闭全局及按 key 的限流；不使用 SDK 重试，以免掩盖失败率。

## Python SDK

```python
import asyncio
from pathlib import Path
from client import InferenceClient


async def main():
    async with InferenceClient(total_timeout=10, max_retries=3) as client:
        result = await client.predict(Path("reports/sample.png").read_bytes())
        print(result)


asyncio.run(main())
```

SDK 仅重试连接错误和 HTTP 429/503/504；4xx 输入错误、500 和读取超时不重试。重试和等待共同消耗总预算。`X-API-Key` 仅作为限流分组标识，不实现身份认证。

## 设计与局限

- [设计与指标语义](docs/DESIGN.md)：执行边界、排队、超时、信号与停机。
- [压测报告](docs/report.md)：仅记录真实结果，区分冒烟与正式性能结论。
- [面试说明](docs/interview.md)：实现依据、取舍与不能声称的结论。
- [开发总纲](AGENT.md)：项目需求；[进度](docs/PROGRESS.md)：阶段提交和验证记录。

本项目是单进程 Uvicorn、单实例限流；对外部署需自行配置网关认证、TLS 和分布式限流。不要将 `EXP_*` 实验开关用于常规服务。CPU RSS 为进程树之和，可能重复计算共享内存；GPU 显存不等于利用率。
