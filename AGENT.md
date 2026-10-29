# 小模型推理服务与并发压测 —— 项目总纲

## 1. 角色与目标
你是资深 Python / AI 系统工程师。请从零实现"小模型推理服务 + 并发压测"项目，作为简历上的学习项目。
重点是「服务化设计 + 并发调优 + 指标观测」，对标 AI 系统软件的工程化场景。每个设计都要能在面试中讲清楚，每个性能结论都要有真实数据支撑。

## 2. 简历原文（实现必须支撑每一句）
小模型推理服务与并发压测（学习项目）｜独立开发 · 服务化 + 性能调优
技术栈：Python、PyTorch、FastAPI、Docker、Locust/JMeter、Linux
把一个开源小模型（如 TinyBERT / ResNet）包装成可用的推理服务，重点练「服务化设计 + 并发调优 + 指标观测」，对标 AI 系统软件的工程化场景。
- 服务化设计：用 FastAPI 封装推理接口，设计请求队列、批量合并（dynamic batching）、超时与重试、限流策略，并用 Docker 打包一键启动。
- 框架使用：用 PyTorch 完成模型加载与推理，理解设备选择（CPU/GPU）、显存占用与算子耗时分布，记录单条与批量推理的延迟差异。
- 并发与性能调优：用线程池/进程池/异步三种方式压测吞吐，对比 QPS、P99 延迟与资源占用，定位瓶颈（CPU 打满 / IO 等待 / 锁竞争）并给出优化前后数据。
- 观测与稳定性：接入日志与指标采集（请求量、耗时、失败率），写一份压测报告，说明稳定性与可扩展性的取舍。

## 3. 技术约束
- Python 3.11、PyTorch 2.x（CPU 版就能跑通全部功能，检测到 CUDA 时自动支持 GPU）、FastAPI + Uvicorn（单进程）、pydantic-settings。
- 主模型：torchvision 的 ResNet18（ImageNet 预训练权重，图像分类）。模型层抽象成统一接口，方便以后接入 TinyBERT。
- 依赖白名单：torch、torchvision、fastapi、uvicorn、pydantic-settings、python-multipart、pillow、numpy、prometheus-client、httpx、psutil；开发和压测：pytest、pytest-asyncio、ruff、locust、pandas、matplotlib、py-spy。新增依赖要在 docs/PROGRESS.md 写明理由。
- 测试必须离线可跑：单元测试用很小的假模型（DummyRunner），不下载任何权重。真实权重由 scripts/download_weights.py 下载到 weights/（不提交 git），加载时用 torch.load(..., weights_only=True)。
- 所有可调参数用环境变量配置，在 .env.example 里列出默认值。
- 标识符用英文，注释和文档用中文；关键函数注明运行在哪个线程或进程、会不会阻塞事件循环。
- 当前环境无法运行 Docker、GPU 或长时间压测时，只完成代码和静态检查，并在 PROGRESS.md 标注"待本地验证"。禁止伪造任何运行结果。

## 4. 目录结构
- app/main.py：FastAPI 应用和生命周期（加载模型、预热、优雅停机）
- app/config.py：配置
- app/api/：路由、请求响应模型、统一错误处理
- app/models/：ModelRunner 抽象（preprocess / infer_batch / postprocess）、resnet.py、dummy.py
- app/serving/：三种服务模式 threadpool.py、processpool.py、batcher.py（异步队列 + 动态批处理）
- app/middleware/：request_id、令牌桶限流、访问日志
- app/observability/：JSON 日志、Prometheus 指标
- client/：Python SDK（超时 + 指数退避重试 + 抖动）
- bench/：模型基准、算子 profiling、locustfile.py、jmeter/、资源采样、一键压测和出图脚本
- deploy/：Dockerfile、Dockerfile.gpu、docker-compose.yml、Prometheus 配置、Grafana 数据源和看板
- tests/、scripts/、reports/（原始数据和图表）、docs/（DESIGN.md、report.md、interview.md、bench-guide.md、PROGRESS.md）

## 5. 接口设计
- POST /v1/models/{model}/predict：multipart 上传图片（jpeg / png，最大 5 MB），可选参数 top_k（默认 5）；可选请求头 X-Request-ID（没有就生成）、X-Timeout-Ms。
  返回 request_id、model、predictions（label + score）、batch_size（这条请求实际所在批次的大小）、timing（queue_ms、preprocess_ms、inference_ms、total_ms）。
- GET /v1/models：已加载的模型及配置（设备、服务模式、批处理参数）。
- GET /healthz（进程存活）、GET /readyz（模型加载并预热完成才返回 200，停机中返回 503）、GET /metrics（Prometheus）。
- 统一错误体 {code, message, request_id}：400 输入非法、413 文件过大、429 限流、503 队列满或未就绪、504 超时、500 推理失败；429 和 503 带 Retry-After。

## 6. 服务化设计要点
- 三种服务模式，用 SERVE_MODE 切换。接口、中间件、指标完全一致，保证对比公平：
  1. threadpool：每个请求在线程池（大小可配）里单条推理；
  2. processpool：ProcessPoolExecutor，用 spawn 启动方式（避免 fork 之后 OpenMP / CUDA 出问题）；每个子进程在 initializer 里各自加载模型、设置 torch 线程数；传给子进程的是图片字节而不是张量，减少序列化开销；
  3. async_batch（默认）：asyncio 有界队列 + 后台批处理协程，凑满 MAX_BATCH_SIZE 或等待超过 MAX_WAIT_MS 就发车；推理在专用推理线程里执行，绝不阻塞事件循环；结果按请求拆开，回填到各自的 Future。
- 请求队列：有界（QUEUE_MAXSIZE），满了直接返回 503，不无限堆积。
- 超时：每个请求有截止时间（X-Timeout-Ms 和默认值取较小的）；组批时丢掉已过期的请求，不浪费算力；等结果超时返回 504。
- 重试：
  - 服务端：批推理遇到 CUDA OOM 时把批次对半拆开重试；遇到其他异常时逐条重试一次，只让坏样本失败，不连累同批的其他请求；
  - 客户端 SDK：只对连接错误、429、503、504 重试，指数退避加随机抖动，遵守 Retry-After，有总超时预算和最大次数；输入错误类的 4xx 绝不重试。
- 限流：令牌桶，一个全局桶，再按 X-API-Key（没有就按客户端 IP）各一个桶，参数可配，超限返回 429。文档里要说明这是单实例限流，多实例需要网关或 Redis。
- 稳定性：启动时预热（每种批大小各跑几次）后才 ready；优雅停机：收到 SIGTERM 后 readyz 返回 503，停止接收新请求，在限定时间内处理完队列里的请求，剩下的返回 503。
- TORCH_THREADS：auto（默认，按"可用核数 ÷ 推理进程数"设置，可用核数要考虑容器 CPU 限额，读取 cgroup 的 cpu.max）、正整数、0（不设置，用于复现实验 A）。
- 其他配置：SERVE_MODE、DEVICE（auto / cpu / cuda）、THREADPOOL_SIZE、PROCESS_WORKERS、MAX_BATCH_SIZE、MAX_WAIT_MS、QUEUE_MAXSIZE、REQUEST_TIMEOUT_MS、RATE_LIMIT_QPS、RATE_LIMIT_BURST、LOG_LEVEL，以及第 8 节的实验开关。

## 7. 观测
- 日志：JSON 结构化日志输出到 stdout，用 QueueHandler 异步写；每个请求一行访问日志（request_id、路由、状态码、各阶段耗时、batch_size、服务模式）。
- 指标（prometheus-client）：请求总数（按路由、状态码）；请求耗时、排队耗时、推理耗时、批大小（直方图）；队列深度、在途请求数（Gauge）；拒绝数（按原因：限流 / 队列满 / 超时）；GPU 显存占用（有 GPU 时）。
- docker compose 自动拉起 Prometheus + Grafana，并预置看板：QPS、P50 / P95 / P99、失败率、队列深度、批大小分布、CPU 和内存。

## 8. 压测与调优要求
- 工具：Locust 为主（headless 模式输出 CSV）；JMeter 提供 .jmx，用非 GUI 模式跑同一场景做交叉验证；资源采样脚本用 psutil 每秒记录服务进程（含子进程）的 CPU 和内存，有 GPU 时同时记录显存和利用率。
- 公平性：服务端用 docker 的 cpus 或 cpuset 固定可用核数，压测端用其余的核；每个点先预热，再固定时长采样；非限流场景要关闭限流，避免误伤；报告里写明测试环境。
- 必测场景：
  1. 模型基准：批大小 1 / 2 / 4 / 8 / 16 / 32 / 64 的单批延迟、单样本平均延迟、吞吐、CPU 内存 / GPU 显存；
  2. 算子分布：torch.profiler 统计批大小 1 和 16 的 Top 15 算子耗时占比，导出 chrome trace；
  3. 三种模式并发扫描：并发 1 / 8 / 32 / 64 / 128 下的 QPS、P50 / P95 / P99、失败率、CPU、内存；
  4. 批处理参数扫描：MAX_BATCH_SIZE × MAX_WAIT_MS，画出吞吐和 P99 的取舍曲线；
  5. 过载测试：请求量超过处理能力时，对比"开 / 关限流和有界队列"下被接受请求的 P99 和失败率；
  6. 稳定性：持续压测 30 分钟，内存和错误率不能持续上涨。
- 瓶颈实验：用实验开关复现，每个都写清现象 → 定位手段和证据（py-spy、top、pidstat -w、vmstat、iostat、torch.profiler）→ 原因 → 修复 → 前后对比：
  - 实验 A「CPU 打满 / 线程过度订阅」：processpool 下各子进程不设置线程数（TORCH_THREADS=0），或在容器 CPU 限额下仍按宿主机核数开线程；修复为按可用核数分配；
  - 实验 B「IO 等待」：每条请求同步写日志并 fsync（EXP_SYNC_LOG=1）；修复为异步日志；
  - 实验 C「锁竞争」：threadpool 下给模型推理加全局锁（EXP_GLOBAL_LOCK=1）；修复为去锁并调好线程数，或改用动态批处理；
  - 实验 D「事件循环阻塞」（加分项）：async_batch 下在事件循环里直接做图片预处理（EXP_INLINE_PREPROCESS=1）；修复为放进线程池；用 py-spy 和 /healthz 延迟来证明。
- 结论以真实数据为准，和预期不符时如实记录并分析原因。所有数据由脚本真实运行产生，原始 CSV 和图表放在 reports/，禁止手填或编造数字。

## 9. 工作方式（每个阶段都必须遵守）
1. 开始前读本文件和 docs/PROGRESS.md，先列出本阶段计划，然后直接实现；真正有歧义时才提问。
2. 只做当前阶段的内容，不提前堆后续功能。
3. 阶段结束前必须通过 ruff check . 和 pytest；禁止删除或弱化测试来"通过"。
4. 阶段结束时更新 docs/PROGRESS.md（完成了什么、怎么验证、已知问题、待本地验证的项），然后 git commit（Conventional Commits）。
5. 最后用不超过 5 行告诉我：做了什么、我怎么手动验证。
