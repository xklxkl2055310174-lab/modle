# 进度与验证记录

## 阶段一：服务、执行模式与离线测试（完成）
- 计划：统一 API/中间件，ResNet18 和 DummyRunner，三种有界队列执行模式，截止时间、故障隔离、限流、指标、SDK。
- 仓库原来仅包含 AGENT.md 和标题 README；无既有代码需要迁移。
- 按项目要求使用 Python 3.11 隔离环境；不引入白名单外的直接运行依赖。
- 尚无性能结论；Docker、CUDA、真实权重和长时间测试均须单独验证。
- 验证：`ruff check .` 通过，`pytest -q` **36 passed in 8.78s**（Python 3.11.16、Windows，DummyRunner，全程无权重下载）。
- 覆盖：三种模式 API、进程全部预热、动态组批、排队超时/取消/队列满恢复、坏图隔离、OOM 二分、单条重试、停机超时、SDK 总预算、双层令牌桶。
- 环境问题：受限沙箱会阻止 asyncio socketpair 的本机回环连接；允许回环连接后测试通过。不是服务代码阻塞。
- 限制：Python 3.11 无法安全抢占正在执行的原生线程/GPU kernel。停机期限内结束等待并拒绝剩余请求；容器最终退出由 stop_grace_period 兜底，详见 DESIGN。

## 阶段二：压测、部署与文档（完成）
- 实现模型批大小基准、CPU/GPU Top 15 和 Chrome trace、Locust 原始逐请求记录、JMeter 场景、进程树资源采样和独立 healthz 探测。
- 实现模式/批参数/过载/稳定性/A-D 瓶颈扫描、实际测量时长统计、成功 P99、绘图与稳定性趋势分析。
- 提供 CPU/GPU Dockerfile、Compose、Prometheus、Grafana 自动预置、Linux/Windows CI。
- 完成中文 README、设计、压测指南、报告模板、面试说明；已下载并校验官方权重（git 忽略）。
- 验证：`ruff check .`、compileall、Grafana JSON / JMeter XML 解析通过；`pytest -q` **37 passed in 19.33s**。
- 直接依赖全部来自白名单。Java/JMeter、Docker 和 Linux 诊断命令为外部工具，不额外引入 Python 依赖。
- 待本地验证：Docker 镜像构建/监控栈、真实 CUDA、JMeter 与 Linux 采样工具。

## 阶段三：真实 CPU 链路与最终验证（实现和本机验证完成）
- 计划：运行真实 ResNet18 模型基准和算子分析、三模式短时 Locust 压测，保存原始数据与图表；复核实现边界后运行全量检查。
- 不将短时冒烟升级为长期稳定性或生产性能结论；30 分钟和完整参数矩阵另标待本地验证。
- 复核修复：Windows 自动线程数考虑 psutil CPU affinity；OOM 拆批前释放异常栈中的设备张量；chunked 上传在扩展缓冲前检查上限；新增慢上传/伪 MIME/像素超限回归。
- 真实运行发现并修复 Windows Locust 默认 GBK 读取中文 pyproject.toml 失败：矩阵子进程统一设置 PYTHONUTF8=1。第一次尝试未产生性能结果，不能计入成功测试。
- 最终验证：`ruff check .`、compileall、`git diff --check` 通过；`pytest -q` **40 passed in 9.92s**。
- 真实模型：torch 2.15.0+cpu / torchvision 0.29.2+cpu，ResNet18 官方权重 SHA256 校验通过；批大小 1/2/4/8/16/32/64 各 1 次预热、3 次采样；batch 1/16 各 2 次 profiler 采样，生成 Top 15 和 Chrome trace。
- 真实 HTTP：三模式各并发 4、3 秒预热、10 秒目标窗口，服务 affinity 0/1/2/3，Locust affinity 4/5，TORCH_THREADS=2；共 **654 次预测、0 失败**。原始 CSV、环境、图表在 `reports/verified/`；报告由 `bench.report` 自动生成。
- 本次线程池成功 QPS 高于动态批处理，报告如实保留；窗口短、桌面机器、每点只跑一次，不能据此声称优化收益或正式容量。
- 实测发现宿主有 RTX 4060 Laptop GPU，但项目验证环境安装的是 CPU wheel；没有验证 CUDA 推理。Docker/JMeter 未安装；仅完成 Docker 配置与 JMeter XML 结构检查。
- 已验证正常服务信号关闭路径（压测脚本自动启动/停止）；预热小像素上限、错误日志阶段字段和压测异常清理进一步修复，全部离线回归通过。
- 明确待本地验证：Docker 构建与 Grafana/Prometheus 启动、真实 CUDA/OOM、JMeter 交叉验证、完整并发/批参数/过载矩阵、A-D 诊断证据及 30 分钟稳定性。相应脚本已交付，没有伪造结果。
