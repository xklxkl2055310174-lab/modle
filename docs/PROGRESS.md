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

## 阶段三：真实 CPU 链路与最终验证（进行中）
- 计划：运行真实 ResNet18 模型基准和算子分析、三模式短时 Locust 压测，保存原始数据与图表；复核实现边界后运行全量检查。
- 不将短时冒烟升级为长期稳定性或生产性能结论；30 分钟和完整参数矩阵另标待本地验证。
- 复核修复：Windows 自动线程数考虑 psutil CPU affinity；OOM 拆批前释放异常栈中的设备张量；chunked 上传在扩展缓冲前检查上限；新增慢上传/伪 MIME/像素超限回归。
- 真实运行发现并修复 Windows Locust 默认 GBK 读取中文 pyproject.toml 失败：矩阵子进程统一设置 PYTHONUTF8=1。第一次尝试未产生性能结果，不能计入成功测试。
