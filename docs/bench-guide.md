# 压测操作手册

所有命令从仓库根运行。先安装 README 中的测试/压测依赖和真实模型，生成 `reports/sample.png`。检查 `MODEL_NAME=resnet18`，明确设备，记录完整软件版本。DummyRunner 仅做功能冒烟，不得作为简历性能数字。

## 模型基准和算子分析

```bash
python -m bench.model_benchmark --batches 1,2,4,8,16,32,64 --iterations 20 --warmup 3
python -m bench.profile_model --batches 1,16 --iterations 5
```

分别输出每次迭代原始 CSV、均值/吞吐、RSS/显存、CPU/GPU Top 15 self-time 算子占比和 Chrome trace。查看 trace 可用 Chrome tracing 或 Perfetto。模型基准不含预处理；如果 OOM，保留已测数据、记录失败批次，不用假数据填空。

## 三种模式和参数矩阵

```bash
python -m bench.run_matrix --scenario modes --warmup 15 --duration 60 --server-cpus 0,1 --client-cpus 2,3
python -m bench.run_matrix --scenario batch --warmup 15 --duration 60 --server-cpus 0,1 --client-cpus 2,3
python -m bench.run_matrix --scenario overload --warmup 15 --duration 60 --server-cpus 0,1 --client-cpus 2,3
python -m bench.run_matrix --scenario stability --server-cpus 0,1 --client-cpus 2,3
python -m bench.run_matrix --scenario bottlenecks --server-cpus 0,1 --client-cpus 2,3
```

- modes：三模式 × 并发 1/8/32/64/128；batch：批大小 1/4/8/16/32/64 × 等待 0/1/5/10/20ms。
- overload：限流开/关 × 队列有界/无界的四组合。无界队列仅用于对照，仍受请求截止时间约束；128 并发属于闭环压力，不能等价于无限到达流量。必须从队列/拒绝率确认实际进入过载。
- stability：默认采样 1800 秒；若指定更短 duration，必须标为冒烟。
- bottlenecks：A/B/C/D 开关前后共 8 点。每个点关闭无关实验开关，保证对照可解释。
- smoke：三模式各并发 4 的快速全链路验证。`--limit N` 只运行前 N 点。

矩阵自行启动服务，应提前停止占用该端口的服务。每点启动、预热完成后独立启动 Locust 测量进程，不混入预热请求。默认关闭全局及按 key 限流，过载场景按案例覆盖。`X-API-Key` 不发送，SDK 重试不参与。

`--server-cpus` 和 `--client-cpus` 根据自己机器实际 CPU 编号调整，必须不重叠。本机脚本使用 psutil affinity，子进程继承；Docker 用 `SERVICE_CPUS` 或自行加 cpuset，将压测放在其他核或机器。CPU 限额不等价于独占 CPU，不能保证稳定时钟和没有邻居干扰。若未指定隔离，environment.json 记录 cpu_isolated=false，数据仅供冒烟。

## 数据口径和画图

每次输出目录唯一；包含 environment.json、summary.csv、每点配置/模型元数据、requests.csv、requests.window.json、Locust 原始 CSV、resources.csv、health.csv 和日志。QPS 用 test_start 到 test_stop 的**实际时长**作为分母，含启动用户和停止时排空窗口；同时保留目标 duration。应确保正式采样窗口足够长，降低边界开销。

`qps` 包含失败请求，`success_qps` 只计成功；p50/p95/p99 包含全部响应，`accepted_p99_ms` 只计成功响应。限流可通过快速拒绝降低总体 P99，不能据此声称推理变快。成功样本为空时 P99 留空/NaN。资源采样首个 CPU 值无基线，不纳入均值；父子进程 RSS 可能重复计算共享页。GPU 利用率来自整张卡，不能归因于单一进程。

```bash
python -m bench.plot reports/runs/<目录>
python -m bench.analyze_stability reports/runs/<稳定性目录>/stability
python -m bench.resources --pid <服务PID> --seconds 60 --output reports/runs/resources.csv
```

稳定性分析输出 RSS 斜率、首尾内存、每分钟失败率和是否达到 30 分钟。斜率不是内存泄漏证明，需结合分配器缓存、平台监控和多次复测。不能预先宣布“内存不涨”。

## JMeter 交叉验证

需要单独安装 Java/JMeter（不作为 Python 依赖）。固定同样服务配置、图片、并发、时长和 CPU 隔离；先运行预热，再测量：

```bash
jmeter -n -t bench/jmeter/predict.jmx -Jhost=127.0.0.1 -Jport=8000 -Jmodel=resnet18 -Jusers=8 -Jduration=15 -Jimage=/absolute/path/sample.png -l reports/warmup.jtl
jmeter -n -t bench/jmeter/predict.jmx -Jhost=127.0.0.1 -Jport=8000 -Jmodel=resnet18 -Jusers=8 -Jduration=60 -Jimage=/absolute/path/sample.png -l reports/measure.jtl -e -o reports/jmeter-html
```

JMeter 与 Locust 可能因连接池/计时口径不同有偏差，应记录并解释，不改数字凑一致。

## 瓶颈定位证据

| 实验 | 故意退化 | 定位手段 | 修复与比较 |
|---|---|---|---|
| A | processpool / TORCH_THREADS=0 | top、pidstat -w、py-spy、cgroup cpu.stat | auto；核数固定后比较吞吐/P99/上下文切换 |
| B | EXP_SYNC_LOG=1 | iostat -x、vmstat、py-spy 中 fsync | 关闭开关，比较 IO 等待与 P99 |
| C | threadpool / EXP_GLOBAL_LOCK=1 | py-spy 线程栈、pidstat -w | 去锁、扫描 torch 线程，或批处理 |
| D | EXP_INLINE_PREPROCESS=1 | health.csv、py-spy | 预处理移回工作线程，看 health P99 |

```bash
py-spy record --pid <PID> --duration 30 -o reports/profile.svg
pidstat -p <PID> -w 1
vmstat 1
iostat -x 1
```

Linux 工具需平台安装；权限不足时如实记录。py-spy 对进程池需分别采样子进程。任何对照都按“观察 → 证据 → 原因假设 → 修复 → 复测”记录；如果预期改善未出现，保留结果并解释固定开销、内存带宽、短测量等替代原因。
