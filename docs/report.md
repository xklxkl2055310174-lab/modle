# 压测报告

## 结论范围

当前先交付可运行的测量代码与离线可靠性测试。真实 CPU 冒烟数据将在最终验证后填入本报告；未测得的数据不填写。短时、同机或未隔离 CPU 的结果不能支持“性能优化提升 X%”或“生产稳定运行”的结论。

## 已验证

- 第一阶段：Python 3.11.16 / Windows，36 项 DummyRunner 离线测试通过。
- 模型、线程/进程/动态批模式的功能与性能测量状态以 docs/PROGRESS.md 最终记录为准。

## 正式实验验收清单

| 场景 | 实现入口 | 实测状态 |
|---|---|---|
| 批大小 1/2/4/8/16/32/64 | bench.model_benchmark | 待最终验证记录 |
| batch 1/16 Top 15 与 trace | bench.profile_model | 待最终验证记录 |
| 三模式 × 并发 1/8/32/64/128 | run_matrix --scenario modes | 待本地完整压测 |
| MAX_BATCH_SIZE × MAX_WAIT_MS | --scenario batch | 待本地完整压测 |
| 限流/有界队列四组合 | --scenario overload | 待本地验证 |
| 30 分钟稳定性 | --scenario stability | 待本地验证 |
| A/B/C/D 前后证据 | --scenario bottlenecks | 待本地验证 |
| JMeter 交叉验证 | bench/jmeter/predict.jmx | 待本地安装运行 |
| Docker/Prometheus/Grafana | deploy/ | 待 Docker 环境验证 |
| CUDA OOM / GPU 利用率 | DEVICE=cuda | 离线模拟分支已测试；真实 GPU 待验证 |

## 每次正式报告需填写

环境（CPU/GPU/内存/OS/包版本/commit）、服务与压测 CPU 隔离、图片数据来源、预热/采样时长、三次以上重复测量波动、原始 CSV 路径。模型基准和 HTTP 基准的计时范围不同，不能混用。

每项瓶颈按“现象 → 定位工具及原始证据 → 原因假设 → 改动 → 同环境复测”填写。失败率、拒绝率、成功 P99 和资源代价必须一起报告；不符合预期的结果也必须保留。
