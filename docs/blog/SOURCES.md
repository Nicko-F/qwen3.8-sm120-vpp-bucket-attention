# 内容与图形来源

## 项目版本

- 仓库：https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention
- 图表读取仓库当前的 [benchmarks](../../benchmarks)，构建时嵌入两种语言的页面。
- 核对日期：2026-11-05

## 实验数据

| 文件 | 对应章节 | 来源 |
| --- | --- | --- |
| [prefill.csv](../../benchmarks/prefill.csv) | Prefill 完整 bank 延迟 | [配置与结果](../results.md) |
| [decode.csv](../../benchmarks/decode.csv) | 完整模型 Decode Step | [配置与结果](../results.md) |
| [http_fusion.csv](../../benchmarks/http_fusion.csv) | 三 Profile HTTP 吞吐 | [配置与结果](../results.md) |

算子融合的区间统计沿用 `docs/results.md`：NormPack 约 33–39%，Front 约 52–55%，完整 XQA Tail 约 0.7–1.3%，完整 GPU Decode 调用 2.04–7.18%。

Prefill 表按每个长度的最佳 CPP/VPP 配置组织。Decode 表使用共同输出窗口的 Step 统计；HTTP 表使用两轮正式 bank 的输出 Token 总数与总时长。完整配置在网页各章的实验折叠区。

## 交互模型

### 图 01–02：CPP / VPP

`BlogModel.schedule(n, growth, kind)` 使用依赖图生成时间线。

- CPP 每卡一个 Stage，VPP 每卡两个 Virtual Stage。
- CPP 单 Stage 成本为 `1 + growth × chunk_index`；VPP 单 Virtual Stage 为其一半。
- VPP 暖机顺序为 7/5/3/1，然后交错执行 Return/Forward 并排空。
- 一个任务同时等待前一层 Stage 的输出和本卡前一项任务的完成。
- 对应源码：`src/qwen_infra/vpp/layout.py`。
- 模型只设置计算节点成本，图中时间采用归一化单位。

### 图 03–04：Rank 与 DMA

- 最终资源：每 Rank 一条 Compute、一条 D2H、一条 H2D，两个共享 Source Slot、三个共享 Receive Slot。
- 每条边的 Host Ring：8 × 16 MiB。
- Source 生命周期跟随 D2H；Host Segment 跟随 H2D；Receive 跟随 Consumer Event。
- 对应文档：`docs/vpp-resources.md`。
- 对应源码：`vpp/native/stages.py`、`dma_edge.cpp`、`device_pool.cpp`。
- 时间线用相对单位展示合法的数据依赖顺序和分段循环。

### 图 05–07：Bucket 与 Merge

`BlogModel.taskPlan(lengths, ctas)` 等价实现区间式 Task 分配：

```text
Tile = 256 tokens
budget = max(2, ceil(total_tiles / ctas))
Task = request 区间与 Bucket 区间的交集
```

- `nativePlan()` 展示项目对照中统一 Split / Cyclic 的工作分配。
- B9 示例使用 16 行 padded descriptor；B32 使用 32 行。
- `tasks`、`requests`、`buckets` 对应原 C++ / Python 描述表。
- Last-CTA 图按 T2、T0、T4、T3、T1 的完成顺序说明归约职责。
- Partial O 是局部归一化输出，Merge 权重为 `l_i × exp(m_i − m)`。
- 对应源码：`kernels/nvfp4_bucket_xqa.py`、`nvfp4_bucket_xqa.cu`、`xqa_task_impl.cuh`。

### 图 08–10：融合

- Norm 图保留 BF16 Residual，并把归一化结果直接打包为 NVFP4。
- Front 图分别展示 Q/K 的 Norm → RoPE，以及 V 的量化写入路径。
- Tail 图在最终 Attention 输出阶段执行 Gate 与 FP8，保留 Partial 合并格式。
- 原生边界图按数据通路组织，Gate / Quant 等编译尾处理合并显示。
- 图中按每卡、每 token 计算一次 BF16 中间张量写回和重读的逻辑字节数。Norm 的 y 为 2 × 5120 × 2 B = 20 KiB，Front 的旋转 K 为 2 × 256 × 2 B = 1 KiB/KV head，Tail 的最终 O 为 2 × 6 × 256 × 2 B = 6 KiB。缓存命中会影响实际 GDDR7 流量。
- 对应源码：`nvfp4_norm_pack.cu`、`nvfp4_attention_front.cu`、`nvfp4_bucket_xqa.cu`、`xqa_task_impl.cuh`。

## 上游与参考阅读

1. VPP 作者代码：
   https://github.com/RookieCoder-Camera/vllm-ascend/blob/c409752993b80524e4af6b98ef3b69a8f9c13f2c/vllm_ascend/distributed/vpp_utils.py
2. VPP 论文入口（沿用项目引用）：https://arxiv.org/abs/2608.26523
3. Tencent HPC-Ops / vLLM Blog：
   https://vllm-project.github.io/2026/07/06/vllm-hpc-ops.html
4. Tencent Assign 实现：
   https://github.com/Tencent/hpc-ops/blob/2a2e26562433a8ba4b504858f1c938eb7612c901/src/attention/decode/sm90/assign_task.cu
5. FA3 变长 Split 元数据：
   https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_prepare_scheduler.cu
6. FA4 变长 Split 元数据：
   https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/prepare_scheduler.py
7. FA4 论文：https://arxiv.org/abs/2603.05451
8. FlashInfer XQA：
   https://github.com/flashinfer-ai/flashinfer/tree/0179e5b312d6aa0c790e8e571b9037bb277f4dbd/csrc/xqa
9. 排版与讲解层次参考：
   https://vllm-project.github.io/2025/09/05/anatomy-of-vllm.html

正文与图形为本 Blog 编写；源码实现的来源、版权和许可证继续遵循各自仓库中的 NOTICE 与 LICENSE。
