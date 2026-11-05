# Qwen3.8 SM120：VPP 与 Bucket Attention

[![CI](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/workflows/ci.yml)
[![GPU: SM120](https://img.shields.io/badge/GPU-SM120-76B900)](docs/installation.md)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**面向 NVIDIA Blackwell 的 Qwen3.8 推理优化，从流水线调度到 CUDA 算子。**
基于 vLLM 与 FlashInfer，结合长上下文 Prefill 的折返式 VPP、Decode 的
Bucket Attention，以及 Attention 和 MLP 边界上的低精度算子融合。

[English](README.md) · [功能](#功能) · [性能](#性能) · [快速开始](#快速开始) ·
[架构](docs/architecture.md) · [Benchmark](benchmarks/README.md)

## 功能

- **折返式 VPP Prefill。** 在四张 GPU 上执行八个虚拟 Stage，改善 Attention
  随上下文增长造成的流水线填充与排空 Bubble。分段 CUDA DMA 通过 Copy Engine
  重叠通信与计算，并使用紧凑的 Rank 级共享缓冲。
- **按 KV 工作量均衡 Decode。** 共享任务规划器将不同请求的 KV 工作量分配给
  CTA；根据负载选择 Native、Aligned 或 Persistent 路径，兼容的 Attention 层
  复用同一份计划。
- **CUDA 算子融合。** 融合 Residual/RMSNorm→NVFP4 打包、QK Norm/RoPE/KV
  写入、Attention 输出/Gate→FP8 量化，减少 Kernel 启动和中间张量读写。
- **接入 vLLM。** 复用 Continuous Batching、采样和 OpenAI 兼容 API，Prefill
  与 Decode 分别提供执行入口，便于独立优化和测量。

| 组件 | 场景 | 实现 | 入口 |
| --- | --- | --- | --- |
| 折返式 VPP | 长上下文 Prefill，TP1/PP4 | 八个虚拟 Stage，分段 DMA 或 NCCL 通信 | `qwen-infra prefill --profile vpp` |
| Bucket Attention | KV 长度不均衡的 Decode | NVFP4 分页 KV、共享规划、FlashInfer XQA 任务执行 | `qwen-infra serve --profile decode-bucket` |
| 算子融合 | Decode 的 Attention 与 MLP 边界 | CUDA C++ 直接生成 NVFP4/FP8 消费者格式 | `qwen-infra serve --profile decode-fused` |
| 原生执行 | 性能对照 | vLLM 服务与 CPP Prefill | `--profile native` / `--profile cpp` |

## 性能

以下结果使用 **四张同 NUMA 的 RTX 5090**，模型为
`nvidia/Qwen3.8-27B-NVFP4`，KV Cache 使用 NVFP4：

| 优化 | 测量范围 | Baseline | 收益 |
| --- | --- | --- | --- |
| VPP Prefill | 四请求，8K–128K 输入，各配置扫描中的最佳值 | CPP | 完整 Bank 延迟降低 **4.47–8.55%** |
| Bucket Attention | 严重 KV 不均衡，B9 / B32，完整模型 Decode | 原生 vLLM | Step 延迟降低 **28.28% / 36.55%** |
| Bucket 服务 | 混合长度 HTTP 到达负载 | 原生 vLLM | 输出吞吐提高 **10–11%** |
| 三项算子融合 | B1/8/16/32，1K/2K 上下文，完整 GPU Decode 调用 | 相同 Bucket 路径 | 延迟降低 **2.04–7.18%** |

绝对耗时和负载定义见[性能结果](docs/results.md)，对比命令见
[Benchmark](benchmarks/README.md)。

## 快速开始

### 安装并启动服务

在 Linux SM120 GPU 机器上运行以下命令，脚本会创建 `.venv`、安装依赖并启动
OpenAI 兼容服务：

```bash
git clone https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention.git
cd qwen3.8-sm120-vpp-bucket-attention
bash scripts/bootstrap.sh --with-toolkit -- serve \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile decode-bucket --tensor-parallel-size 4 \
  --max-model-len 32768 --host 127.0.0.1 --port 8000
```

`--model` 支持本地权重目录；通过 `CUDA_VISIBLE_DEVICES` 选择设备。已有 CUDA
工具链时可以省略 `--with-toolkit`。详细配置见[安装文档](docs/installation.md)。

### 启用融合算子

```bash
.venv/bin/qwen-infra serve --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile decode-fused --tensor-parallel-size 4 \
  --max-model-len 32768 --host 127.0.0.1 --port 8000
```

### 运行 Prefill

用 VPP 处理四个 8K Prompt，每个请求生成一个 Token：

```bash
.venv/bin/qwen-infra prefill --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile vpp --chunk-size 2048 --prompt-tokens 8192 --requests 4
```

使用 `--profile cpp` 运行原生流水线对照。VPP 的 DMA 路径每个 Rank 共享
两个输出 Slot 和三个输入 Slot，Stream、预取与缓冲生命周期见
[通信设计](docs/vpp-resources.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [安装](docs/installation.md) | 环境安装、启动参数、支持配置 |
| [架构](docs/architecture.md) | VPP、Bucket Attention、融合与精度 |
| [集成设计](docs/design.md) | vLLM 接入点与源码导航 |
| [VPP 通信](docs/vpp-resources.md) | 搬运流水、Event、Credit 与显存预算 |
| [性能结果](docs/results.md) | Prefill、Decode、算子与 HTTP 数据 |
| [Benchmark](benchmarks/README.md) | 对比命令与数据表 |
| [测试](docs/validation.md) | CPU、CUDA 与模型集成测试 |

## 致谢

本项目基于 [vLLM](https://github.com/vllm-project/vllm)、
[FlashInfer](https://github.com/flashinfer-ai/flashinfer)、
[腾讯 HPC-Ops](https://github.com/Tencent/hpc-ops) 和
[VPP 论文与实现](https://arxiv.org/abs/2608.26523) 开展。
算法来源与适配说明见[参考资料](docs/references.md)。

欢迎提交 Issue 和 PR，参见[贡献指南](CONTRIBUTING.md)。
项目采用 [Apache-2.0](LICENSE) 许可证，第三方来源保留在
[NOTICE](NOTICE) 与 [licenses/](licenses/) 中。
