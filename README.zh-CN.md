# Qwen3.8 SM120：VPP 与 Bucket Attention

[![CI](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/actions/workflows/ci.yml)

[English](README.md) · [架构](docs/architecture.md) ·
[实验结果](docs/results.md) · [安装](docs/installation.md)

基于 **vLLM 与 FlashInfer** 的 Qwen3.8 推理优化，包含三个层级：

- **Prefill：**四卡折返式 VPP，配合 CUDA 分段 DMA 或 NCCL 边界传输；DMA 路径
  每个 Rank 共享 2 个输出 Slot 与 3 个输入 Slot，见[资源模型](docs/vpp-resources.md)。
- **Decode：**按 KV 工作量分桶，跨兼容层复用计划，保留原生回退。
- **算子：**Residual/RMSNorm→NVFP4、QK Norm/RoPE/KV 写入、Attention 输出与 FP8 量化融合。

## 快速开始

在 Linux NVIDIA GPU 机器上克隆，使用一条部署命令创建私有环境并启动服务：

```bash
git clone https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention.git
cd qwen3.8-sm120-vpp-bucket-attention
bash scripts/bootstrap.sh --with-toolkit -- serve \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --profile decode-bucket --tensor-parallel-size 4 \
  --max-model-len 32768 --host 127.0.0.1 --port 8000
```

`--model` 可以是自己的模型目录或 Hugging Face 模型 ID。设备通过标准
`CUDA_VISIBLE_DEVICES` 选择；缓存位置可配置。代码不要求作者的目录、环境名、
固定 GPU 编号、历史实验文件或源码哈希。
已测权重是 [NVIDIA 的混合 NVFP4 版本](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)，
已有本地副本时直接传目录即可复用。
已有兼容 CUDA 工具链时可以省略 `--with-toolkit`；该选项只安装到私有环境中。

```bash
# 不加载权重的依赖诊断
.venv/bin/qwen-infra doctor

# 原生路径；或在 Bucket 上启用融合候选
.venv/bin/qwen-infra serve --model /path/to/model --profile native
.venv/bin/qwen-infra serve --model /path/to/model \
  --profile decode-fused --tensor-parallel-size 4

# 四卡 VPP，执行 Prefill 并生成一个 token
.venv/bin/qwen-infra prefill --model /path/to/Qwen-NVFP4 \
  --profile vpp --chunk-size 2048 --prompt-tokens 8192 --requests 1
```

加上 `--dry-run` 可以先查看启动配置。服务请求、调度和 OpenAI 兼容 API 仍由
vLLM 负责。VPP 当前是明确受支持形状的 P+1 路径；算子支持范围见安装文档，
不把某一台实验机的配置当作所有机器的要求。

## 已有收益

| 层级 | 已完成实验的观察 |
| --- | --- |
| Prefill | B4、8K–128K，最佳已测 VPP 对最佳 CPP 的完整 bank 延迟降低 **4.47–8.55%**。 |
| Decode | 严重 KV 不均衡的 B9/B32 完整模型窗口，步延迟降低 **28.28% / 36.55%**。 |
| HTTP | 有限混合负载 bank 的 Bucket 输出吞吐提高约 **10–11%**；均匀负载接近持平。 |
| 融合 | 受控 1K/2K、B1/8/16/32 GPU 调用，相对同一 Bucket 路径降低 **2.04–7.18%**。 |
| 融合接入 HTTP | 额外吞吐变化 **−1.46% 至 +1.76%**，尚无普遍服务收益。 |

这些数字来自整理前完成的研究实验。新包的安装与 smoke 验证另行记录，不能混作
重新跑过完整性能曲线。各层的分母不同，收益不能直接相加。完整口径、负结果和
适用范围见[实验结果](docs/results.md)。

## 仓库内容

代码在 `src/qwen_infra/`，启动脚本在 `scripts/`，测试在 `tests/`，架构与结果在
`docs/` 和 `benchmarks/`。不包含权重、环境缓存、私有 NVIDIA 原件或历史研究流水账。
项目使用 [Apache-2.0](LICENSE)，第三方来源与许可证见 [NOTICE](NOTICE)。
