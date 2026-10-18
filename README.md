# Bharath's AI Infrastructure Playbook (`BRs-AI-Infra-Playbook`)

> **Frontier GPU & TPU Architecture Guides, Systems Optimizations, and Production Write-Ups**  
> Maintained by [Bharath Rathinam](https://github.com/engineerbharath12) · Solutions Architect & AI Systems Specialist

Welcome to the **AI Infrastructure Playbook**. This repository is an engineering and architectural knowledge base designed for AI infrastructure engineers, ML platform teams, and technical leadership scaling foundation model pre-training and high-throughput inference across **Google Cloud TPU** and **NVIDIA GPU** superclusters.

Every article in this series is grounded in empirical customer deployments and production optimizations, complete with embedded architectural diagrams, exact deployment manifests, and mathematical first principles.

---

## Series Structure

```text
BRs-AI-Infra-Playbook/
├── TPU/                                                    # Cloud TPU Track
│   ├── README.md                                          # TPU Series Index & Summaries
│   ├── 01-pathways-elastic-training/                      # Article 1: Multi-Slice Pathways Elasticity
│   ├── 02-tpu-pallas-kernel-optimizations/                # Article 2: Custom TPU Kernels in Pallas
│   ├── 03-whisper-tpu-part2-accuracy-latency-throughput/  # Article 3: Whisper on TPU (Part 2)
│   └── 04-whisper-tpu-part1-deep-dive/                    # Article 4: Whisper on TPU (Part 1)
└── GPU/                                                    # GPU Track
    ├── README.md                                          # GPU Series Index & Summaries
    ├── 01-diffusion-caching-and-video-quality/            # Article 1: Diffusion Denoising Caching & VBench
    ├── 02-video-generation-mid-tier-blackwell-clusters/   # Article 2: Video Gen on Mid-Tier Blackwell
    ├── 03-vision-transformers-nvidia-media-engines-deepcache/ # Article 3: NVDEC & DeepCache ViT Optimization
    └── 04-70b-model-training-mfu-optimization/            # Article 4: 70B Training MFU & Goodput
```

---

## All Articles by Hardware Platform

### [Google Cloud TPU Series](./TPU/README.md)
| # | Article Title | Description |
| :-: | :--- | :--- |
| **01** | **[How to Protect Multi-Slice LLM Pre-Training with Pathways and Cloud TPU](./TPU/01-pathways-elastic-training/README.md)** | Decouples the training coordinator from TPU slices to isolate hardware failures, eliminate recompilation stalls, and maintain continuous multi-slice training goodput. |
| **02** | **[Unlocking Kernel-Level Optimizations on TPUs using Pallas](./TPU/02-tpu-pallas-kernel-optimizations/README.md)** | Guides engineers through authoring custom TPU kernels in Pallas by managing VMEM/HBM memory hierarchies and DMA pipelining to hit peak hardware rooflines. |
| **03** | **[Unlocking Supersonic Speech Recognition: Part 2 — Accuracy, Latency & Throughput](./TPU/03-whisper-tpu-part2-accuracy-latency-throughput/README.md)** | Analyzes the production trade-offs between word error rate, real-time factor (RTFx), and tail latency when scaling concurrent Whisper speech pipelines on Cloud TPU. |
| **04** | **[Unlocking Supersonic Speech Recognition: Optimizing Whisper on TPUs (Part 1)](./TPU/04-whisper-tpu-part1-deep-dive/README.md)** | Re-architects OpenAI Whisper in native JAX to saturate TPU Matrix Multiply Units (MXUs), breaking sequential decoding bottlenecks to achieve over 1,500x RTFx. |

### [NVIDIA GPU Series](./GPU/README.md)
| # | Article Title | Description |
| :-: | :--- | :--- |
| **01** | **[Beyond the Hardware: A Technical Guide to Diffusion Caching Methods](./GPU/01-diffusion-caching-and-video-quality/README.md)** | Evaluates dynamic feature-caching algorithms for Diffusion Transformers (DiT) and establishes quantitative VBench metrics to accelerate video generation without visual degradation. |
| **02** | **[Breaking the Economic Barrier: Video Generation on Mid-Tier Blackwell Clusters](./GPU/02-video-generation-mid-tier-blackwell-clusters/README.md)** | Serves large-scale video diffusion ensembles cost-effectively on mid-tier Blackwell infrastructure by combining Ulysses sequence parallelism with direct PCIe switch P2P collectives. |
| **03** | **[Accelerating Vision Transformers: Breaking the I/O Wall with NVIDIA Media Engines](./GPU/03-vision-transformers-nvidia-media-engines-deepcache/README.md)** | Eliminates CPU-to-GPU data movement bottlenecks in vision transformer pipelines by coupling NVDEC hardware decoding with zero-copy CUDA memory residency and DeepCache reuse. |
| **04** | **[The Road to the Golden State: Optimizing 70B Model Training by Focusing on MFU](./GPU/04-70b-model-training-mfu-optimization/README.md)** | Details an empirical optimization methodology for 70B LLM pre-training on 512-GPU superclusters that raises Model Flops Utilization (MFU) from 32% to over 54% through collective overlap. |
