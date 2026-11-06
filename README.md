# Bharath's AI Infrastructure Playbook

> **Frontier GPU & TPU Architecture Guides, Systems Optimizations, and Production Write-Ups**  
> Maintained by [Bharath Rathinam](https://github.com/engineerbharath12) · Solutions Architect & AI Systems Specialist

Welcome to the **AI Infrastructure Playbook**. This repository is an engineering and architectural knowledge base designed for AI infrastructure engineers, ML platform teams, and technical leadership scaling foundation model pre-training, multi-modal pipelines, and high-throughput inference across **Google Cloud TPU**, **NVIDIA GPU**, and **Accelerator Diversity** architectures.

Every article in this series is grounded in empirical customer deployments and production optimizations, complete with embedded architectural diagrams, exact deployment manifests, and mathematical first principles.

---

## Playbook Tracks

| Track | Focus Area | Articles |
| :--- | :--- | :-: |
| **[Google Cloud TPU Series](./TPU/)** | Cloud TPU supercluster architecture, Pathways elasticity, Pallas kernels, and high-throughput serving | **4** |
| **[NVIDIA GPU Series](./GPU/)** | Multi-GPU training MFU, NVLink/PCIe P2P collectives, hardware media pipelines, and low-latency inference | **4** |
| **[Accelerator Diversity Series](./Accelerator-Diversity/)** | Democratizing ML frameworks across training and inference (`TorchTPU`, open compilers, and portable runtimes) | **2** |

---

## Series Structure

```text
BRs-AI-Infra-Playbook/
├── TPU/                                             # Cloud TPU Track
│   ├── README.md
│   ├── 01-pathways-elastic-training/
│   ├── 02-tpu-pallas-kernel-optimizations/
│   ├── 03-whisper-tpu-part2-accuracy-latency-throughput/
│   └── 04-whisper-tpu-part1-deep-dive/
├── GPU/                                             # NVIDIA GPU Track
│   ├── README.md
│   ├── 01-diffusion-caching-and-video-quality/
│   ├── 02-video-generation-mid-tier-blackwell-clusters/
│   ├── 03-vision-transformers-nvidia-media-engines-deepcache/
│   └── 04-70b-model-training-mfu-optimization/
└── Accelerator-Diversity/                           # Framework Portability
    ├── README.md
    ├── 01-pytorch-on-tpu-evolution-torchtpu/
    └── 02-torchtpu-video-preprocessing-accelerator-diversity/
```

---

## All Articles by Track & Platform

### [Google Cloud TPU Series](./TPU/)

| # | Article Title | Description |
| :-: | :--- | :--- |
| **01** | **[How to Protect Multi-Slice LLM Pre-Training with Pathways and Cloud TPU](./TPU/01-pathways-elastic-training/)** | Decouples the training coordinator from TPU slices to isolate hardware failures, eliminate recompilation stalls, and maintain continuous multi-slice training goodput. |
| **02** | **[Unlocking Kernel-Level Optimizations on TPUs using Pallas](./TPU/02-tpu-pallas-kernel-optimizations/)** | Guides engineers through authoring custom TPU kernels in Pallas by managing VMEM/HBM memory hierarchies and DMA pipelining to hit peak hardware rooflines. |
| **03** | **[Unlocking Supersonic Speech Recognition: Part 2 — Accuracy, Latency & Throughput](./TPU/03-whisper-tpu-part2-accuracy-latency-throughput/)** | Analyzes the production trade-offs between word error rate, real-time factor (RTFx), and tail latency when scaling concurrent Whisper speech pipelines on Cloud TPU. |
| **04** | **[Unlocking Supersonic Speech Recognition: Optimizing Whisper on TPUs (Part 1)](./TPU/04-whisper-tpu-part1-deep-dive/)** | Re-architects OpenAI Whisper in native JAX to saturate TPU Matrix Multiply Units (MXUs), breaking sequential decoding bottlenecks to achieve over 1,500x RTFx. |

### [NVIDIA GPU Series](./GPU/)

| # | Article Title | Description |
| :-: | :--- | :--- |
| **01** | **[Beyond the Hardware: A Technical Guide to Diffusion Caching Methods](./GPU/01-diffusion-caching-and-video-quality/)** | Evaluates dynamic feature-caching algorithms for Diffusion Transformers (DiT) and establishes quantitative VBench metrics to accelerate video generation without visual degradation. |
| **02** | **[Breaking the Economic Barrier: Video Generation on Mid-Tier Blackwell Clusters](./GPU/02-video-generation-mid-tier-blackwell-clusters/)** | Serves large-scale video diffusion ensembles cost-effectively on mid-tier Blackwell infrastructure by combining Ulysses sequence parallelism with direct PCIe switch P2P collectives. |
| **03** | **[Accelerating Vision Transformers: Breaking the I/O Wall with NVIDIA Media Engines](./GPU/03-vision-transformers-nvidia-media-engines-deepcache/)** | Eliminates CPU-to-GPU data movement bottlenecks in vision transformer pipelines by coupling NVDEC hardware decoding with zero-copy CUDA memory residency and DeepCache reuse. |
| **04** | **[The Road to the Golden State: Optimizing 70B Model Training by Focusing on MFU](./GPU/04-70b-model-training-mfu-optimization/)** | Details an empirical optimization methodology for 70B LLM pre-training on 512-GPU superclusters that raises Model Flops Utilization (MFU) from 32% to over 54% through collective overlap. |

### [Accelerator Diversity Series](./Accelerator-Diversity/)

> **Democratizing ML Frameworks Across Training & Inference:** Architectural guides, compiler deep dives, and production blueprints exploring how **TorchTPU**, open compiler stacks (`PyTorch 2.x`, `StableHLO`, `XLA`), cross-platform training and serving runtimes (`vLLM`, `SGLang`, `JAX`), and multi-accelerator orchestration democratize ML frameworks—enabling engineering teams to develop, train, and serve AI workloads interchangeably across **Google Cloud TPUs** and **NVIDIA GPUs** without vendor or ecosystem lock-in.

| # | Article Title | Description |
| :-: | :--- | :--- |
| **01** | **[The evolution of PyTorch on Cloud TPUs: From JAX and PyTorch/XLA to TorchAX and native TorchTPU](./Accelerator-Diversity/01-pytorch-on-tpu-evolution-torchtpu/)** | Traces how PyTorch execution on Cloud TPUs evolved from lazy XLA graph tracing and runtime JAX shims to native TorchTPU lowering via PrimTorch's ~250 Core ATen primitives. |
| **02** | **[Running GPU-first PyTorch video pipelines on TorchTPU: Split CPU-TPU decoding and accelerator diversity](./Accelerator-Diversity/02-torchtpu-video-preprocessing-accelerator-diversity/)** | Bridges the absence of hardware NVDEC engines on Cloud TPUs using compact uint8 YUV420p split decoding and XLA MXU preprocessing inside a 90/10 unified GPU/TPU codebase. |
