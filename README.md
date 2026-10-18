# Bharath's AI Infrastructure Playbook (`BRs-AI-Infra-Playbook`)

> **Frontier GPU & TPU Architecture Guides, Systems Optimizations, and Production Write-Ups**  
> Maintained by [Bharath Rathinam](https://github.com/engineerbharath12) · Solutions Architect & AI Systems Specialist

Welcome to the **AI Infrastructure Playbook**. This repository is an engineering and architectural knowledge base designed for AI infrastructure engineers, ML platform teams, and technical leadership scaling foundation model pre-training and high-throughput inference across **Google Cloud TPU** and **NVIDIA GPU** superclusters.

Every article in this series is grounded in empirical customer deployments and production optimizations, complete with embedded architectural diagrams, exact deployment manifests, and mathematical first principles.

---

## 🗂 Series Structure

```text
BRs-AI-Infra-Playbook/
├── TPU/                                                    # Google Cloud TPU Track
│   ├── README.md                                          # TPU Series Index & Roadmap
│   ├── 01-pathways-elastic-training/                      # Article 1: Multi-Slice Pathways Elasticity
│   ├── 02-tpu-pallas-kernel-optimizations/                # Article 2: Custom TPU Kernels in Pallas
│   ├── 03-whisper-tpu-part2-accuracy-latency-throughput/  # Article 3: Whisper on TPU (Part 2)
│   └── 04-whisper-tpu-part1-deep-dive/                    # Article 4: Whisper on TPU (Part 1)
└── GPU/                                                    # NVIDIA GPU Track
    ├── README.md                                          # GPU Series Index & Roadmap
    ├── 01-diffusion-caching-and-video-quality/            # Article 1: Diffusion Denoising Caching & VBench
    ├── 02-video-generation-mid-tier-blackwell-clusters/   # Article 2: Video Gen on Mid-Tier Blackwell
    ├── 03-vision-transformers-nvidia-media-engines-deepcache/ # Article 3: NVDEC & DeepCache ViT Optimization
    └── 04-70b-model-training-mfu-optimization/            # Article 4: 70B Training MFU & Goodput
```

---

## 📚 All Articles by Hardware Platform

### 🚀 [Google Cloud TPU Series](./TPU/README.md)
| # | Article Title | Focus Area | Status |
| :-: | :--- | :--- | :---: |
| **01** | **[How to Protect Multi-Slice LLM Pre-Training with Pathways and Cloud TPU](./TPU/01-pathways-elastic-training/README.md)** | Single-Controller Elasticity, Replica Resize, Kueue Two-Tier Timeout, GCS HNS | **Published** |
| **02** | **[Unlocking Kernel-Level Optimizations on TPUs using Pallas](./TPU/02-tpu-pallas-kernel-optimizations/README.md)** | Custom Pallas Kernels, VMEM/HBM Memory Hierarchy, Lattice Boltzmann | **Medium** |
| **03** | **[Unlocking Supersonic Speech Recognition: Part 2 — Accuracy, Latency & Throughput](./TPU/03-whisper-tpu-part2-accuracy-latency-throughput/README.md)** | Whisper ASR Serving, High-Throughput Batching, RTFx Tuning | **Medium** |
| **04** | **[Unlocking Supersonic Speech Recognition: Optimizing Whisper on TPUs (Part 1)](./TPU/04-whisper-tpu-part1-deep-dive/README.md)** | Native JAX Whisper, Matrix Multiply Units (MXUs), 1500x RTFx | **Medium** |

### ⚡ [NVIDIA GPU Series](./GPU/README.md)
| # | Article Title | Focus Area | Status |
| :-: | :--- | :--- | :---: |
| **01** | **[Beyond the Hardware: A Technical Guide to Diffusion Caching Methods](./GPU/01-diffusion-caching-and-video-quality/README.md)** | DiT Denoising Caching, First-Block Caching, TeaCache, VBench | **Medium** |
| **02** | **[Breaking the Economic Barrier: Video Generation on Mid-Tier Blackwell Clusters](./GPU/02-video-generation-mid-tier-blackwell-clusters/README.md)** | Wan 2.2 / Hunyuan, G4 RTX 6000 Ada, PCIe Gen5 Switch P2P | **Medium** |
| **03** | **[Accelerating Vision Transformers: Breaking the I/O Wall with NVIDIA Media Engines](./GPU/03-vision-transformers-nvidia-media-engines-deepcache/README.md)** | NVDEC Hardware Decoders, Zero-Copy Tensors, DeepCache ViT | **Medium** |
| **04** | **[The Road to the Golden State: Optimizing 70B Model Training by Focusing on MFU](./GPU/04-70b-model-training-mfu-optimization/README.md)** | 512x H100 Superclusters, MFU 32% -> 54%, 3D Parallelism Overlap | **Medium** |
