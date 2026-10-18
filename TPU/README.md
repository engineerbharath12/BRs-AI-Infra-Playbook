# Google Cloud TPU Architecture & Distributed Training Series

> **Frontier Scale Pre-Training, Supercluster Resiliency, and Systems Optimizations on Cloud TPU**  
> Part of [Bharath's AI Infrastructure Playbook (`BRs-AI-Infra-Playbook`)](../)

Welcome to the **Google Cloud TPU Track**. Cloud TPU superclusters (`v6e`, `v5p`, and next-gen `v7x Ironwood`) provide industry-leading compute density and power efficiency for training foundation models. However, scaling synchronous training across multiple accelerator slices over optical Inter-Chip Interconnect (ICI) tori and Megascale Data Center Networks (DCN) requires deep systems engineering to survive routine cluster disruptions without sacrificing compute goodput.

---

## Published Articles


### 1. [How to Protect Multi-Slice LLM Pre-Training from Cluster Disruptions with Google Cloud Pathways and Cloud TPU](./01-pathways-elastic-training/)
* **Topic:** Single-Controller Elasticity, Replica Resize, Pause/Resume, Kueue Two-Tier Eviction, GCS HNS/OCDBT
* **Hardware:** Cloud TPU v6e (`v6e-256` multi-slice topology)
* **Summary:** Decouples the Python coordinator onto a CPU node pool (`restartCount = 0`), isolating the failure blast radius to a single slice and cutting recovery overhead by skipping pod bootstrapping and XLA recompilation.

---


### 2. [Unlocking Kernel-Level Optimizations on TPUs using Pallas: A Getting Started Guide](./02-tpu-pallas-kernel-optimizations/)
* **Topic:** Custom XLA Kernel Engineering, Pallas DSL, Mosaic GPU/TPU, Memory Hierarchy (VMEM vs. HBM)
* **Hardware:** Cloud TPU v4 / v5e / v6e
* **Summary:** A practitioner's guide to writing custom low-level TPU kernels in Pallas—bypassing high-level compiler heuristics to achieve peak compute roofline on matrix-vector products and scientific lattice computations.

---


### 3. [Unlocking Supersonic Speech Recognition: Part 2 — Balancing Accuracy, Latency, and Throughput on TPUs](./03-whisper-tpu-part2-accuracy-latency-throughput/)
* **Topic:** Large-Scale ASR Serving, High-Concurrency Batching, RTFx Optimization, Dynamic Latency Tuning
* **Hardware:** Cloud TPU v5e / v6e
* **Summary:** Deep dive into production trade-offs between word error rate (WER), real-time factor (RTFx), and P99 tail latency when serving OpenAI Whisper at scale on Cloud TPU.

---


### 4. [Unlocking Supersonic Speech Recognition: A Deep Dive into Optimizing Whisper on TPUs](./04-whisper-tpu-part1-deep-dive/)
* **Topic:** JAX Model Sharding, Speculative Decoding, Mel-Filterbank Vectorization, FlashAttention on TPU
* **Hardware:** Cloud TPU v5e / v6e
* **Summary:** How rewriting Whisper in native JAX and leveraging TPU Matrix Multiply Units (MXUs) broke past CPU/GPU latency bottlenecks to achieve over 1,500x Real-Time Factor (RTFx).
