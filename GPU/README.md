# NVIDIA GPU Supercomputing & Inference Systems Series

> **High-Performance Multi-GPU Distributed Training, NCCL Tuning, and Low-Latency Serving**  
> Maintained by [Bharath Rathinam](https://github.com/engineerbharath12) · Part of [Bharath's AI Infrastructure Playbook (`BRs-AI-Infra-Playbook`)](../)

Welcome to the **NVIDIA GPU Track**. Scaling generative inference and multi-billion-parameter foundation model training across Google Cloud's A3 Ultra (H200 SXM5), A3 Mega (H100), and next-generation G4 (RTX 6000 Ada / Blackwell) families requires hardware-aware orchestration across PCIe switches, NVLink interconnects, and local NVMe storage.

Every guide in this track is grounded in empirical benchmarks, architectural trade-offs, and production recipes.

---

## Track Structure

```text
GPU/
├── README.md
├── 01-diffusion-caching-and-video-quality/
├── 02-video-generation-mid-tier-blackwell-clusters/
├── 03-vision-transformers-nvidia-media-engines-deepcache/
└── 04-70b-model-training-mfu-optimization/
```

---

## All Articles in This Track

| # | Article Title | Description |
| :-: | :--- | :--- |
| **01** | **[Beyond the Hardware: A Technical Guide to Diffusion Caching Methods](./01-diffusion-caching-and-video-quality/)** | Evaluates dynamic feature-caching algorithms for Diffusion Transformers (DiT) and establishes quantitative VBench metrics to accelerate video generation without visual degradation. |
| **02** | **[Breaking the Economic Barrier: Video Generation on Mid-Tier Blackwell Clusters](./02-video-generation-mid-tier-blackwell-clusters/)** | Serves large-scale video diffusion ensembles cost-effectively on mid-tier Blackwell infrastructure by combining Ulysses sequence parallelism with direct PCIe switch P2P collectives. |
| **03** | **[Accelerating Vision Transformers: Breaking the I/O Wall with NVIDIA Media Engines](./03-vision-transformers-nvidia-media-engines-deepcache/)** | Eliminates CPU-to-GPU data movement bottlenecks in vision transformer pipelines by coupling NVDEC hardware decoding with zero-copy CUDA memory residency and DeepCache reuse. |
| **04** | **[The Road to the Golden State: Optimizing 70B Model Training by Focusing on MFU](./04-70b-model-training-mfu-optimization/)** | Details an empirical optimization methodology for 70B LLM pre-training on 512-GPU superclusters that raises Model Flops Utilization (MFU) from 32% to over 54% through collective overlap. |

---

## Article Summaries & Hardware Profiles

### 1. [Beyond the Hardware: A Technical Guide to Diffusion Caching Methods and Video Quality Determination](./01-diffusion-caching-and-video-quality/)
* **Topic:** Diffusion Acceleration, First-Block Caching (FBC), TeaCache, Context-Aware Token Pruning, VBench
* **Hardware:** NVIDIA H100 / RTX 6000 Ada / Blackwell
* **Summary:** Mathematical and architectural guide to taming the quadratic cost of diffusion transformer (DiT) denoising steps using dynamic caching algorithms without visual degradation.

---

### 2. [Breaking the Economic Barrier: High-Efficiency Video Generation on Mid-Tier Blackwell Clusters](./02-video-generation-mid-tier-blackwell-clusters/)
* **Topic:** Video Diffusion (Wan 2.2 / Hunyuan), G4 VM Architecture, PCIe Gen5 Switch P2P, Ulysses Context Parallelism
* **Hardware:** Google Cloud G4 instances (NVIDIA RTX 6000 Ada / Blackwell architecture)
* **Summary:** How to serve massive 14B–19B video diffusion ensembles at a fraction of H100 cost by unlocking direct PCIe P2P collectives and avoiding CPU root complex round-trips.

---

### 3. [Accelerating Vision Transformers: Breaking the I/O Wall with NVIDIA Media Engines & DeepCache Architecture](./03-vision-transformers-nvidia-media-engines-deepcache/)
* **Topic:** Hardware Video Decoders (NVDEC), Zero-Copy Tensor Streams, DeepCache, Spatial-Temporal Latent Sharding
* **Hardware:** NVIDIA L4 / L40S / H100
* **Summary:** Eliminating the hidden CPU-to-GPU data movement bottleneck in multi-modal vision pipelines by coupling NVDEC hardware decoding with zero-copy CUDA memory residency.

---

### 4. [The Road to the Golden State: How We Optimized 70B Model Training by Focusing on MFU to Save Millions](./04-70b-model-training-mfu-optimization/)
* **Topic:** Model Flops Utilization (MFU), 70B Pre-Training, 3D Parallelism, Communication-Computation Overlap, Goodput
* **Hardware:** 512x NVIDIA H100 / H200 Superclusters
* **Summary:** How analyzing hardware execution timelines and optimizing distributed collective communications elevated instantaneous MFU from ~32% to >54%, shaving weeks off a multi-million-dollar training run.
