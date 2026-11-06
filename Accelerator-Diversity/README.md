# Accelerator Diversity Series

> **Democratizing ML Frameworks Across Training & Inference on Google Cloud TPU and NVIDIA GPU**  
> Maintained by [Bharath Rathinam](https://github.com/engineerbharath12) · Part of [Bharath's AI Infrastructure Playbook (`BRs-AI-Infra-Playbook`)](../)

Welcome to the **Accelerator Diversity Track**. Scaling modern AI infrastructure requires decoupling model development from single-vendor hardware lock-in so engineering teams can schedule **both training and inference** portably across heterogeneous accelerator fleets.

Every guide in this track focuses on democratizing ML frameworks—exploring how **TorchTPU**, open compiler stacks (`PyTorch 2.x`, `StableHLO`, `XLA`), portable kernel abstractions (`Pallas`, `Triton`), unified training and serving runtimes (`vLLM`, `SGLang`, `JAX`), and multi-accelerator orchestration allow shared codebases to run interchangeably across **Google Cloud TPUs** and **NVIDIA GPUs**.

---

## Track Structure

```text
Accelerator-Diversity/
├── README.md
├── 01-pytorch-on-tpu-evolution-torchtpu/
└── 02-torchtpu-video-preprocessing-accelerator-diversity/
```

---

## All Articles in This Track

| # | Article Title | Description |
| :-: | :--- | :--- |
| **01** | **[The evolution of PyTorch on Cloud TPUs: From JAX and PyTorch/XLA to TorchAX and native TorchTPU](./01-pytorch-on-tpu-evolution-torchtpu/)** | Traces how PyTorch execution on Cloud TPUs evolved from lazy XLA graph tracing and runtime JAX shims to native TorchTPU lowering via PrimTorch's ~250 Core ATen primitives. |
| **02** | **[Running GPU-first PyTorch video pipelines on TorchTPU: Split CPU-TPU decoding and accelerator diversity](./02-torchtpu-video-preprocessing-accelerator-diversity/)** | Bridges the absence of hardware NVDEC engines on Cloud TPUs using compact uint8 YUV420p split decoding and XLA MXU preprocessing inside a 90/10 unified GPU/TPU codebase. |

---

## Article Summaries & Hardware Profiles

### 1. [The evolution of PyTorch on Cloud TPUs: From JAX and PyTorch/XLA to TorchAX and native TorchTPU](./01-pytorch-on-tpu-evolution-torchtpu/)
* **Topic:** PrimTorch & Core ATen (~250 Ops), PrivateUse1 C++ Dispatch, PyTorch/XLA vs. TorchAX vs. Native TorchTPU, StableHLO & XLA VMEM Fusion, Pallas Interop
* **Hardware:** Google Cloud TPU v6e (Trillium) / TPU v7 (Ironwood)
* **Summary:** Architectural deep dive into the PyTorch 2.x operator funnel and the three generations of PyTorch on Cloud TPUs—explaining how native TorchTPU unifies first-class PyTorch eager and torch.compile execution with XLA whole-graph VMEM fusion and custom Pallas kernels.

---

### 2. [Running GPU-first PyTorch video pipelines on TorchTPU: Bridging hardware NVDEC with split CPU-TPU decoding and a common codebase](./02-torchtpu-video-preprocessing-accelerator-diversity/)
* **Topic:** Hardware NVDEC vs. Split CPU-TPU Decode, Planar uint8 YUV420p PCIe Optimization, Fused XLA MXU Bilinear Resize, Static Batch Bucketing, 90/10 Accelerator Diversity Adapter
* **Hardware:** Google Cloud G4 (RTX PRO 6000 Blackwell) & Cloud TPU v6e (Trillium)
* **Summary:** Shows how to run GPU-first PyTorch video pipelines on Cloud TPUs without hardware media decoders—cutting PCIe frame bandwidth by 75%–87.5% and boosting throughput by +122% over host-CPU preprocessing while maintaining a single 90/10 codebase across GPU and TPU.
