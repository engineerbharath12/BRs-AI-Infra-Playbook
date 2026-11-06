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
└── 01-pytorch-on-tpu-evolution-torchtpu/
```

---

## All Articles in This Track

| # | Article Title | Description |
| :-: | :--- | :--- |
| **01** | **[The evolution of PyTorch on Cloud TPUs: From JAX and PyTorch/XLA to TorchAX and native TorchTPU](./01-pytorch-on-tpu-evolution-torchtpu/)** | Traces how PyTorch execution on Cloud TPUs evolved from lazy XLA graph tracing and runtime JAX shims to native TorchTPU lowering via PrimTorch's ~250 Core ATen primitives. |

---

## Article Summaries & Hardware Profiles

### 1. [The evolution of PyTorch on Cloud TPUs: From JAX and PyTorch/XLA to TorchAX and native TorchTPU](./01-pytorch-on-tpu-evolution-torchtpu/)
* **Topic:** PrimTorch & Core ATen (~250 Ops), PrivateUse1 C++ Dispatch, PyTorch/XLA vs. TorchAX vs. Native TorchTPU, StableHLO & XLA VMEM Fusion, Pallas Interop
* **Hardware:** Google Cloud TPU v6e (Trillium) / TPU v7 (Ironwood)
* **Summary:** Architectural deep dive into the PyTorch 2.x operator funnel and the three generations of PyTorch on Cloud TPUs—explaining how native TorchTPU unifies first-class PyTorch eager and torch.compile execution with XLA whole-graph VMEM fusion and custom Pallas kernels.
