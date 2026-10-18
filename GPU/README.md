# NVIDIA GPU Supercomputing & Inference Systems Series

> **High-Performance Multi-GPU Distributed Training, NCCL Tuning, and Low-Latency Serving**  
> Part of [Bharath's AI Infrastructure Playbook (`BRs-AI-Infra-Playbook`)](../../README.md)

Welcome to the **NVIDIA GPU Track**. Scaling GPU workloads across Google Cloud's A3 Ultra (H200 SXM5), A3 Mega (H100), and next-generation G4 (RTX 6000 Ada / Blackwell) families requires precise orchestration across PCIe switches, NVLink networks, and remote storage file systems.

---

## 📖 Upcoming Articles & Guides

### 1. Accelerating Multi-GPU Collective Bandwidth on Google Cloud G4 VMs via PCIe Switch P2P (`NCCL_P2P_LEVEL`)
* **Status:** ⏳ *In Draft*
* **Hardware:** Google Cloud G4 instances (NVIDIA RTX 6000 Ada / Blackwell architecture)
* **Abstract:** Exploring how enabling direct Peer-to-Peer traffic across PCIe switches by bypassing the CPU Root Complex doubles NCCL collective communication bandwidth on multi-GPU instances.

---

### 2. Disaggregated Prefix Caching and LMCache Tier-3 Local NVMe Offload on GKE
* **Status:** ⏳ *In Draft*
* **Frameworks:** vLLM, GKE Inference Gateway, Endpoint Picker (EPP), LMCache
* **Abstract:** A production benchmark guide on cutting Time-To-First-Token (TTFT) and improving throughput by pairing Kubernetes Gateway API prefix routing with multi-tier local NVMe KV-cache spillover.
