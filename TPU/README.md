# Google Cloud TPU Architecture & Distributed Training Series

> **Frontier Scale Pre-Training, Supercluster Resiliency, and Systems Optimizations on Cloud TPU**  
> Part of [Bharath's AI Infrastructure Playbook (`BRs-AI-Infra-Playbook`)](../../README.md)

Welcome to the **Google Cloud TPU Track**. Cloud TPU superclusters (`v6e`, `v5p`, and next-gen `v7x Ironwood`) provide world-leading compute density and power efficiency for training foundation models. However, scaling synchronous training across multiple accelerator slices over optical Inter-Chip Interconnect (ICI) tori and Megascale Data Center Networks (DCN) requires deep systems engineering to survive routine cluster disruptions without sacrificing compute goodput.

---

## 📖 Published & Upcoming Articles

### 1. [How to Protect Multi-Slice LLM Pre-Training from Cluster Disruptions with Google Cloud Pathways and Cloud TPU](./01-pathways-elastic-training/README.md)
* **Status:** ✅ **Published**
* **Reading Time:** ~7 min read
* **Hardware:** Cloud TPU v6e (`v6e-256` multi-slice topology)
* **Abstract:** In traditional Multi-Controller JAX (McJAX), a single host hardware fault, OS upgrade, or Spot preemption crashes 100% of the job across all slices. By adopting **Google Cloud Pathways with MaxText**, teams decouple the Python coordinator onto a CPU node pool (`restartCount = 0`), isolating the failure blast radius to a single slice. This guide details **Option 1 (Replica Resize)** for unbroken hero runs, **Option 2 (Pause and Resume)** for fixed-batch invariance, two-tier timeout escalation with **Kueue `waitForPodsReady`**, and **GCS HNS + OCDBT** storage optimizations.
* **Diagrams:**
  * [Figure 1: McJAX Failure Cascade vs. Pathways Single-Controller Elasticity](./01-pathways-elastic-training/diagrams/mcjax_vs_pathways_architecture.png)
  * [Figure 2: Elastic Recovery Modes & Two-Tier Timeout Escalation](./01-pathways-elastic-training/diagrams/elastic_modes_decision_flow.png)

---

### 2. Multi-Tier Checkpointing (MTC): Staging Checkpoints in Host RAM and Adjacent Peer Nodes Over DCN
* **Status:** ⏳ *In Review (Part 2 of Pathways Elasticity)*
* **Abstract:** Why write every checkpoint to remote object storage when an adjacent TPU host is microseconds away? This article explores how combining Pathways elasticity with Multi-Tier Checkpointing stages high-frequency checkpoints in local `/dev/shm` and replicates state to peer nodes over Megascale DCN, unlocking near-instantaneous rewind recovery.

---

### 3. Scaling PyGrain Data Loaders on Multi-Host Cloud TPU v6e Without Host RAM OOMs
* **Status:** ⏳ *In Development*
* **Abstract:** Root-cause analysis and architectural guide to solving host RAM exhaustion in distributed PyGrain multi-domain pipelines through ArrayRecord offloaded indexing, thread pool limits, and lazy dataset iterator sharding.
