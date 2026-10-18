# Bharath's AI Infrastructure Playbook (`BRs-AI-Infra-Playbook`)

> **Frontier GPU & TPU Architecture Guides, Systems Optimizations, and Production Write-Ups**  
> Maintained by [Bharath Rathinam](https://github.com/engineerbharath12) · Solutions Architect & AI Systems Specialist

Welcome to the **AI Infrastructure Playbook**. This repository is an engineering and architectural knowledge base designed for AI infrastructure engineers, ML platform teams, and technical leadership scaling foundation model pre-training and high-throughput inference across **Google Cloud TPU** and **NVIDIA GPU** superclusters.

Every article in this series is grounded in empirical customer deployments, featuring:
* **Production Runbooks & Recipes:** Ready-to-apply configurations, flags, and deployment manifests.
* **Declarative Vector Diagrams:** Native 16:9 Bento architecture and workflow diagrams built with [Dendrite](https://dendrite-758054785671.cr.gclb.goog/), exported in PNG, SVG, and Draw.io formats.
* **Mathematical First Principles:** Rigorous stage-elimination recovery matrices, collective communication bounds, and storage I/O formulations.

---

## 🗂 Series Structure

The playbook is organized by compute architecture:

```text
BRs-AI-Infra-Playbook/
├── TPU/                                        # Google Cloud TPU Track
│   ├── README.md                              # TPU Series Index & Roadmap
│   └── 01-pathways-elastic-training/          # Article 1: Multi-Slice Pathways Elasticity
│       ├── README.md                          # Full technical write-up with embedded figures
│       └── diagrams/                          # High-resolution PNG, SVG, Dendrite DSL, Draw.io
└── GPU/                                        # NVIDIA GPU Track
    └── README.md                              # GPU Series Index & Roadmap
```

---

## 📚 Curated Articles

### 🚀 [Google Cloud TPU Series](./TPU/README.md)
| # | Article Title | Focus Area | Status |
| :-: | :--- | :--- | :---: |
| **01** | **[How to Protect Multi-Slice LLM Pre-Training from Cluster Disruptions with Google Cloud Pathways and Cloud TPU](./TPU/01-pathways-elastic-training/README.md)** | Single-Controller Elasticity, Replica Resize, Pause/Resume, Kueue Two-Tier Timeout, GCS HNS/OCDBT | **Published** |
| **02** | **Multi-Tier Checkpointing (MTC): Staging Checkpoints in Host RAM and Adjacent Peer Nodes Over DCN** | In-Memory Checkpointing, Microsecond Recovery, Peer-Node Replication | *Coming Soon (Part 2)* |
| **03** | **Scaling PyGrain Data Loaders on Multi-Host Cloud TPU v6e Without Host RAM OOMs** | PyGrain, ArrayRecord, Memory Profiling, Distributed Data Ingestion | *Coming Soon* |

### ⚡ [NVIDIA GPU Series](./GPU/README.md)
| # | Article Title | Focus Area | Status |
| :-: | :--- | :--- | :---: |
| **01** | **Accelerating Multi-GPU Collective Bandwidth on Google Cloud G4 VMs via PCIe Switch P2P (`NCCL_P2P_LEVEL`)** | Blackwell / Ada Lovelace, PCIe Gen5 Switch, NCCL Collective Tuning | *In Progress* |
| **02** | **Disaggregated Prefix Caching and LMCache Tier-3 Local NVMe Offload on GKE** | vLLM, Inference Gateway, Prefix Routing, TTFT & Goodput Optimization | *Coming Soon* |

---

## 🎨 Diagram & Asset Standards
All architectural diagrams in this playbook are rendered with [Dendrite](https://dendrite-758054785671.cr.gclb.goog/) using the 16:9 Bento Matrix standard. Each article's `./diagrams/` folder includes:
* **High-DPI PNG (`@2x` Retina):** Optimized for direct copy/paste into Google Docs and publishing to Medium.
* **Lossless SVG:** Scalable vector format for presentations and high-resolution zooming.
* **Editable Draw.io XML (`.drawio`):** Editable directly in [diagrams.net (Draw.io)](https://app.diagrams.net/).
* **Declarative Dendrite DSL (`.dendrite`):** Declarative source code for live modification in Dendrite Studio.
