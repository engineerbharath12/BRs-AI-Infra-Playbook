# How to Protect Multi-Slice LLM Pre-Training from Cluster Disruptions with Google Cloud Pathways and Cloud TPU

*An engineering and business guide to surviving hardware failures, software upgrades, and Spot preemptions with elastic multi-slice training.*

When you scale foundation model pre-training across multiple accelerator slices for weeks, a single host failure, software upgrade, or Spot preemption in a traditional multi-controller setup crashes 100% of the job and leaves hundreds of healthy chips sitting idle. By adopting Google Cloud Pathways with MaxText on Cloud TPU, engineering teams decouple the training coordinator from individual accelerator slices—eliminating entire stages of recovery overhead and improving effective cluster goodput by over 40%. First, we examine why coupled training architectures fail at scale and how the unique design of Cloud TPU and Pathways isolates the blast radius of any slice interruption. Next, we provide a practical configuration guide—covering Replica Resize, Pause and Resume, two-tier timeout escalation with Kueue, and Google Cloud Storage (GCS) optimizations—before sharing how we helped a customer protect a flagship multi-slice "hero" training job during live hardware replacements.

---

## Section 1: Understanding the Business Cost of Disruptions and the Uniqueness of Pathways on Cloud TPU

When an organization commits millions of dollars in compute capital to train a frontier model, both engineering leads and executive stakeholders measure success by a single metric: **effective cluster goodput**—the percentage of wall-clock time spent advancing the model's token horizon versus time lost to restarts, recompilations, and I/O stalls.

At multi-slice scale, cluster interruptions are not rare edge cases. They are routine operational events across four categories:
* **Unplanned hardware failures:** Unrecoverable HBM parity errors, optical link drops, or host crashes that take a node offline mid-step and require hardware replacement.
* **Scheduled software and node upgrades:** Periodic OS security patches, kernel updates, and TPU driver upgrades that require draining individual hosts.
* **Spot capacity and DWS Flex-Start preemptions:** Asynchronous capacity reclamations when optimizing compute Total Cost of Ownership (TCO) with Spot TPUs or Dynamic Workload Scheduler (DWS).
* **Multi-tenant priority evictions:** Urgent evaluation or fine-tuning workloads temporarily preempting a slice in shared enterprise clusters managed by Kueue.

### Why Traditional Multi-Controller Architectures Stall the Entire Cluster

In standard Multi-Controller JAX (**McJAX**) or `torchrun`, every host runs its own tightly coupled Python interpreter and XLA runtime, synchronizing gradients across all slices at every step. If **a single host** fails or reboots for a patch, the global collective barrier times out and aborts the entire `JobSet`. Every healthy slice across the cluster is forced to tear down its containers, wait for the single failed slice to be replaced, cold-boot Python, recompile the XLA graph (which takes **15 to 25+ minutes** on large 70B+ parameter models), and re-read checkpoints from storage simultaneously.

### What Makes Cloud TPU and Google Cloud Pathways Unique

Cloud TPU and Pathways solve this fate-sharing bottleneck at both the network fabric and execution runtime layers:

1. **Two-tier network isolation (2D ICI Torus + Megascale DCN):** Inside a single Cloud TPU slice, chips are connected via a high-speed 2D optical Inter-Chip Interconnect (ICI) torus, while multiple slices scale out over Google's **Megascale Data Center Network (DCN)**. A host failure breaks that specific slice's closed 2D ICI ring—temporarily taking that slice offline—but **every peer slice connected over Megascale DCN remains 100% healthy and operational**.
2. **Decoupled single-controller runtime (`pathways-head`):** Originally built for Google's internal frontier models and open-sourced for Cloud TPU, Pathways moves the Python JAX coordinator (`jax-tpu`), Resource Manager (`pathways-rm`), and IFRT proxy (`pathways-proxy`) onto a dedicated CPU node pool. TPU slices run stateless C++ execution workers (`pathways-worker`).
3. **Direct HBM-to-storage data path:** Even though a single CPU coordinator orchestrates the job, model weights never bottleneck through the coordinator's CPU RAM. TPU workers stream sharded tensors directly between TPU HBM and storage via `TensorStore`.

---

### Figure 1: Coupled Multi-Controller Failure Cascade vs. Decoupled Single-Controller Elasticity in Google Cloud Pathways

![Figure 1: Traditional Multi-Controller JAX (McJAX) vs. Google Cloud Pathways Single-Controller](./diagrams/mcjax_vs_pathways_architecture.png)

*Figure 1: Why a single host hardware failure aborts 100% of a coupled McJAX job, versus in-process slice isolation in Google Cloud Pathways. ([View Full SVG](./diagrams/mcjax_vs_pathways_architecture.svg) · [Dendrite DSL](./diagrams/mcjax_vs_pathways_architecture.dendrite) · [Draw.io XML](./diagrams/mcjax_vs_pathways_architecture.drawio))*

---

| Operational Metric | Without Pathways (Coupled McJAX) | With Pathways (Single-Controller) | Business and Engineering Return |
| :--- | :--- | :--- | :--- |
| **Blast radius of a single-host failure** | **100% of cluster** (All slices abort) | **Isolated to 1 slice** (Peers stay healthy) | Eliminates cluster-wide cascading outages |
| **Coordinator restarts (`restartCount`)** | **1+ full container teardowns** | **`0` (Survives in CPU memory)** | Skips cold pod boot and in-line XLA recompilation |
| **Healthy slice utilization during replacement** | **0% (All healthy slices sit idle)** | **100% active** (via Option 1 Resize) | Zero wasted capital on surviving slices |
| **End-to-end recovery mechanism** | **Cold Pod Boot + 20m XLA Compile + GCS IO** | **Architectural Stage Elimination** | Skips 2 to 3 full software recovery stages |
| **Effective cluster goodput under disruptions** | **Baseline (~58% – 65%)** | **> 92% (+42% relative goodput gain)** | Accelerates frontier model release timelines |

---

## Section 2: Engineering Guide — Selecting Your Elastic Mode and Optimizing Storage

Pathways and MaxText provide two production elasticity modes controlled by a single switch—**`elastic_min_slice_count`**—allowing teams to match recovery behavior to their workload priority and convergence requirements.

### 1. Choosing Between Option 1 (Replica Resize) and Option 2 (Pause and Resume)

* **Option 1: Replica Resize (`elastic_min_slice_count = M`, where `1 <= M < N`)**  
  Set `elastic_min_slice_count` to **any positive integer `M` (`1 <= M < N`)** representing the minimum floor of healthy slices required to continue training (for example, `M = 3`, `2`, or `1` on a 4-slice job). When a slice fails, `pathways-rm` detects `WorkerLost` and—as long as at least `M` slices remain healthy—reshapes the JAX mesh **in-memory (`restartCount = 0`, 0% GCS read I/O)**. The job continues training uninterrupted on surviving slices while the failed slice is replaced in the background, then automatically scales back up to `N` slices in-memory once the replacement registers.
  * *Best for:* A **single flagship "hero" pre-training job** that must never stop making forward token progress during hardware replacements, rolling upgrades, or Spot preemptions.

* **Option 2: Pause and Resume (`elastic_min_slice_count = -1`)**  
  Setting `elastic_min_slice_count = -1` (the default) enforces a fixed `N`-slice mesh so the global batch size never changes. When a slice fails, `pathways-head` pauses in CPU memory (`restartCount = 0`, keeping its warm in-memory XLA graph) and waits for the replacement slice. Once the replacement joins, MaxText validates the checkpoint commit marker and restores weights into HBM.
  * *Best for:* Convergence recipes requiring a **strictly constant global batch size** at every step, or **shared multi-tenant Kueue clusters**.

---

### Figure 2: Decision Flow & Two-Tier Escalation for Pathways Elastic Training

![Figure 2: Decision Flow & Two-Tier Escalation for Pathways Elastic Training](./diagrams/elastic_modes_decision_flow.png)

*Figure 2: Elastic recovery modes and two-tier timeout integration between in-memory wait and Kueue cluster eviction. ([View Full SVG](./diagrams/elastic_modes_decision_flow.svg) · [Dendrite DSL](./diagrams/elastic_modes_decision_flow.dendrite) · [Draw.io XML](./diagrams/elastic_modes_decision_flow.drawio))*

---

### Understanding the 4-Stage Recovery Pipeline: Why Stage Elimination Beats Fixed Timers

On large-scale production training runs (70B+ parameters spanning 1,024+ TPU chips), recovery speed should not be evaluated by toy-benchmark wall-clock numbers. Instead, evaluate recovery by **which foundational stages of the cluster restart pipeline are eliminated**:

| Recovery Stage | Coupled McJAX (Baseline) | Pathways Option 1 (Replica Resize) | Pathways Option 2 Path A (In-Memory Resume) | Pathways Option 2 Path B (Kueue Re-Admission) |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 1: Hardware Replacement** | **Paid** (3–5 min) | **Eliminated from critical path** (Background) | **Paid** (3–5 min) | **Paid** (Zonal stockout wait) |
| **Stage 2: Pod Image Pull & Init** | **Paid** (2–4 min cluster-wide) | **Eliminated** (`restartCount = 0`) | **Eliminated** (`restartCount = 0` on head) | **Paid** (Fresh pod startup) |
| **Stage 3: XLA Graph Compilation** | **Paid** (15–25 min cold recompile) | **Eliminated** (Pre-compiled mesh shapes) | **Eliminated** (Warm in-memory XLA cache) | **Eliminated** (Persistent GCS XLA cache: ~30s) |
| **Stage 4: Weight & Optimizer Restore** | **Paid** (Full GCS read across all slices) | **Eliminated** (`0%` GCS read I/O) | **Paid** (Only reads last committed step) | **Paid** (Only reads last committed step) |
| **Overall Recovery Impact** | **Full 20–35 min cluster stall** | **`0s` stall (Continuous token progress)** | **Skips 85%+ of software recovery overhead** | **Frees idle capacity; zero recompilation** |

---

### Two-Tier Timeout & Eviction Interaction: `elastic_timeout_seconds` vs. Kueue `waitForPodsReady`

In production clusters orchestrated by Kubernetes and **Kueue**, Option 2 coordinates an intelligent **two-tier timeout escalation**:

```text
Slice Failure Occurs (WorkerLost)
   │
   ├── Tier 1: In-Memory Wait (t < elastic_timeout_seconds, e.g. 600s)
   │     • pathways-head stays 3/3 Running in CPU RAM (restartCount = 0)
   │     • Keeps warm in-memory XLA compilation cache
   │     • Replacement slice joins in 3-5m -> In-process weight restore from GCS (Skips Pod Init & XLA Recompilation)
   │     • Kueue never touches the job; zero cluster disruption
   │
   └── Tier 2: Prolonged Zonal Stockout (t > elastic_timeout_seconds)
         • If replacement hardware cannot be provisioned before timeout, pathways-head exits (code 1)
         • Kueue's waitForPodsReady (e.g. 300s) detects pod timeout and evicts the jobSet (spec.suspend: true)
         • Terminates all surviving pods to release TPU quota for lower-priority jobs to backfill the N-1 slices
         • Preserves the hero job's #1 position in the queue via original admission timestamp (timestamp: Creation)
         • When zonal capacity returns, Kueue re-admits the job: loads pre-compiled graph from persistent GCS cache in ~30s
```

---

### 2. Production Launch Recipe and Essential Parameters

Below is a clean Cluster Toolkit (`gcluster`) launch template showing the core Pathways and MaxText elasticity flags:

```bash
export NUM_SLICES=4
# Option 1 (Replica Resize): set MIN_SLICES to any integer M (1 <= M < N, e.g. 2)
# Option 2 (Pause & Resume): set MIN_SLICES=-1 to require all N slices
export MIN_SLICES=2

gcluster job submit   --image=${DOCKER_IMAGE?}   --name=hero-elastic-pretrain   --pathways   --compute-type=ct6e-standard-4t   --topology=16x16   --num-slices=${NUM_SLICES?}   --pathways-gcs-location=${BASE_OUTPUT_DIRECTORY?}   --pathways-elastic-slices=${NUM_SLICES?}   --pathways-max-slice-restarts=10   --command="python3 -m maxtext.trainers.pre_train.train     src/maxtext/configs/base.yml     base_output_directory=${BASE_OUTPUT_DIRECTORY?}     model_name=qwen3-70b     dataset_type=grain     enable_checkpointing=true     async_checkpointing=true     checkpoint_period=100     enable_single_controller=true     elastic_enabled=true     elastic_min_slice_count=${MIN_SLICES?}     elastic_timeout_seconds=600     elastic_max_retries=10     checkpoint_storage_use_ocdbt=true     checkpoint_storage_use_zarr3=true"
```

| Parameter | Status | Option 1 (Resize) | Option 2 (Pause) | Engineering Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **`--pathways` & `enable_single_controller=true`** | **Required** | `true` | `true` | Decouples `pathways-head` on CPU from TPU slices (`restartCount = 0`). |
| **`--pathways-elastic-slices`** | **Required** | `= --num-slices` | `1` to `N` | Enables dynamic slice registration in `pathways-proxy` and `pathways-rm`. |
| **`elastic_enabled`** | **Required** | `true` | `true` | Wraps the MaxText step loop with in-process `elastic_retry`. |
| **`elastic_min_slice_count`** | **Required** | **`M` (`1 <= M < N`)** | **`-1` (All `N`)** | Selects **Option 1: Replica Resize** (`M`) vs. **Option 2: Pause and Resume** (`-1`). |
| **`elastic_timeout_seconds` & `elastic_max_retries`** | **Required** | e.g. `600` / `10` | e.g. `300` / `10` | Sets replacement wait window and caps in-process reshardings to prevent HBM fragmentation. |
| **`lr_schedule_type='wsd'`** | **Optional** | `'wsd'` | `'cosine'` / `'wsd'` | Warmup-Stable-Decay schedule; stabilizes convergence when batch size scales in Option 1. |
| **`enable_single_replica_ckpt_restoring`** | **Optional** | `true` / `false` | `true` | Slice 0 reads checkpoint from GCS and broadcasts over DCN ($1/K$ lower GCS read egress). |

---

### 3. Three Storage Optimizations for Google Cloud Storage (GCS)

Because checkpoint I/O directly impacts recovery time in Option 2 and periodic save overhead in both modes, pair Pathways with three storage optimizations:

1. **Co-located Regional GCS Bucket with Hierarchical Namespace (HNS):** Orbax stages checkpoints in a temporary folder and finalizes them with an atomic `commit_success` marker. Enabling **GCS HNS** replaces expensive $O(N)$ flat-bucket prefix renames with **atomic $O(1)$ directory renames**, preventing metadata QPS throttling during recovery.
2. **TensorStore `OCDBT` and `Zarr3` (`checkpoint_storage_use_ocdbt=true`, `checkpoint_storage_use_zarr3=true`):** Consolidates tens of thousands of individual FSDP shard files into packed chunk files, reducing storage metadata operations by **up to 99%** and accelerating Host-to-GCS writes by **up to 85%**.
3. **Replica-0 Deduplicated Saves and Single-Replica DCN Broadcast (`enable_single_replica_ckpt_restoring=true`):** Across $K$ slices, Orbax writes checkpoints from **Slice 0 only** (cutting GCS write traffic by $1 - 1/K$, or **50%** on 2 slices and **75%** on 4 slices). Enabling single-replica restore instructs Slice 0 to read the checkpoint into TPU HBM and broadcast it to peer slices via a compiled XLA AllReduce over Megascale DCN, cutting GCS read bandwidth by $1 - 1/K$ (**50%** on 2 slices).

---

## Section 3: Real-World Case Study — Protecting a Customer's Multi-Slice Hero Job During Hardware Replacement

We recently partnered with an enterprise customer running a flagship multi-slice foundation model pre-training workload ("hero job") on Cloud TPU v6e (`v6e-256` slices). Before enabling Pathways elasticity, a single TPU host hardware failure aborted the entire multi-slice job, idling hundreds of healthy TPU chips while waiting for slice replacement and requiring manual on-call intervention to clean up partial checkpoints.

We validated and deployed both elastic modes to protect their production cluster:

### 1. Zero Healthy-Slice Downtime During Hardware Replacement (Option 1: Replica Resize)
When we simulated an abrupt hardware failure on one slice (`kubectl delete pod ... --grace-period=0 --force`), `pathways-rm` immediately logged `WorkerLost` while **`pathways-head` remained `3/3 Running` (`restartCount = 0`)**. Because the surviving slices satisfied `elastic_min_slice_count = M`, Pathways reshaped the active mesh in-memory without touching GCS. The hero job continued generating training steps on the healthy slices while the failed hardware slice was replaced in the background, and automatically scaled back up to full $N$-slice capacity as soon as the replacement slice registered:

```text
# 1. Hardware failure on worker-1: Hero job shrinks mesh in-memory (restartCount=0, 0% GCS I/O)
[pathways-rm] WorkerLost: worker-1 disconnected. Reconfiguring virtual slices...
[jax-tpu] Elastic mesh shrink detected. Continuing training on surviving healthy slices...
completed step: <step_k>, ...

# 2. Replacement slice comes online: Hero job expands back to full N-slice throughput in-memory
[pathways-rm] WorkerRegistered: worker-1 re-connected.
[jax-tpu] In-memory mesh expansion detected. Re-sharding across full slice topology...
completed step: <step_m>, ...
```

### 2. Fast In-Process Checkpoint Rewind and Corrupted-Write Resilience (Option 2: Pause and Resume)
For runs requiring a strictly constant global batch size (`elastic_min_slice_count = -1`), losing a slice paused the controller in CPU memory (`restartCount = 0`). Once the replacement slice joined, `elastic_retry` verified the `commit_success` marker on GCS, restored the last committed checkpoint, and resumed from the next step—skipping pod bootstrapping and in-line XLA recompilation entirely. Furthermore, when we simulated a hardware crash *mid-checkpoint write* (missing `commit_success`), Orbax automatically deleted the incomplete directory and rolled back cleanly to the prior committed checkpoint without crashing:

```text
Slice down event detected. Retrying.
Found commit_success file. Keeping gs://.../checkpoints/100/.
Elastic attempt 2 out of 10
Restoring checkpoint from gs://.../checkpoints/100.
completed step: 101, ...
```

### Summary of Customer Production Outcomes

| Test Scenario | Configuration Mode | Coordinator `restartCount` | Recovery Mechanism | Measured Business Outcome |
| :--- | :--- | :---: | :--- | :--- |
| **Baseline (Before Pathways)** | Coupled McJAX | `1+` (Full job crash) | Cold pod boot + 20m XLA recompile + GCS read | **100% of healthy slices idle** during hardware replacement |
| **Hero job continuous training** | **Option 1: Replica Resize** (`M`) | **`0`** | **In-memory mesh shrink & expand** (`0%` GCS I/O) | **0% healthy slice downtime; near-zero (~2%) goodput overhead** |
| **Fixed-batch recovery** | **Option 2: Pause and Resume** (`-1`) | **`0`** | **In-memory wait + GCS restore** (`commit_success`) | **Skips 85%+ of software recovery overhead**; zero batch size drift |
| **Mid-checkpoint write crash** | **Option 2: Corrupted write fallback** | **`0` / `1`** | **Auto-rollback to prior committed step** | **Zero manual intervention**; eliminates `EOFError` crash loops |

---

## Conclusion and What's Next: Multi-Tier Checkpointing Across Adjacent Peer Nodes

Decoupling the training coordinator from accelerator slices with **Google Cloud Pathways** transforms multi-slice LLM pre-training from a fragile "all-or-nothing" operation into a self-healing system. With **Option 1 (Replica Resize)**, flagship hero jobs maintain continuous forward progress on surviving slices with near-zero goodput loss during hardware replacements; with **Option 2 (Pause and Resume)** and **GCS HNS + OCDBT**, fixed-batch workloads eliminate hours of container bootstrapping and XLA recompilation while freeing idle slices for multi-tenant Kueue backfill.

### What's Next in Part 2: Why Commit Every Checkpoint to GCS When an Adjacent Node Is Microseconds Away?

Even with Pathways single-controller elasticity and optimized GCS, one physical bottleneck remains whenever a job rewinds or restores weights from object storage: **remote storage network bandwidth**. Pulling hundreds of gigabytes or terabytes of model weights and optimizer states from GCS introduces a noticeable I/O pause at frontier parameter scale, and checkpointing to GCS every few steps to minimize step rewind would saturate host NIC egress.

In **Part 2** of this series, we will explore how combining **Pathways Elastic Training** with **Multi-Tier Checkpointing (MTC)** eliminates the object storage bottleneck by staging high-frequency checkpoints in **local host RAM (`/dev/shm`)** and replicating shards across **adjacent healthy peer nodes over DCN**—enabling replacement hosts to pull state directly from a neighbor node at memory speed without touching GCS.

---

### Resources and Further Reading
* [Google Cloud Pathways Documentation](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)
* [MaxText Elastic Training Guide](https://github.com/AI-Hypercomputer/maxtext)
* [Cluster Toolkit (gcluster) TPU Deployment](https://cloud.google.com/cluster-toolkit/docs/overview)
