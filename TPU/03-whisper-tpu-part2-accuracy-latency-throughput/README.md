# Unlocking Supersonic Speech Recognition: Part 2 — Balancing Accuracy, Latency, and Throughput on TPUs

> **Author:** [Bharath Rathinam](https://github.com/engineerbharath12) · **Published:** Fri, 26 Dec 2025 12:06:46 GMT  
> **Track:** TPU Supercomputing & Systems Architecture

---

### Unlocking Supersonic Speech Recognition: Part 2 — Balancing Accuracy, Latency, and Throughput on TPUs

In [Part 1 of this series](https://medium.com/@engineerbharath/unlocking-supersonic-speech-recognition-a-deep-dive-into-optimizing-whisper-on-tpus-6b8d734993d9), we achieved a massive milestone: optimizing OpenAI’s Whisper model on Google Cloud TPUs (v5e/v6e) to reach staggering throughput numbers (RTFx). We proved that JAX, combined with TPUs, could transcribe audio at supersonic speeds.

But after publishing, a common theme emerged in the community feedback. We kept hearing the same questions: *“Sure, it’s fast, but is it accurate?”* *“Does the speed-up hurt the transcription quality?”* *“What about real-time latency for online users?”*

Throughput (how much audio you can process per hour) is king for batch jobs, but **Latency** (how fast you get a response) and **Word Error Rate (WER)** (how correct that response is) are the gatekeepers for production.

In Part 2, we dive deep into the new benchmarks added to the [Whisper-on-TPU repository](https://github.com/engineerbharath12/whisper-tpu-google), specifically focusing on measuring Quality (WER) and Responsiveness (E2E Latency).

### 1. The Quality Metric: Word Error Rate (WER)

Speed is meaningless if the output is gibberish. To ensure our optimizations didn’t break the model, we implemented a robust WER benchmarking suite.

### How We Measure It

We use the industry-standard JiWER library. However, comparing raw text is tricky. If the model outputs “Hello, world!” and the ground truth is “hello world”, a naive comparison sees 2 errors (punctuation and capitalization).

Our implementation normalizes the text before comparison:

* **Lowercasing:** Hello -> hello
* **Punctuation Removal:** world! -> world
* **Whitespace Standardization:** Removing double spaces.

This ensures we are measuring *semantic* accuracy, not formatting preferences.

### The “Speed-Up” Trade-off

One of our key optimizations for throughput was **Audio Acceleration**. By speeding up the input audio (e.g., using ffmpeg filters) before feeding it to Whisper, we compress the time dimension, effectively allowing the model to process more content per second.

But does making everyone sound like a chipmunk hurt accuracy?

**The Verdict:** Surprisingly, no. We found that a speed factor of **2.0x to 2.2x** results in a negligible impact on WER.

* **Baseline (1.0x):** ~6–8% WER
* **Accelerated (2.2x):** ~9–10% WER

### The “10% Rule” and LLMs

Why is a slightly higher WER acceptable? In modern AI pipelines, ASR (Automatic Speech Recognition) is rarely the final step. The text is usually fed into an LLM for sentiment analysis, summarization, or RAG (Retrieval Augmented Generation).

LLMs are incredibly robust to phonetic errors. If Whisper transcribes *“The cat sat on the mat”* as *“The cat sat on the map”* due to audio compression, a downstream LLM processing the context often recovers the meaning or ignores the noise. As long as WER stays within the **10% range**, the utility of the transcription remains high, while the cost of inference drops by **50% or more** due to the 2.2x throughput boost.

### 2. The Responsiveness Metric: End-to-End (E2E) Latency

While throughput measures the size of the pipe, latency measures how fast the water flows through it. For a user waiting for a voice command response, **latency is the only metric that matters.**

### The Batching Dilemma

In Part 1, we maximized throughput by using massive batch sizes (e.g., batch_size=80). This saturates the TPU cores efficiently. However, for online inference, large batches introduce a phenomenon known as **Head-of-Line Blocking**.

Imagine a bus (Batch Size 80) waiting to fill up before leaving.

* **Request 1** arrives at t=0s.
* **Request 80** arrives at t=2s.
* The bus leaves at t=2s.
* Inference takes 1s.

For **Request 80**, latency is 1s (Processing). For **Request 1**, latency is 3s (2s Waiting + 1s Processing).

The second batch cannot start until the first batch is completely finished. In a streaming scenario, this wait time kills the user experience.

### Optimizing for Online Inference

To solve this, we introduced e2e_latency.py to benchmark distinct concurrency levels.

**Our findings for online processing:**

* **Smaller Batches:** Reducing batch size (e.g., to 4 or 8) drastically reduces the “wait time,” even if it slightly lowers overall TPU utilization.
* **Bucket Batching:** JAX requires static shapes for compilation. We use “Bucket Batching” (e.g., buckets of [4, 40, 80]).
* If 3 requests arrive, we pad to 4 and run immediately.
* If 50 requests arrive, we run a batch of 40 and a batch of 4 (padded to 40? No, padded to next bucket).

This approach — implemented in our pipeline_online.py — allows us to balance the raw power of the TPU with the responsiveness required for real-time apps.

### 3. Critical Fixes: Timestamps & Stride

One of the most complex challenges in long-form audio on TPUs is “Stitching.” When processing a 1-hour file, we chop it into 30-second chunks. If we just concatenate the text, we often lose words at the boundaries.

We utilize a **strided** approach (e.g., processing seconds 0–30, then 27–57). To make this work, we need accurate timestamps to identify the overlap.

**The Fix:** We discovered that without explicitly forcing return_timestamps=True in the Flax generation config, the striding logic would fail to align chunks, leading to high WER. We also implemented a custom **TPU-compatible feature extractor** (using Slaney normalization and clamping) to ensure the numerical precision on the TPU matched the CPU implementation exactly.

This ensures that our **2200x RTFx** speed on long audio doesn’t come at the cost of missing sentences.

### Conclusion

Optimizing for TPUs is a multi-dimensional puzzle.

* **Throughput:** Solved via JAX pmap, compilation, and Audio Speedup (2.2x).
* **Accuracy:** Verified via JiWER, proving 2.2x speedup is production-safe.
* **Latency:** Managed via Bucket Batching and intelligent queue management.

By adding these benchmarks to the repository, we’ve moved from a “proof of concept” speed demonstration to a production-ready ASR engine.

**Code & Benchmarks:** The full implementation, including the new WER and Latency scripts, is available in the dec26_wer+e2elatency branch of the repository.

[**Check out the GitHub Repo**](https://github.com/engineerbharath12/whisper-tpu-google)

---

### Links & References
* **Part of:** [Bharath's AI Infrastructure Playbook (BRs-AI-Infra-Playbook)](../../README.md)
