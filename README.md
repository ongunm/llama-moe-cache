## License
Dual licensed — AGPL v3 for open source use, commercial license available.  
See [COMMERCIAL.md](COMMERCIAL.md) for details.

Copyright (C) 2026 Ongun Manav



# FATE: Fast Expert Caching for MoE Inference in llama.cpp

A ~500-line C++ extension to [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds a GPU-resident expert cache with cross-layer and temporal predictive prefetching for Mixture-of-Experts models. Run models that exceed VRAM at up to 2x the speed of standard offloading — no training, no profiling, no special hardware.

## Files Changed

All FATE logic is contained in 5 files. The rest of the repository is unmodified llama.cpp.

| File | Status | Description |
|---|---|---|
| `src/llama-fate.h` | **New** | FATE system structures: GPU pool, prefetcher, prediction state |
| `src/llama-fate.cpp` | **New** | Core implementation: pool management, cross-layer + temporal prediction, prefetch engine, expert hook |
| `src/llama-context.cpp` | Modified | FATE initialization and expert hook registration |
| `ggml/src/ggml-backend.cpp` | Modified | Expert hook callback in the ggml scheduler — intercepts per-expert copies for caching |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Modified | CUDA prefetch stream, async H2D transfers, GPU-side barriers, pinned memory helpers |

Total addition: ~500 lines of C++. No new dependencies.

## Results

**Qwen3-30B-A3B Q4_K_M (18.6 GB model) on RTX 4070 Ti (12 GB VRAM)**

| | Vanilla Offloading | FATE |
|---|---|---|
| Generation speed | 33.74 t/s | **64.45 t/s** |
| Speedup | 1.0x | **1.91x** |
| Cache hit rate | — | **99.50%** |
| Cache hits / misses | — | 75,690 / 384 |
| Training required | — | **None** |

The model is 1.55x larger than VRAM. It does not fit. FATE runs it nearly twice as fast as vanilla llama.cpp's CPU offloading.

**Mixtral-8x7B-Instruct Q4_K_M (26.5 GB model) on RTX 4070 Ti (12 GB VRAM)**

| | FATE |
|---|---|
| Cache hit rate | **99.94%** |
| Hits / misses | 39,162 / 24 |

Tested across two different MoE architectures with zero code changes between them.

## Usage

Build llama.cpp as normal, then add `--fate` and `--fate-cache <MB>`:

```bash
./build/bin/llama-completion \
  -m model.gguf \
  --fate --fate-cache 2048 \
  -p "Your prompt here" \
  -n 100
```

`--fate-cache` sets the GPU pool size in MB. Set it as large as your free VRAM allows after model loading. The pool should ideally be larger than the per-token working set (see Scaling section below).

## How It Works

1. **GPU Pool**: A persistent VRAM buffer holds cached expert tensors. On a cache hit, the expert is copied via fast GPU-internal D2D transfer (~500 GB/s) instead of slow PCIe H2D (~25 GB/s). That is 20x faster per byte.

2. **Cross-Layer Prediction**: Experts activated at layer N predict which experts layer N+1 will need. The predicted experts are prefetched into the pool before they are accessed.

3. **Temporal Prediction**: Experts activated at layer N for the previous token predict which experts layer N will need for the current token. MoE routing is temporally stable — the same experts tend to fire across consecutive tokens.

4. **Prefetch Stream**: Predicted experts are transferred from CPU to GPU on a dedicated CUDA stream, overlapping with GPU compute on the main stream. A GPU-side barrier ensures data is ready before the main stream reads it.

No training. No offline profiling. No learned predictors. The prediction is a zero-shot heuristic based on the structural properties of MoE routing.

## Scaling: From 18 GB to 1 Trillion Parameters

Modern MoE models activate only a small fraction of their experts per token. The sparser the model, the smaller the working set that must fit in the GPU cache:

| Model | Total Params | Active / Token | Sparsity | Size (Q4) | Working Set |
|---|---|---|---|---|---|
| Mixtral-8x7B | 47B | 13B | 25% | 26 GB | 7.3 GB |
| Qwen3-30B-A3B | 30.5B | 3.3B | 6.25% | 18.6 GB | 1.4 GB |
| DeepSeek-V2 | 236B | 21B | 3.75% | ~120 GB | ~4 GB |
| DeepSeek-V3 | 671B | 37B | 3.1% | ~350 GB | ~11 GB |

FATE requires three things to fit in VRAM: non-expert weights, KV cache, and the expert pool (which must hold the working set). Everything else lives in system RAM. Total model size is bounded only by RAM.

**What this means in practice:**

- A **120 GB model** (DeepSeek-V2, GPT-4-class) needs ~12 GB VRAM + 128 GB RAM. A single consumer GPU.
- A **350 GB model** (DeepSeek-V3, state-of-the-art) needs ~24 GB VRAM + 384 GB RAM. A single workstation GPU.
- A **1 trillion+ parameter model** at 3% sparsity needs ~32 GB VRAM + 512 GB RAM. One workstation, not a cluster.

The VRAM requirement stays nearly flat as model size grows because the working set scales with active parameters (which stay small), not total parameters. The industry trend toward higher sparsity (256+ experts, top-8 routing) makes this approach more effective with every new model generation.

## Comparison to Existing Work

| System | Public Code | Hit Rate | Speedup | Prediction | Framework |
|---|---|---|---|---|---|
| FATE paper (Fang et al. 2025) | No | 99.08% | ~2-4x | Cross-layer gate (CPU) | Custom PyTorch |
| HOBBIT (Tang et al. 2024) | No | N/R | 9.93x | Stacked gating | llama.cpp (8K LoC, unreleased) |
| QuantumLeap/ExpertFlow | Yes | 75-85% | 2.3x | Unknown | llama.cpp fork |
| vLLM PR #37190 | Open PR | N/R | ~2x | None | vLLM (Python) |
| llama.cpp Issue #20757 | Request only | — | — | — | — |
| **This repo** | **Yes** | **99.50%** | **1.91x** | **Cross-layer + temporal** | **llama.cpp (~500 LoC)** |

This is the first public, working C++ implementation of predictive expert caching for llama.cpp.

## Known Limitations

- **Prompt evaluation is slower** (4 t/s vs 56 t/s vanilla). The expert hook processes experts individually during prefill, breaking batch copy optimization. Fix planned: bypass FATE during prefill.
- **mmap prevents pinned memory**. `cudaHostRegister` fails on mmap'd pages, forcing a staging copy. Using `--no-mmap` would enable direct async DMA (not yet implemented).
- **LRU eviction only**. ARC or LFRU would improve hit rates under memory pressure. Current 99.5% hit rate makes this low priority.

## Reports

- [FATE_RESULTS_QWEN3.md](FATE_RESULTS_QWEN3.md) — Benchmark results and analysis for Qwen3-30B-A3B
- [FATE_IMPLEMENTATION_REPORT.md](FATE_IMPLEMENTATION_REPORT.md) — Full technical report: architecture, root cause analysis, hypotheses, scaling model

## References

1. Fang, Z., et al. "FATE: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate." arXiv:2502.12224v2, 2025.
2. Tang, P., et al. "HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference." arXiv:2411.01433v2, 2024.
3. Zhu, S., et al. "Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models." arXiv:2511.10676v1, 2025.

## License

Dual licensed — AGPL v3 for open source use, commercial license available. See [COMMERCIAL.md](COMMERCIAL.md) for details.

FATE additions are Copyright (C) 2026 Ongun Manav. The underlying llama.cpp code retains its original MIT license.
